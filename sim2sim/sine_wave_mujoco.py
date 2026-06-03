#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run an open-loop sine-wave yaw gait in MuJoCo and optionally record video."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import imageio
import mujoco
import numpy as np

try:
    import mujoco.viewer

    HAS_VIEWER = True
except Exception:
    HAS_VIEWER = False


def wrap_to_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def parse_bias_schedule(schedule: str) -> tuple[tuple[float, float], ...]:
    entries: list[tuple[float, float]] = []
    if not schedule:
        return tuple(entries)

    for raw_entry in schedule.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        try:
            time_s, bias = entry.split(":", maxsplit=1)
            time_s_f = float(time_s)
            bias_f = float(bias)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid bias schedule entry '{entry}'. Use comma-separated time:bias pairs, e.g. 0:0.1,3:0.0."
            ) from exc
        if time_s_f < 0.0:
            raise argparse.ArgumentTypeError(f"Bias schedule time must be non-negative, got {time_s_f}.")
        entries.append((time_s_f, bias_f))

    entries.sort(key=lambda item: item[0])
    return tuple(entries)


def scheduled_bias(sim_t: float, default_bias: float, schedule: tuple[tuple[float, float], ...]) -> float:
    current = float(default_bias)
    for start_s, bias in schedule:
        if sim_t + 1.0e-9 < start_s:
            break
        current = float(bias)
    return current


def command_theta_body(cmd_vx: float, cmd_vy: float, mode: str) -> float:
    cmd_vel = float(math.hypot(cmd_vx, cmd_vy))
    if cmd_vel <= 1.0e-6:
        return 0.0
    if mode == "body_lateral":
        return float(math.atan2(cmd_vy, abs(cmd_vx)))
    if mode == "velocity_vector":
        return float(math.atan2(cmd_vy, cmd_vx))
    raise ValueError(f"Unsupported heading command mode: {mode}")


def heading_gain_sign(cmd_vx: float, theta_source: str, theta_deadband_speed: float, reverse_heading_gain: bool) -> float:
    if reverse_heading_gain and theta_source == "heading" and cmd_vx < -theta_deadband_speed:
        return -1.0
    return 1.0


@dataclass
class SineWaveCfg:
    base_body_name: str = "base_link"
    controlled_joints: Tuple[str, ...] = ("yaw1", "yaw2", "yaw3", "yaw4", "yaw5", "yaw6", "yaw7")
    virtual_chassis_bodies: Tuple[str, ...] = (
        "base_link",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "link8",
        "link9",
        "link10",
        "link11",
        "link12",
        "link13",
        "link14",
    )
    action_scale: float = 0.25
    default_joint_angles: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class VirtualChassisState:
    origin_w: np.ndarray
    lin_vel_vc: np.ndarray
    ang_vel_vc: np.ndarray
    lin_vel_w: np.ndarray
    axes_w: np.ndarray
    heading_theta_w: float


class MujocoSineWaveRunner:
    def __init__(
        self,
        mjcf_path: str,
        cfg: SineWaveCfg,
        headless: bool,
        record_video: bool,
        video_path: str,
        video_fps: float,
    ):
        self.cfg = cfg
        self.headless = headless
        self.record_video = record_video
        self.video_path = video_path
        self.video_fps = video_fps

        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.mj_dt = float(self.model.opt.timestep)

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body_name)
        if self.base_body_id < 0:
            raise ValueError(f"Base body '{cfg.base_body_name}' not found in MJCF.")

        self.vc_body_ids = []
        for body_name in cfg.virtual_chassis_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"Virtual chassis body '{body_name}' not found in MJCF.")
            self.vc_body_ids.append(body_id)
        self.vc_body_ids = np.array(self.vc_body_ids, dtype=np.int32)
        self.prev_vc_axes_w = np.zeros((3, 3), dtype=np.float64)
        self.has_prev_vc_axes = False

        self.joint_ids = []
        self.qpos_adr = []
        self.qvel_adr = []
        for joint_name in cfg.controlled_joints:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' not found in MJCF.")
            self.joint_ids.append(joint_id)
            self.qpos_adr.append(int(self.model.jnt_qposadr[joint_id]))
            self.qvel_adr.append(int(self.model.jnt_dofadr[joint_id]))
        self.joint_ids = np.array(self.joint_ids, dtype=np.int32)
        self.qpos_adr = np.array(self.qpos_adr, dtype=np.int32)
        self.qvel_adr = np.array(self.qvel_adr, dtype=np.int32)

        self.act_ids = self._map_actuators_to_joints(self.joint_ids)
        self.ctrl_low, self.ctrl_high = self._get_ctrl_ranges(self.act_ids)
        self.q_default = np.array(cfg.default_joint_angles, dtype=np.float32)

        self.viewer = None
        self.renderer = None
        self._video_frames: List[np.ndarray] = []
        self._log_rows: List[dict[str, float]] = []
        self._last_cmd_vx = 0.0
        self._last_cmd_vy = 0.0
        self._last_cmd_vel = 0.0
        self._last_cmd_theta = 0.0
        self._last_cmd_theta_body = 0.0
        self._last_bias = 0.0
        self._last_frequency = 0.0
        self._theta_source = "vc_velocity"
        if self.record_video:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            print(f"[Runner] Video recording enabled, output: {self.video_path}")

        print("[Runner]")
        print(f"  MJCF             = {mjcf_path}")
        print(f"  MuJoCo dt        = {self.mj_dt:.6f}s")
        print(f"  Controlled joints= {cfg.controlled_joints}")
        print(f"  Actuator mapping = {list(zip(cfg.controlled_joints, self.act_ids.tolist()))}")

    def _map_actuators_to_joints(self, joint_ids: np.ndarray) -> np.ndarray:
        if self.model.nu <= 0:
            raise RuntimeError("Model has no actuators.")
        trnid = np.array(self.model.actuator_trnid, dtype=np.int32)
        joint_to_act = {}
        for act_id in range(self.model.nu):
            joint_id = int(trnid[act_id, 0])
            if joint_id >= 0:
                joint_to_act[joint_id] = act_id

        act_ids = []
        missing = []
        for joint_id in joint_ids:
            joint_id = int(joint_id)
            if joint_id not in joint_to_act:
                missing.append(joint_id)
            else:
                act_ids.append(joint_to_act[joint_id])

        if missing:
            names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or str(joint_id)
                for joint_id in missing
            ]
            raise RuntimeError(f"Cannot find actuators for joints: {names}")
        return np.array(act_ids, dtype=np.int32)

    def _get_ctrl_ranges(self, act_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ctrl_range = np.array(self.model.actuator_ctrlrange, dtype=np.float32)
        low = ctrl_range[act_ids, 0].copy()
        high = ctrl_range[act_ids, 1].copy()
        for index in range(len(low)):
            if np.isclose(low[index], 0.0) and np.isclose(high[index], 0.0):
                low[index], high[index] = -1.0e9, 1.0e9
        return low, high

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for index, qpos_adr in enumerate(self.qpos_adr):
            self.data.qpos[qpos_adr] = float(self.q_default[index])
        for qvel_adr in self.qvel_adr:
            self.data.qvel[qvel_adr] = 0.0
        self.has_prev_vc_axes = False
        self.prev_vc_axes_w[:] = 0.0
        self._log_rows.clear()
        self._last_cmd_vx = 0.0
        self._last_cmd_vy = 0.0
        self._last_cmd_vel = 0.0
        self._last_cmd_theta = 0.0
        self._last_cmd_theta_body = 0.0
        self._last_bias = 0.0
        self._last_frequency = 0.0
        self._theta_source = "vc_velocity"
        mujoco.mj_forward(self.model, self.data)

    def _apply_joint_targets(self, q_target: np.ndarray) -> None:
        q_target = np.clip(q_target, self.ctrl_low, self.ctrl_high).astype(np.float32)
        self.data.ctrl[self.act_ids] = q_target

    def _capture_video_frame(self) -> None:
        if not self.record_video:
            return
        self.renderer.update_scene(self.data)
        robot_pos = self.data.xpos[self.base_body_id].copy()
        cam_pos = robot_pos + np.array([-0.5, 0.0, 2.0])
        forward = np.array([0.0, 0.0, -1.0])
        up = np.array([0.0, 1.0, 0.0])
        for cam in self.renderer.scene.camera:
            cam.pos[:] = cam_pos
            cam.forward[:] = forward
            cam.up[:] = up
        self._video_frames.append(self.renderer.render())

    def _compute_virtual_chassis_state(self) -> VirtualChassisState:
        body_pos_w = np.array(self.data.xpos[self.vc_body_ids], dtype=np.float64)
        body_com_w = np.array(self.data.xipos[self.vc_body_ids], dtype=np.float64)
        body_cvel = np.array(self.data.cvel[self.vc_body_ids], dtype=np.float64)
        body_ang_vel_w = body_cvel[:, 0:3]
        body_lin_vel_w = body_cvel[:, 3:6] + np.cross(body_ang_vel_w, body_pos_w - body_com_w)

        origin_w = body_pos_w.mean(axis=0)
        centered = body_pos_w - origin_w
        axes_w, _, _ = np.linalg.svd(centered.T, full_matrices=False)

        if self.has_prev_vc_axes:
            dots = np.sum(axes_w * self.prev_vc_axes_w, axis=0)
            axes_w = axes_w * np.where(dots >= 0.0, 1.0, -1.0)
        else:
            head_to_tail_w = body_pos_w[-1] - body_pos_w[0]
            if float(np.dot(axes_w[:, 0], head_to_tail_w)) < 0.0:
                axes_w[:, 0] = -axes_w[:, 0]

            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if float(np.dot(axes_w[:, 2], world_up)) < 0.0:
                axes_w[:, 2] = -axes_w[:, 2]

        if np.linalg.det(axes_w) < 0.0:
            axes_w[:, 1] = -axes_w[:, 1]

        vc_lin_vel_w = body_lin_vel_w.mean(axis=0)
        vc_ang_vel_w = body_ang_vel_w.mean(axis=0)
        lin_vel_vc = axes_w.T @ vc_lin_vel_w
        ang_vel_vc = axes_w.T @ vc_ang_vel_w
        heading_theta_w = float(math.atan2(axes_w[1, 0], axes_w[0, 0]))

        self.prev_vc_axes_w = axes_w
        self.has_prev_vc_axes = True
        return VirtualChassisState(
            origin_w=origin_w,
            lin_vel_vc=lin_vel_vc,
            ang_vel_vc=ang_vel_vc,
            lin_vel_w=vc_lin_vel_w,
            axes_w=axes_w,
            heading_theta_w=heading_theta_w,
        )

    @staticmethod
    def _theta_from_state(state: VirtualChassisState, theta_source: str, deadband_speed: float = 1.0e-6) -> float:
        if theta_source == "heading":
            return state.heading_theta_w

        vc_vel = float(np.linalg.norm(state.lin_vel_vc[:2]))
        if vc_vel < deadband_speed:
            return 0.0
        return float(math.atan2(state.lin_vel_vc[1], state.lin_vel_vc[0]))

    def _log_step(self, sim_t: float) -> None:
        state = self._compute_virtual_chassis_state()
        origin_w = state.origin_w
        lin_vel_vc = state.lin_vel_vc
        ang_vel_z_vc = float(state.ang_vel_vc[2])
        base_pos_w = np.array(self.data.xpos[self.base_body_id], dtype=np.float64)
        base_vel_w = np.array(self.data.qvel[0:3], dtype=np.float64)
        vc_vel = float(np.linalg.norm(lin_vel_vc[:2]))
        vc_theta = float(math.atan2(lin_vel_vc[1], lin_vel_vc[0])) if vc_vel > 1.0e-6 else 0.0
        world_vel = float(np.linalg.norm(state.lin_vel_w[:2]))
        world_vel_theta = float(math.atan2(state.lin_vel_w[1], state.lin_vel_w[0])) if world_vel > 1.0e-6 else 0.0
        theta_feedback = self._theta_from_state(state, self._theta_source)
        self._log_rows.append(
            {
                "time": float(sim_t),
                "cmd_vx": float(self._last_cmd_vx),
                "cmd_vy": float(self._last_cmd_vy),
                "cmd_vel": float(self._last_cmd_vel),
                "cmd_theta": float(self._last_cmd_theta),
                "cmd_theta_body": float(self._last_cmd_theta_body),
                "theta_feedback": float(theta_feedback),
                "theta_source": self._theta_source,
                "vc_x": float(origin_w[0]),
                "vc_y": float(origin_w[1]),
                "vc_vx": float(lin_vel_vc[0]),
                "vc_vy": float(lin_vel_vc[1]),
                "vc_vel": float(vc_vel),
                "vc_theta": float(vc_theta),
                "vc_heading_theta_w": float(state.heading_theta_w),
                "world_vel_theta": float(world_vel_theta),
                "world_vx": float(state.lin_vel_w[0]),
                "world_vy": float(state.lin_vel_w[1]),
                "vc_wz": float(ang_vel_z_vc),
                "bias": float(self._last_bias),
                "frequency": float(self._last_frequency),
                "base_x": float(base_pos_w[0]),
                "base_y": float(base_pos_w[1]),
                "base_vx_w": float(base_vel_w[0]),
                "base_vy_w": float(base_vel_w[1]),
                "base_vz_w": float(base_vel_w[2]),
            }
        )

    def run(
        self,
        seconds: float,
        amplitude: float,
        frequency: float,
        phase_lag: float,
        auto_phase_lag: bool,
        positive_vx_phase_sign: float,
        bias: float,
        bias_schedule: tuple[tuple[float, float], ...],
        controller: bool,
        cmd_vx: float,
        cmd_vy: float,
        control_dt: float,
        controller_start: float,
        theta_kp: float,
        theta_kd: float,
        heading_command_mode: str,
        reverse_heading_gain: bool,
        speed_kp: float,
        speed_kd: float,
        theta_source: str,
        min_frequency: float,
        max_frequency: float,
        max_bias: float,
        max_bias_rate: float,
        theta_speed_gate: float,
        theta_filter_tau: float,
        theta_deadband_speed: float,
        log_path: str | None,
        plot_path: str | None,
        theta_plot_path: str | None,
        log_warmup: float,
        realtime: bool,
        realtime_factor: float,
        lead: float,
    ) -> None:
        self.reset()

        if (not self.headless) and HAS_VIEWER:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        steps = int(seconds / self.mj_dt)
        frame_interval = max(1, int(round(1.0 / (self.video_fps * self.mj_dt))))
        control_interval = max(1, int(round(control_dt / self.mj_dt)))
        joint_index = np.arange(len(self.cfg.controlled_joints), dtype=np.float32)
        t_wall0 = time.perf_counter()
        t0 = time.time()
        sim_t = 0.0
        oscillator_phase = 0.0
        current_bias = scheduled_bias(0.0, float(bias), bias_schedule)
        current_frequency = float(frequency)
        prev_theta_error = 0.0
        prev_speed_error = 0.0
        filtered_theta_error = 0.0
        cmd_vel = float(math.hypot(cmd_vx, cmd_vy))
        cmd_theta_body = command_theta_body(cmd_vx, cmd_vy, heading_command_mode)
        base_phase_lag = abs(float(phase_lag))
        current_phase_lag = float(phase_lag)
        if auto_phase_lag:
            if cmd_vx > theta_deadband_speed:
                current_phase_lag = float(np.sign(positive_vx_phase_sign) * base_phase_lag)
            elif cmd_vx < -theta_deadband_speed:
                current_phase_lag = float(-np.sign(positive_vx_phase_sign) * base_phase_lag)
            else:
                current_phase_lag = 0.0
        self._last_cmd_vx = float(cmd_vx)
        self._last_cmd_vy = float(cmd_vy)
        self._last_cmd_vel = cmd_vel
        self._last_cmd_theta_body = cmd_theta_body
        self._last_bias = current_bias
        self._last_frequency = current_frequency
        self._theta_source = theta_source

        initial_state = self._compute_virtual_chassis_state()
        if theta_source == "heading":
            cmd_theta = wrap_to_pi(initial_state.heading_theta_w + cmd_theta_body)
        else:
            cmd_theta = cmd_theta_body
        self._last_cmd_theta = cmd_theta

        print(
            f"[Sine] amplitude={amplitude:.3f} rad, frequency={frequency:.3f} Hz, "
            f"phase_lag={current_phase_lag:+.6f} rad, bias={current_bias:+.3f} rad"
        )
        if bias_schedule:
            print(f"[Sine] bias_schedule={list(bias_schedule)}")
        if controller:
            print(
                f"[Controller] cmd_vx={cmd_vx:+.3f}, cmd_vy={cmd_vy:+.3f}, "
                f"cmd_vel={cmd_vel:.3f}, cmd_theta={cmd_theta:+.3f}, theta_source={theta_source}"
            )

        for step in range(steps):
            if bias_schedule and not controller:
                current_bias = scheduled_bias(sim_t, float(bias), bias_schedule)

            if controller and sim_t >= controller_start and step % control_interval == 0:
                state = self._compute_virtual_chassis_state()
                lin_vel_vc = state.lin_vel_vc
                vc_vel = float(np.linalg.norm(lin_vel_vc[:2]))
                theta_feedback = self._theta_from_state(state, theta_source, theta_deadband_speed)

                theta_error = wrap_to_pi(cmd_theta - theta_feedback) if cmd_vel >= theta_deadband_speed else 0.0
                speed_error = cmd_vel - vc_vel
                if theta_filter_tau > 0.0:
                    alpha = control_dt / (theta_filter_tau + control_dt)
                    filtered_theta_error = wrap_to_pi(
                        filtered_theta_error + alpha * wrap_to_pi(theta_error - filtered_theta_error)
                    )
                else:
                    filtered_theta_error = theta_error
                theta_derivative = (theta_error - prev_theta_error) / max(control_dt, 1.0e-6)
                speed_derivative = (speed_error - prev_speed_error) / max(control_dt, 1.0e-6)
                prev_theta_error = theta_error
                prev_speed_error = speed_error

                gate = np.clip((vc_vel - theta_deadband_speed) / max(theta_speed_gate - theta_deadband_speed, 1.0e-6), 0.0, 1.0)
                bias_target = heading_gain_sign(
                    cmd_vx, theta_source, theta_deadband_speed, reverse_heading_gain
                ) * gate * (theta_kp * filtered_theta_error + theta_kd * theta_derivative)
                bias_target = float(np.clip(bias_target, -max_bias, max_bias))
                max_delta_bias = max_bias_rate * control_dt
                current_bias += float(np.clip(bias_target - current_bias, -max_delta_bias, max_delta_bias))
                current_frequency = np.clip(
                    frequency + speed_kp * speed_error + speed_kd * speed_derivative,
                    min_frequency,
                    max_frequency,
                )
                if cmd_vel < theta_deadband_speed:
                    current_frequency = 0.0
                    current_bias = 0.0
            self._last_bias = float(current_bias)
            self._last_frequency = float(current_frequency)

            oscillator_phase += 2.0 * math.pi * current_frequency * self.mj_dt
            q_target = self.q_default + current_bias + amplitude * np.sin(
                oscillator_phase + joint_index * current_phase_lag
            )
            self._apply_joint_targets(q_target)
            self._log_step(sim_t)
            mujoco.mj_step(self.model, self.data)

            if self.record_video and step % frame_interval == 0:
                self._capture_video_frame()

            sim_t += self.mj_dt

            if realtime:
                target_wall = sim_t / max(realtime_factor, 1.0e-6)
                wall_elapsed = time.perf_counter() - t_wall0
                sleep_s = target_wall - wall_elapsed - lead
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

            if self.viewer is not None:
                self.viewer.sync()

        if self.viewer is not None:
            self.viewer.close()

        wall = time.time() - t0
        print(f"[Done] Sim {seconds:.2f}s, wall {wall:.2f}s, RTF={seconds / max(wall, 1.0e-6):.2f}x")

        if self.record_video:
            self._save_video()
        self._print_velocity_summary(log_warmup)
        if log_path:
            self._save_csv(log_path)
        if plot_path:
            self._save_velocity_plot(plot_path)
        if theta_plot_path:
            self._save_theta_plot(theta_plot_path)

    def _save_video(self) -> None:
        os.makedirs(os.path.dirname(self.video_path) or ".", exist_ok=True)
        if not self._video_frames:
            print("[Video] No frames captured, skip saving.")
            return
        writer = imageio.get_writer(self.video_path, fps=self.video_fps, format="FFMPEG", codec="libx264")
        for frame in self._video_frames:
            writer.append_data(frame)
        writer.close()
        print(f"[Video] Saved {len(self._video_frames)} frames to {self.video_path}")

    def _print_velocity_summary(self, warmup: float) -> None:
        rows = [row for row in self._log_rows if row["time"] >= warmup]
        if not rows:
            rows = self._log_rows
        if not rows:
            print("[Velocity] No log rows collected.")
            return

        vc_vx = np.array([row["vc_vx"] for row in rows], dtype=np.float64)
        vc_vy = np.array([row["vc_vy"] for row in rows], dtype=np.float64)
        vc_wz = np.array([row["vc_wz"] for row in rows], dtype=np.float64)
        vc_vel = np.array([row["vc_vel"] for row in rows], dtype=np.float64)
        frequency = np.array([row["frequency"] for row in rows], dtype=np.float64)
        bias = np.array([row["bias"] for row in rows], dtype=np.float64)
        base_vx = np.array([row["base_vx_w"] for row in rows], dtype=np.float64)
        base_vy = np.array([row["base_vy_w"] for row in rows], dtype=np.float64)
        first = rows[0]
        last = rows[-1]
        duration = max(last["time"] - first["time"], 1.0e-6)
        print("[Velocity]")
        print(f"  warmup ignored       = {warmup:.3f}s")
        print(f"  mean vc_vx/vy/wz     = {vc_vx.mean():+.6f}, {vc_vy.mean():+.6f}, {vc_wz.mean():+.6f}")
        print(f"  mean vc_vel          = {vc_vel.mean():+.6f}")
        print(f"  mean frequency/bias  = {frequency.mean():+.6f}, {bias.mean():+.6f}")
        print(f"  mean world base vx/vy = {base_vx.mean():+.6f}, {base_vy.mean():+.6f}")
        print(f"  vc displacement x/y  = {last['vc_x'] - first['vc_x']:+.6f}, {last['vc_y'] - first['vc_y']:+.6f}")
        print(f"  vc displacement / s  = {(last['vc_x'] - first['vc_x']) / duration:+.6f}, {(last['vc_y'] - first['vc_y']) / duration:+.6f}")

    def _save_csv(self, log_path: str) -> None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if not self._log_rows:
            print("[CSV] No rows captured, skip saving.")
            return
        fieldnames = list(self._log_rows[0].keys())
        with open(log_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._log_rows)
        print(f"[CSV] Saved {len(self._log_rows)} rows to {log_path}")

    def _save_velocity_plot(self, plot_path: str) -> None:
        if not self._log_rows:
            print("[Plot] No rows captured, skip saving.")
            return

        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
        t = np.array([row["time"] for row in self._log_rows], dtype=np.float64)
        cmd_vx = np.array([row["cmd_vx"] for row in self._log_rows], dtype=np.float64)
        cmd_vy = np.array([row["cmd_vy"] for row in self._log_rows], dtype=np.float64)
        vx_key = "world_vx" if "world_vx" in self._log_rows[0] else "vc_vx"
        vy_key = "world_vy" if "world_vy" in self._log_rows[0] else "vc_vy"
        actual_vx = np.array([row[vx_key] for row in self._log_rows], dtype=np.float64)
        actual_vy = np.array([row[vy_key] for row in self._log_rows], dtype=np.float64)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, cmd_vx, label="cmd_vx", linewidth=1.8)
        axes[0].plot(t, actual_vx, label=vx_key, linewidth=1.2)
        axes[0].set_ylabel("vx (m/s)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(t, cmd_vy, label="cmd_vy", linewidth=1.8)
        axes[1].plot(t, actual_vy, label=vy_key, linewidth=1.2)
        axes[1].set_ylabel("vy (m/s)")
        axes[1].set_xlabel("time (s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"[Plot] Saved velocity tracking plot to {plot_path}")

    def _save_theta_plot(self, plot_path: str) -> None:
        if not self._log_rows:
            print("[Plot] No rows captured, skip saving theta plot.")
            return

        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
        t = np.array([row["time"] for row in self._log_rows], dtype=np.float64)
        cmd_theta = np.array([row["cmd_theta"] for row in self._log_rows], dtype=np.float64)
        feedback_key = "theta_feedback" if "theta_feedback" in self._log_rows[0] else "vc_theta"
        theta_feedback = np.array([row[feedback_key] for row in self._log_rows], dtype=np.float64)
        theta_error = np.array(
            [wrap_to_pi(cmd - actual) for cmd, actual in zip(cmd_theta, theta_feedback)], dtype=np.float64
        )
        theta_feedback_plot = cmd_theta - theta_error

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, cmd_theta, label="cmd_theta", linewidth=1.8)
        axes[0].plot(t, theta_feedback_plot, label=feedback_key, linewidth=1.2)
        axes[0].set_ylabel("theta (rad)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(t, theta_error, label="theta_error", linewidth=1.2)
        axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axes[1].set_ylabel("error (rad)")
        axes[1].set_xlabel("time (s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"[Plot] Saved theta tracking plot to {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mjcf",
        type=str,
        default="source/snake_project/snake_project/data/Snake/14DOF-DW.xml",
        help="Path to MuJoCo MJCF.",
    )
    parser.add_argument("--headless", type=int, default=0, help="1=headless, 0=viewer")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--amplitude", type=float, default=0.20, help="Yaw sine amplitude in radians.")
    parser.add_argument("--frequency", type=float, default=0.8, help="Sine frequency in Hz.")
    parser.add_argument("--phase_lag", type=float, default=math.pi / 3.0, help="Joint-position spatial phase lag.")
    parser.add_argument("--auto_phase_lag", type=int, default=0, help="1=flip phase_lag sign from cmd_vx, 0=use fixed sign.")
    parser.add_argument(
        "--positive_vx_phase_sign",
        type=float,
        default=1.0,
        help="Phase-lag sign used when cmd_vx is positive and --auto_phase_lag=1.",
    )
    parser.add_argument("--bias", type=float, default=0.0, help="Constant yaw offset added to all controlled joints.")
    parser.add_argument(
        "--bias_schedule",
        type=parse_bias_schedule,
        default=tuple(),
        help="Optional comma-separated open-loop bias schedule as time:bias pairs, e.g. 0:0.1,3:0.0.",
    )
    parser.add_argument("--controller", type=int, default=0, help="1=closed-loop polar velocity controller, 0=open-loop.")
    parser.add_argument("--cmd_vx", type=float, default=0.2, help="Velocity command x in virtual-chassis frame.")
    parser.add_argument("--cmd_vy", type=float, default=0.0, help="Velocity command y in virtual-chassis frame.")
    parser.add_argument("--control_dt", type=float, default=0.02, help="Controller update period in seconds.")
    parser.add_argument("--controller_start", type=float, default=0.5, help="Seconds to run open-loop before enabling control.")
    parser.add_argument("--theta_kp", type=float, default=0.4, help="Direction P gain: theta error -> bias.")
    parser.add_argument("--theta_kd", type=float, default=0.0, help="Direction D gain: theta error derivative -> bias.")
    parser.add_argument(
        "--heading_command_mode",
        type=str,
        choices=("body_lateral", "velocity_vector"),
        default="body_lateral",
        help=(
            "How heading mode maps (cmd_vx, cmd_vy) to heading offset. "
            "body_lateral treats negative vx as reverse motion without a 180-degree heading target."
        ),
    )
    parser.add_argument(
        "--reverse_heading_gain",
        type=int,
        default=1,
        help="Flip heading-control gain when cmd_vx is negative, preventing reverse commands from over-turning.",
    )
    parser.add_argument("--speed_kp", type=float, default=1.0, help="Speed P gain: speed error -> frequency.")
    parser.add_argument("--speed_kd", type=float, default=0.0, help="Speed D gain: speed error derivative -> frequency.")
    parser.add_argument(
        "--theta_source",
        type=str,
        choices=("vc_velocity", "heading"),
        default="vc_velocity",
        help="Theta feedback source: vc_velocity uses body-frame velocity angle, heading uses world-frame virtual-chassis heading.",
    )
    parser.add_argument("--min_frequency", type=float, default=0.0, help="Minimum controller frequency in Hz.")
    parser.add_argument("--max_frequency", type=float, default=2.0, help="Maximum controller frequency in Hz.")
    parser.add_argument("--max_bias", type=float, default=0.35, help="Maximum absolute controller bias in radians.")
    parser.add_argument("--max_bias_rate", type=float, default=0.25, help="Maximum bias slew rate in rad/s.")
    parser.add_argument("--theta_speed_gate", type=float, default=0.15, help="Speed where theta control reaches full strength.")
    parser.add_argument("--theta_filter_tau", type=float, default=0.4, help="Low-pass time constant for theta error in seconds.")
    parser.add_argument(
        "--theta_deadband_speed",
        type=float,
        default=0.03,
        help="Below this speed, direction angle is treated as unreliable.",
    )
    parser.add_argument("--log_path", type=str, default="", help="Optional CSV path for velocity logs.")
    parser.add_argument("--plot_path", type=str, default="", help="Optional PNG path for vx/vy tracking plot.")
    parser.add_argument("--theta_plot_path", type=str, default="", help="Optional PNG path for theta tracking plot.")
    parser.add_argument("--log_warmup", type=float, default=2.0, help="Seconds to ignore in printed velocity summary.")
    parser.add_argument("--realtime", type=int, default=1, help="1=pace to real-time, 0=run as fast as possible")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor.")
    parser.add_argument("--lead", type=float, default=0.001, help="Sleep lead margin in seconds.")
    parser.add_argument("--record_video", type=int, default=0, help="1=record simulation video, 0=no recording")
    parser.add_argument("--video_path", type=str, default="sim2sim/videos/sine_wave.mp4")
    parser.add_argument("--video_fps", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = MujocoSineWaveRunner(
        mjcf_path=args.mjcf,
        cfg=SineWaveCfg(),
        headless=bool(args.headless),
        record_video=bool(args.record_video),
        video_path=args.video_path,
        video_fps=args.video_fps,
    )
    runner.run(
        seconds=float(args.seconds),
        amplitude=float(args.amplitude),
        frequency=float(args.frequency),
        phase_lag=float(args.phase_lag),
        auto_phase_lag=bool(args.auto_phase_lag),
        positive_vx_phase_sign=float(args.positive_vx_phase_sign),
        bias=float(args.bias),
        bias_schedule=tuple(args.bias_schedule),
        controller=bool(args.controller),
        cmd_vx=float(args.cmd_vx),
        cmd_vy=float(args.cmd_vy),
        control_dt=float(args.control_dt),
        controller_start=float(args.controller_start),
        theta_kp=float(args.theta_kp),
        theta_kd=float(args.theta_kd),
        heading_command_mode=str(args.heading_command_mode),
        reverse_heading_gain=bool(args.reverse_heading_gain),
        speed_kp=float(args.speed_kp),
        speed_kd=float(args.speed_kd),
        theta_source=str(args.theta_source),
        min_frequency=float(args.min_frequency),
        max_frequency=float(args.max_frequency),
        max_bias=float(args.max_bias),
        max_bias_rate=float(args.max_bias_rate),
        theta_speed_gate=float(args.theta_speed_gate),
        theta_filter_tau=float(args.theta_filter_tau),
        theta_deadband_speed=float(args.theta_deadband_speed),
        log_path=str(args.log_path) or None,
        plot_path=str(args.plot_path) or None,
        theta_plot_path=str(args.theta_plot_path) or None,
        log_warmup=float(args.log_warmup),
        realtime=bool(args.realtime),
        realtime_factor=float(args.rtf),
        lead=float(args.lead),
    )


if __name__ == "__main__":
    main()
