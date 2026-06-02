#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run an open-loop sine-wave yaw gait in MuJoCo and optionally record video."""

from __future__ import annotations

import argparse
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


@dataclass
class SineWaveCfg:
    base_body_name: str = "base_link"
    controlled_joints: Tuple[str, ...] = ("yaw1", "yaw2", "yaw3", "yaw4", "yaw5", "yaw6", "yaw7")
    action_scale: float = 0.25
    default_joint_angles: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


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

    def run(
        self,
        seconds: float,
        amplitude: float,
        frequency: float,
        phase_lag: float,
        realtime: bool,
        realtime_factor: float,
        lead: float,
    ) -> None:
        self.reset()

        if (not self.headless) and HAS_VIEWER:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        steps = int(seconds / self.mj_dt)
        frame_interval = max(1, int(round(1.0 / (self.video_fps * self.mj_dt))))
        joint_index = np.arange(len(self.cfg.controlled_joints), dtype=np.float32)
        omega = 2.0 * math.pi * frequency
        t_wall0 = time.perf_counter()
        t0 = time.time()
        sim_t = 0.0

        print(
            f"[Sine] amplitude={amplitude:.3f} rad, frequency={frequency:.3f} Hz, "
            f"phase_lag={phase_lag:+.6f} rad"
        )

        for step in range(steps):
            q_target = self.q_default + amplitude * np.sin(omega * sim_t + joint_index * phase_lag)
            self._apply_joint_targets(q_target)
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
        realtime=bool(args.realtime),
        realtime_factor=float(args.rtf),
        lead=float(args.lead),
    )


if __name__ == "__main__":
    main()
