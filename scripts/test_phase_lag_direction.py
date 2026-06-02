# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Test which adjacent-joint phase-lag sign produces positive virtual-chassis vx."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open-loop phase-lag direction test for the snake task.")
parser.add_argument("--task", type=str, default="Snake-VelocityTracking-Flat-Play-v0", help="Task name.")
parser.add_argument("--seconds", type=float, default=8.0, help="Seconds to simulate for each phase sign.")
parser.add_argument("--warmup", type=float, default=2.0, help="Seconds to ignore before averaging velocity.")
parser.add_argument("--amplitude", type=float, default=0.20, help="Yaw target amplitude in radians.")
parser.add_argument("--frequency", type=float, default=0.8, help="Sine-wave frequency in Hz.")
parser.add_argument("--target_phase_lag", type=float, default=math.pi / 3.0, help="Reward-frame phase lag to test.")
parser.add_argument("--action_scale", type=float, default=0.25, help="JointPositionAction scale from env config.")
parser.add_argument("--cmd_vx", type=float, default=0.2, help="Command vx value to stamp into the command term.")
parser.add_argument("--cmd_vy", type=float, default=0.0, help="Command vy value to stamp into the command term.")
parser.add_argument("--cmd_wz", type=float, default=0.0, help="Command wz value to stamp into the command term.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import snake_project.tasks  # noqa: F401


def _stamp_command(command_term, cmd_vx: float, cmd_vy: float, cmd_wz: float) -> None:
    env_ids = torch.arange(command_term.num_envs, device=command_term.device)
    command_term.vel_command_b[env_ids, 0] = cmd_vx
    command_term.vel_command_b[env_ids, 1] = cmd_vy
    command_term.vel_command_b[env_ids, 2] = cmd_wz


def _run_phase_test(env, command_term, reward_phase_lag: float) -> tuple[float, float]:
    base_env = env.unwrapped
    device = base_env.device
    num_actions = env.action_space.shape[-1]
    joint_ids = torch.arange(num_actions, device=device, dtype=torch.float32).unsqueeze(0)
    raw_amplitude = args_cli.amplitude / args_cli.action_scale
    omega = 2.0 * math.pi * args_cli.frequency
    dt = base_env.step_dt
    total_steps = int(args_cli.seconds / dt)
    warmup_steps = int(args_cli.warmup / dt)

    # reward phase = atan2(qdot, q), so a desired reward lag of +phi is produced
    # by a joint-position wave with spatial phase -phi.
    action_spatial_phase = -reward_phase_lag
    vx_samples = []
    origin_start_x = None
    origin_end_x = None

    env.reset()
    for step in range(total_steps):
        _stamp_command(command_term, args_cli.cmd_vx, args_cli.cmd_vy, args_cli.cmd_wz)
        t = step * dt
        actions = raw_amplitude * torch.sin(omega * t + joint_ids * action_spatial_phase)
        env.step(actions)
        origin_w, _, lin_vel_vc, _ = command_term._compute_virtual_state()
        if step == warmup_steps:
            origin_start_x = float(origin_w[0, 0].item())
        if step >= warmup_steps:
            vx_samples.append(float(lin_vel_vc[0, 0].item()))
            origin_end_x = float(origin_w[0, 0].item())

    mean_vx = sum(vx_samples) / max(len(vx_samples), 1)
    dx = 0.0 if origin_start_x is None or origin_end_x is None else origin_end_x - origin_start_x
    return mean_vx, dx


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    command_term = env.unwrapped.command_manager.get_term("base_velocity")

    positive_lag = abs(args_cli.target_phase_lag)
    results = []
    for reward_phase_lag in (positive_lag, -positive_lag):
        mean_vx, dx = _run_phase_test(env, command_term, reward_phase_lag)
        results.append((reward_phase_lag, mean_vx, dx))
        print(f"reward_phase_lag={reward_phase_lag:+.6f} mean_vx_vc={mean_vx:+.6f} dx_world={dx:+.6f}")

    best = max(results, key=lambda item: item[1])
    print(f"positive-vx phase sign candidate: {'+1.0' if best[0] > 0.0 else '-1.0'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
