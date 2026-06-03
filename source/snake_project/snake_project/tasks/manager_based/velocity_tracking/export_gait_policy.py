#!/usr/bin/env python3
"""Export a teacher-compatible JIT policy for the sine-gait action term.

The trained actor outputs [frequency, bias]. This exporter wraps that actor so
the saved JIT policy accepts the standard 30-D observation and returns the 7-D
joint action expected by the unmodified MuJoCo sim2sim scripts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = next((p for p in THIS_FILE.parents if (p / "scripts" / "rsl_rl" / "cli_args.py").exists()), None)
if REPO_ROOT is None:
    raise RuntimeError("Could not locate Snake_Project repository root.")

RSL_RL_SCRIPTS = REPO_ROOT / "scripts" / "rsl_rl"
LOCAL_SOURCE = REPO_ROOT / "source" / "snake_project"
if str(RSL_RL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPTS))
if str(LOCAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(LOCAL_SOURCE))

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Export a wrapped sine-gait JIT policy.")
parser.add_argument("--task", type=str, default="Snake-VelocityTracking-Flat-Play-v0", help="Task name.")
parser.add_argument("--output", type=str, default=None, help="Output JIT policy path. Defaults near the checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments used to instantiate the runner.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import snake_project.tasks  # noqa: F401
from snake_project.tasks.manager_based.velocity_tracking.gait_policy_export import export_sine_gait_policy_as_jit
from snake_project.tasks.manager_based.velocity_tracking.mdp.gait_actions import SineGaitAction


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.checkpoint is None:
        raise ValueError("Please provide an RSL-RL checkpoint with --checkpoint.")
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    output_path = args_cli.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(checkpoint_path), "exported", "policy.pt")
    output_path = os.path.abspath(output_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(checkpoint_path)

        policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        joint_pos_action = env.unwrapped.action_manager.get_term("joint_pos")
        if not isinstance(joint_pos_action, SineGaitAction):
            raise TypeError("The task action term 'joint_pos' is not SineGaitAction.")

        export_sine_gait_policy_as_jit(
            policy_nn,
            normalizer=normalizer,
            action_cfg=joint_pos_action.cfg,
            path=os.path.dirname(output_path),
            filename=os.path.basename(output_path),
        )

        scripted = torch.jit.load(output_path, map_location="cpu")
        test_out = scripted(torch.zeros(1, 30))
        if tuple(test_out.shape) != (1, len(joint_pos_action.cfg.joint_names)):
            raise RuntimeError(f"Unexpected wrapped policy output shape: {tuple(test_out.shape)}")
        print(f"[INFO] Exported teacher-compatible policy: {output_path}")
        print(f"[INFO] Shape check: obs(1, 30) -> action{tuple(test_out.shape)}")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
