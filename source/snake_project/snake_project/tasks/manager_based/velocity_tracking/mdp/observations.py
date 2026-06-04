from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import observations as builtin_obs
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def last_raw_actions(env: "ManagerBasedEnv", action_name: str = "joint_pos") -> torch.Tensor:
    return torch.nan_to_num(env.action_manager.get_term(action_name).raw_actions, nan=0.0, posinf=0.0, neginf=0.0)


def gait_parameters(env: "ManagerBasedEnv", action_name: str = "joint_pos") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    frequency = getattr(action_term, "current_frequency", None)
    bias = getattr(action_term, "current_bias", None)
    if frequency is None or bias is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    params = torch.cat([frequency, bias], dim=1)
    return torch.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)


def episode_time(env: "ManagerBasedEnv") -> torch.Tensor:
    time_s = env.episode_length_buf.to(dtype=torch.float32, device=env.device).unsqueeze(1) * float(env.step_dt)
    return torch.nan_to_num(time_s, nan=0.0, posinf=0.0, neginf=0.0)


def base_ang_vel(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.base_ang_vel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def projected_gravity(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.projected_gravity(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def joint_pos_rel(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.joint_pos_rel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def joint_vel_rel(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.joint_vel_rel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def generated_commands(env: "ManagerBasedEnv", command_name: str) -> torch.Tensor:
    return torch.nan_to_num(builtin_obs.generated_commands(env, command_name), nan=0.0, posinf=0.0, neginf=0.0)
