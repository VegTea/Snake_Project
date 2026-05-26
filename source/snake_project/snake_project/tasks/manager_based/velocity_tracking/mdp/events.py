from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_snake_state(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    joint_position_range: tuple[float, float] = (-0.05, 0.05),
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Reset the snake near its nominal state with legacy plane-task randomization."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    asset: Articulation = env.scene[asset_cfg.name]

    # 域随机化参数默认值：位姿/速度/关节位置均匀采样范围
    pose_range = pose_range or {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.141592653589793, 3.141592653589793)}
    velocity_range = velocity_range or {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
    }

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]

    # --- 域随机化：初始位姿 ---
    # x, y 位置在指定范围内均匀采样，模拟每次部署位置不同
    root_state[:, 0] += sample_uniform(*pose_range["x"], (env_ids.numel(),), device=env.device)
    root_state[:, 1] += sample_uniform(*pose_range["y"], (env_ids.numel(),), device=env.device)
    # yaw 朝向在指定范围内均匀采样，模拟每次部署朝向不同
    yaw = sample_uniform(*pose_range["yaw"], (env_ids.numel(),), device=env.device)
    root_state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

    # --- 域随机化：初始速度 ---
    # 线速度(x/y/z)和角速度(roll/pitch/yaw)在指定范围内均匀采样，模拟非静止初始状态
    root_state[:, 7] = sample_uniform(*velocity_range["x"], (env_ids.numel(),), device=env.device)
    root_state[:, 8] = sample_uniform(*velocity_range["y"], (env_ids.numel(),), device=env.device)
    root_state[:, 9] = sample_uniform(*velocity_range["z"], (env_ids.numel(),), device=env.device)
    root_state[:, 10] = sample_uniform(*velocity_range["roll"], (env_ids.numel(),), device=env.device)
    root_state[:, 11] = sample_uniform(*velocity_range["pitch"], (env_ids.numel(),), device=env.device)
    root_state[:, 12] = sample_uniform(*velocity_range["yaw"], (env_ids.numel(),), device=env.device)

    # --- 域随机化：关节角度扰动 ---
    # 在默认关节角度上叠加均匀噪声，模拟关节初始微小偏移
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(asset.data.default_joint_vel[env_ids])
    if asset_cfg.joint_ids == slice(None):
        randomized_joint_ids = torch.arange(asset.num_joints, device=env.device, dtype=torch.long)
    elif isinstance(asset_cfg.joint_ids, torch.Tensor):
        randomized_joint_ids = asset_cfg.joint_ids.to(device=env.device, dtype=torch.long)
    else:
        randomized_joint_ids = torch.tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)
    if randomized_joint_ids.numel() > 0:
        joint_pos[:, randomized_joint_ids] += sample_uniform(
            joint_position_range[0],
            joint_position_range[1],
            (env_ids.numel(), randomized_joint_ids.numel()),
            device=env.device,
        )

    # 将随机化后的状态写入仿真器
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
