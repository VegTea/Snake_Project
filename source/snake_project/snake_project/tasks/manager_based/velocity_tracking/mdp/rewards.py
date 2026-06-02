"""
最终奖励函数由以下 11 项加权求和构成：

 1. track_lin_vel_xy_exp      权重  5.0   虚拟底盘框架下平面线速度追踪奖励（指数核 - 线性惩罚）
 2. track_ang_vel_z_exp       权重  1.0   虚拟底盘框架下偏航角速度追踪奖励（指数核）
 3. ang_vel_xy_l2             权重 -0.05  对虚拟底盘xy轴角速度的L2惩罚
 4. joint_torques_l2          权重 -1e-4  对关节力矩的L2惩罚（节能正则）
 5. joint_acc_l2              权重 -2.5e-7 对关节加速度的L2惩罚（平滑正则）
 6. raw_action_rate           权重 -0.01  对原始动作变化速率的L2惩罚（动作平滑）
 7. joint_amplitude           权重  0.2   对关节持续运动幅度的奖励（鼓励持续运动）
 8. phase_propagation         权重  0.4   对相邻关节速度方向交替的奖励（相位传播/蜿蜒步态）
 9. motion_coordination       权重 -0.5   对所有关节同时同向运动的惩罚（抑制直线蠕动）
10. is_terminated             权重 -10.0  终止惩罚（激励存活）
11. contact_penalty           权重 -5.0   对虚拟底盘连杆接触地面的惩罚（抑制非蜿蜒接触）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .virtual_chassis import compute_virtual_chassis_command_terms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(num_envs: int, device: torch.device | str, env_ids) -> torch.Tensor | None:
    if env_ids is None or isinstance(env_ids, slice):
        return None
    if isinstance(env_ids, torch.Tensor):
        return env_ids
    return torch.tensor(env_ids, device=device, dtype=torch.long)


def joint_amplitude(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """对关节持续运动幅度的奖励：对主动关节速度绝对值取均值，鼓励持续运动而非静止。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.mean(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def motion_coordination(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """对所有关节同向弯曲/运动的惩罚：关节位置符号与速度符号越一致则惩罚值越大，抑制直线蠕动。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    pos_sign_mean = torch.abs(torch.mean(torch.sign(joint_pos), dim=1))
    vel_sign_mean = torch.abs(torch.mean(torch.sign(joint_vel), dim=1))
    return (pos_sign_mean + vel_sign_mean) / 2.0


def phase_propagation(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """对相邻关节速度方向交替（相位传播）的奖励：相邻关节速度方向相反时奖励高，鼓励蜿蜒步态。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    if joint_vel.shape[1] < 2:
        return torch.zeros(env.num_envs, device=env.device)

    vel_product = joint_vel[:, :-1] * joint_vel[:, 1:]
    vel_mag = torch.abs(joint_vel[:, :-1]) * torch.abs(joint_vel[:, 1:]) + 1.0e-6
    normalized_product = vel_product / vel_mag
    return -torch.mean(normalized_product, dim=1)


class RawActionRatePenalty(ManagerTermBase):
    """对原始动作变化速率的L2惩罚：对相邻时间步原始动作差值的平方和进行惩罚，平滑动作序列。"""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.prev_raw_action = None
        params = getattr(cfg, "params", {}) or {}
        self.rate_clip = float(params.get("rate_clip", 10.0))

    def reset(self, env_ids=None) -> dict[str, float]:
        if self.prev_raw_action is None:
            return {}
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_raw_action.zero_()
        else:
            self.prev_raw_action[env_ids] = 0.0
        return {}

    def __call__(self, env: "ManagerBasedRLEnv", action_term_name: str = "joint_pos") -> torch.Tensor:
        action_term = env.action_manager.get_term(action_term_name)
        raw_action = action_term.raw_actions
        if self.prev_raw_action is None:
            self.prev_raw_action = torch.zeros_like(raw_action)
        delta = raw_action - self.prev_raw_action
        if self.rate_clip > 0.0:
            delta = torch.clamp(delta, min=-self.rate_clip, max=self.rate_clip)
        self.prev_raw_action.copy_(raw_action)
        return torch.sum(torch.square(delta), dim=1)


class VirtualChassisTrackLinVelXYExp(ManagerTermBase):
    """对虚拟底盘平面线速度追踪的奖励：指数核奖励（exp(-error^2/std^2)）- 线性惩罚项，追踪速度指令。"""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> dict[str, float]:
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
        return {}

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str,
        std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        linear_coef: float = 0.0,
        reward_clip_min: float = -20.0,
    ) -> torch.Tensor:
        body_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]
        body_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, :]
        body_ang_vel_w = self.asset.data.body_ang_vel_w[:, self.asset_cfg.body_ids, :]

        if not (torch.isfinite(body_pos_w).all() and torch.isfinite(body_lin_vel_w).all() and torch.isfinite(body_ang_vel_w).all()):
            return torch.zeros(self.num_envs, device=self.device)

        _, axes_w, actual_lin_vel_vc, _ = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(actual_lin_vel_vc).all():
            return torch.zeros(self.num_envs, device=self.device)

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        lin_vel_error = torch.sum(
            torch.square(env.command_manager.get_command(command_name)[:, :2] - actual_lin_vel_vc[:, :2]),
            dim=1,
        )
        exp_reward = torch.exp(-lin_vel_error / std**2)
        lin_penalty = linear_coef * torch.sqrt(lin_vel_error)
        raw_reward = exp_reward - lin_penalty
        if reward_clip_min is not None:
            return torch.clamp(raw_reward, min=reward_clip_min)
        return raw_reward


class VirtualChassisTrackLinVelXExp(ManagerTermBase):
    """对虚拟底盘 x 方向线速度追踪的奖励：指数核奖励 - 线性惩罚项。"""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> dict[str, float]:
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
        return {}

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str,
        std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        linear_coef: float = 0.0,
        reward_clip_min: float = -20.0,
    ) -> torch.Tensor:
        body_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]
        body_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, :]
        body_ang_vel_w = self.asset.data.body_ang_vel_w[:, self.asset_cfg.body_ids, :]

        if not (torch.isfinite(body_pos_w).all() and torch.isfinite(body_lin_vel_w).all() and torch.isfinite(body_ang_vel_w).all()):
            return torch.zeros(self.num_envs, device=self.device)

        _, axes_w, actual_lin_vel_vc, _ = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(actual_lin_vel_vc).all():
            return torch.zeros(self.num_envs, device=self.device)

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - actual_lin_vel_vc[:, 0])
        exp_reward = torch.exp(-lin_vel_error / std**2)
        lin_penalty = linear_coef * torch.sqrt(lin_vel_error)
        raw_reward = exp_reward - lin_penalty
        if reward_clip_min is not None:
            return torch.clamp(raw_reward, min=reward_clip_min)
        return raw_reward


class VirtualChassisTrackLinVelYExp(ManagerTermBase):
    """对虚拟底盘 y 方向线速度追踪的奖励：指数核奖励 - 线性惩罚项。"""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> dict[str, float]:
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
        return {}

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str,
        std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        linear_coef: float = 0.0,
        reward_clip_min: float = -20.0,
    ) -> torch.Tensor:
        body_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]
        body_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, :]
        body_ang_vel_w = self.asset.data.body_ang_vel_w[:, self.asset_cfg.body_ids, :]

        if not (torch.isfinite(body_pos_w).all() and torch.isfinite(body_lin_vel_w).all() and torch.isfinite(body_ang_vel_w).all()):
            return torch.zeros(self.num_envs, device=self.device)

        _, axes_w, actual_lin_vel_vc, _ = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(actual_lin_vel_vc).all():
            return torch.zeros(self.num_envs, device=self.device)

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 1] - actual_lin_vel_vc[:, 1])
        exp_reward = torch.exp(-lin_vel_error / std**2)
        lin_penalty = linear_coef * torch.sqrt(lin_vel_error)
        raw_reward = exp_reward - lin_penalty
        if reward_clip_min is not None:
            return torch.clamp(raw_reward, min=reward_clip_min)
        return raw_reward


class VirtualChassisTrackAngVelZExp(ManagerTermBase):
    """对虚拟底盘偏航角速度追踪的奖励：指数核奖励 exp(-error^2/std^2)，追踪偏航角速度指令。"""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> dict[str, float]:
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
        return {}

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str,
        std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        body_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]
        body_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, :]
        body_ang_vel_w = self.asset.data.body_ang_vel_w[:, self.asset_cfg.body_ids, :]

        if not (torch.isfinite(body_pos_w).all() and torch.isfinite(body_lin_vel_w).all() and torch.isfinite(body_ang_vel_w).all()):
            return torch.zeros(self.num_envs, device=self.device)

        _, axes_w, _, actual_ang_vel_z_vc = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(actual_ang_vel_z_vc).all():
            return torch.zeros(self.num_envs, device=self.device)

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - actual_ang_vel_z_vc)
        return torch.exp(-ang_vel_error / std**2)


class VirtualChassisTrackPlanarVelL2(ManagerTermBase):
    """Negative planar velocity tracking error matching the evaluation MAE metric.

    Computes: -sqrt((cmd_vx - vc_vx)^2 + (cmd_vy - vc_vy)^2 + vc_wz^2)

    This replaces the exponential-kernel tracking rewards with a direct L2 error
    penalty that provides linear gradient at any tracking accuracy level,
    consistent with the sim2sim planner_MAE metric.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> dict[str, float]:
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
        return {}

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        soft_clamp: float = 2.0,
    ) -> torch.Tensor:
        body_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]
        body_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, :]
        body_ang_vel_w = self.asset.data.body_ang_vel_w[:, self.asset_cfg.body_ids, :]

        if not (torch.isfinite(body_pos_w).all() and torch.isfinite(body_lin_vel_w).all() and torch.isfinite(body_ang_vel_w).all()):
            return torch.zeros(self.num_envs, device=self.device)

        _, axes_w, actual_lin_vel_vc, actual_ang_vel_z_vc = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(actual_lin_vel_vc).all():
            return torch.zeros(self.num_envs, device=self.device)

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        cmd = env.command_manager.get_command(command_name)
        planar_error = torch.sqrt(
            torch.sum(torch.square(cmd[:, :2] - actual_lin_vel_vc[:, :2]), dim=1)
            + torch.square(actual_ang_vel_z_vc)
        )
        if soft_clamp is not None and soft_clamp > 0.0:
            planar_error = soft_clamp * torch.tanh(planar_error / soft_clamp)
        return -planar_error


class VirtualChassisAngVelZL2(ManagerTermBase):
    """Penalize z-axis angular velocity of the virtual chassis using L2 squared kernel.

    Since the command for ang_vel_z is always 0, this term directly penalizes
    unwanted yaw rotation of the virtual chassis.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> dict[str, float]:
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
        return {}

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        max_penalty: float = 10.0,
    ) -> torch.Tensor:
        body_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]
        body_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, :]
        body_ang_vel_w = self.asset.data.body_ang_vel_w[:, self.asset_cfg.body_ids, :]

        if not (torch.isfinite(body_pos_w).all() and torch.isfinite(body_lin_vel_w).all() and torch.isfinite(body_ang_vel_w).all()):
            return torch.zeros(self.num_envs, device=self.device)

        _, axes_w, _, actual_ang_vel_z_vc = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all():
            return torch.zeros(self.num_envs, device=self.device)

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        penalty = torch.square(actual_ang_vel_z_vc)
        if max_penalty is not None:
            return torch.clamp(penalty, max=max_penalty)
        return penalty


class VirtualChassisAngVelXYL2(ManagerTermBase):
    """对虚拟底盘xy轴角速度的L2惩罚：将各连杆平均角速度投影到虚拟底盘坐标系后取xy分量的L2范数平方。

    与内置的基于根连杆坐标系的惩罚不同，本项对所有虚拟底盘连杆取均值后投影，对蛇形运动更具物理意义。
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> dict[str, float]:
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
        return {}

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        max_penalty: float = 10.0,
    ) -> torch.Tensor:
        body_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]
        body_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, :]
        body_ang_vel_w = self.asset.data.body_ang_vel_w[:, self.asset_cfg.body_ids, :]

        if not (torch.isfinite(body_pos_w).all() and torch.isfinite(body_lin_vel_w).all() and torch.isfinite(body_ang_vel_w).all()):
            return torch.zeros(self.num_envs, device=self.device)

        _, axes_w, _, _ = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all():
            return torch.zeros(self.num_envs, device=self.device)

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        vc_ang_vel_w = body_ang_vel_w.mean(dim=1)
        ang_vel_vc = torch.einsum("bij,bj->bi", axes_w.transpose(1, 2), vc_ang_vel_w)
        raw_penalty = torch.sum(torch.square(ang_vel_vc[:, :2]), dim=1)
        if max_penalty is not None:
            return torch.clamp(raw_penalty, max=max_penalty)
        return raw_penalty


def contact_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.0,
) -> torch.Tensor:
    """对虚拟底盘连杆接触地面的惩罚：当指定连杆的接触力大于阈值时给予惩罚，抑制非蜿蜒运动相关的身体接触。"""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]
    forces_on_bodies = net_forces[:, sensor_cfg.body_ids, :]
    in_contact = torch.any(torch.norm(forces_on_bodies, dim=-1) > threshold, dim=1)
    return in_contact.float()
