from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class SineGaitAction(ActionTerm):
    """Action term that maps policy outputs [frequency, bias] to a sine yaw gait."""

    def __init__(self, cfg: "SineGaitActionCfg", env):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        self._raw_policy_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._raw_joint_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_joint_actions)
        self._phase = torch.zeros(self.num_envs, 1, device=self.device)
        self._joint_index = torch.arange(self._num_joints, device=self.device, dtype=torch.float32).unsqueeze(0)
        self._default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        # Keep the PolicyCfg last_actions term at 7 dimensions for README compatibility.
        return self._raw_joint_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_policy_actions[:] = actions

        raw_frequency = actions[:, 0:1]
        raw_bias = actions[:, 1:2]
        frequency = self.cfg.frequency_min + 0.5 * (torch.tanh(raw_frequency) + 1.0) * (
            self.cfg.frequency_max - self.cfg.frequency_min
        )
        bias = self.cfg.bias_max * torch.tanh(raw_bias)

        command_vx = self._env.command_manager.get_command(self.cfg.command_name)[:, 0:1]
        phase_sign = torch.where(
            command_vx > self.cfg.command_deadband,
            torch.ones_like(command_vx) * float(self.cfg.positive_vx_phase_sign),
            torch.where(
                command_vx < -self.cfg.command_deadband,
                torch.ones_like(command_vx) * -float(self.cfg.positive_vx_phase_sign),
                torch.zeros_like(command_vx),
            ),
        )
        spatial_phase_lag = abs(float(self.cfg.phase_lag)) * phase_sign

        self._phase = torch.remainder(
            self._phase + 2.0 * math.pi * frequency * float(self._env.step_dt),
            2.0 * math.pi,
        )
        q_offset = float(self.cfg.amplitude) * torch.sin(self._phase + self._joint_index * spatial_phase_lag) + bias
        q_target = self._default_joint_pos + q_offset
        q_target = torch.clamp(q_target, min=float(self.cfg.target_clip[0]), max=float(self.cfg.target_clip[1]))

        self._processed_actions = q_target
        self._raw_joint_actions = (q_target - self._default_joint_pos) / float(self.cfg.action_scale)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_policy_actions[env_ids] = 0.0
        self._raw_joint_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._default_joint_pos[env_ids]
        self._phase[env_ids] = 0.0


@configclass
class SineGaitActionCfg(ActionTermCfg):
    """Configuration for the two-parameter sine-gait action term."""

    class_type: type[ActionTerm] = SineGaitAction
    joint_names: list[str] = None
    preserve_order: bool = True
    action_scale: float = 0.25
    amplitude: float = 0.20
    phase_lag: float = math.pi / 3.0
    frequency_min: float = 0.0
    frequency_max: float = 2.0
    bias_max: float = 0.35
    command_name: str = "base_velocity"
    command_deadband: float = 0.03
    positive_vx_phase_sign: float = 1.0
    target_clip: tuple[float, float] = (-1.57, 1.57)
