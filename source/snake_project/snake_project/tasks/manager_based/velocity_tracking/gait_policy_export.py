from __future__ import annotations

import copy
import math
import os

import torch


class SineGaitPolicyExporter(torch.nn.Module):
    """TorchScript wrapper that converts [frequency, bias] actor outputs to 7 joint actions."""

    def __init__(
        self,
        policy,
        normalizer=None,
        amplitude: float = 0.20,
        phase_lag: float = math.pi / 3.0,
        action_scale: float = 0.25,
        frequency_min: float = 0.0,
        moving_frequency_min: float = 0.1,
        frequency_max: float = 2.0,
        bias_max: float = 0.25,
        bias_gate_speed: float = 0.08,
        max_bias_rate: float = 0.15,
        policy_dt: float = 0.02,
        command_deadband: float = 0.03,
        moving_command_deadband: float = 0.03,
        positive_vx_phase_sign: float = 1.0,
        num_joints: int = 7,
    ):
        super().__init__()
        if getattr(policy, "is_recurrent", False):
            raise ValueError("SineGaitPolicyExporter currently supports non-recurrent actor policies only.")
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()

        self.amplitude = float(amplitude)
        self.phase_lag = float(abs(phase_lag))
        self.action_scale = float(action_scale)
        self.frequency_min = float(frequency_min)
        self.moving_frequency_min = float(moving_frequency_min)
        self.frequency_max = float(frequency_max)
        self.bias_max = float(bias_max)
        self.bias_gate_speed = float(bias_gate_speed)
        self.max_bias_rate = float(max_bias_rate)
        self.policy_dt = float(policy_dt)
        self.command_deadband = float(command_deadband)
        self.moving_command_deadband = float(moving_command_deadband)
        self.positive_vx_phase_sign = float(positive_vx_phase_sign)

        self.register_buffer("phase", torch.zeros(1, 1))
        self.register_buffer("bias", torch.zeros(1, 1))
        self.register_buffer("frequency", torch.zeros(1, 1))
        self.register_buffer("time", torch.zeros(1, 1))
        self.register_buffer("joint_index", torch.arange(num_joints, dtype=torch.float32).unsqueeze(0))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        if obs.shape[1] == 12:
            actor_obs = obs
        else:
            gait_params = torch.cat([self.frequency.expand(batch, 1), self.bias.expand(batch, 1)], dim=1)
            episode_time = self.time.expand(batch, 1)
            actor_obs = torch.cat([obs[:, :9], gait_params, episode_time], dim=1)
        raw_action = self.actor(self.normalizer(actor_obs))
        raw_frequency = raw_action[:, 0:1]
        raw_bias = raw_action[:, 1:2]

        cmd_vx = obs[:, 6:7]
        cmd_vy = obs[:, 7:8]
        command_speed = torch.norm(obs[:, 6:8], dim=1, keepdim=True)
        bias_gate = torch.clamp(command_speed / self.bias_gate_speed, min=0.0, max=1.0)
        bias_target = bias_gate * self.bias_max * torch.tanh(raw_bias)
        current_bias = self.bias.expand_as(bias_target)
        next_bias = current_bias + torch.clamp(
            bias_target - current_bias,
            min=-self.max_bias_rate * self.policy_dt,
            max=self.max_bias_rate * self.policy_dt,
        )
        self.bias[:] = next_bias[:1]

        frequency_min = torch.where(
            command_speed > self.moving_command_deadband,
            torch.ones_like(command_speed) * self.moving_frequency_min,
            torch.ones_like(command_speed) * self.frequency_min,
        )
        frequency_max = torch.ones_like(command_speed) * self.frequency_max
        moving_frequency = frequency_min + 0.5 * (torch.tanh(raw_frequency) + 1.0) * (frequency_max - frequency_min)
        frequency = torch.where(
            command_speed > self.moving_command_deadband,
            moving_frequency,
            torch.zeros_like(moving_frequency),
        )
        self.frequency[:] = frequency[:1]

        positive = torch.ones_like(cmd_vx) * self.positive_vx_phase_sign
        phase_sign = torch.where(
            cmd_vx > self.command_deadband,
            positive,
            torch.where(
                cmd_vx < -self.command_deadband,
                -positive,
                torch.where(torch.abs(cmd_vy) > self.command_deadband, positive, torch.zeros_like(cmd_vx)),
            ),
        )
        spatial_phase_lag = self.phase_lag * phase_sign

        next_phase = torch.remainder(self.phase + 2.0 * math.pi * frequency * self.policy_dt, 2.0 * math.pi)
        self.phase[:] = next_phase[:1]
        self.time[:] = self.time + self.policy_dt

        joint_action = next_bias + self.amplitude * torch.sin(next_phase + self.joint_index * spatial_phase_lag)
        return joint_action / self.action_scale

    @torch.jit.export
    def reset(self):
        self.phase[:] = 0.0
        self.bias[:] = 0.0
        self.frequency[:] = 0.0
        self.time[:] = 0.0


def export_sine_gait_policy_as_jit(policy, normalizer, action_cfg, path: str, filename: str = "policy.pt") -> None:
    exporter = SineGaitPolicyExporter(
        policy=policy,
        normalizer=normalizer,
        amplitude=action_cfg.amplitude,
        phase_lag=action_cfg.phase_lag,
        action_scale=action_cfg.action_scale,
        frequency_min=action_cfg.frequency_min,
        moving_frequency_min=action_cfg.moving_frequency_min,
        frequency_max=action_cfg.frequency_max,
        bias_max=action_cfg.bias_max,
        bias_gate_speed=action_cfg.bias_gate_speed,
        max_bias_rate=action_cfg.max_bias_rate,
        policy_dt=0.02,
        command_deadband=action_cfg.command_deadband,
        moving_command_deadband=action_cfg.moving_command_deadband,
        positive_vx_phase_sign=action_cfg.positive_vx_phase_sign,
        num_joints=len(action_cfg.joint_names),
    )
    os.makedirs(path, exist_ok=True)
    exporter.to("cpu")
    scripted = torch.jit.script(exporter)
    scripted.save(os.path.join(path, filename))
