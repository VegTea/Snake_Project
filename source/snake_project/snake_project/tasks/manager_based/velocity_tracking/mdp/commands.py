from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .virtual_chassis import (
    arrow_quat_from_virtual_velocity,
    compute_virtual_chassis_command_terms,
    quat_from_axes_w,
)


class SnakeVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with legacy snake-task sampling semantics."""

    cfg: "SnakeVelocityCommandCfg"

    def __init__(self, cfg: "SnakeVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.current_lin_vel_x_range = list(cfg.ranges.lin_vel_x)
        self.current_lin_vel_y_range = list(cfg.ranges.lin_vel_y)

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.current_lin_vel_x_range)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.current_lin_vel_y_range)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        super()._update_command()
        small_planar = torch.norm(self.vel_command_b[:, :2], dim=1) <= self.cfg.planar_zero_threshold
        self.vel_command_b[small_planar, :2] = 0.0

    def expand_lin_vel_x(self, step_size: float, max_curriculum: float) -> tuple[float, float]:
        self.current_lin_vel_x_range[0] = round(max(self.current_lin_vel_x_range[0] - step_size, -max_curriculum), 4)
        self.current_lin_vel_x_range[1] = round(min(self.current_lin_vel_x_range[1] + step_size, max_curriculum), 4)
        self.cfg.ranges.lin_vel_x = tuple(self.current_lin_vel_x_range)
        return tuple(self.current_lin_vel_x_range)

    def expand_lin_vel_y(self, step_size: float, max_curriculum: float) -> tuple[float, float]:
        self.current_lin_vel_y_range[0] = round(max(self.current_lin_vel_y_range[0] - step_size, -max_curriculum), 4)
        self.current_lin_vel_y_range[1] = round(min(self.current_lin_vel_y_range[1] + step_size, max_curriculum), 4)
        self.cfg.ranges.lin_vel_y = tuple(self.current_lin_vel_y_range)
        return tuple(self.current_lin_vel_y_range)

    def shrink_lin_vel_x(self, step_size: float, min_curriculum: float) -> tuple[float, float]:
        self.current_lin_vel_x_range[0] = round(min(self.current_lin_vel_x_range[0] + step_size, -min_curriculum), 4)
        self.current_lin_vel_x_range[1] = round(max(self.current_lin_vel_x_range[1] - step_size, min_curriculum), 4)
        self.cfg.ranges.lin_vel_x = tuple(self.current_lin_vel_x_range)
        return tuple(self.current_lin_vel_x_range)

    def shrink_lin_vel_y(self, step_size: float, min_curriculum: float) -> tuple[float, float]:
        self.current_lin_vel_y_range[0] = round(min(self.current_lin_vel_y_range[0] + step_size, -min_curriculum), 4)
        self.current_lin_vel_y_range[1] = round(max(self.current_lin_vel_y_range[1] - step_size, min_curriculum), 4)
        self.cfg.ranges.lin_vel_y = tuple(self.current_lin_vel_y_range)
        return tuple(self.current_lin_vel_y_range)


@configclass
class SnakeVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the legacy snake velocity command."""

    class_type: type = SnakeVelocityCommand
    planar_zero_threshold: float = 0.2


class VirtualChassisVelocityCommand(SnakeVelocityCommand):
    """Velocity command interpreted and visualized in the virtual chassis frame."""

    cfg: "VirtualChassisVelocityCommandCfg"

    def __init__(self, cfg: "VirtualChassisVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.metrics["plannar_MAE"] = torch.zeros(self.num_envs, device=self.device)
        if not 0 <= cfg.debug_vis_env_idx < self.num_envs:
            raise ValueError(
                f"debug_vis_env_idx={cfg.debug_vis_env_idx} is out of range for num_envs={self.num_envs}."
            )
        self._body_ids, _ = self.robot.find_bodies(list(cfg.body_names), preserve_order=True)
        self._body_ids = torch.tensor(self._body_ids, device=self.device, dtype=torch.long)
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._vis_prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self._vis_has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> dict[str, float]:
        extras = super().reset(env_ids=env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
            self._vis_prev_axes_w.zero_()
            self._vis_has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
            self._vis_prev_axes_w[env_ids] = 0.0
            self._vis_has_prev_axes[env_ids] = False
        return extras

    def _compute_virtual_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        body_pos_w = self.robot.data.body_pos_w[:, self._body_ids, :]
        body_lin_vel_w = self.robot.data.body_lin_vel_w[:, self._body_ids, :]
        body_ang_vel_w = self.robot.data.body_ang_vel_w[:, self._body_ids, :]

        if not torch.isfinite(body_pos_w).all():
            return (
                torch.zeros_like(body_pos_w.mean(dim=1)),
                torch.zeros_like(self.prev_axes_w),
                torch.zeros(self.num_envs, 3, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
            )

        origin_w, axes_w, lin_vel_vc, ang_vel_z_vc = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(lin_vel_vc).all():
            return origin_w, axes_w, lin_vel_vc, ang_vel_z_vc

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        return origin_w, axes_w, lin_vel_vc, ang_vel_z_vc

    def _compute_virtual_state_for_env_ids(
        self,
        env_ids: torch.Tensor,
        prev_axes_w: torch.Tensor,
        has_prev_axes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        body_pos_w = self.robot.data.body_pos_w[env_ids][:, self._body_ids, :]
        body_lin_vel_w = self.robot.data.body_lin_vel_w[env_ids][:, self._body_ids, :]
        body_ang_vel_w = self.robot.data.body_ang_vel_w[env_ids][:, self._body_ids, :]
        return compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=prev_axes_w[env_ids],
            has_prev=has_prev_axes[env_ids],
        )

    def _update_metrics(self):
        _, _, lin_vel_vc, ang_vel_z_vc = self._compute_virtual_state()
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        planar_error = self.vel_command_b[:, :2] - lin_vel_vc[:, :2]
        self.metrics["error_vel_xy"] += torch.norm(planar_error, dim=-1) / max_command_step
        self.metrics["plannar_MAE"] += torch.sqrt(
            torch.sum(torch.square(planar_error), dim=-1) + torch.square(ang_vel_z_vc)
        ) / max_command_step
        self.metrics["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 2] - ang_vel_z_vc) / max_command_step

    def _update_command(self):
        if self.cfg.heading_command:
            _, axes_w, _, _ = self._compute_virtual_state()
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if env_ids.numel() > 0:
                vc_x_axis_w = axes_w[env_ids, :, 0]
                heading_w = torch.atan2(vc_x_axis_w[:, 1], vc_x_axis_w[:, 0])
                heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - heading_w)
                self.vel_command_b[env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
        small_planar = torch.norm(self.vel_command_b[:, :2], dim=1) <= self.cfg.planar_zero_threshold
        self.vel_command_b[small_planar, :2] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "virtual_chassis_visualizer"):
                self.virtual_chassis_visualizer = VisualizationMarkers(self.cfg.virtual_chassis_frame_cfg)
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.virtual_chassis_visualizer.set_visibility(True)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "virtual_chassis_visualizer"):
                self.virtual_chassis_visualizer.set_visibility(False)
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        env_ids = torch.tensor([self.cfg.debug_vis_env_idx], device=self.device, dtype=torch.long)
        origin_w, axes_w, lin_vel_vc, _ = self._compute_virtual_state_for_env_ids(
            env_ids=env_ids,
            prev_axes_w=self._vis_prev_axes_w,
            has_prev_axes=self._vis_has_prev_axes,
        )
        vc_quat_w = quat_from_axes_w(axes_w)
        marker_pos_w = origin_w.clone()
        marker_pos_w[:, 2] += self.cfg.velocity_marker_z_offset
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_vc_velocity_to_arrow(axes_w, self.command[env_ids, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_vc_velocity_to_arrow(axes_w, lin_vel_vc[:, :2])
        self.virtual_chassis_visualizer.visualize(
            translations=origin_w,
            orientations=vc_quat_w,
            marker_indices=torch.zeros(1, dtype=torch.int32, device=self.device),
        )
        self.goal_vel_visualizer.visualize(
            translations=marker_pos_w,
            orientations=vel_des_arrow_quat,
            scales=vel_des_arrow_scale,
        )
        self.current_vel_visualizer.visualize(
            translations=marker_pos_w,
            orientations=vel_arrow_quat,
            scales=vel_arrow_scale,
        )
        self._vis_prev_axes_w[env_ids] = axes_w
        self._vis_has_prev_axes[env_ids] = True

    def _resolve_vc_velocity_to_arrow(self, axes_w: torch.Tensor, velocity_vc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        arrow_scale = torch.ones(velocity_vc.shape[0], 3, device=self.device)
        max_speed = max(float(self.cfg.velocity_marker_max_speed), 1.0e-6)
        arrow_scale[:, 0] = torch.linalg.norm(velocity_vc, dim=1).clamp(max=max_speed) / max_speed
        arrow_quat = arrow_quat_from_virtual_velocity(axes_w, velocity_vc)
        return arrow_scale, arrow_quat


@configclass
class VirtualChassisVelocityCommandCfg(SnakeVelocityCommandCfg):
    """Configuration for virtual chassis velocity command tracking."""

    class_type: type = VirtualChassisVelocityCommand
    body_names: tuple[str, ...] = ()
    debug_vis_env_idx: int = 0
    velocity_marker_max_speed: float = 0.75
    velocity_marker_z_offset: float = 0.10
    virtual_chassis_frame_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/virtual_chassis"
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/virtual_velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/virtual_velocity_current"
    )

    virtual_chassis_frame_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.12, 0.12, 0.12)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.12, 0.12, 0.12)
    goal_vel_visualizer_cfg.markers["arrow"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
    current_vel_visualizer_cfg.markers["arrow"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
