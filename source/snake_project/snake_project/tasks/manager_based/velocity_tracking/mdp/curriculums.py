from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_velocity_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    command_name: str = "base_velocity",
    reward_term_name: str = "track_lin_vel_xy_exp",
    max_curriculum: float = 0.4,
    min_curriculum: float = 0.1,
    step_size: float = 0.05,
    threshold_ratio: float = 0.8,
    ema_decay: float = 0.05,
    min_env_count: int = 10,
) -> dict[str, float]:
    """Symmetrically expand/shrink the x/y command ranges using EMA-smoothed reward.

    Uses exponential moving average over per-episode tracking rewards.  Only updates
    the range when at least ``min_env_count`` environments have finished, and the
    EMA crosses the expand / shrink thresholds.
    """
    command_term = env.command_manager.get_term(command_name)
    x_min, x_max = command_term.current_lin_vel_x_range
    y_min, y_max = command_term.current_lin_vel_y_range

    if env_ids is None or len(env_ids) == 0:
        return {
            "lin_vel_x_min": x_min, "lin_vel_x_max": x_max,
            "lin_vel_y_min": y_min, "lin_vel_y_max": y_max,
        }

    episode_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
    mean_tracking_reward = float(torch.mean(episode_sums) / env.max_episode_length_s)
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    threshold = threshold_ratio * reward_cfg.weight

    if not hasattr(command_term, "_tracking_reward_ema"):
        command_term._tracking_reward_ema = mean_tracking_reward
    else:
        command_term._tracking_reward_ema = (
            (1.0 - ema_decay) * command_term._tracking_reward_ema + ema_decay * mean_tracking_reward
        )
    ema = command_term._tracking_reward_ema

    if len(env_ids) >= min_env_count:
        if ema > threshold:
            x_min, x_max = command_term.expand_lin_vel_x(step_size=step_size, max_curriculum=max_curriculum)
            y_min, y_max = command_term.expand_lin_vel_y(step_size=step_size, max_curriculum=max_curriculum)
        elif ema < 0.6 * threshold:
            x_min, x_max = command_term.shrink_lin_vel_x(step_size=step_size, min_curriculum=min_curriculum)
            y_min, y_max = command_term.shrink_lin_vel_y(step_size=step_size, min_curriculum=min_curriculum)

    return {
        "lin_vel_x_min": x_min, "lin_vel_x_max": x_max,
        "lin_vel_y_min": y_min, "lin_vel_y_max": y_max,
        "mean_tracking_reward": ema,
    }


def reward_weight_stage_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    gait_steps: int = 2400,
    transition_steps: int = 4800,
    gait_iterations: int | None = None,
    transition_iterations: int | None = None,
    steps_per_iteration: int = 24,
    gait_weights: dict[str, float] | None = None,
    velocity_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Schedule reward weights from gait formation to velocity tracking.

    When ``gait_iterations`` and ``transition_iterations`` are provided, the
    schedule is expressed in PPO iterations by converting ``env.common_step_counter``
    with ``steps_per_iteration``.  Otherwise it falls back to environment steps.
    """
    if gait_weights is None:
        gait_weights = {
            "track_lin_vel_xy_exp": 0.0,
            "sine_wave_position": 2.0,
            "sine_wave_velocity": 0.5,
            "phase_propagation": 3.0,
            "joint_amplitude": 0.3,
        }
    if velocity_weights is None:
        velocity_weights = {
            "track_lin_vel_xy_exp": 5.0,
            "sine_wave_position": 0.0,
            "sine_wave_velocity": 0.0,
            "phase_propagation": 0.4,
            "joint_amplitude": 0.2,
        }

    step = int(env.common_step_counter)
    if gait_iterations is not None:
        progress = step / max(int(steps_per_iteration), 1)
        gait_duration = float(gait_iterations)
        transition_duration = float(transition_iterations or 0)
        progress_key = "iteration"
    else:
        progress = float(step)
        gait_duration = float(gait_steps)
        transition_duration = float(transition_steps)
        progress_key = "step"

    if progress <= gait_duration:
        alpha = 0.0
    elif transition_duration <= 0:
        alpha = 1.0
    else:
        alpha = min(max((progress - gait_duration) / transition_duration, 0.0), 1.0)

    updated_weights = {"alpha": alpha, "step": float(step), progress_key: float(progress)}
    for term_name, gait_weight in gait_weights.items():
        if term_name not in velocity_weights:
            continue
        velocity_weight = velocity_weights[term_name]
        weight = (1.0 - alpha) * gait_weight + alpha * velocity_weight
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if term_cfg.weight != weight:
            term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, term_cfg)
        updated_weights[term_name] = weight

    return updated_weights
