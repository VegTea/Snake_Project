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


def command_lin_vel_x_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    command_name: str = "base_velocity",
    reward_term_name: str = "track_lin_vel_x_exp",
    max_curriculum: float = 0.4,
    min_curriculum: float = 0.1,
    step_size: float = 0.05,
    threshold_ratio: float = 0.8,
    ema_decay: float = 0.05,
    min_env_count: int = 10,
) -> dict[str, float]:
    """Expand/shrink only the x command range using EMA-smoothed x tracking reward."""
    command_term = env.command_manager.get_term(command_name)
    x_min, x_max = command_term.current_lin_vel_x_range

    if env_ids is None or len(env_ids) == 0:
        return {"lin_vel_x_min": x_min, "lin_vel_x_max": x_max}

    episode_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
    mean_tracking_reward = float(torch.mean(episode_sums) / env.max_episode_length_s)
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    threshold = threshold_ratio * reward_cfg.weight

    if not hasattr(command_term, "_tracking_reward_x_ema"):
        command_term._tracking_reward_x_ema = mean_tracking_reward
    else:
        command_term._tracking_reward_x_ema = (
            (1.0 - ema_decay) * command_term._tracking_reward_x_ema + ema_decay * mean_tracking_reward
        )
    ema = command_term._tracking_reward_x_ema

    if len(env_ids) >= min_env_count:
        if ema > threshold:
            x_min, x_max = command_term.expand_lin_vel_x(step_size=step_size, max_curriculum=max_curriculum)
        elif ema < 0.6 * threshold:
            x_min, x_max = command_term.shrink_lin_vel_x(step_size=step_size, min_curriculum=min_curriculum)

    return {
        "lin_vel_x_min": x_min,
        "lin_vel_x_max": x_max,
        "mean_tracking_reward_x": ema,
    }


def command_lin_vel_y_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    command_name: str = "base_velocity",
    reward_term_name: str = "track_lin_vel_y_exp",
    max_curriculum: float = 0.4,
    min_curriculum: float = 0.1,
    step_size: float = 0.05,
    threshold_ratio: float = 0.8,
    ema_decay: float = 0.05,
    min_env_count: int = 10,
) -> dict[str, float]:
    """Expand/shrink only the y command range using EMA-smoothed y tracking reward."""
    command_term = env.command_manager.get_term(command_name)
    y_min, y_max = command_term.current_lin_vel_y_range

    if env_ids is None or len(env_ids) == 0:
        return {"lin_vel_y_min": y_min, "lin_vel_y_max": y_max}

    episode_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
    mean_tracking_reward = float(torch.mean(episode_sums) / env.max_episode_length_s)
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    threshold = threshold_ratio * reward_cfg.weight

    if not hasattr(command_term, "_tracking_reward_y_ema"):
        command_term._tracking_reward_y_ema = mean_tracking_reward
    else:
        command_term._tracking_reward_y_ema = (
            (1.0 - ema_decay) * command_term._tracking_reward_y_ema + ema_decay * mean_tracking_reward
        )
    ema = command_term._tracking_reward_y_ema

    if len(env_ids) >= min_env_count:
        if ema > threshold:
            y_min, y_max = command_term.expand_lin_vel_y(step_size=step_size, max_curriculum=max_curriculum)
        elif ema < 0.6 * threshold:
            y_min, y_max = command_term.shrink_lin_vel_y(step_size=step_size, min_curriculum=min_curriculum)

    return {
        "lin_vel_y_min": y_min,
        "lin_vel_y_max": y_max,
        "mean_tracking_reward_y": ema,
    }
