python scripts/rsl_rl/play.py \
    --task Snake-VelocityTracking-Flat-Play-v0 \
    --checkpoint logs/rsl_rl/snake_velocity_flat_tracking/2026-05-26_02-39-23/model_9000.pt \
    --video \
    --video_length 750 \
    --headless \
    --manual_command \
    --cmd_vx 0.3 \
    --cmd_vy 0 \
    env.episode_length_s=15.0
export MUJOCO_GL=egl
python sim2sim/sim2sim_mujoco.py \
    --cmd_vx 0.3 \
    --cmd_vy 0.0 \
    --headless 1 \
    --policy logs/rsl_rl/snake_velocity_flat_tracking/2026-05-26_02-39-23/exported/policy.pt
