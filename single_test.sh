export MUJOCO_GL=egl
python sim2sim/sim2sim_mujoco.py \
    --cmd_vx 0.3 \
    --cmd_vy 0.0 \
    --headless 1 \
    --record_video 1 \
    --video_path sim2sim/videos/rollout.mp4 \
    --seconds 15.0 \
    --policy logs/rsl_rl/snake_velocity_flat_tracking/2026-05-26_02-39-23/exported/policy.pt