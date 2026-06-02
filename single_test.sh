LOG_DIR=$1
export MUJOCO_GL=egl
python sim2sim/sim2sim_mujoco.py \
    --cmd_vx 0.3 \
    --cmd_vy 0.0 \
    --headless 1 \
    --record_video 1 \
    --video_path sim2sim/videos/rollout.mp4 \
    --seconds 15.0 \
    --policy "${LOG_DIR}/exported/policy.pt"