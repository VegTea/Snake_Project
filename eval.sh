LOG_DIR="logs/rsl_rl/snake_velocity_flat_tracking/2026-05-28_01-13-46_exp"

python scripts/rsl_rl/play.py \
    --task Snake-VelocityTracking-Flat-Play-v0 \
    --checkpoint "${LOG_DIR}/model_4000.pt" \
    --video \
    --video_length 150 \
    --camera_top_down \
    --headless \
    --manual_command \
    --cmd_vx 0.3 \
    --cmd_vy 0 \
    env.episode_length_s=3.0

export MUJOCO_GL=egl
python sim2sim/sim2sim_eval.py \
    --policy "${LOG_DIR}/exported/policy.pt"
python sim2sim/calc_weighted_mae.py \
    sim2sim/eval_output/data/eval_mae.csv