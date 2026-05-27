python scripts/rsl_rl/play.py \
    --task Snake-VelocityTracking-Flat-Play-v0 \
    --checkpoint logs/rsl_rl/snake_velocity_flat_tracking/2026-05-27_09-39-17_exp3/model_1000.pt \
    --video \
    --video_length 750 \
    --headless \
    --manual_command \
    --cmd_vx 0.3 \
    --cmd_vy 0 \
    env.episode_length_s=15.0
export MUJOCO_GL=egl
python sim2sim/sim2sim_eval.py \
    --policy logs/rsl_rl/snake_velocity_flat_tracking/2026-05-27_09-39-17_exp3/exported/policy.pt
python sim2sim/calc_weighted_mae.py \
    sim2sim/eval_output/data/eval_mae.csv