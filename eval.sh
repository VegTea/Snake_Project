CHECKPOINT=$1
LOG_DIR="${LOG_DIR:-$(dirname "${CHECKPOINT}")}"

python scripts/rsl_rl/play.py \
    --task Snake-VelocityTracking-Flat-Play-v0 \
    --checkpoint "${CHECKPOINT}" \
    --video \
    --video_length 750 \
    --headless \
    --manual_command \
    --cmd_vx 0.3 \
    --cmd_vy 0 \
    env.episode_length_s=15.0
export MUJOCO_GL=egl
python sim2sim/sim2sim_eval.py \
    --policy "${LOG_DIR}/exported/policy.pt"
python sim2sim/calc_weighted_mae.py \
    --csv_path "${LOG_DIR}/eval_output/data/eval_mae.csv"