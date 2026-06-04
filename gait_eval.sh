CHECKPOINT=$1
LOG_DIR="${LOG_DIR:-$(dirname "${CHECKPOINT}")}"

ACCEPT_EULA=Y python source/snake_project/snake_project/tasks/manager_based/velocity_tracking/export_gait_policy.py \
  --task Snake-VelocityTracking-Flat-Play-v0 \
  --checkpoint "${CHECKPOINT}" \
  --output "${LOG_DIR}/exported/policy.pt" \
  --num_envs 1 \
  --headless

export MUJOCO_GL=egl
python sim2sim/sim2sim_eval.py \
    --output-dir "${LOG_DIR}/eval_output" \
    --policy "${LOG_DIR}/exported/policy.pt"
python sim2sim/calc_weighted_mae.py \
    --csv_path "${LOG_DIR}/eval_output/data/eval_mae.csv"