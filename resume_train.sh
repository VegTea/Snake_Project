CHECKPOINT=$1
CHECKPOINT_NAME=$(basename "${CHECKPOINT}")
RUN_NAME=$(basename "$(dirname "${CHECKPOINT}")")
LOG_DIR=$(dirname "${CHECKPOINT}")

export PYTHONPATH="$(pwd)/source/snake_project:${PYTHONPATH}"
python scripts/rsl_rl/train.py \
    --task Snake-VelocityTracking-Flat-v0 \
    --num_envs 4096 \
    --headless \
    --resume \
    --logger wandb \
    --run_name "${RUN_NAME}_end" \
    --load_run "${RUN_NAME}" \
    --checkpoint "${CHECKPOINT_NAME}" \
    --max_iterations 5000 \
    agent.experiment_name=snake_velocity_flat_tracking