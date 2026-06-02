CHECKPOINT=$1
LOG_DIR="${LOG_DIR:-$(dirname "${CHECKPOINT}")}"
RUN_NAME=
CHECK_POINT_NAME=

python scripts/rsl_rl/train.py \
    --task Snake-VelocityTracking-Flat-v0 \
    --num_envs 4096 \
    --headless \
    --resume \
    --load_run 2026-05-25_13-52-55 \
    --checkpoint model_4999.pt \
    --max_iterations 6999 \
    agent.experiment_name=snake_velocity_flat_tracking
