RUN_NAME=$1
python scripts/rsl_rl/train.py \
    --task Snake-VelocityTracking-Flat-v0 \
    --num_envs 4096 \
    --headless \
    --run_name "${RUN_NAME}" \
    --max_iterations 5000 \
    --logger wandb \
    --log_project_name Snake_Project \
    agent.save_interval=1000