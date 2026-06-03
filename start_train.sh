export PYTHONPATH="$(pwd)/source/snake_project:${PYTHONPATH}"

python scripts/rsl_rl/train.py \
    --task Snake-VelocityTracking-Flat-v0 \
    --num_envs 4096 \
    --headless \
    --run_name exp14 \
    --max_iterations 10000 \
    --logger wandb \
    --log_project_name Snake_Project \
    agent.save_interval=500
