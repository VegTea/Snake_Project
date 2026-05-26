python scripts/rsl_rl/train.py \
    --task Snake-VelocityTracking-Flat-v0 \
    --num_envs 4096 \
    --headless \
    --run_name exp1 \
    --max_iterations 15000 \
    agent.save_interval=1000