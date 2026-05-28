python scripts/rsl_rl/train.py \
    --task Snake-VelocityTracking-Flat-v0 \
    --num_envs 6000 \
    --headless \
    --resume \
    --run_name exp \
    --load_run 2026-05-27_14-01-10_exp \
    --checkpoint model_3500.pt \
    --max_iterations 7000 \
    --logger wandb \
    --log_project_name Snake_Project \
    agent.save_interval=500
