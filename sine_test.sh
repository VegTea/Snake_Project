export MUJOCO_GL=egl
python sim2sim/sine_wave_mujoco.py \
    --phase_lag 1.0471975512 \
    --bias 0.0 \
    --bias_schedule 0:0.1,3:0.0 \
    --amplitude 0.20 \
    --frequency 0.1 \
    --headless 1 \
    --record_video 1 \
    --video_path sim2sim/videos/sine_wave_pi_over_3.mp4 \
    --log_path sim2sim/videos/sine_wave_pi_over_3.csv \
    --log_warmup 2.0 \
    --seconds 13.0

#q_i(t) = A * sin(2*pi*f*t + i*pi/3)
