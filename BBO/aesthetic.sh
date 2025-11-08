# SQDF-CM-Buffer : Bootstrap
CUDA_VISIBLE_DEVICES=0,3 accelerate launch --main_process_port 29501 online/sqdf_cm_buffer_main.py --config config/Bootstrap.py:aesthetic_sqdf_cm_buffer \
    --config.num_epochs=20 \
    --config.num_outer_eval_imgs=32 \
    --config.sqdf_alpha=1 \
    --config.sample.batch_size_per_gpu_available=8 \
    --config.buffer_size=1024 \
    --config.buffer_method='reward+gamma_PER' \
    --config.seed=0 \
    --config.buffer_init_per_loop=True 

# # SQDF-CM-Buffer : UCB
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29502 online/sqdf_cm_buffer_main.py --config config/UCB.py:aesthetic_sqdf_cm_buffer \
#     --config.num_epochs=20 \
#     --config.num_outer_eval_imgs=32 \
#     --config.sqdf_alpha=1 \
#     --config.sample.batch_size_per_gpu_available=8 \
#     --config.buffer_size=1024 \
#     --config.buffer_method='reward+gamma_PER' \
#     --config.seed=0 \
#     --config.buffer_init_per_loop=True 

