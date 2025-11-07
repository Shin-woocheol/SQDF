# i have tested the code with 128 batch size, i.e 4 gpus x 8 batch size x 4 gradient accumulation steps, however you can change the batch size 
# or batch size division as per your requirements
# #* sqdf buffer debug
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29501 main.py \
    --num_epochs=2000 \
    --train_gradient_accumulation_steps=2 \
    --backprop_strategy='sqdf' \
    --sample_num_steps=50 \
    --sqdf_gamma=0.90 \
    --sqdf_alpha=2 \
    --sqdf_step_min=0 \
    --sqdf_step_max=50 \
    --reward_fn='hps' \
    --prompt_fn='hps_v2_all' \
    --train_batch_size=2 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb" \
    --use_custom_eval_prompts \
    --use_buffer=True \
    --buffer_size=1024 \
    --replay_ratio=1 \
    --lora_rank=8 \
    # --use_consistency_model=False \
    # --train_learning_rate=5e-4 

#* sqdf buffer
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29515 main.py \
#     --num_epochs=500 \
#     --train_gradient_accumulation_steps=16 \
#     --backprop_strategy='sqdf' \
#     --sqdf_gamma=0.90 \
#     --sqdf_alpha=0.005 \
#     --reward_fn='hps' \
#     --prompt_fn='hps_v2_all' \
#     --train_batch_size=4 \
#     --tracker_project_name="hps" \
#     --log_with="wandb" \
#     --use_custom_eval_prompts \
#     --use_buffer=True \
#     --buffer_size=4096 \
#     --replay_ratio=1 \
#     --save_img_freq=50 \
#     --lora_rank=32 \
#     --train_learning_rate=0.0002 \
#     # --use_consistency_model=False \

# #* draft setting
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29510 main.py \
#     --num_epochs=500 \
#     --train_gradient_accumulation_steps=16 \
#     --backprop_strategy='fixed' \
#     --backward_step=49 \
#     --sample_num_steps=50 \
#     --sample_eta=0.0 \
#     --reward_fn='hps' \
#     --prompt_fn='hps_v2_all' \
#     --train_batch_size=8 \
#     --tracker_project_name="hps" \
#     --log_with="wandb" \
#     --use_custom_eval_prompts \
#     --save_img_freq=50 \
#     --use_consistency_model=False \
#     --lora_rank=32 \
#     --train_learning_rate=2e-4 \

# # #* ReFL
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29501 main.py \
#     --num_epochs=500 \
#     --train_gradient_accumulation_steps=32 \
#     --backprop_strategy='refl' \
#     --refl_step_min=40 \
#     --refl_step_max=50 \
#     --sample_num_steps=50 \
#     --reward_fn='hps' \
#     --prompt_fn='hps_v2_all' \
#     --train_batch_size=4 \
#     --tracker_project_name="hps" \
#     --log_with="wandb" \
#     --use_custom_eval_prompts \
#     --train_learning_rate=2e-4 \
#     --use_custom_eval_prompts \
#     --save_img_freq=50 \
#     --use_consistency_model=False \
#     --lora_rank=32 \