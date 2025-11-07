#############################SQDF#############################
#* sqdf cm,buffer 
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29541 main.py \
    --num_epochs=2000 \
    --train_gradient_accumulation_steps=8 \
    --backprop_strategy='sqdf' \
    --sample_num_steps=50 \
    --sqdf_gamma=0.90 \
    --sqdf_alpha=2.0 \
    --sqdf_step_min=0 \
    --sqdf_step_max=50 \
    --reward_fn='aesthetic' \
    --prompt_fn='simple_animals' \
    --train_batch_size=4 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb" \
    --use_custom_eval_prompts \
    --use_buffer=True \
    --buffer_size=1024 \
    --replay_ratio=1 \
    --seed=0 \

#* sqdf buf
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --main_process_port 29514 main.py \
#     --num_epochs=2000 \
#     --train_gradient_accumulation_steps=16 \
#     --backprop_strategy='sqdf' \
#     --sample_num_steps=50 \
#     --sqdf_gamma=0.9 \
#     --sqdf_alpha=1.5 \
#     --sqdf_step_min=0 \
#     --sqdf_step_max=50 \
#     --reward_fn='aesthetic' \
#     --prompt_fn='simple_animals' \
#     --train_batch_size=2 \
#     --tracker_project_name="stable_diffusion_training" \
#     --log_with="wandb" \
#     --use_custom_eval_prompts \
#     --save_img_freq=40 \
#     --use_buffer=True \
#     --train_on_policy_batch_size=1 \
#     --buffer_size=1024 \
#     --replay_ratio=1 \
#     --use_consistency_model=False \
#     --seed=0 \

#* sqdf cm
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --main_process_port 29513 main.py \
#     --num_epochs=2000 \
#     --train_gradient_accumulation_steps=16 \
#     --backprop_strategy='sqdf' \
#     --sample_num_steps=50 \
#     --sqdf_gamma=0.9 \
#     --sqdf_alpha=2.0 \
#     --sqdf_step_min=0 \
#     --sqdf_step_max=50 \
#     --reward_fn='aesthetic' \
#     --prompt_fn='simple_animals' \
#     --train_batch_size=2 \
#     --tracker_project_name="stable_diffusion_training" \
#     --log_with="wandb" \
#     --use_custom_eval_prompts \
#     --save_img_freq=40 \
#     --use_buffer=False \
#     --use_consistency_model=True \
#     --seed=0 \




#############################BASELINE#############################
#* draft
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --main_process_port 29506 main.py \
#     --num_epochs=2000 \
#     --train_gradient_accumulation_steps=8 \
#     --backprop_strategy='fixed' \
#     --backward_step=49 \
#     --sample_num_steps=50 \
#     --reward_fn='aesthetic' \
#     --prompt_fn='simple_animals' \
#     --train_batch_size=4 \
#     --tracker_project_name="stable_diffusion_training" \
#     --log_with="wandb" \
#     --use_custom_eval_prompts \
#     --save_img_freq=2 \
#     --train_learning_rate=4e-4 \
#     --use_consistency_model=False \
#     --seed=1 \

# # #* draftlv
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --main_process_port 29510 main.py \
#     --num_epochs=2000 \
#     --train_gradient_accumulation_steps=16 \
#     --train_gradient_accumulation_steps=16 \
#     --backprop_strategy='draftlv' \
#     --backward_step=49 \
#     --sample_num_steps=50 \
#     --reward_fn='aesthetic' \
#     --prompt_fn='simple_animals' \
#     --train_batch_size=2 \
#     --train_batch_size=2 \
#     --tracker_project_name="stable_diffusion_training" \
#     --log_with="wandb" \
#     --use_custom_eval_prompts \
#     --draftlv_loop=2 \
#     --use_consistency_model=False \
#     --train_learning_rate=4e-4 \
#     --save_img_freq=5 \
#     --seed=0 \

# #* ReFL
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29599 main.py \
#     --num_epochs=2000 \
#     --train_gradient_accumulation_steps=16 \
#     --backprop_strategy='refl' \
#     --refl_step_min=40 \
#     --refl_step_max=50 \
#     --sample_num_steps=50 \
#     --reward_fn='aesthetic' \
#     --prompt_fn='simple_animals' \
#     --train_batch_size=2 \
#     --tracker_project_name="stable_diffusion_training" \
#     --use_custom_eval_prompts \
#     --log_with="wandb" \
#     --use_consistency_model=False \
#     --train_learning_rate=1e-5 \
#     --save_img_freq=40 \
#     --seed=2 \


# * draft+KL
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29503 main.py \
#     --num_epochs=2000 \
#     --train_gradient_accumulation_steps=2 \
#     --backprop_strategy='draft+' \
#     --backward_step=49 \
#     --sample_num_steps=50 \
#     --sqdf_alpha=0.035 \
#     --sample_eta=1.0 \
#     --reward_fn='aesthetic' \
#     --prompt_fn='simple_animals' \
#     --train_batch_size=8 \
#     --tracker_project_name="stable_diffusion_training" \
#     --log_with="wandb" \
#     --use_custom_eval_prompts \
#     --save_img_freq=100 \
#     --use_consistency_model=False \
#     --seed=2 \
#     --train_learning_rate=0.0004 \

#######################################################