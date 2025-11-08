import torch
from PIL import Image
import sys
import shutil
import os
import copy
import gc
cwd = os.getcwd()
sys.path.append(cwd)

from tqdm import tqdm
import random
from collections import defaultdict
import prompts as prompts_file
import torch.distributed as dist
import numpy as np
import torch.utils.checkpoint as checkpoint
import wandb
import contextlib
import torchvision
from transformers import AutoProcessor, AutoModel
import sys
from diffusers.models.attention_processor import LoRAAttnProcessor, AttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import datetime

from accelerate.logging import get_logger    
from accelerate import Accelerator
from absl import app, flags
from ml_collections import config_flags
import time

from diffusers_patch.ddim_with_kl import ddim_step_KL
from online.model_utils import generate_embeds_fn, evaluate_loss_fn, evaluate, prepare_pipeline_sqdf, generate_new_x_, online_aesthetic_loss_fn
from online.dataset import D_explored

## 
from diffusers import LCMScheduler, AutoPipelineForText2Image
from sqdf_utils import sqdf_pipeline_step_with_grad, buffer_sqdf_pipeline_step_with_grad

## Buffer
from sqdf_utils import get_on_policy_data, on_policy_data_store, combine_meta_data, get_off_policy_data

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/online.py:aesthetic", "Training configuration.")

from accelerate.utils import set_seed, ProjectConfiguration
logger = get_logger(__name__)
    


def main(_):
    config = FLAGS.config
    
    
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

    # define run name
    if accelerator.num_processes > 1:
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S") if accelerator.is_main_process else ""
        obj = [unique_id]
        dist.broadcast_object_list(obj, src=0)
        config.unique_id = obj[0]
    
    if config.use_consistency_model:
        if config.buffer_init_per_loop:
            config.run_name = f"sqdf_CM_initBuffer{config.buffer_size}_{config.buffer_method}_{config.train.optimism}_{config.reward_fn}_inner{config.num_epochs}_a{config.sqdf_alpha}_bs{config.train.total_batch_size}_seed{config.seed}"
        else:
            config.run_name = f"sqdf_CM_Buffer{config.buffer_size}_{config.buffer_method}_{config.train.optimism}_{config.reward_fn}_inner{config.num_epochs}_a{config.sqdf_alpha}_bs{config.train.total_batch_size}_seed{config.seed}"
    
    else:
        config.run_name = f"sqdf_{config.train.optimism}_{config.reward_fn}_inner{config.num_epochs}_a{config.sqdf_alpha}_bs{config.train.total_batch_size}_seed{config.seed}"

    if not config.run_name:
        config.run_name = config.unique_id
    else:
        config.run_name += "_" + config.unique_id


    if accelerator.is_main_process:
        wandb_args = {}
        wandb_args["name"] = config.run_name
        if config.debug:
            wandb_args.update({'mode':"disabled"})        
        accelerator.init_trackers(
            project_name="Online", config=config.to_dict(), init_kwargs={"wandb": wandb_args}
        )

        accelerator.project_configuration.project_dir = os.path.join(config.logdir, config.run_name)
        accelerator.project_configuration.logging_dir = os.path.join(config.logdir, wandb.run.name)    


    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    if config.pretrained.model.endswith(".safetensors") or config.pretrained.model.endswith(".ckpt"):
        pipeline = StableDiffusionPipeline.from_single_file(config.pretrained.model)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)

    # freeze parameters of models to save more memory
    inference_dtype = torch.float32
    # inference_dtype = torch.float16

    unet_list, Unet2d_models = prepare_pipeline_sqdf(pipeline, accelerator, config, inference_dtype)

    # load consistency model
    if config.use_consistency_model:

        model_id   = "runwayml/stable-diffusion-v1-5"
        adapter_id = "latent-consistency/lcm-lora-sdv1-5"

        pipe_cm = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        pipe_cm.scheduler = LCMScheduler.from_config(pipe_cm.scheduler.config)
        pipe_cm.load_lora_weights(adapter_id)   
        pipe_cm.fuse_lora()

        consistency_unet = pipe_cm.unet.eval().to(accelerator.device, dtype=inference_dtype)
        consistency_unet.requires_grad_(False)
        consistency_scheduler = pipe_cm.scheduler
        consistency_scheduler.set_timesteps(config.steps, device=accelerator.device)
        consistency_scheduler.alphas_cumprod = consistency_scheduler.alphas_cumprod.to(accelerator.device)

        del pipe_cm
        torch.cuda.empty_cache()
    
    else:
        consistency_unet = None
        consistency_scheduler = None

    


    # make buffer_dir
    config.buffer_dir = os.path.join("buffer", config.run_name)
    os.makedirs(config.buffer_dir, exist_ok=True)
    config.num_stored_traj = 0
    config.buf_size_per_gpu = config.buffer_size // accelerator.num_processes


    embedding_fn = generate_embeds_fn(device = accelerator.device, torch_dtype = inference_dtype)    
    
    online_loss_fn = online_aesthetic_loss_fn(grad_scale=config.grad_scale,
                                    aesthetic_target=config.aesthetic_target,
                                    config=config,
                                    accelerator = accelerator,
                                    torch_dtype = inference_dtype,
                                    device = accelerator.device)
    if config.reward_fn == 'aesthetic':
        eval_loss_fn = evaluate_loss_fn(grad_scale=config.grad_scale,
                                    aesthetic_target=config.aesthetic_target,
                                    accelerator = accelerator,
                                    torch_dtype = inference_dtype,
                                    device = accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True 

    prompt_fn = getattr(prompts_file, config.prompt_fn)
    samping_prompt_fn = getattr(prompts_file, config.samping_prompt_fn)

    if config.eval_prompt_fn == '':
        eval_prompt_fn = prompt_fn
    else:
        eval_prompt_fn = getattr(prompts_file, config.eval_prompt_fn)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
            pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
        )[0]

    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size_per_gpu_available, 1, 1)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size_per_gpu_available, 1, 1)

    autocast = contextlib.nullcontext          
    #################### TRAINING ####################        

    num_fresh_samples = config.num_samples  # 64 samples take 4 minutes to generate
    assert len(num_fresh_samples) == config.train.num_outer_loop, "Number of outer loops must match the number of data counts"
    
    exp_dataset = D_explored(config, accelerator.device).to(accelerator.device, dtype=inference_dtype)
    exp_dataset.model = accelerator.prepare(exp_dataset.model)
    
    # set embedding function for dataset to convert images internally
    exp_dataset.embedding_fn = embedding_fn

    global_step = 0
    for outer_loop in range(config.train.num_outer_loop):        
        ##### Generate a new sample x(i) ∼ p(i)(x) by running {p(i) and get a feedback y(i) =r(x(i)) + ε.
        # fix pre-trained unet
        ref_unet = unet_list[0]
        ref_unet.eval()
        for param in ref_unet.parameters():
                param.requires_grad = False
        

        # define current, training unet
        current_unet = unet_list[outer_loop]
        training_unet = unet_list[outer_loop+1]

        num_new_x = num_fresh_samples[outer_loop]
        print(num_new_x)

        current_unet.eval()
        
        # Freeze the parameter of current model
        if outer_loop == 0:
            for param in current_unet.parameters():
                param.requires_grad = False
        else:
            for name, attn_processor in current_unet.named_children():
                for param in attn_processor.parameters():
                    param.requires_grad = False
        logger.info(f"Freezing current model: {outer_loop}")
        logger.info(f"Start training model: {outer_loop+1}")
        
        if outer_loop > 0: # load the previous model to the training model
            logger.info(f"Load previous model: {outer_loop} weight to training model: {outer_loop+1}")
            training_unet.load_state_dict(current_unet.state_dict())
        
        for name, attn_processor in training_unet.named_children():
                for param in attn_processor.parameters():
                    assert param.requires_grad == True, "All LoRA parameters should be trainable"
        
        if outer_loop == 0 and 'restore_initial_data_from' in config.keys():
            logger.info(f"Restore initial data from {config.restore_initial_data_from}")
            all_new_x = torch.load(config.restore_initial_data_from)
            all_new_x = all_new_x.to(accelerator.device)
        
        else:
            if config.reward_fn == 'aesthetic':
                new_x = generate_new_x_(
                    current_unet, 
                    num_new_x // config.train.num_gpus, 
                    pipeline, 
                    accelerator, 
                    config, 
                    inference_dtype, 
                    samping_prompt_fn, 
                    sample_neg_prompt_embeds, 
                    embedding_fn)  
                all_new_x = accelerator.gather(new_x)  # gather samples and distribute to all GPUs
                assert(len(all_new_x) == num_new_x), "Number of fresh online samples does not match the target number" 

                    
        ##### Construct a new dataset: D(i) = D(i−1) + (x(i), y(i))
        exp_dataset.update(all_new_x)
        
        del all_new_x
        
        # Train a pessimistic reward model r(x; D(i)) and a pessimistic bonus term g(i)(x; D(i))
        if config.train.optimism in ['none', 'UCB']:
            exp_dataset.train_MLP(accelerator, config)
        elif config.train.optimism == 'bootstrap':
            exp_dataset.train_bootstrap(accelerator, config)
        else:
            raise ValueError(f"Unknown optimism {config.train.optimism}")
        
        if accelerator.num_processes > 1:
            # sanity check model weight sync
            if config.train.optimism == 'bootstrap':
                print(f"Process {accelerator.process_index} model 0 layer 0 bias: {exp_dataset.model.module.models[0].layers[0].bias.data}")
            else:
                print(f"Process {accelerator.process_index} layer 0 bias: {exp_dataset.model.module.layers[0].bias.data}")
            print(f"Process {accelerator.process_index} x: {exp_dataset.x.shape}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        ##### Update a diffusion model as {p(i)} by finetuning.
        optimizer = torch.optim.AdamW(
            training_unet.parameters(), # filter(lambda p: p.requires_grad, training_unet.parameters()), 
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )

        # Prepare everything with our `accelerator`.
        training_unet, optimizer = accelerator.prepare(training_unet, optimizer)
        # optimizer = accelerator.prepare(optimizer)
        
        timesteps = pipeline.scheduler.timesteps #[981, 961, 941, 921,]
           
        with open(f"./assets/{config.eval_prompt_fn}.txt", 'r', encoding='utf-8') as f:
            eval_prompts = [line.strip() for line in f if line.strip()]
        config.max_vis_images = len(eval_prompts)


        # if initialize buffer for every outer loop #################################
        if config.buffer_init_per_loop:
            if outer_loop > 0:
                if accelerator.is_main_process:
                    config.buffer_dir = os.path.join("buffer", config.run_name)
                    print("--removing buffer...--")
                    try:
                        shutil.rmtree(config.buffer_dir)
                    except FileNotFoundError:
                        pass
                accelerator.wait_for_everyone()
            os.makedirs(config.buffer_dir, exist_ok=True)
            config.num_stored_traj = 0
            config.buf_size_per_gpu = config.buffer_size // accelerator.num_processes
        ################################################################################

        for epoch in list(range(0, config.num_epochs)):
            training_unet.train()
            info = defaultdict(list)
            info_vis = defaultdict(list)
            image_vis_list = []

            steps_per_inner_epoch = config.train.total_samples_per_epoch // config.train.total_batch_size
            for step in list(range(0, steps_per_inner_epoch)):
                if config.buffer_init_per_loop:
                    buffer_step = global_step % 20
                else:
                    buffer_step = global_step
                ### 
                # get_on-policy traj
                on_policy_data = get_on_policy_data(buffer_step, # global_step
                                                    config,
                                                    accelerator, 
                                                    training_unet,
                                                    pipeline,
                                                    inference_dtype,
                                                    prompt_fn,
                                                    train_neg_prompt_embeds,
                                                    online_loss_fn,
                                                    exp_dataset)

                on_policy_data_store(config, accelerator, on_policy_data, buffer_step)

                # wait generating metadata
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    combine_meta_data(config, accelerator)
                accelerator.wait_for_everyone()

                # get off_policy_data
                off_policy_data = get_off_policy_data(config, accelerator)
                # for _ in range(config.replay_ratio):
                #         off_policy_data = get_off_policy_data()
                #         global_step = self.step(epoch, global_step, off_policy_data)
                #         if global_step >= epochs:
                #             print(f"global_step: {global_step} reached to epochs: {epochs}")
                #             return


                for inner_iters in tqdm(
                        list(range(config.train.data_loader_iterations // steps_per_inner_epoch)),
                        position=0,
                        disable=not accelerator.is_local_main_process
                    ):
                    latent = torch.randn((config.train.batch_size_per_gpu_available, 4, 64, 64),
                        device=accelerator.device, dtype=inference_dtype)    

                    if accelerator.is_main_process:
                        logger.info(f"{config.run_name.rsplit('/', 1)[0]} Loop={outer_loop}/Epoch={epoch}/Steps={global_step}/Iter={inner_iters}: training")

                    if config.buffer_method == "reward+gamma_PER":
                        timestep_indices = off_policy_data.timestep_indices[inner_iters] 
                    else:
                        timestep_indices = None


                    prompts = off_policy_data.prompts[inner_iters]

                    prompt_ids = pipeline.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=pipeline.tokenizer.model_max_length,
                    ).input_ids.to(accelerator.device)   

                    # pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
                    prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
                    
                
                    with accelerator.accumulate(training_unet):
                        with autocast():
                            with torch.enable_grad(): # important b/c don't have on by default in module 
                                sqdf_output_dict = {}    
                                sqdf_output = buffer_sqdf_pipeline_step_with_grad(
                                                                            config,
                                                                            accelerator,
                                                                            training_unet,
                                                                            ref_unet,
                                                                            consistency_unet,
                                                                            pipeline.vae,
                                                                            pipeline.image_processor,
                                                                            pipeline.scheduler,
                                                                            consistency_scheduler,
                                                                            eta = 1.0,
                                                                            latents = latent,
                                                                            prompt_embeds = prompt_embeds,
                                                                            train_neg_prompt_embeds = train_neg_prompt_embeds,
                                                                            use_consistency_model = config.use_consistency_model,
                                                                            all_latents=off_policy_data.latents[inner_iters],
                                                                            timestep_indices=timestep_indices  
                                                                            )   
                                
                                sqdf_output_dict["image"] = sqdf_output.images
                                sqdf_output_dict["KL_loss_term"] = sqdf_output.KL_loss_term
                                sqdf_output_dict["time_tensor"] = sqdf_output.time_tensor
                                kl_vis = accelerator.gather(sqdf_output_dict["KL_loss_term"]) # for logging

                                # query to reward proxy 
                                loss, rewards = online_loss_fn(sqdf_output_dict["image"], config, exp_dataset)

                                # calculate loss
                                kl_loss_term = sqdf_output_dict["KL_loss_term"].to(rewards.device)
                                timestep = sqdf_output_dict["time_tensor"].to(rewards.device)

                                loss = -((config.sqdf_gamma ** (timestep)) * rewards - config.sqdf_alpha * kl_loss_term)
                                loss = loss.mean()

                                rewards_mean = rewards.mean()
                                rewards_std = rewards.std()
                            
                                # for logging
                                info["loss"].append(loss)
                                info["KL-entropy"].append(kl_vis.mean())
                                
                                info["rewards"].append(rewards_mean)
                                info["rewards_std"].append(rewards_std)
                                
                                # backward pass
                                accelerator.backward(loss)
                                if accelerator.sync_gradients:
                                    accelerator.clip_grad_norm_(training_unet.parameters(), config.train.max_grad_norm)
                                optimizer.step()
                                optimizer.zero_grad()                        

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    # generate image per data_loader_iterations
                    if accelerator.sync_gradients:
                        assert (
                            inner_iters + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training and evaluation
                        if config.visualize_eval and (global_step % config.vis_freq ==0): 

                            all_eval_images = []
                            all_eval_rewards = []
                            if config.same_evaluation:
                                generator = torch.cuda.manual_seed(config.seed)
                                latent = torch.randn((config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype, generator=generator)    
                            else:
                                latent = torch.randn((config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)                                
                            with torch.no_grad():

                                num_iter = config.max_vis_images // config.train.batch_size_per_gpu_available

                                for index in range(num_iter):
                                    ims, rewards = evaluate(
                                        training_unet,
                                        latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)],
                                        train_neg_prompt_embeds,
                                        eval_prompts[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)], 
                                        pipeline.vae,
                                        pipeline.text_encoder,
                                        pipeline.tokenizer,
                                        pipeline.scheduler, 
                                        accelerator, 
                                        inference_dtype,
                                        config,
                                        eval_loss_fn,
                                        )
                                    
                                    all_eval_images.append(ims)
                                    all_eval_rewards.append(rewards)
                                    
                            eval_rewards = torch.cat(all_eval_rewards)
                            eval_reward_mean = eval_rewards.mean()
                            eval_reward_std = eval_rewards.std()
                            eval_images = torch.cat(all_eval_images)
                            eval_image_vis = []
                            if accelerator.is_main_process:
                                name_val = config.run_name
                                log_dir = f"logs/{name_val}/eval_vis"
                                os.makedirs(log_dir, exist_ok=True)
                                for i, eval_image in enumerate(eval_images):
                                    eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                                    pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                                    prompt = eval_prompts[i]
                                    pil.save(f"{log_dir}/{outer_loop:01d}_{epoch:03d}_{inner_iters:03d}_{i:03d}_{prompt}.png")
                                    pil = pil.resize((256, 256))
                                    reward = eval_rewards[i]
                                    eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))                    
                                accelerator.log({"eval_images": eval_image_vis},step=global_step)
                        
                        logger.info("Logging")
                        
                        info = {k: torch.stack(v).mean() for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        logger.info(f"loss: {info['loss']}, rewards: {info['rewards']}")

                        info.update({"outer_loop": outer_loop,
                                    "epoch": epoch, 
                                    "inner_epoch": inner_iters,
                                    "eval_rewards":eval_reward_mean,
                                    "eval_rewards_std":eval_reward_std,
                                    "dataset_size": len(exp_dataset),
                                    "dataset_y_avg": torch.mean(exp_dataset.y),
                                    })
                        accelerator.log(info, step=global_step)

                        # if config.visualize_train:
                        #     ims = torch.cat(info_vis["image"])
                        #     rewards = torch.cat(info_vis["rewards_img"])
                        #     prompts = info_vis["prompts"]
                        #     images  = []
                        #     for i, image in enumerate(ims):
                        #         image = (image.clone().detach() / 2 + 0.5).clamp(0, 1)
                        #         pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                        #         pil = pil.resize((256, 256))
                        #         prompt = prompts[i]
                        #         reward = rewards[i]
                        #         images.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))
                            
                        #     accelerator.log(
                        #         {"images": images},
                        #         step=global_step,
                        #     )

                        global_step += 1
                        info = defaultdict(list)

                # make sure we did an optimization step at the end of the inner epoch
                assert accelerator.sync_gradients
                ## Saving ckpt
                # if epoch % config.save_freq == 0 and accelerator.is_main_process:
                #     def save_model_hook(models, weights, output_dir):
                #         if isinstance(models[-1], AttnProcsLayers):
                #             Unet2d_models[outer_loop+1].save_attn_procs(output_dir)
                #         else:
                #             raise ValueError(f"Unknown model type {type(models[-1])}")
                #         for _ in range(len(weights)):
                #             weights.pop()
                #     accelerator.register_save_state_pre_hook(save_model_hook)
                #     accelerator.save_state()

        # generate image for evaluation per each outer_loop 
        num_repeats = config.num_outer_eval_imgs // accelerator.num_processes
        outer_loop_eval_rewards = []

        # load eval_prompts
        with open(f"./assets/{config.eval_prompt_fn}.txt", 'r', encoding='utf-8') as f:
            outer_eval_prompts = [line.strip() for line in f if line.strip()]

        for i in range(num_repeats):
            # fix seed for each GPU
            seed = 0 + accelerator.process_index + i * accelerator.num_processes
            print(f"[Rank: {accelerator.process_index}, Seed: {seed}]")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # loop for generate all eval prompts
            for start_index in range(0, len(outer_eval_prompts), config.train.batch_size_per_gpu_available):
                end_index = min(start_index + config.train.batch_size_per_gpu_available, len(outer_eval_prompts))
                num_latent = end_index - start_index

                latent_batch = torch.randn(
                    (num_latent, 4, 64, 64),
                    device=accelerator.device,
                    dtype=inference_dtype
                )

                prompts_batch = outer_eval_prompts[start_index:end_index]
                eval_neg_prompt_embeds = neg_prompt_embed.repeat(num_latent, 1, 1)

                with torch.no_grad():  
                    ims, rewards = evaluate(
                        training_unet,
                        latent_batch,
                        eval_neg_prompt_embeds,
                        prompts_batch, 
                        pipeline.vae,
                        pipeline.text_encoder,
                        pipeline.tokenizer,
                        pipeline.scheduler, 
                        accelerator, 
                        inference_dtype,
                        config,
                        eval_loss_fn,
                        )
                    
                    name_val = config.run_name
                    log_dir = f"logs/{name_val}/epoch{outer_loop}"
                    os.makedirs(log_dir, exist_ok=True)
                    for i, eval_image in enumerate(ims):
                        eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                        pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                        prompt = prompts_batch[i]
                        reward = rewards[i]
                        pil.save(f"{log_dir}/{prompt}_{seed}_{reward}.png")
                        outer_loop_eval_rewards.append(reward)            

        # gather eval_reward and logging
        accelerator.wait_for_everyone()        
        avg_outer_loop_eval_reward = torch.tensor(outer_loop_eval_rewards, device=accelerator.device).mean()
        avg_outer_loop_eval_reward = accelerator.gather(avg_outer_loop_eval_reward).mean().item()    
                    
        seed = random.randint(0, 100)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed) 
                    
        info["outer_eval_reward_mean"] = avg_outer_loop_eval_reward
        accelerator.log({"outer_eval_reward_mean": avg_outer_loop_eval_reward}, step=global_step)

        del optimizer 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  

if __name__ == "__main__":
    app.run(main)
