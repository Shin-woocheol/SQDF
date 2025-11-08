import copy
import torch
from PIL import Image
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)

from aesthetic_scorer import AestheticScorerDiff, online_AestheticScorerDiff
from tqdm import tqdm
import random
from collections import defaultdict
import prompts as prompts_file
import numpy as np
import torch.utils.checkpoint as checkpoint
import wandb
import contextlib
import torchvision
from transformers import AutoProcessor, AutoModel
import sys
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import AttnProcessor2_0, LoRAAttnProcessor2_0
from diffusers.loaders import AttnProcsLayers
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import datetime
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from accelerate.logging import get_logger    
from accelerate import Accelerator
from absl import app, flags
from ml_collections import config_flags

from diffusers_patch.ddim_with_kl import ddim_step_KL

import io

def online_aesthetic_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     config=None,
                     device=None,
                     accelerator=None,
                     torch_dtype=None
                     ):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = online_AestheticScorerDiff(dtype=torch_dtype, config=config).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    scorer.eval()
    
    for param in scorer.parameters():
        assert not param.requires_grad, "Scorer should not have any trainable parameters"

    target_size = 224
    def loss_fn(im_pix_un, config, D_exp):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size, antialias=False)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        
        rewards = scorer(im_pix, config, D_exp)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn

def evaluate_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     device=None,
                     accelerator=None,
                     torch_dtype=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    target_size = 224
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size, antialias=False)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards,_ = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn

def evaluate(training_unet,
             latent,
             train_neg_prompt_embeds,
             prompts, 
             vae,
             text_encoder,
             tokenizer,
             scheduler, 
             accelerator, 
             inference_dtype, 
             config, 
             loss_fn):
    
    prompt_ids = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).input_ids.to(accelerator.device)       
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device)
    prompt_embeds = text_encoder(prompt_ids)[0]         
    
    all_rgbs_t = []
    for i, t in tqdm(
        enumerate(scheduler.timesteps), 
        total=len(scheduler.timesteps),
        disable=not accelerator.is_local_main_process
        ):
        t = torch.tensor([t],
                            dtype=inference_dtype,
                            device=latent.device)
        t = t.repeat(config.train.batch_size_per_gpu_available)

        noise_pred_uncond = training_unet(latent, t, train_neg_prompt_embeds).sample
        noise_pred_cond = training_unet(latent, t, prompt_embeds).sample
                
        grad = (noise_pred_cond - noise_pred_uncond)
        noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
        latent = scheduler.step(noise_pred, t[0].long(), latent, config.sample_eta).prev_sample
        
    ims = vae.decode(latent.to(vae.dtype) / 0.18215).sample
    if config.reward_fn == 'hps':
        loss, rewards = loss_fn(ims, prompts)
    else:    
        _, rewards = loss_fn(ims)
    return ims, rewards

def generate_embeds_fn(device=None, torch_dtype=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    target_size = 224
    def embedding_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size, antialias=False)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        embeds = scorer.generate_feats(im_pix)
        return embeds
    return embedding_fn

def generate_new_x_(current_unet, 
            num_new_x, 
            pipeline, 
            accelerator, 
            config, 
            inference_dtype, 
            prompt_fn, 
            sample_neg_prompt_embeds, 
            embedding_fn,
            ):
    
    all_latents = torch.randn((num_new_x, 4, 64, 64), device=accelerator.device, dtype=inference_dtype) 

    all_prompts, _ = zip(
        *[('A stunning beautiful oil painting of a ' + prompt_fn()[0] + ', cinematic lighting, golden hour light.', {}) 
            if random.random() < config.good_prompt_prop else prompt_fn() for _ in range(num_new_x)]
    )    
    all_embeds = []
    
    with torch.no_grad():
        for index in tqdm(range(num_new_x // config.sample.batch_size_per_gpu_available),
                            total=num_new_x // config.sample.batch_size_per_gpu_available,
                            desc="Obtain fresh samples and feedbacks",
                            disable=not accelerator.is_local_main_process
                        ):
            latent = all_latents[config.sample.batch_size_per_gpu_available*index:config.sample.batch_size_per_gpu_available *(index+1)]
            prompts = all_prompts[config.sample.batch_size_per_gpu_available*index:config.sample.batch_size_per_gpu_available *(index+1)]
            
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)   
                
            pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
            
            for i, t in tqdm(
                enumerate(pipeline.scheduler.timesteps), 
                total=len(pipeline.scheduler.timesteps),
                disable=not accelerator.is_local_main_process
                ):
                t = torch.tensor([t],
                                    dtype=inference_dtype,
                                    device=latent.device)
                t = t.repeat(config.sample.batch_size_per_gpu_available)

                noise_pred_uncond = current_unet(latent, t, sample_neg_prompt_embeds).sample
                noise_pred_cond = current_unet(latent, t, prompt_embeds).sample
                        
                grad = (noise_pred_cond - noise_pred_uncond)
                noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
                
                latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent, config.sample_eta).prev_sample
            
            ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
            
            # for i in range(ims.shape[0]):
            #         eval_image = (ims[i,:,:,:].clone().detach() / 2 + 0.5).clamp(0, 1)
            #         pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            #         pil.save(f"./model/{i:03d}_{prompts[i]}.png")

            embeds = embedding_fn(ims)
            assert embeds.shape[0] == config.sample.batch_size_per_gpu_available
            assert embeds.shape[1] == 768
            all_embeds.append(embeds)
    return torch.cat(all_embeds, dim=0)

def prepare_pipeline_sqdf(pipeline, accelerator, config, inference_dtype):
    import copy, torch, weakref
    from diffusers import DDIMScheduler
    from diffusers.models.attention_processor import AttnProcessor2_0, LoRAAttnProcessor2_0
    from diffusers.loaders import AttnProcsLayers

    # ---- freeze base modules ----
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1, disable=not accelerator.is_local_main_process,
        leave=False, desc="Timestep", dynamic_ncols=True,
    )

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(config.steps, device=accelerator.device)

    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    try:
        pipeline.unet.set_attn_processor(AttnProcessor2_0())
        print("AttnProcessor2_0")
    except Exception:
        pass

    # deep copy unet 
    unets = [copy.deepcopy(pipeline.unet) for _ in range(4)]
    for u in unets:
        u.to(accelerator.device, dtype=inference_dtype)
        u.requires_grad_(False)   

    def is_self_attn(name: str) -> bool:
        return ".attn1" in name  # attn1=self, attn2=cross

    def block_hidden_size(unet, name: str) -> int:
        if name.startswith("mid_block"):
            return unet.config.block_out_channels[-1]
        if name.startswith("down_blocks"):
            bid = int(name.split(".")[1])
            return unet.config.block_out_channels[bid]
        if name.startswith("up_blocks"):
            bid = int(name.split(".")[1])
            return list(reversed(unet.config.block_out_channels))[bid]
        return unet.config.block_out_channels[-1]

    
    class LoRAWrapped(AttnProcsLayers):
        def __init__(self, procs, core_unet):
            super().__init__(procs)                    
            object.__setattr__(self, "_unet_ref", weakref.ref(core_unet)) 
        def forward(self, *args, **kwargs):
            unet = self._unet_ref()
            return unet(*args, **kwargs)

    def attach_lora_and_wrap(unet):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_dim = None if is_self_attn(name) else unet.config.cross_attention_dim
            hs = block_hidden_size(unet, name)
            attn_procs[name] = LoRAAttnProcessor2_0(hidden_size=hs, cross_attention_dim=cross_dim)
        unet.set_attn_processor(attn_procs)

        for proc in unet.attn_processors.values():
            for p in proc.parameters():
                p.requires_grad_(True)

        return LoRAWrapped(unet.attn_processors, unet)

    wrappers = [attach_lora_and_wrap(u) for u in unets]

    return [pipeline.unet, *wrappers], [pipeline.unet, *unets]