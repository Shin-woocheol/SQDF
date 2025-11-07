import os
import textwrap
import random
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Union
from warnings import warn
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils
from accelerate import Accelerator
from accelerate.logging import get_logger
from aesthetic_scorer import AestheticScorerDiff
from prompts import sanitize_prompt

from accelerate.utils import ProjectConfiguration, set_seed
from transformers import is_wandb_available
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import torchvision
import torchvision.utils as vutils
from sd_pipeline import DiffusionPipeline
from config.alignprop_config import AlignPropConfig
from trl.trainer import BaseTrainer
from tqdm import tqdm
import torch.distributed as dist

# add for eval_img_logging
from itertools import cycle, islice
from collections import defaultdict

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json

if is_wandb_available():
    import wandb

logger = get_logger(__name__)

@dataclass
class SampleData:
    latents: torch.Tensor  
    prompts: List[str]  
    rewards: Optional[List[float]] = None 
    timestep_indices: Optional[List[int]] = None

def hps_loss_fn(inference_dtype=None, device=None):
    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        'laion2B-s32B-b79K',
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )    
    
    tokenizer = get_tokenizer(model_name)
    
    link = "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"
    import os
    import requests
    from tqdm import tqdm

    # Create the directory if it doesn't exist
    os.makedirs(os.path.expanduser('~/.cache/hpsv2'), exist_ok=True)
    checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"

    # Download the file if it doesn't exist
    if not os.path.exists(checkpoint_path):
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(checkpoint_path, 'wb') as file, tqdm(
            desc="Downloading HPS_v2_compressed.pt",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
    
    
    # force download of model via score
    hpsv2.score([], "")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def loss_fn(im_pix, prompts):    
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return  loss, scores
    
    return loss_fn


def clip_score(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from scorers.clip_scorer import CLIPScorer

    scorer = CLIPScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn


def aesthetic_loss_fn(aesthetic_target=None,
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
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn

def pickScore_loss_fn(
    inference_dtype=None, 
    device=None, 
    return_loss=True, 
):
    from scorers.PickScore_scorer import PickScoreScorer

    # scorer = PickScoreScorer(dtype=torch.float32, device=device)
    scorer = PickScoreScorer(dtype=inference_dtype, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn

def clip_plus_aesthetic_loss_fn(aesthetic_target=None,
                                grad_scale=0,
                                device=None,
                                accelerator=None,
                                torch_dtype=None,
                                weight=0.5):
    
    clip_scorer = clip_score(torch_dtype, device, return_loss=True)
    aesthetic_scorer = aesthetic_loss_fn(aesthetic_target, grad_scale, device, accelerator, torch_dtype)
    
    def loss_fn(images, prompts):
        clip_loss, clip_reward = clip_scorer(images, prompts)
        aesthetic_loss, aesthetic_reward = aesthetic_scorer(images)
        return clip_loss + aesthetic_loss, (1-weight) * 20 * clip_reward + weight * aesthetic_reward, clip_reward, aesthetic_reward #by DAS paper
    
    return loss_fn



class AlignPropTrainer(BaseTrainer):
    """
    The AlignPropTrainer uses Deep Diffusion Policy Optimization to optimise diffusion models.
    Note, this trainer is heavily inspired by the work here: https://github.com/mihirp1998/AlignProp/
    As of now only Stable Diffusion based pipelines are supported

    Attributes:
        config (`AlignPropConfig`):
            Configuration object for AlignPropTrainer. Check the documentation of `PPOConfig` for more details.
        reward_function (`Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor]`):
            Reward function to be used
        prompt_function (`Callable[[], Tuple[str, Any]]`):
            Function to generate prompts to guide model
        sd_pipeline (`DiffusionPipeline`):
            Stable Diffusion pipeline to be used for training.
        image_samples_hook (`Optional[Callable[[Any, Any, Any], Any]]`):
            Hook to be called to log images
    """

    _tag_names = ["trl", "alignprop"]

    def __init__(
        self,
        config: AlignPropConfig,
        prompt_function: Callable[[], Tuple[str, Any]],
        sd_pipeline: DiffusionPipeline,
        image_samples_hook: Optional[Callable[[Any, Any, Any], Any]] = None,
    ):
        from datetime import datetime
        # self.now = datetime.now().strftime("%m%d_%H%M")
        
        if image_samples_hook is None:
            warn("No image_samples_hook provided; no images will be logged")

        self.prompt_fn = prompt_function
        
        self.config = config
        self.image_samples_callback = image_samples_hook
        self.best = float('-inf')

        accelerator_project_config = ProjectConfiguration(**self.config.project_kwargs)

        if self.config.resume_from:
            self.config.resume_from = os.path.normpath(os.path.expanduser(self.config.resume_from))
            if "checkpoint_" not in os.path.basename(self.config.resume_from):
                # get the most recent checkpoint in this directory
                checkpoints = list(
                    filter(
                        lambda x: "checkpoint_" in x,
                        os.listdir(self.config.resume_from),
                    )
                )
                if len(checkpoints) == 0:
                    raise ValueError(f"No checkpoints found in {self.config.resume_from}")
                checkpoint_numbers = sorted([int(x.split("_")[-1]) for x in checkpoints])
                self.config.resume_from = os.path.join(
                    self.config.resume_from,
                    f"checkpoint_{checkpoint_numbers[-1]}",
                )

                accelerator_project_config.iteration = checkpoint_numbers[-1] + 1


        self.accelerator = Accelerator(
            log_with=self.config.log_with,
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_project_config,
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=self.config.train_gradient_accumulation_steps,
            **self.config.accelerator_kwargs,
        )

        if self.accelerator.num_processes > 1:
            # Ensure all ranks share the same timestamp identifier
            now = self.config.now if self.accelerator.is_main_process else ""
            obj = [now]
            dist.broadcast_object_list(obj, src=0)
            self.config.now = obj[0]
        self.now = self.config.now

        # print self.now for each GPU (debug)
        rank = self.accelerator.process_index
        print(f"[Rank {rank}] self.now = {self.now}")

        if self.config.log_with == 'wandb':
            if self.config.use_consistency_model:
                run_name = f"{self.now}_sqdf-cm_{self.config.reward_fn}_bs{self.config.train_batch_size * self.config.train_gradient_accumulation_steps * self.accelerator.num_processes}_a{self.config.sqdf_alpha}_g{self.config.sqdf_gamma}_lr{self.config.train_learning_rate}_e{self.config.sample_eta}"
        
            elif self.config.use_n_step_sampling:
                run_name = f"{self.now}_sqdf-{self.config.n_steps}-step_{self.config.reward_fn}_bs{self.config.train_batch_size * self.config.train_gradient_accumulation_steps * self.accelerator.num_processes}_a{self.config.sqdf_alpha}_g{self.config.sqdf_gamma}_lr{self.config.train_learning_rate}_e{self.config.sample_eta}"

            else:
                run_name = f"{self.now}_{self.config.backprop_strategy}_{self.config.reward_fn}_bs{self.config.train_batch_size * self.config.train_gradient_accumulation_steps * self.accelerator.num_processes}_a{self.config.sqdf_alpha}_g{self.config.sqdf_gamma}_lr{self.config.train_learning_rate}_e{self.config.sample_eta}"

            if self.config.use_buffer:
                run_name += f"_buf{self.config.buffer_size}_{self.config.replay_ratio}"
                if self.config.buffer_method == "on_policy":
                    run_name += f"_on_policy"
                elif self.config.buffer_method == "reward+gamma_PER":
                    run_name += f"_reward+gamma_PER"

            run_name += f"_seed{self.config.seed}"
            group_name = f"{self.config.backprop_strategy}"

            self.config.tracker_kwargs.setdefault("wandb", {})
            self.config.tracker_kwargs["wandb"].setdefault("name", run_name)
            self.config.tracker_kwargs["wandb"].setdefault("group", group_name)


        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        config_dict = config.to_dict()
        config_dict['checkpoints_dir'] = self.config.project_kwargs['project_dir']

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                self.config.tracker_project_name,
                config=dict(alignprop_trainer_config=config_dict)
                if not is_using_tensorboard
                else config.to_dict(),
                init_kwargs=self.config.tracker_kwargs,
            )

        logger.info(f"\n{config}")

        set_seed(self.config.seed, device_specific=True)

        self.sd_pipeline = sd_pipeline

        self.sd_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32

        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)
        if self.config.backprop_strategy == 'sqdf' or self.config.backprop_strategy == 'draft+':
            self.sd_pipeline.ref_unet.to(self.accelerator.device, dtype=inference_dtype)

        trainable_layers = self.sd_pipeline.get_trainable_layers()

        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.optimizer = self._setup_optimizer(
            trainable_layers.parameters() if not isinstance(trainable_layers, list) else trainable_layers
        )

        self.neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""] if self.config.negative_prompts is None else self.config.negative_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]

        # NOTE: for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = self.sd_pipeline.autocast or self.accelerator.autocast

        if hasattr(self.sd_pipeline, "use_lora") and self.sd_pipeline.use_lora:
            if self.config.use_consistency_model:
                unet, consistency_unet, self.optimizer = self.accelerator.prepare(trainable_layers, sd_pipeline.consistency_unet, self.optimizer)
            else:
                unet, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
            
            self.trainable_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
        else:
            if self.config.use_consistency_model:
                self.trainable_layers, consistency_unet, self.optimizer = self.accelerator.prepare(trainable_layers, sd_pipeline.consistency_unet, self.optimizer)
            else:
                self.trainable_layers, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
        
        if self.config.reward_fn=='hps':
            self.loss_fn = hps_loss_fn(inference_dtype, self.accelerator.device)
        elif self.config.reward_fn=='aesthetic': # easthetic
            self.loss_fn = aesthetic_loss_fn(grad_scale=self.config.grad_scale,
                                            aesthetic_target=self.config.aesthetic_target,
                                            accelerator = self.accelerator,
                                            torch_dtype = inference_dtype,
                                            device = self.accelerator.device) 
        elif self.config.reward_fn == 'clip_plus_aesthetic':
            self.loss_fn = clip_plus_aesthetic_loss_fn(grad_scale=self.config.grad_scale,
                                            aesthetic_target=self.config.aesthetic_target,
                                            accelerator = self.accelerator,
                                            torch_dtype = inference_dtype,
                                            device = self.accelerator.device)
        elif self.config.reward_fn == 'pickScore':
            self.loss_fn = pickScore_loss_fn(inference_dtype, self.accelerator.device)
        else:
            raise NotImplementedError
        if config.resume_from:
            logger.info(f"Resuming from {config.resume_from}")
            self.accelerator.load_state(config.resume_from)
            self.first_epoch = int(config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0

        if self.config.use_custom_eval_prompts:
            self.eval_prompts = self.config.custom_eval_prompts
            self.eval_prompt_metadata = [{} for _ in range(len(self.eval_prompts))]
        else:
            self.eval_prompts, self.eval_prompt_metadata = zip(*[self.prompt_fn() for _ in range(config.train_batch_size * config.log_image_iter)])
        self.log_image_iter = config.log_image_iter

        #* Buffer
        if self.config.use_buffer:
            self.buffer_dir = os.path.join("buffer", self.now)
            os.makedirs(self.buffer_dir, exist_ok=True)

            self.num_stored_traj = 0
            self.buf_size_per_gpu = self.config.buffer_size // self.accelerator.num_processes
            print(f"[Rank {self.accelerator.process_index}] buf_size_per_gpu: {self.buf_size_per_gpu}")
            assert self.config.buffer_size % self.accelerator.num_processes == 0, f"buffer_size ({self.config.buffer_size}) must be divisible by number of processes ({self.accelerator.num_processes})"
        else:
            self.buffer_dir = None

        # Save config to JSON file
        if self.accelerator.is_main_process and not self.config.test_mode:
            config_dict = self.config.to_dict()
            config_path = os.path.join(self.config.project_kwargs['project_dir'], "config.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"Config saved to {config_path}")

    def step(self, epoch: int, global_step: int, all_data : SampleData = None):
        """
        Perform a single step of training.

        Args:
            epoch (int): The current epoch.
            global_step (int): The current global step.

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.
            - If `self.image_samples_callback` is not None, it will be called with the prompt_image_pairs, global_step, and the accelerator tracker.

        Returns:
            global_step (int): The updated global step.
        """
        info = defaultdict(list)
        print(f"Epoch: {epoch}, Global Step: {global_step}")

        self.sd_pipeline.unet.train()

        for i in range(self.config.train_gradient_accumulation_steps):
            with self.accelerator.accumulate(self.sd_pipeline.unet), self.autocast(), torch.enable_grad():
                if self.config.use_buffer:
                    if self.config.buffer_method == 'reward+gamma_PER':
                        timestep_indices = all_data.timestep_indices[i]
                    else:
                        timestep_indices = None
                    prompt_image_pairs = self._generate_samples(
                        batch_size=self.config.train_batch_size,
                        prompts=all_data.prompts[i],
                        all_latents=all_data.latents[i],
                        timestep_indices=timestep_indices,
                    )
                else:
                    prompt_image_pairs = self._generate_samples(
                        batch_size=self.config.train_batch_size,
                    )
                
                if self.config.reward_fn in ["hps", "pickScore"]:
                    if self.config.backprop_strategy == 'draftlv':
                        prompts = prompt_image_pairs["prompts"]
                        prompts = [p for p in prompts for _ in range(self.config.draftlv_loop)]
                        loss, rewards = self.loss_fn(prompt_image_pairs["images"], prompts)
                    else:
                        loss, rewards = self.loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
                elif self.config.reward_fn == 'clip_plus_aesthetic':
                    loss, rewards, clip_reward, aesthetic_reward = self.loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
                else:
                    loss, rewards = self.loss_fn(prompt_image_pairs["images"])

                self.accelerator.wait_for_everyone()
                # Loss and metrics
                rewards_vis = self.accelerator.gather(rewards).detach().cpu().numpy()
                if self.config.reward_fn == 'clip_plus_aesthetic':
                    clip_rewards_vis = self.accelerator.gather(clip_reward).detach().cpu().numpy()
                    aesthetic_rewards_vis = self.accelerator.gather(aesthetic_reward).detach().cpu().numpy()
                if self.config.backprop_strategy == 'sqdf' or self.config.backprop_strategy == 'draft+':
                    kl_vis = self.accelerator.gather(prompt_image_pairs["KL_loss_term"]).detach().cpu().numpy()
                
                self.accelerator.wait_for_everyone()
                
                if self.config.backprop_strategy == 'sqdf':
                    mask = (rewards > 0).float()
                    kl_loss_term = prompt_image_pairs["KL_loss_term"].to(rewards.device)
                    timestep = prompt_image_pairs["time_tensor"].to(rewards.device)
                    loss = -((self.config.sqdf_gamma ** (timestep)) * rewards - self.config.sqdf_alpha * kl_loss_term)
                    loss = loss * mask 
                    loss = loss.mean()
                elif self.config.backprop_strategy == 'draft+':
                    kl_loss_term = prompt_image_pairs["KL_loss_term"].to(rewards.device)
                    loss = -(rewards - self.config.sqdf_alpha * kl_loss_term)
                    loss = loss.mean()
                elif self.config.backprop_strategy == "draftlv":
                    loss = loss.view(self.config.train_batch_size, self.config.draftlv_loop)
                    loss = loss.sum(dim=1).mean()
                else:
                    loss = loss.mean()
                
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.trainable_layers.parameters()
                        if not isinstance(self.trainable_layers, list)
                        else self.trainable_layers,
                        self.config.train_max_grad_norm,
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            info["reward_mean"].append(rewards_vis.mean())
            info["reward_std"].append(rewards_vis.std())
            if self.config.reward_fn == 'clip_plus_aesthetic':
                info["clip_reward_mean"].append(clip_rewards_vis.mean())
                info["aesthetic_reward_mean"].append(aesthetic_rewards_vis.mean())
            if self.config.backprop_strategy == 'sqdf' or self.config.backprop_strategy == 'draft+':
                info["kl_loss_term"].append(kl_vis.mean())
            info["loss"].append(loss.item())

        # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            # log training-related stuff
            info = {k: torch.mean(torch.tensor(v)) for k, v in info.items()}
            info.update({"epoch": global_step})
            self.accelerator.log(info, step=global_step)
            global_step += 1
            info = defaultdict(list)
        else:
            raise ValueError(
                "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
            )
        # Logs generated images
        if self.image_samples_callback is not None and global_step % self.config.log_image_freq == 0 and self.accelerator.is_main_process and self.accelerator.trackers:
            print("Logging images")
            # Fix the random seed for reproducibility
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            
            all_prompt_image_pairs = []
            for i in range(self.log_image_iter):
                start_idx = i * self.config.train_batch_size
                end_idx = start_idx + self.config.train_batch_size

                if self.config.use_custom_eval_prompts:
                    sliced_prompts = self.eval_prompts[start_idx:min(end_idx, len(self.eval_prompts))]
                else:
                    sliced_prompts = self.eval_prompts[start_idx:end_idx]

                prompt_image_pairs_eval = self._generate_samples(
                    batch_size=len(sliced_prompts), with_grad=False, prompts=sliced_prompts
                )
                all_prompt_image_pairs.append(prompt_image_pairs_eval)
            
            combined_pairs = {
                "images": torch.cat([pairs["images"] for pairs in all_prompt_image_pairs], dim=0),
                "prompts": [prompt for pairs in all_prompt_image_pairs for prompt in pairs["prompts"]],
                "prompt_metadata": [meta for pairs in all_prompt_image_pairs for meta in pairs["prompt_metadata"]]
            }
            
            self.image_samples_callback(combined_pairs, global_step, self.accelerator.trackers[0])
            seed = random.randint(0, 100)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)     

       
        if epoch != 0 and epoch % self.config.save_freq == 0:
            checkpoint_dir = f"checkpoints/{self.now}/epoch{epoch}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            print("Saving checkpoint")
            self.accelerator.save_state(output_dir=checkpoint_dir)
        print("Step Done")
        self.accelerator.wait_for_everyone()
        return global_step

    def _setup_optimizer(self, trainable_layers_parameters):
        if self.config.train_use_8bit_adam:
            import bitsandbytes

            optimizer_cls = bitsandbytes.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
        )

    def _save_model_hook(self, models, weights, output_dir):
        self.sd_pipeline.save_checkpoint([models[0]], [weights[0]], output_dir)
        weights.clear()

    def _load_model_hook(self, models, input_dir):
        self.sd_pipeline.load_checkpoint([models[0]], input_dir)
        models.clear()

    def _generate_samples(self, batch_size, with_grad=True, prompts=None, on_policy_sampling=False, all_latents=None, timestep_indices=None):
        """
        Generate samples from the model

        Args:
            batch_size (int): Batch size to use for sampling
            with_grad (bool): Whether the generated RGBs should have gradients attached to it.
            prompts (List[str], optional): Prompts to use for generation
            on_policy_batch_size (int, optional): On-policy batch size for buffer usage

        Returns:
            prompt_image_pairs (Dict[Any])
        """
        prompt_image_pairs = {}

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

        if prompts is None:
            prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(batch_size)])
        else:
            prompt_metadata = [{} for _ in range(batch_size)]

        prompt_ids = self.sd_pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.sd_pipeline.tokenizer.model_max_length,
        ).input_ids.to(self.accelerator.device)

        prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

        if with_grad:
            if not self.config.use_buffer:
                sd_output = self.sd_pipeline.rgb_with_grad(
                    prompt_embeds=prompt_embeds, 
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    backprop_strategy=self.config.backprop_strategy,
                    backprop_kwargs=self.config.backprop_kwargs[self.config.backprop_strategy],
                    output_type="pt",
                    mode=self.config.backprop_strategy,
                    use_consistency_model=self.config.use_consistency_model,
                )
            elif self.config.use_n_step_sampling:
                sd_output = self.sd_pipeline.rgb_with_grad(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    backprop_strategy=self.config.backprop_strategy,
                    backprop_kwargs=self.config.backprop_kwargs[self.config.backprop_strategy],
                    output_type="pt",
                    mode="sqdf_n_step",
                    use_consistency_model=self.config.use_consistency_model,
                    all_latents=all_latents,
                    buffer_method=self.config.buffer_method,
                    gamma=self.config.sqdf_gamma,
                    timestep_indices=timestep_indices,
                    n_steps=self.config.n_steps,
                )

            else:
                sd_output = self.sd_pipeline.rgb_with_grad(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    backprop_strategy=self.config.backprop_strategy,
                    backprop_kwargs=self.config.backprop_kwargs[self.config.backprop_strategy],
                    output_type="pt",
                    mode="sqdf_buffer",
                    use_consistency_model=self.config.use_consistency_model,
                    all_latents=all_latents,
                    buffer_method=self.config.buffer_method,
                    gamma=self.config.sqdf_gamma,
                    timestep_indices=timestep_indices,
                )
        else:
            sd_output = self.sd_pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=self.config.sample_num_steps,
                guidance_scale=self.config.sample_guidance_scale,
                eta=self.config.sample_eta,
                output_type="pt",
            )
            if on_policy_sampling:
                prompt_image_pairs["all_latents"] = sd_output.latents[0:self.config.sample_num_steps]
        
        if self.config.backprop_strategy == 'sqdf' and with_grad:
            images = sd_output.images
            time_tensor = sd_output.time_tensor
            KL_loss_term = sd_output.KL_loss_term
            prompt_image_pairs["time_tensor"] = time_tensor
            prompt_image_pairs["KL_loss_term"] = KL_loss_term
        elif self.config.backprop_strategy == 'draft+' and with_grad:
            images = sd_output.images
            KL_loss_term = sd_output.KL_loss_term
            prompt_image_pairs["KL_loss_term"] = KL_loss_term
        else:
            images = sd_output.images

        prompt_image_pairs["images"] = images
        prompt_image_pairs["prompts"] = prompts
        prompt_image_pairs["prompt_metadata"] = prompt_metadata

        return prompt_image_pairs

    def train(self, epochs: Optional[int] = None):
        """
        Train the model for a given number of epochs
        """
        global_step = 0
        if epochs is None:
            epochs = self.config.num_epochs
        for epoch in range(self.first_epoch, epochs):
            if self.config.use_buffer:
                # get on-policy data
                on_policy_data = self.get_on_policy_data(global_step)
                self.on_policy_data_store(on_policy_data, global_step)

                # wait generating metadata
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.combine_meta_data()
                self.accelerator.wait_for_everyone()

                for _ in range(self.config.replay_ratio):
                    off_policy_data = self.get_off_policy_data()
                    global_step = self.step(epoch, global_step, off_policy_data)
                    if global_step >= epochs:
                        print(f"global_step: {global_step} reached to epochs: {epochs}")
                        return
            else:
                global_step = self.step(epoch, global_step)
        

    def _save_pretrained(self, save_directory):
        self.sd_pipeline.save_pretrained(save_directory)
        self.create_model_card()

    def get_on_policy_data(self, epoch : int):
        latnets_list = []
        prompts_list = []
        rewards_list = []
        G = self.config.train_gradient_accumulation_steps
        B = self.config.train_batch_size
        total_on_policy_batch_size = B * G
        print(f"[Rank {self.accelerator.process_index}] will generate {total_on_policy_batch_size} on-policy trajectories")

        num_iterations = (total_on_policy_batch_size + self.config.train_batch_size - 1) // self.config.train_batch_size
        num_iterations = int(num_iterations)
        
        for i in range(num_iterations):
            # On the last iteration, generate only the remaining number
            current_batch_size = min(self.config.train_batch_size, total_on_policy_batch_size - i * self.config.train_batch_size)
            
            #get all_latents, images, prompts
            prompt_image_pairs = self._generate_samples(
                batch_size=current_batch_size,
                with_grad=False,
                on_policy_sampling=True,
            )
            #get rewards
            if self.config.reward_fn in ["hps", "pickScore"]:
                _, rewards = self.loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
            elif self.config.reward_fn == "ImageReward":
                _, rewards = self.loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
                rewards = F.relu(rewards) 
            elif self.config.reward_fn == 'clip_plus_aesthetic':
                _, rewards, _, _ = self.loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
            else:
                _, rewards = self.loss_fn(prompt_image_pairs["images"])

            #change all latents format to [batch_size, num_timesteps, channels, height, width]
            all_latents = prompt_image_pairs["all_latents"]
            all_latents = torch.stack(all_latents, dim=0) #* (num_timesteps, batch, channel, height, width)
            all_latents = all_latents.permute(1, 0, 2, 3, 4) #* (batch, num_timesteps, channel, height, width)
            
            latnets_list.append(all_latents)
            prompts_list.extend(prompt_image_pairs["prompts"])
            rewards_list.extend(rewards.tolist())

    # Concatenate all latents into a single tensor
        combined_latents = torch.cat(latnets_list, dim=0)  # (total_on_policy_batch_size, num_timesteps, channels, height, width)
        # combined_latents = combined_latents.view(G, B, *combined_latents.shape[1:])

        # prompts = [prompts_list[i * B : (i+1) * B] for i in range(G)] 
        # rewards = [rewards_list[i * B : (i+1) * B] for i in range(G)]

        sample_data = SampleData(
            latents=combined_latents, # (total_on_policy_batch_size, num_timesteps, channels, height, width)
            prompts=prompts_list, # (total_on_policy_batch_size, )
            rewards=rewards_list, # (total_on_policy_batch_size,)
        )

        return sample_data
    
    def on_policy_data_store(self, on_policy_data : SampleData, epoch : int):
        rank = self.accelerator.process_index
        meta_path = os.path.join(self.buffer_dir, f"metadata_rank{rank}.pt")
        
        if os.path.exists(meta_path):
            meta_data = torch.load(meta_path)
        else:
            meta_data = []

        G = self.config.train_gradient_accumulation_steps
        B = self.config.train_batch_size

        # store on-policy per trajectory data to buffer 
        for i in range(int(G * B)):
            traj_idx = self.num_stored_traj % self.buf_size_per_gpu
            file_name = f"traj_rank{rank}_{traj_idx:06d}.pt"
            file_path = os.path.join(self.buffer_dir, file_name)

            buffer_dict = {
                "latents": on_policy_data.latents[i].detach().contiguous().clone().cpu(),
                "prompts": on_policy_data.prompts[i],
                "rewards": on_policy_data.rewards[i],
                "epoch": epoch,
            }

            torch.save(buffer_dict, file_path)

            meta_dict = {
                "file_name": file_name,
                "reward": float(buffer_dict["rewards"]),
                "epoch": epoch,
                "prompt": buffer_dict["prompts"],
            }

            if len(meta_data) < self.buf_size_per_gpu:
                meta_data.append(meta_dict)
            else:
                meta_data[traj_idx] = meta_dict

            self.num_stored_traj += 1

        torch.save(meta_data, meta_path)

    def combine_meta_data(self):
        meta_data = []
        for rank in range(self.accelerator.num_processes):
            meta_path = os.path.join(self.buffer_dir, f"metadata_rank{rank}.pt")
            if os.path.exists(meta_path):
                meta = torch.load(meta_path)
                meta_data.extend(meta)
                print(f"[Rank {rank}] meta_data: {len(meta)}")
        print(f"[Rank {rank}] all_meta_data: {len(meta_data)}")
        torch.save(meta_data, os.path.join(self.buffer_dir, "metadata.pt"))

    def get_off_policy_data(self):
        # load meta data
        meta_path = os.path.join(self.buffer_dir, "metadata.pt")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        
        meta_data = torch.load(meta_path)

        total_available_traj = len(meta_data)

        # calculate total number of trajectories to load
        G = self.config.train_gradient_accumulation_steps
        B_off = self.config.train_batch_size
        N = G * B_off

        print(f"[Rank {self.accelerator.process_index}] will sample {N} off-policy trajectories")

        if total_available_traj < N:
            raise ValueError(f"Not enough trajectories available in buffer. Need {N} but only {total_available_traj} available.")
        
        # sample N trajectories
        if self.config.buffer_method == 'on-policy':
            sampled_meta = sorted(meta_data, key=lambda x: x["epoch"], reverse=True)[:N]
        elif self.config.buffer_method == 'reward+gamma_PER':
            sampled_meta, timestep_indices = self.sample_reward_gamma_PER(meta_data, N)
        else: # random
            sampled_meta = random.sample(meta_data, N)

        # load traj
        latents = []
        prompts = []

        for meta in sampled_meta:
            file_path = os.path.join(self.buffer_dir, meta["file_name"])
            buffer_dict = torch.load(file_path, map_location=self.accelerator.device)

            latents.append(buffer_dict["latents"])
            prompts.append(buffer_dict["prompts"])

        # change latents format to [batch_size, num_timesteps, channels, height, width]
        latents = torch.stack(latents, dim=0)
        latents = latents.view(G, B_off, *latents.shape[1:])

        prompts = [prompts[i * B_off : (i+1) * B_off] for i in range(G)]

        sample_data = SampleData(
            latents=latents, # (accumulation, off_policy_batch, num_timesteps, channels, height, width)
            prompts=prompts, # (accumulation, off_policy_batch)
        )
        if self.config.buffer_method == 'reward+gamma_PER':
            timestep_indices = [timestep_indices[i * B_off : (i+1) * B_off] for i in range(G)]
            sample_data.timestep_indices = timestep_indices
        return sample_data


    def sample_reward_gamma_PER(self, meta_data, N):
        '''
        Sample trajectory indices proportional to reward * gamma^(num_timesteps-1).
        '''
        # number of trajectories in buffer
        num_traj = len(meta_data)
        if num_traj == 0:
            raise ValueError("No trajectories available in meta_data.")

        # timesteps per trajectory (use configured sampling steps)
        T = int(self.config.sample_num_steps)
        gamma = float(self.config.sqdf_gamma)

        # rewards per trajectory (shape: [num_traj])
        rewards = torch.tensor([float(m["reward"]) for m in meta_data], dtype=torch.float32)
        rewards = torch.clamp_min(rewards, 0.0)  # ensure non-negative

        # flip gamma
        gamma_weights = torch.pow(torch.tensor(gamma, dtype=torch.float32), torch.arange(T, dtype=torch.float32)).flip(dims=[0])

        # build 2D matrix: [num_traj, T] with values = reward * gamma^t
        weight_matrix = rewards.view(num_traj, 1) * gamma_weights.view(1, T)

        # flatten and normalize to probabilities
        flat_weights = weight_matrix.flatten()
        total = flat_weights.sum()
        probs = flat_weights / total

        # sample N indices (with replacement to allow duplicates if needed)
        N = int(N)
        flat_indices = torch.multinomial(probs, num_samples=N, replacement=True)

        # map flat indices back to (traj_idx, timestep_idx)
        traj_indices = (flat_indices // T).tolist()
        timestep_indices = (flat_indices % T).tolist()

        # collect sampled meta corresponding to traj indices
        sampled_meta = [meta_data[i] for i in traj_indices]

        # return meta list and their per-trajectory timestep indices
        return sampled_meta, timestep_indices
