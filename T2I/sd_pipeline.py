import contextlib
import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from transformers.utils import is_peft_available

from trl.core import randn_tensor
from trl.models.sd_utils import convert_state_dict_to_diffusers
from trl.models.modeling_sd_base import DefaultDDPOStableDiffusionPipeline, DDPOSchedulerOutput, _left_broadcast, _get_variance
from diffusers import LCMScheduler, AutoPipelineForText2Image

from diffusers.utils import BaseOutput

import inspect

if is_peft_available():
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

@dataclass
class DDPOSchedulerOutput:
    """
    Output class for the diffusers scheduler to be finetuned with the DDPO trainer

    Args:
        latents (`torch.Tensor`):
            Predicted sample at the previous timestep. Shape: `(batch_size, num_channels, height, width)`
        log_probs (`torch.Tensor`):
            Log probability of the above mentioned sample. Shape: `(batch_size)`
    """

    latents: torch.Tensor
    log_probs: torch.Tensor

@dataclass
class DDPOPipelineOutput:
    """
    Output class for the diffusers pipeline to be finetuned with the DDPO trainer

    Args:
        images (`torch.Tensor`):
            The generated images.
        latents (`List[torch.Tensor]`):
            The latents used to generate the images.
        log_probs (`List[torch.Tensor]`):
            The log probabilities of the latents.

    """

    images: torch.Tensor
    latents: torch.Tensor
    log_probs: torch.Tensor
    KL_loss_term: Optional[torch.Tensor] = None

@dataclass
class SQDFSchedulerOutput:
    """
    Output class for the diffusers scheduler to be finetuned with the DDPO trainer

    Args:
        latents (`torch.Tensor`):
            Predicted sample at the previous timestep. Shape: `(batch_size, num_channels, height, width)`
        log_probs (`torch.Tensor`):
            Log probability of the above mentioned sample. Shape: `(batch_size)`
    """
    prev_sample: torch.Tensor #* x_t-1
    # x0_hat: torch.Tensor #* x_t-1 -> x_0^hat
    KL_loss_term: torch.Tensor #* KL loss
    
@dataclass
class SQDFPipelineOutput:
    """
    Output class for the diffusers pipeline to be finetuned with the DDPO trainer

    Args:
        images (`torch.Tensor`):
            The generated images.
        latents (`List[torch.Tensor]`):
            The latents used to generate the images.
        log_probs (`List[torch.Tensor]`):
            The log probabilities of the latents.

    """

    images: torch.Tensor
    KL_loss_term: torch.Tensor
    time_tensor: torch.Tensor
    all_latents: Optional[torch.Tensor] = None
    pos_prompt_embeds: Optional[torch.Tensor] = None

class LCMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    denoised: Optional[torch.Tensor] = None


def make_uniform_subsequences(
    timesteps: torch.LongTensor,              # e.g. [981, 961, ..., 21, 1]
    timestep_index: torch.LongTensor,         # shape: [B], values in [0, len(timesteps)-1]
    n_steps: int
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Returns:
        sub_vals:   [B, n_steps] actual scheduler timestep values (descending), last is 0
        sub_index:  [B, n_steps] indices into the (appended) timesteps vector
    Notes:
        - Uniform spacing is done in *index* space between start_idx and end_idx (the 0-timestep).
        - If n_steps is larger than the remaining distance, the tail may repeat the last index.
    """
    assert n_steps >= 2, "n_steps must be >= 2 (start and 0 included)."
    n_steps = n_steps + 1
    device = timesteps.device
    B = timestep_index.shape[0]

    # 1) append 0 to timesteps if needed
    if timesteps[-1].item() != 0:
        # common schedulers end with 1; we just append 0 as requested
        timesteps = torch.cat([timesteps, torch.tensor([0], device=device, dtype=timesteps.dtype)], dim=0)

    T = timesteps.shape[0] - 1  # index of the appended 0
    # safety: original indices are in [0, T-1] because we appended one element
    assert timestep_index.min().item() >= 0 and timestep_index.max().item() <= T-1
  
    # 2) make evenly spaced indices in [start_idx, T] per batch
    start_idx_f = timestep_index.to(torch.float32)                     # [B]
    end_idx_f   = torch.full_like(start_idx_f, float(T), device=device)               # [B]
    grid        = torch.linspace(0, 1, steps=n_steps, device=device)   # [n_steps], 0..1

    # broadcast to [B, n_steps]
    idx_f = start_idx_f[:, None] + (end_idx_f - start_idx_f)[:, None] * grid[None, :]

    # 3) round to nearest integer indices and clamp to [0, T]
    idx = torch.round(idx_f).to(torch.long).clamp_(min=0, max=T)       # [B, n_steps]

    # 4) enforce strictly non-decreasing indices to avoid duplicates in the middle
    #    (if the gap is too small vs n_steps, the tail may stick at T)
    if n_steps > 1:
        # ensure idx[:, i] >= idx[:, i-1]
        for i in range(1, n_steps):
            idx[:, i] = torch.maximum(idx[:, i], idx[:, i-1])

    # 5) force the last index to be T (the appended 0)
    idx[:, -1] = T

    # 6) map to actual timestep values (descending); last column will be 0
    sub_vals = timesteps[idx]  # [B, n_steps]

    return sub_vals, idx

def sqdf_step_masked(
    self,
    model_output: torch.FloatTensor,
    t_cur: torch.Tensor,                                      # [B] (981, 961, ...)
    t_next: torch.Tensor,                                     # [B] (961, 941, ...)
    sample: torch.FloatTensor, #* latents x_t
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    B = sample.shape[0]

    # 0) masking for t_cur == t_next 
    step_mask = (t_cur != t_next)                              # [B] bool
    if not step_mask.any():
        return sample  # return all elements if all elements are skipped

    idx = step_mask.nonzero(as_tuple=True)[0]                  # batch_index for step
    keep = (~step_mask).nonzero(as_tuple=True)[0]              # batch_index for skip

    # 1) get sub-batch for step
    x  = sample[idx]                                           # [b,C,H,W]
    tc = t_cur[idx]                                            # [b]
    tn = t_next[idx]                                           # [b]
    model_output_sub = model_output[idx]

    # 3) gather alpha/beta (gather requires CPU indices)
    tc_cpu, tn_cpu = tc.to(torch.long).cpu(), tn.to(torch.long).cpu()
    alpha_prod_t = self.alphas_cumprod.gather(0, tc_cpu).to(device=x.device, dtype=x.dtype)   # [b]
    alpha_prod_t_prev = self.alphas_cumprod.gather(0, tn_cpu).to(device=x.device, dtype=x.dtype)   # [b]

    # broadcasting: [b] -> [b,1,1,1]
    alpha_prod_t = _left_broadcast(alpha_prod_t, x.shape).to(x.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, x.shape).to(x.device)
    beta_prod_t = 1 - alpha_prod_t

    # 4) x0^hat & epsilon
    if self.config.prediction_type == "epsilon":
        pred_original_x = (x - beta_prod_t ** (0.5) * model_output_sub) / alpha_prod_t ** (0.5) #* x0^hat
        pred_epsilon = model_output_sub
    else:
        raise ValueError(f"Unknown prediction_type: {self.config.prediction_type}")

    # 5) Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_x = self._threshold_sample(pred_original_x)
    elif self.config.clip_sample:
        pred_original_x = pred_original_x.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    if use_clipped_model_output:
        pred_epsilon = (x - alpha_prod_t.sqrt() * pred_original_x) / beta_prod_t.sqrt()

    # 6) std_dev_t = 0 since we use DDIM sampling
    std_dev_t = eta

    # 7) DDIM step
    pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * pred_epsilon

    prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_x + pred_sample_direction
    prev_sample = prev_sample_mean

    # 8) combine results (skip elements are kept)
    out = sample.clone()
    out[idx] = prev_sample.to(sample.dtype)
    
    return out

def last_latent_prediction(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
) -> DDPOSchedulerOutput:

    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # Get alpha and beta
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    beta_prod_t = 1 - alpha_prod_t

    # Compute x_0 estimate
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(
            f"Unsupported prediction_type: {self.config.prediction_type}"
        )

    return DDPOSchedulerOutput(
        latents=pred_original_sample.type(sample.dtype),
        log_probs=torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)  # dummy
    )


def one_step_noising(
        self,
        timestep: int, # t+1 (1)
        sample: torch.FloatTensor, # x0
):
    """
    get alpha, sigma for DDIM sampling of specific timestep
    Args: timestep (t)
    output : alpha_t+1, sigma_t+1
    """
    # 1. compute alphas, betas(=σ_t)
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    beta_prod_t = 1 - alpha_prod_t  

    # 2. make epsilon
    noise = randn_tensor(
            sample.shape,
            generator=None,
            device=sample.device,
            dtype=sample.dtype,
        )

    # 3. one step forward process
    x1 = alpha_prod_t ** (0.5) * sample + beta_prod_t ** (0.5) * noise

    return x1


def scheduler_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> DDPOSchedulerOutput:
    """

    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
            predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
            can directly provide the noise for the variance itself. This is useful for methods such as
            CycleDiffusion. (https://huggingface.co/papers/2210.05559)

    Returns:
        `DDPOSchedulerOutput`: the predicted sample at the previous timestep and the log probability of the sample
    """

    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://huggingface.co/papers/2010.02502
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device) #* (batch, channel, height, width)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://huggingface.co/papers/2010.02502
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep) #* sigma^2
    std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://huggingface.co/papers/2010.02502
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://huggingface.co/papers/2010.02502
    prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # log prob of prev_sample given prev_sample_mean and std_dev_t
    # log_prob = (
    #     -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2)+ 0.000001)
    #     - torch.log(std_dev_t)
    #     - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))
    # )
    # add small epsilon for numerical stability
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2)+ 0.000001)
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    # prev_sample : x_t-1
    # log_prob : log p(x_t-1 | x_t)

    return DDPOSchedulerOutput(prev_sample.type(sample.dtype), log_prob) 

def sqdf_scheduler_step(
    self,
    model_output: torch.FloatTensor,
    ref_model_output: torch.FloatTensor,
    timestep: torch.Tensor,
    sample: torch.FloatTensor, #* latents x_t
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> SQDFSchedulerOutput:
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://huggingface.co/papers/2010.02502
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu()) #* scalar
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    # broadcast alpha term for loss computation
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device) #* (batch, channel, height, width)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device) #* (batch, channel, height, width)

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://huggingface.co/papers/2010.02502
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5) #* x0^hat
        pred_epsilon = model_output
        ref_pred_original_sample = (sample - beta_prod_t ** (0.5) * ref_model_output) / alpha_prod_t ** (0.5)
        ref_pred_epsilon = ref_model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        ref_pred_original_sample = model_output
        ref_pred_epsilon = (sample - alpha_prod_t ** (0.5) * ref_pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        ref_pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * ref_model_output
        ref_pred_epsilon = (alpha_prod_t**0.5) * ref_model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
        ref_pred_original_sample = self._threshold_sample(ref_pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )
        ref_pred_original_sample = ref_pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep)  # scalar based on timesteps only
    std_dev_t = eta * variance ** (0.5)
    # broadcast sigma term for loss computation
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device) #* (batch, channel, height, width)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        ref_pred_epsilon = (sample - alpha_prod_t ** (0.5) * ref_pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://huggingface.co/papers/2010.02502
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon #* (batch, channel, height, width)
    ref_pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * ref_pred_epsilon #* (batch, channel, height, width)


    # 7. compute x_t without "random noise" of formula (12) from https://huggingface.co/papers/2010.02502
    prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction #* (batch, channel, height, width)
    #* mu_ref
    ref_prev_sample_mean = alpha_prod_t_prev ** (0.5) * ref_pred_original_sample + ref_pred_sample_direction #* (batch, channel, height, width)

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        #* x_t-1
        prev_sample = prev_sample_mean + std_dev_t * variance_noise #* (batch, channel, height, width)

    scalar = 1 / (std_dev_t.view(std_dev_t.shape[0], -1).mean(dim=1)**2)
    raw = (ref_prev_sample_mean - prev_sample_mean)**2 #* (batch, channel, height, width)
    raw = raw.view(raw.shape[0], -1).mean(dim=1)
    KL_loss_term = scalar * raw
    return SQDFSchedulerOutput(prev_sample.type(sample.dtype), KL_loss_term)

def lcm_scheduler_step_from_xt_to_x0(
    self,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    return_dict: bool = True,
) -> Union[LCMSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.Tensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
    Returns:
        [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.
    """

    # 2. compute alphas, betas
    self.alphas_cumprod = self.alphas_cumprod.to(sample.device)
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    beta_prod_t = 1 - alpha_prod_t

    # 3. Get scalings for boundary conditions
    c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
    c_skip = _left_broadcast(c_skip, sample.shape).to(sample.device)
    c_out = _left_broadcast(c_out, sample.shape).to(sample.device)

    # 4. Compute the predicted original sample x_0 based on the model parameterization
    if self.config.prediction_type == "epsilon":  # noise-prediction
        predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif self.config.prediction_type == "sample":  # x-prediction
        predicted_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":  # v-prediction
        predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction` for `LCMScheduler`."
        )

    # 5. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        predicted_original_sample = self._threshold_sample(predicted_original_sample)
    elif self.config.clip_sample:
        predicted_original_sample = predicted_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 6. Denoise model output using boundary conditions
    denoised = c_out * predicted_original_sample + c_skip * sample

    return denoised

def sqdf_pipeline_step_with_grad(
    pipeline,
    prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    gradient_checkpoint: bool = False,
    backprop_strategy: str = 'gaussian',
    backprop_kwargs: Dict[str, Any] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    use_consistency_model: bool = True,
):    
    if height is None and width is None:
        height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    
    with torch.no_grad():
        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = pipeline._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )


        prompt_embeds = pipeline._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps

        if use_consistency_model:
            pipeline.consistency_scheduler.set_timesteps(num_inference_steps, device=device)
            consistency_timesteps = pipeline.consistency_scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
    # 6. Denoising loop

    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    all_latents = [] 
    all_latents.append(latents)
    all_log_probs = []
    with torch.no_grad():
        with pipeline.progress_bar(total=num_inference_steps-1) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual

                if gradient_checkpoint:
                    noise_pred = checkpoint.checkpoint(
                        pipeline.unet,
                        latent_model_input,
                        t,
                        prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        use_reentrant=False,
                    )[0]
                else:
                    noise_pred = pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                
                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                # compute the previous noisy sample x_t -> x_t-1
                scheduler_output = scheduler_step(pipeline.scheduler, noise_pred, t, latents, eta)
                latents = scheduler_output.latents #*(batch, channel, height, width)
                log_prob = scheduler_output.log_probs

                all_latents.append(latents)
                all_log_probs.append(log_prob)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                if i == len(timesteps) - 2:  
                    break
    
    all_latents = torch.stack(all_latents, dim=0) #* (num_timesteps, batch, channel, height, width)
    all_latents = all_latents.permute(1, 0, 2, 3, 4) #* (batch, num_timesteps, channel, height, width)
    neg_prompt_embeds, pos_prompt_embeds = prompt_embeds.chunk(2)

    timestep_index = torch.randint(0, num_inference_steps, (batch_size,), device=latents.device)

    latents = all_latents[torch.arange(batch_size), timestep_index]
    timestep_gamma = num_inference_steps - timestep_index - 1
    t = pipeline.scheduler.timesteps[timestep_index]
    
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

    t_expanded = torch.cat([t] * 2) if do_classifier_free_guidance else t
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t_expanded)

    if gradient_checkpoint: #* action
        noise_pred = checkpoint.checkpoint(
            pipeline.unet,
            latent_model_input,
            t_expanded,
            prompt_embeds,
        )[0] #* (batchx2, channel, height, width)
        ref_noise_pred = checkpoint.checkpoint(
            pipeline.ref_unet,
            latent_model_input,
            t_expanded,
            prompt_embeds,
        )[0]
    else:
        noise_pred = pipeline.unet(
            latent_model_input,
            t_expanded,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]
        ref_noise_pred = pipeline.ref_unet(
            latent_model_input,
            t_expanded,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]

    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) #* (batch, channel, height, width)
        ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
        ref_noise_pred = ref_noise_pred_uncond + guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond) #* (batch, channel, height, width)

    if do_classifier_free_guidance and guidance_rescale > 0.0:
        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
        ref_noise_pred = rescale_noise_cfg(ref_noise_pred, ref_noise_pred_text, guidance_rescale=guidance_rescale)

    scheduler_output = sqdf_scheduler_step(pipeline.scheduler, noise_pred, ref_noise_pred, t, latents, eta)

    KL_loss_term = scheduler_output.KL_loss_term #* (batch)


    prev_timestep_index = torch.clamp(timestep_index + 1, max=pipeline.scheduler.timesteps.shape[0] - 1)

    if use_consistency_model: # default = True

        latent_model_input = torch.cat([scheduler_output.prev_sample] * 2) if do_classifier_free_guidance else scheduler_output.prev_sample
        # latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t) 
        # latent_model_input = scheduler_output.prev_sample

        prev_t = pipeline.scheduler.timesteps[prev_timestep_index]
        diff = torch.abs(prev_t.unsqueeze(1) - torch.tensor(consistency_timesteps).unsqueeze(0))
        closest_idx = torch.argmin(diff, dim=1)
        t_mapped = consistency_timesteps[closest_idx]
        t_expanded = torch.cat([t_mapped] * 2) if do_classifier_free_guidance else t_mapped
    
        # 
        cm_guidance_scale = 1
        timestep_cond = None
        if pipeline.consistency_unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(cm_guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipeline.consistency_unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latent_model_input.dtype)

        # pipeline.consistency_unet.to(device=device, dtype=latent_model_input.dtype)

        # latent_model_input : x_t-1
        model_pred = pipeline.consistency_unet(
            latent_model_input,
            t_expanded,
            timestep_cond=timestep_cond,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]

        if do_classifier_free_guidance:
            model_pred_uncond, model_pred_text = model_pred.chunk(2)
            model_pred = model_pred_uncond + cm_guidance_scale * (model_pred_text - model_pred_uncond)
        if do_classifier_free_guidance and guidance_rescale > 0.0:
            model_pred = rescale_noise_cfg(model_pred, model_pred_text, guidance_rescale=guidance_rescale)

        x_0_hat = lcm_scheduler_step_from_xt_to_x0(pipeline.consistency_scheduler,
                                            model_pred,
                                            t_mapped,
                                            scheduler_output.prev_sample)

    else:
        # x_0 = scheduler_output.x0_hat #* (batch, channel, height, width)   
        latent_model_input = torch.cat([scheduler_output.prev_sample] * 2) if do_classifier_free_guidance else scheduler_output.prev_sample

        t_expanded = torch.cat([pipeline.scheduler.timesteps[prev_timestep_index]] * 2) if do_classifier_free_guidance else pipeline.scheduler.timesteps[prev_timestep_index]

        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t_expanded)
             
        ref_noise_pred = pipeline.ref_unet(
            latent_model_input,
            t_expanded,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]
        if do_classifier_free_guidance:
            ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
            ref_noise_pred = ref_noise_pred_uncond + guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond)
        if do_classifier_free_guidance and guidance_rescale > 0.0:
            ref_noise_pred = rescale_noise_cfg(ref_noise_pred, ref_noise_pred_text, guidance_rescale=guidance_rescale)
        predicted_latents = last_latent_prediction(pipeline.scheduler, ref_noise_pred, pipeline.scheduler.timesteps[prev_timestep_index], scheduler_output.prev_sample)
        x_0_hat = predicted_latents.latents #*(batch, channel, height, width)
    
    mask = (timestep_gamma == 0).view(-1, 1, 1, 1)
    x_0 = torch.where(mask, scheduler_output.prev_sample, x_0_hat)

    x_0 = x_0.to(dtype=pipeline.vae.dtype, device=pipeline.vae.device)

    image = pipeline.vae.decode(x_0 / pipeline.vae.config.scaling_factor, return_dict=False)[0] #* (batch, channel, height, width)
    do_denormalize = [True] * image.shape[0]
    image = pipeline.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload last model to CPU
    if hasattr(pipeline, "final_offload_hook") and pipeline.final_offload_hook is not None:
        pipeline.final_offload_hook.offload()
    
    return SQDFPipelineOutput(image, KL_loss_term, timestep_gamma)

def buffer_sqdf_pipeline_step_with_grad(
    pipeline,
    prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    gradient_checkpoint: bool = False,
    backprop_strategy: str = 'gaussian',
    backprop_kwargs: Dict[str, Any] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    use_consistency_model: bool = True,
    all_latents: Optional[torch.FloatTensor] = None,
    buffer_method: str = '',
    gamma: float = 0.9,
    timestep_indices: Optional[torch.LongTensor] = None,
):    
    if height is None and width is None:
        height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    
    with torch.no_grad():
        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = pipeline._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_embeds = pipeline._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps

        if use_consistency_model:
            pipeline.consistency_scheduler.set_timesteps(num_inference_steps, device=device)
            consistency_timesteps = pipeline.consistency_scheduler.timesteps
    
    if buffer_method == 'reward+gamma_PER':
        timestep_index = torch.tensor(timestep_indices, device=device)
    else:
        timestep_index = torch.randint(0, num_inference_steps, (batch_size,), device=device)
        
    latents = all_latents[torch.arange(batch_size), timestep_index]

    timestep_gamma = num_inference_steps - timestep_index - 1
    t = pipeline.scheduler.timesteps[timestep_index]
    
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

    t_expanded = torch.cat([t] * 2) if do_classifier_free_guidance else t
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t_expanded)

    if gradient_checkpoint: #* action
        noise_pred = checkpoint.checkpoint(
            pipeline.unet,
            latent_model_input,
            t_expanded,
            prompt_embeds,
        )[0] #* (batchx2, channel, height, width)
        ref_noise_pred = checkpoint.checkpoint(
            pipeline.ref_unet,
            latent_model_input,
            t_expanded,
            prompt_embeds,
        )[0]
    else:
        noise_pred = pipeline.unet(
            latent_model_input,
            t_expanded,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]
        ref_noise_pred = pipeline.ref_unet(
            latent_model_input,
            t_expanded,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]

    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) #* (batch, channel, height, width)
        ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
        ref_noise_pred = ref_noise_pred_uncond + guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond) #* (batch, channel, height, width)

    if do_classifier_free_guidance and guidance_rescale > 0.0:
        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
        ref_noise_pred = rescale_noise_cfg(ref_noise_pred, ref_noise_pred_text, guidance_rescale=guidance_rescale)

    scheduler_output = sqdf_scheduler_step(pipeline.scheduler, noise_pred, ref_noise_pred, t, latents, eta)

    KL_loss_term = scheduler_output.KL_loss_term #* (batch)

    prev_timestep_index = torch.clamp(timestep_index + 1, max=pipeline.scheduler.timesteps.shape[0] - 1)

    if use_consistency_model: # default = True
        latent_model_input = torch.cat([scheduler_output.prev_sample] * 2) if do_classifier_free_guidance else scheduler_output.prev_sample
        # latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        # latent_model_input = scheduler_output.prev_sample

        prev_t = pipeline.scheduler.timesteps[prev_timestep_index]
        diff = torch.abs(prev_t.unsqueeze(1) - torch.tensor(consistency_timesteps).unsqueeze(0))
        closest_idx = torch.argmin(diff, dim=1)
        t_mapped = consistency_timesteps[closest_idx]
        t_expanded = torch.cat([t_mapped] * 2) if do_classifier_free_guidance else t_mapped
    
        # 
        cm_guidance_scale = 1
        timestep_cond = None
        if pipeline.consistency_unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(cm_guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipeline.consistency_unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latent_model_input.dtype)

        
        # pipeline.consistency_unet.to(device=device, dtype=latent_model_input.dtype)

        # latent_model_input : x_t-1
        model_pred = pipeline.consistency_unet(
            latent_model_input,
            t_expanded,
            timestep_cond=timestep_cond,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]

        if do_classifier_free_guidance:
            model_pred_uncond, model_pred_text = model_pred.chunk(2)
            model_pred = model_pred_uncond + cm_guidance_scale * (model_pred_text - model_pred_uncond)
        if do_classifier_free_guidance and guidance_rescale > 0.0:
            model_pred = rescale_noise_cfg(model_pred, model_pred_text, guidance_rescale=guidance_rescale)

        x_0_hat = lcm_scheduler_step_from_xt_to_x0(pipeline.consistency_scheduler,
                                            model_pred,
                                            t_mapped,
                                            scheduler_output.prev_sample)

    else:
        # x_0 = scheduler_output.x0_hat #* (batch, channel, height, width)   
        latent_model_input = torch.cat([scheduler_output.prev_sample] * 2) if do_classifier_free_guidance else scheduler_output.prev_sample

        t_expanded = torch.cat([pipeline.scheduler.timesteps[prev_timestep_index]] * 2) if do_classifier_free_guidance else pipeline.scheduler.timesteps[prev_timestep_index]

        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t_expanded)
               
        ref_noise_pred = pipeline.ref_unet(
            latent_model_input,
            t_expanded,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]
        if do_classifier_free_guidance:
            ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
            ref_noise_pred = ref_noise_pred_uncond + guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond)
        if do_classifier_free_guidance and guidance_rescale > 0.0:
            ref_noise_pred = rescale_noise_cfg(ref_noise_pred, ref_noise_pred_text, guidance_rescale=guidance_rescale)
        predicted_latents = last_latent_prediction(pipeline.scheduler, ref_noise_pred, pipeline.scheduler.timesteps[prev_timestep_index], scheduler_output.prev_sample)
        x_0_hat = predicted_latents.latents #*(batch, channel, height, width)
    
    mask = (timestep_gamma == 0).view(-1, 1, 1, 1)
    x_0 = torch.where(mask.to(scheduler_output.prev_sample.device), scheduler_output.prev_sample, x_0_hat)


    x_0 = x_0.to(dtype=pipeline.vae.dtype, device=pipeline.vae.device)

    image = pipeline.vae.decode(x_0 / pipeline.vae.config.scaling_factor, return_dict=False)[0] #* (batch, channel, height, width)
    do_denormalize = [True] * image.shape[0]
    image = pipeline.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload last model to CPU
    if hasattr(pipeline, "final_offload_hook") and pipeline.final_offload_hook is not None:
        pipeline.final_offload_hook.offload()

    return SQDFPipelineOutput(image, KL_loss_term, timestep_gamma)


def buffer_sqdf_pipeline_n_step_sampling(
    pipeline,
    prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    gradient_checkpoint: bool = False,
    backprop_strategy: str = 'gaussian',
    backprop_kwargs: Dict[str, Any] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    use_consistency_model: bool = True,
    all_latents: Optional[torch.FloatTensor] = None,
    buffer_method: str = '',
    gamma: float = 0.9,
    timestep_indices: Optional[torch.LongTensor] = None,
    n_steps: int = 4,
):    
    if height is None and width is None:
        height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    
    with torch.no_grad():
        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = pipeline._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_embeds = pipeline._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps

        if use_consistency_model:
            pipeline.consistency_scheduler.set_timesteps(num_inference_steps, device=device)
            consistency_timesteps = pipeline.consistency_scheduler.timesteps
    
    if buffer_method == 'reward+gamma_PER':
        timestep_index = torch.tensor(timestep_indices, device=device)
    else:
        timestep_index = torch.randint(0, num_inference_steps, (batch_size,), device=device)
      
        
    latents = all_latents[torch.arange(batch_size), timestep_index]

    timestep_gamma = num_inference_steps - timestep_index - 1
    t = pipeline.scheduler.timesteps[timestep_index]
    
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

    t_expanded = torch.cat([t] * 2) if do_classifier_free_guidance else t
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t_expanded)

    if gradient_checkpoint: #* action
        noise_pred = checkpoint.checkpoint(
            pipeline.unet,
            latent_model_input,
            t_expanded,
            prompt_embeds,
        )[0] #* (batchx2, channel, height, width)
        ref_noise_pred = checkpoint.checkpoint(
            pipeline.ref_unet,
            latent_model_input,
            t_expanded,
            prompt_embeds,
        )[0]
    else:
        noise_pred = pipeline.unet(
            latent_model_input,
            t_expanded,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]
        ref_noise_pred = pipeline.ref_unet(
            latent_model_input,
            t_expanded,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]

    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) #* (batch, channel, height, width)
        ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
        ref_noise_pred = ref_noise_pred_uncond + guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond) #* (batch, channel, height, width)

    if do_classifier_free_guidance and guidance_rescale > 0.0:
        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
        ref_noise_pred = rescale_noise_cfg(ref_noise_pred, ref_noise_pred_text, guidance_rescale=guidance_rescale)

    scheduler_output = sqdf_scheduler_step(pipeline.scheduler, noise_pred, ref_noise_pred, t, latents, eta)

    KL_loss_term = scheduler_output.KL_loss_term #* (batch)

    prev_timestep_index = torch.clamp(timestep_index + 1, max=pipeline.scheduler.timesteps.shape[0] - 1)

    # make sub_sequence for n-step DDIM sampling
    sub_sequence_timesteps, idx = make_uniform_subsequences(pipeline.scheduler.timesteps, prev_timestep_index, n_steps)


    # n-step DDIM sampling
    B, K = sub_sequence_timesteps.shape
    sub_sequence_sample = scheduler_output.prev_sample
    for k in range(K-1):
        t_cur = sub_sequence_timesteps[:, k]
        t_next = sub_sequence_timesteps[:, k+1]

        if (t_cur == t_next).all():
            continue

        latent_model_input = torch.cat([sub_sequence_sample] * 2) if do_classifier_free_guidance else sub_sequence_sample
        t_expanded = torch.cat([t_cur] * 2) if do_classifier_free_guidance else t_cur
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t_expanded)

        ref_noise_pred = pipeline.ref_unet(
            latent_model_input,
            t_expanded,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]
        if do_classifier_free_guidance:
            ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
            ref_noise_pred = ref_noise_pred_uncond + guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond)
        if do_classifier_free_guidance and guidance_rescale > 0.0:
            ref_noise_pred = rescale_noise_cfg(ref_noise_pred, ref_noise_pred_text, guidance_rescale=guidance_rescale)

        sub_sequence_sample = sqdf_step_masked(pipeline.scheduler, ref_noise_pred, t_cur, t_next, sub_sequence_sample)
    
    x_0 = sub_sequence_sample
    x_0 = x_0.to(dtype=pipeline.vae.dtype, device=pipeline.vae.device)

    image = pipeline.vae.decode(x_0 / pipeline.vae.config.scaling_factor, return_dict=False)[0] #* (batch, channel, height, width)
    do_denormalize = [True] * image.shape[0]
    image = pipeline.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    # Offload last model to CPU
    if hasattr(pipeline, "final_offload_hook") and pipeline.final_offload_hook is not None:
        pipeline.final_offload_hook.offload()

    return SQDFPipelineOutput(image, KL_loss_term, timestep_gamma)


def pipeline_step_with_grad(
    pipeline,
    prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    gradient_checkpoint: bool = True,
    backprop_strategy: str = 'gaussian',
    backprop_kwargs: Dict[str, Any] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    use_consistency_model: bool = False,
):
    r"""
    Function to get RGB image with gradients attached to the model weights.

    Args:
        prompt (`str` or `List[str]`, *optional*, defaults to `None`):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds` instead.
        height (`int`, *optional*, defaults to `pipeline.unet.config.sample_size * pipeline.vae_scale_factor`):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to `pipeline.unet.config.sample_size * pipeline.vae_scale_factor`):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to `50`):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to `7.5`):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        gradient_checkpoint (`bool`, *optional*, defaults to True):
            Adds gradient checkpointing to Unet forward pass. Reduces GPU memory consumption while slightly increasing the training time.
        backprop_strategy (`str`, *optional*, defaults to 'gaussian'):
            Strategy for backpropagation. Options: 'gaussian', 'uniform', 'fixed'.
        backprop_kwargs (`dict`, *optional*):
            Additional keyword arguments for backpropagation.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `pipeline.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.

    Examples:

    Returns:
        `DDPOPipelineOutput`: The generated image, the predicted latents used to generate the image and the associated log probabilities
    """

    backprop_timestep = -1
    
    while backprop_timestep >= num_inference_steps or backprop_timestep < 0:    
        if backprop_strategy == 'gaussian':
            backprop_timestep = int(torch.distributions.Normal(backprop_kwargs['mean'], backprop_kwargs['std']).sample().item())
        elif backprop_strategy == 'uniform':
            backprop_timestep = int(torch.randint(backprop_kwargs['min'], backprop_kwargs['max'], (1,)).item())
        elif backprop_strategy == 'fixed':
            backprop_timestep = int(backprop_kwargs['value'])
        elif backprop_strategy == 'refl':
            backprop_timestep = int(torch.randint(backprop_kwargs['min'], backprop_kwargs['max'], (1,)).item()) #* [min, max)
        elif backprop_strategy == 'draft+':
            backprop_timestep = int(backprop_kwargs['value'])
        elif backprop_strategy == 'draftlv':
            backprop_timestep = int(backprop_kwargs['value'])
            draftlv_loop = int(backprop_kwargs['loop'])
    
    if height is None and width is None:
        height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    
    with torch.no_grad():
        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = pipeline._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = pipeline._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps #* (num_inference_steps,)

        # 5. Prepare latent variables
        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents( #* (batch, channel, height, width)
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    if backprop_strategy == 'draft+':
        all_kl_terms = []
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            if gradient_checkpoint:
                noise_pred = checkpoint.checkpoint(
                    pipeline.unet,
                    latent_model_input,
                    t,
                    prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    use_reentrant=False,
                )[0]
            else:
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
            
            if i < backprop_timestep:
                noise_pred = noise_pred.detach()

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://huggingface.co/papers/2305.08891
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # draft+KL 50
            if backprop_strategy == 'draft+':
                if gradient_checkpoint: #* action
                    noise_pred_for_kl = checkpoint.checkpoint(
                        pipeline.unet,
                        latent_model_input,
                        t,
                        prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        use_reentrant=False,
                    )[0] #* (batchx2, channel, height, width)
                    with torch.no_grad():
                        ref_noise_pred = checkpoint.checkpoint(
                            pipeline.ref_unet,
                            latent_model_input,
                            t,
                            prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            use_reentrant=False,
                        )[0]
                else:
                    noise_pred_for_kl = pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                    with torch.no_grad():
                        ref_noise_pred = pipeline.ref_unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            return_dict=False,
                        )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond_for_kl, noise_pred_text_for_kl = noise_pred_for_kl.chunk(2)
                    noise_pred_for_kl = noise_pred_uncond_for_kl + guidance_scale * (noise_pred_text_for_kl - noise_pred_uncond_for_kl) #* (batch, channel, height, width)
                    ref_noise_pred_uncond_for_kl, ref_noise_pred_text_for_kl = ref_noise_pred.chunk(2)
                    ref_noise_pred_for_kl = ref_noise_pred_uncond_for_kl + guidance_scale * (ref_noise_pred_text_for_kl - ref_noise_pred_uncond_for_kl) #* (batch, channel, height, width)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred_for_kl = rescale_noise_cfg(noise_pred_for_kl, noise_pred_text_for_kl, guidance_rescale=guidance_rescale)
                    ref_noise_pred_for_kl = rescale_noise_cfg(ref_noise_pred_for_kl, ref_noise_pred_text_for_kl, guidance_rescale=guidance_rescale)

                scheduler_output = sqdf_scheduler_step(pipeline.scheduler, noise_pred_for_kl, ref_noise_pred_for_kl, t, latents, eta=1.0)
                KL_loss_term = scheduler_output.KL_loss_term #* (batch,)

                all_kl_terms.append(KL_loss_term)


            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = scheduler_step(pipeline.scheduler, noise_pred, t, latents, eta)
            latents = scheduler_output.latents #*(batch, channel, height, width)
            log_prob = scheduler_output.log_probs

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
            
            if backprop_strategy == 'refl' and i == backprop_timestep: 
                break

            if backprop_strategy == "draftlv" and i == backprop_timestep:
                noise_pred = noise_pred.detach()

    if backprop_strategy == 'draft+':
        total_KL_loss_term = torch.stack(all_kl_terms, dim=0).sum(dim=0)  # (T, B) -> (B,)

    if backprop_strategy == 'refl' and backprop_timestep < num_inference_steps - 1: 
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, pipeline.scheduler.timesteps[backprop_timestep])
        noise_pred = pipeline.unet( 
            latent_model_input,
            pipeline.scheduler.timesteps[backprop_timestep],
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        if do_classifier_free_guidance and guidance_rescale > 0.0:
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
        predicted_latents = last_latent_prediction(pipeline.scheduler, noise_pred, pipeline.scheduler.timesteps[backprop_timestep], latents)
        latents = predicted_latents.latents #*(batch, channel, height, width)

    if backprop_strategy == "draftlv":
        latents_list = []
        # draftlv loop 
        for i in range(draftlv_loop):
            x1 = one_step_noising(pipeline.scheduler, t, latents)
            predicted_latents = last_latent_prediction(pipeline.scheduler, noise_pred, t, x1)
            latents_list.append(predicted_latents.latents)
        
        # overwrite latents 
        latents = torch.cat(latents_list, dim=0) # [B*2, C, H, W]
        latents = latents.to(dtype=pipeline.vae.dtype)

    if not output_type == "latent":
        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = pipeline.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    image = pipeline.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload last model to CPU
    if hasattr(pipeline, "final_offload_hook") and pipeline.final_offload_hook is not None:
        pipeline.final_offload_hook.offload()

    if backprop_strategy == 'draft+':
        return DDPOPipelineOutput(image, all_latents, all_log_probs, total_KL_loss_term) 
    else:
        return DDPOPipelineOutput(image, all_latents, all_log_probs) 


class DiffusionPipeline(DefaultDDPOStableDiffusionPipeline):
    def __init__(self, pretrained_model_name: str, pretrained_model_revision: str = "main", use_lora: bool = True, use_consistency_model: bool = True, lora_rank: int = 4, backprop_strategy: str = ''):
        super().__init__(pretrained_model_name,pretrained_model_revision=pretrained_model_revision,use_lora=use_lora)
        
        self.lora_rank = lora_rank

        if backprop_strategy == 'sqdf' or backprop_strategy == 'draft+':
            self.sd_pipeline.ref_unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name,
                revision=pretrained_model_revision,
                subfolder="unet",
            ).eval()
            self.sd_pipeline.ref_unet.requires_grad_(False)
            self.sd_pipeline.ref_unet.to(self.unet.device)
            self.ref_unet = self.sd_pipeline.ref_unet

       # --------------  consistency model ------------------------
        if use_consistency_model:
            model_id = "runwayml/stable-diffusion-v1-5"
            adapter_id = "latent-consistency/lcm-lora-sdv1-5"

            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

            # load and fuse lcm lora
            pipe.load_lora_weights(adapter_id)
            pipe.fuse_lora()

            # add consistency_unet & LCMScheduler to sd_pipeline
            self.sd_pipeline.consistency_unet = pipe.unet.eval()
            self.sd_pipeline.consistency_unet.requires_grad_(False)
            self.sd_pipeline.consistency_unet.to(self.unet.device)
            self.consistency_unet = self.sd_pipeline.consistency_unet

            self.sd_pipeline.consistency_scheduler = pipe.scheduler
            self.consistency_scheduler = self.sd_pipeline.consistency_scheduler

    def get_trainable_layers(self):
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_rank, 
                lora_alpha=self.lora_rank, 
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.sd_pipeline.unet.add_adapter(lora_config)

            # To avoid accelerate unscaling problems in FP16.
            for param in self.sd_pipeline.unet.parameters():
                # only upcast trainable parameters (LoRA) into fp32
                if param.requires_grad:
                    param.data = param.to(torch.float32)
            return self.sd_pipeline.unet
        else:
            return self.sd_pipeline.unet

    def rgb_with_grad(self, *args, **kwargs) -> DDPOPipelineOutput:
        mode = kwargs.pop("mode", None)
        if mode == 'sqdf':
            return sqdf_pipeline_step_with_grad(self.sd_pipeline, *args, **kwargs)  
        elif mode == 'sqdf_buffer':
            return buffer_sqdf_pipeline_step_with_grad(self.sd_pipeline, *args, **kwargs)
        elif mode == 'sqdf_n_step':
            return buffer_sqdf_pipeline_n_step_sampling(self.sd_pipeline, *args, **kwargs)
        else:
            return pipeline_step_with_grad(self.sd_pipeline, *args, **kwargs)    
