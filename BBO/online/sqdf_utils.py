import contextlib
import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm

from diffusers.utils import BaseOutput
from PIL import Image


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
class SQDFSchedulerOutput:
    """
    Output class for the diffusers scheduler to be finetuned with the DDPO trainer

    Args:
        latents (`torch.Tensor`):
            Predicted sample at the previous timestep. Shape: `(batch_size, num_channels, height, width)`
        log_probs (`torch.Tensor`):
            Log probability of the above mentioned sample. Shape: `(batch_size)`
    """
    prev_sample: torch.Tensor 
    KL_loss_term: torch.Tensor 

@dataclass
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

@dataclass
class SampleData:
    latents: torch.Tensor  
    prompts: List[str]  
    rewards: Optional[List[float]] = None 


def _left_broadcast(input_tensor, shape):
    """
    As opposed to the default direction of broadcasting (right to left), this function broadcasts
    from left to right
        Args:
            input_tensor (`torch.FloatTensor`): is the tensor to broadcast
            shape (`Tuple[int]`): is the shape to broadcast to
    """
    input_ndim = input_tensor.ndim
    if input_ndim > len(shape):
        raise ValueError(
            "The number of dimensions of the tensor to broadcast cannot be greater than the length of the shape to broadcast to"
        )
    return input_tensor.reshape(input_tensor.shape + (1,) * (len(shape) - input_ndim)).broadcast_to(shape)

def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        self.alphas_cumprod.gather(0, prev_timestep),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List[torch.Generator], torch.Generator]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
) -> torch.Tensor:
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                warnings.warn(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

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
    
    # print(f"pipeline : {self.alphas_cumprod.device}")
    # print(f"timestep : {prev_timestep.device}")

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep) #* scalar
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        self.alphas_cumprod.gather(0, prev_timestep),
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
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2)+ 0.000001)
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

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
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep) #* scalar
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        self.alphas_cumprod.gather(0, prev_timestep),
        self.final_alpha_cumprod,
    )

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
    variance = _get_variance(self, timestep, prev_timestep) # scalar
    std_dev_t = eta * variance ** (0.5)

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
        
        prev_sample = prev_sample_mean + std_dev_t * variance_noise #* (batch, channel, height, width)

    # KL_loss_term 
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

    # # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
    # # Noise is not used on the final timestep of the timestep schedule.
    # # This also means that noise is not used for one-step sampling.
    # if self.step_index != self.num_inference_steps - 1:
    #     noise = randn_tensor(
    #         model_output.shape, generator=generator, device=model_output.device, dtype=denoised.dtype
    #     )
    #     prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
    # else:
    #     prev_sample = denoised

    # # upon completion increase step index by one
    # self._step_index += 1

    # if not return_dict:
    #     return (prev_sample, denoised)

    # return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)
    return denoised

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
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep)
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

def sqdf_pipeline_step_with_grad(
    config,
    accelerator,
    training_unet,
    ref_unet,
    consistency_unet,
    vae,
    image_processor,
    scheduler,
    consistency_scheduler,
    eta: float = 1.0,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    train_neg_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    use_consistency_model: bool = True,
    do_sqdf1: bool = False
):    
    
    # DDPM Scheduler
    timesteps = scheduler.timesteps
    num_inference_steps = len(timesteps)

    # 
    batch_size = latents.shape[0]

    # 6. Denoising loop
    all_latents = [] 
    all_latents.append(latents)
    all_log_probs = []
    with torch.no_grad():
            for i, t in tqdm(
                enumerate(timesteps), 
                total=len(timesteps),
                disable=not accelerator.is_local_main_process,
                ):
                
                # predict the noise residual
                if config.grad_checkpoint:
                    noise_pred_uncond = checkpoint.checkpoint(training_unet, latents, t, train_neg_prompt_embeds, use_reentrant=False).sample
                    noise_pred_cond = checkpoint.checkpoint(training_unet, latents, t, prompt_embeds, use_reentrant=False).sample

                else:
                    noise_pred_uncond = training_unet(latents, t, train_neg_prompt_embeds).sample
                    noise_pred_cond = training_unet(latents, t, prompt_embeds).sample
                
                # perform guidance
                grad = (noise_pred_cond - noise_pred_uncond)
                
                
                noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
                
                # compute the previous noisy sample x_t -> x_t-1 
                scheduler_output = scheduler_step(scheduler, noise_pred, t, latents, eta)
                latents = scheduler_output.latents #*(batch, channel, height, width)
                # log_prob = scheduler_output.log_probs

                all_latents.append(latents)
                # all_log_probs.append(log_prob)

                if i == len(timesteps) - 2: 
                    break
    
    all_latents = torch.stack(all_latents, dim=0) #* (num_timesteps, batch, channel, height, width)
    all_latents = all_latents.permute(1, 0, 2, 3, 4) #* (batch, num_timesteps, channel, height, width)
    # neg_prompt_embeds, pos_prompt_embeds = prompt_embeds.chunk(2)

    # sampling from on-policy data by randomly choose t
    if do_sqdf1:
        timestep_index = torch.tensor(49).repeat(batch_size)
    else:
        timestep_index = torch.randint(0, num_inference_steps, (batch_size,), device=latents.device)

    # x_t -> x_t-1 by training-unet
    latents = all_latents[torch.arange(batch_size), timestep_index]
    # _gstat(latents, "x_t", accelerator)
    # _add_grad_hook(latents, "x_t", accelerator)

    timestep_gamma = num_inference_steps - timestep_index - 1
    t = scheduler.timesteps[timestep_index]
    
    if config.grad_checkpoint:
        noise_pred_uncond = checkpoint.checkpoint(training_unet, latents, t, train_neg_prompt_embeds, use_reentrant=False).sample
        noise_pred_cond = checkpoint.checkpoint(training_unet, latents, t, prompt_embeds, use_reentrant=False).sample
        
        ref_noise_pred_uncond = checkpoint.checkpoint(ref_unet,latents, t, train_neg_prompt_embeds, use_reentrant=False).sample
        ref_noise_pred_cond = checkpoint.checkpoint(ref_unet,latents, t, prompt_embeds, use_reentrant=False).sample
        
    else:
        noise_pred_uncond = training_unet(latents, t, train_neg_prompt_embeds).sample
        noise_pred_cond = training_unet(latents, t, prompt_embeds).sample
    
        ref_noise_pred_uncond = ref_unet(latents, t, train_neg_prompt_embeds).sample
        ref_noise_pred_cond = ref_unet(latents, t, prompt_embeds).sample

    grad = (noise_pred_cond - noise_pred_uncond)
    old_grad = (ref_noise_pred_cond - ref_noise_pred_uncond)
    
    noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
    ref_noise_pred = ref_noise_pred_uncond + config.sd_guidance_scale * old_grad 

    
    scheduler_output = sqdf_scheduler_step(scheduler, noise_pred, ref_noise_pred, t, latents, eta)
    KL_loss_term = scheduler_output.KL_loss_term #* (batch)


    # x_t-1 -> x0 by ref unet
    prev_timestep_index = torch.clamp(timestep_index + 1, max=scheduler.timesteps.shape[0] - 1)

    if use_consistency_model: # default = True
        consistency_timesteps = consistency_scheduler.timesteps
        latent_model_input = torch.cat([scheduler_output.prev_sample] * 2) # for classifier-free guidance
    
        # 
        prev_t = scheduler.timesteps[prev_timestep_index]
        diff = torch.abs(prev_t.unsqueeze(1) - torch.tensor(consistency_timesteps).unsqueeze(0))
        closest_idx = torch.argmin(diff, dim=1)
        t_mapped = consistency_timesteps[closest_idx]
        t_expanded = torch.cat([t_mapped] * 2) # for classifier-free guidance
        enc_2b = torch.cat([train_neg_prompt_embeds, prompt_embeds], dim=0) 
        # 
        cm_guidance_scale = 1
        timestep_cond = None

        # latent_model_input : x_t-1
        model_pred = consistency_unet(
            latent_model_input,
            t_expanded,
            timestep_cond=timestep_cond,
            encoder_hidden_states=enc_2b,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]

        model_pred_uncond, model_pred_text = model_pred.chunk(2)
        model_pred = model_pred_uncond + cm_guidance_scale * (model_pred_text - model_pred_uncond)
        

        x_0_hat = lcm_scheduler_step_from_xt_to_x0(consistency_scheduler,
                                            model_pred,
                                            t_mapped,
                                            scheduler_output.prev_sample)

    else:
        # x_0 = scheduler_output.x0_hat #* (batch, channel, height, width)   
        latent_model_input = scheduler_output.prev_sample 
        t_expanded = scheduler.timesteps[prev_timestep_index]


        if config.grad_checkpoint:            
            ref_noise_pred_uncond = checkpoint.checkpoint(ref_unet,latent_model_input, t_expanded, train_neg_prompt_embeds, use_reentrant=False).sample
            ref_noise_pred_cond = checkpoint.checkpoint(ref_unet,latent_model_input, t_expanded, prompt_embeds, use_reentrant=False).sample
        
        else:
            ref_noise_pred_uncond = ref_unet(latent_model_input, t_expanded, train_neg_prompt_embeds).sample
            ref_noise_pred_cond = ref_unet(latent_model_input, t_expanded, prompt_embeds).sample


        old_grad = (ref_noise_pred_cond - ref_noise_pred_uncond)
        ref_noise_pred = ref_noise_pred_uncond + config.sd_guidance_scale * old_grad 
        
        predicted_latents = last_latent_prediction(scheduler, ref_noise_pred, scheduler.timesteps[prev_timestep_index], scheduler_output.prev_sample)
        x_0_hat = predicted_latents.latents #*(batch, channel, height, width)
    
    mask = (timestep_gamma == 0).view(-1, 1, 1, 1)
    x_0 = torch.where(mask, scheduler_output.prev_sample, x_0_hat)
    image = vae.decode(x_0.to(vae.dtype) / 0.18215).sample

    return SQDFPipelineOutput(image, KL_loss_term, timestep_gamma)

def buffer_sqdf_pipeline_step_with_grad(
    config,
    accelerator,
    training_unet,
    ref_unet,
    consistency_unet,
    vae,
    image_processor,
    scheduler,
    consistency_scheduler,
    eta: float = 1.0,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    train_neg_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    use_consistency_model: bool = True,
    do_sqdf1: bool = False,
    all_latents: Optional[torch.FloatTensor] = None,
    timestep_indices: Optional[torch.LongTensor] = None,
):    
    # DDPM Scheduler
    timesteps = scheduler.timesteps
    num_inference_steps = len(timesteps)

    # 
    batch_size = prompt_embeds.shape[0]

    if config.buffer_method == 'reward+gamma_PER':
        timestep_index = torch.tensor(timestep_indices, device=latents.device)
    else:
        timestep_index = torch.randint(0, num_inference_steps, (batch_size,), device=latents.device)
    latents = all_latents[torch.arange(batch_size), timestep_index]

    timestep_gamma = num_inference_steps - timestep_index - 1
    t = scheduler.timesteps[timestep_index]
    
    if config.grad_checkpoint:
        noise_pred_uncond = checkpoint.checkpoint(training_unet, latents, t, train_neg_prompt_embeds, use_reentrant=False).sample
        noise_pred_cond = checkpoint.checkpoint(training_unet, latents, t, prompt_embeds, use_reentrant=False).sample
        
        ref_noise_pred_uncond = checkpoint.checkpoint(ref_unet,latents, t, train_neg_prompt_embeds, use_reentrant=False).sample
        ref_noise_pred_cond = checkpoint.checkpoint(ref_unet,latents, t, prompt_embeds, use_reentrant=False).sample
        
    else:
        noise_pred_uncond = training_unet(latents, t, train_neg_prompt_embeds).sample
        noise_pred_cond = training_unet(latents, t, prompt_embeds).sample
    
        ref_noise_pred_uncond = ref_unet(latents, t, train_neg_prompt_embeds).sample
        ref_noise_pred_cond = ref_unet(latents, t, prompt_embeds).sample

    grad = (noise_pred_cond - noise_pred_uncond)
    old_grad = (ref_noise_pred_cond - ref_noise_pred_uncond)
    
    noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
    ref_noise_pred = ref_noise_pred_uncond + config.sd_guidance_scale * old_grad 

    
    scheduler_output = sqdf_scheduler_step(scheduler, noise_pred, ref_noise_pred, t, latents, eta)
    KL_loss_term = scheduler_output.KL_loss_term #* (batch)

    # x_t-1 -> x0 by ref unet
    prev_timestep_index = torch.clamp(timestep_index + 1, max=scheduler.timesteps.shape[0] - 1)

    if use_consistency_model: # default = True
        consistency_timesteps = consistency_scheduler.timesteps
        latent_model_input = torch.cat([scheduler_output.prev_sample] * 2) # for classifier-free guidance
    
        prev_t = scheduler.timesteps[prev_timestep_index]
        diff = torch.abs(prev_t.unsqueeze(1) - torch.tensor(consistency_timesteps).unsqueeze(0))
        closest_idx = torch.argmin(diff, dim=1)
        t_mapped = consistency_timesteps[closest_idx]
        t_expanded = torch.cat([t_mapped] * 2) # for classifier-free guidance
        enc_2b = torch.cat([train_neg_prompt_embeds, prompt_embeds], dim=0) 
        # 
        cm_guidance_scale = 1
        timestep_cond = None

        # latent_model_input : x_t-1
        model_pred = consistency_unet(
            latent_model_input,
            t_expanded,
            timestep_cond=timestep_cond,
            encoder_hidden_states=enc_2b,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]

        model_pred_uncond, model_pred_text = model_pred.chunk(2)
        model_pred = model_pred_uncond + cm_guidance_scale * (model_pred_text - model_pred_uncond)
        

        x_0_hat = lcm_scheduler_step_from_xt_to_x0(consistency_scheduler,
                                            model_pred,
                                            t_mapped,
                                            scheduler_output.prev_sample)

    else:
        # x_0 = scheduler_output.x0_hat #* (batch, channel, height, width)   
        latent_model_input = scheduler_output.prev_sample 
        t_expanded = scheduler.timesteps[prev_timestep_index]


        if config.grad_checkpoint:            
            ref_noise_pred_uncond = checkpoint.checkpoint(ref_unet,latent_model_input, t_expanded, train_neg_prompt_embeds, use_reentrant=False).sample
            ref_noise_pred_cond = checkpoint.checkpoint(ref_unet,latent_model_input, t_expanded, prompt_embeds, use_reentrant=False).sample
        
        else:
            ref_noise_pred_uncond = ref_unet(latent_model_input, t_expanded, train_neg_prompt_embeds).sample
            ref_noise_pred_cond = ref_unet(latent_model_input, t_expanded, prompt_embeds).sample


        old_grad = (ref_noise_pred_cond - ref_noise_pred_uncond)
        ref_noise_pred = ref_noise_pred_uncond + config.sd_guidance_scale * old_grad 
        
        predicted_latents = last_latent_prediction(scheduler, ref_noise_pred, scheduler.timesteps[prev_timestep_index], scheduler_output.prev_sample)
        x_0_hat = predicted_latents.latents #*(batch, channel, height, width)
    
    mask = (timestep_gamma == 0).view(-1, 1, 1, 1)
    x_0 = torch.where(mask, scheduler_output.prev_sample, x_0_hat)
    image = vae.decode(x_0.to(vae.dtype) / 0.18215).sample

    return SQDFPipelineOutput(image, KL_loss_term, timestep_gamma)

def _generate_sample(
    config,
    accelerator,
    training_unet,
    pipeline,
    eta: float = 1.0,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    train_neg_prompt_embeds: Optional[torch.FloatTensor] = None,
):    
    
    # DDPM Scheduler
    timesteps = pipeline.scheduler.timesteps
    num_inference_steps = len(timesteps)
    # 
    batch_size = latents.shape[0]

    # 6. Denoising loop
    all_latents = []
    all_latents.append(latents)
    with torch.no_grad():
            for i, t in tqdm(
                enumerate(timesteps), 
                total=len(timesteps),
                disable=not accelerator.is_local_main_process,
                ):
                
                # predict the noise residual
                if config.grad_checkpoint:
                    noise_pred_uncond = checkpoint.checkpoint(training_unet, latents, t, train_neg_prompt_embeds, use_reentrant=False).sample
                    noise_pred_cond = checkpoint.checkpoint(training_unet, latents, t, prompt_embeds, use_reentrant=False).sample

                else:
                    noise_pred_uncond = training_unet(latents, t, train_neg_prompt_embeds).sample
                    noise_pred_cond = training_unet(latents, t, prompt_embeds).sample
                
                # perform guidance
                grad = (noise_pred_cond - noise_pred_uncond)
                
                
                noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
                
                # compute the previous noisy sample x_t -> x_t-1 
                scheduler_output = scheduler_step(pipeline.scheduler, noise_pred, t, latents, eta)
                latents = scheduler_output.latents #*(batch, channel, height, width)
                # log_prob = scheduler_output.log_probs

                all_latents.append(latents)
                # all_log_probs.append(log_prob)
    imgs = pipeline.vae.decode(latents.to(pipeline.vae.dtype) / 0.18215).sample

    all_latents = all_latents[0:num_inference_steps]
    all_latents = torch.stack(all_latents, dim=0) #* (num_timesteps, batch, channel, height, width)
    all_latents = all_latents.permute(1, 0, 2, 3, 4)


    return imgs, all_latents

def get_on_policy_data(epoch, # global_step
                       config,
                       accelerator, 
                       training_unet,
                       pipeline,
                       inference_dtype,
                       prompt_fn,
                       train_neg_prompt_embeds,
                       online_loss_fn,
                       exp_dataset):
    latnets_list = []
    prompts_list = []
    rewards_list = []
    G = config.train.gradient_accumulation_steps # accumulation

    if epoch > 0:
        B = config.train.on_policy_batch_size
    else:
        B = config.train.batch_size_per_gpu_available
    total_on_policy_batch_size = B * G
    print(f"[Rank {accelerator.process_index}] will generate {int(total_on_policy_batch_size)} on-policy trajectories")

    num_iterations = (total_on_policy_batch_size + config.train.batch_size_per_gpu_available - 1) // config.train.batch_size_per_gpu_available
    num_iterations = int(num_iterations)


    for i in range(num_iterations):
        current_batch_size = min(config.train.batch_size_per_gpu_available, total_on_policy_batch_size - i * config.train.batch_size_per_gpu_available)
        
        latent = torch.randn((current_batch_size, 4, 64, 64),
        device=accelerator.device, dtype=inference_dtype)    

        prompts, prompt_metadata = zip(
            *[prompt_fn() for _ in range(current_batch_size)]
        )

        prompt_ids = pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)   

        # pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0]  

        imgs, all_latents = _generate_sample(
                                config,
                                accelerator,
                                training_unet,
                                pipeline,
                                eta = 1.0,
                                latents = latent,
                                prompt_embeds = prompt_embeds,
                                train_neg_prompt_embeds = train_neg_prompt_embeds,
                            )

        _, rewards = online_loss_fn(imgs, config, exp_dataset)

        latnets_list.append(all_latents)
        prompts_list.extend(prompts)
        rewards_list.extend(rewards.tolist())


    combined_latents = torch.cat(latnets_list, dim=0)  # (total_on_policy_batch_size, num_timesteps, channels, height, width)

    sample_data = SampleData( 
        latents=combined_latents, # (total_on_policy_batch_size, num_timesteps, channels, height, width)
        prompts=prompts_list, # (total_on_policy_batch_size, )
        rewards=rewards_list, # (total_on_policy_batch_size,)
    )

    return sample_data

def on_policy_data_store(config,
                         accelerator, 
                         on_policy_data : SampleData, 
                         epoch : int):
    rank = accelerator.process_index
    meta_path = os.path.join(config.buffer_dir, f"metadata_rank{rank}.pt")
    
    if os.path.exists(meta_path):
        meta_data = torch.load(meta_path)
    else:
        meta_data = []

    G = config.train.gradient_accumulation_steps # accumulation
    if epoch > 0:
        B = config.train.on_policy_batch_size
    else:
        B = config.train.batch_size_per_gpu_available

    # store on-policy per trajectory data to buffer 
    for i in range(int(G * B)):
            traj_idx = config.num_stored_traj % config.buf_size_per_gpu
            file_name = f"traj_rank{rank}_{traj_idx:06d}.pt"
            file_path = os.path.join(config.buffer_dir, file_name)

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

            if len(meta_data) < config.buf_size_per_gpu:
                meta_data.append(meta_dict)
            else:
                meta_data[traj_idx] = meta_dict

            config.num_stored_traj += 1

    torch.save(meta_data, meta_path)

def combine_meta_data(config,
                      accelerator):
    meta_data = []
    for rank in range(accelerator.num_processes):
        meta_path = os.path.join(config.buffer_dir, f"metadata_rank{rank}.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path)
            meta_data.extend(meta)
            print(f"[Rank {rank}] meta_data: {len(meta)}")
    print(f"[Rank {rank}] all_meta_data: {len(meta_data)}")
    torch.save(meta_data, os.path.join(config.buffer_dir, "metadata.pt"))

def sample_reward_gamma_PER(config, meta_data, N):
    # number of trajectories in buffer
    num_traj = len(meta_data)
    if num_traj == 0:
        raise ValueError("No trajectories available in meta_data.")

    # timesteps per trajectory (use configured sampling steps)
    T = int(50)
    gamma = float(config.sqdf_gamma)

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

def get_off_policy_data(config,
                        accelerator,):
    # load meta data
    meta_path = os.path.join(config.buffer_dir, "metadata.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")
    
    meta_data = torch.load(meta_path)

    total_available_traj = len(meta_data)

    # calculate total number of trajectories to load
    G = config.train.gradient_accumulation_steps
    B_off = config.train.batch_size_per_gpu_available
    N = G * B_off

    print(f"[Rank {accelerator.process_index}] will sample {N} off-policy trajectories")

    if total_available_traj < N:
        raise ValueError(f"Not enough trajectories available in buffer. Need {N} but only {total_available_traj} available.")
    
    ############################
    # sample N trajectories
    if config.buffer_method == 'random':
        sampled_meta = random.sample(meta_data, N)
    elif config.buffer_method == 'on-policy':
        sampled_meta = sorted(meta_data, key=lambda x: x["epoch"], reverse=True)[:N]
    elif config.buffer_method == 'reward':
        sampled_meta = sorted(meta_data, key=lambda x: x["reward"], reverse=True)[:N]
    elif config.buffer_method == 'on-policy+random':
        latest_epoch = max(m["epoch"] for m in meta_data)
        current = [m for m in meta_data if m["epoch"] == latest_epoch]
        others  = [m for m in meta_data if m["epoch"] != latest_epoch]
        if len(current) >= N:
            sampled_meta = random.sample(current, N)
        else:
            need = N - len(current)
            sampled_meta = current + random.sample(others, need)
    elif config.buffer_method == 'reward+gamma_PER':
        sampled_meta, timestep_indices = sample_reward_gamma_PER(config, meta_data, N)
    ############################

    # load traj
    latents = []
    prompts = []

    for meta in sampled_meta:
        file_path = os.path.join(config.buffer_dir, meta["file_name"])
        buffer_dict = torch.load(file_path, map_location=accelerator.device)

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
    if config.buffer_method == 'reward+gamma_PER':
        timestep_indices = [timestep_indices[i * B_off : (i+1) * B_off] for i in range(G)]
        sample_data.timestep_indices = timestep_indices

    return sample_data