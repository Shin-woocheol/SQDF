import os
import sys
import warnings
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple, List

from transformers import is_bitsandbytes_available, is_torchvision_available
from datetime import datetime
from trl.core import flatten_dict
from datetime import datetime


@dataclass
class AlignPropConfig:
    r"""
    Configuration class for the [`AlignPropTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(sys.argv[0])[: -len(".py")]`):
            Name of this experiment (defaults to the file name without the extension).
        run_name (`str`, *optional*, defaults to `""`):
            Name of this run.
        log_with (`Optional[Literal["wandb", "tensorboard"]]`, *optional*, defaults to `None`):
            Log with either `"wandb"` or `"tensorboard"`. Check
            [tracking](https://huggingface.co/docs/accelerate/usage_guides/tracking) for more details.
        log_image_freq (`int`, *optional*, defaults to `1`):
            Frequency for logging images.
        log_image_iter (`int`, *optional*, defaults to `1`):
            Number of times to repeat image logging.
        tracker_kwargs (`Dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the tracker (e.g., `wandb_project`).
        accelerator_kwargs (`Dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator.
        project_kwargs (`Dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator project config (e.g., `logging_dir`).
        tracker_project_name (`str`, *optional*, defaults to `"trl"`):
            Name of project to use for tracking.
        logdir (`str`, *optional*, defaults to `"logs"`):
            Top-level logging directory for checkpoint saving.
        num_epochs (`int`, *optional*, defaults to `100`):
            Number of epochs to train.
        save_freq (`int`, *optional*, defaults to `1`):
            Number of epochs between saving model checkpoints.
        num_checkpoint_limit (`int`, *optional*, defaults to `5`):
            Number of checkpoints to keep before overwriting old ones.
        mixed_precision (`str`, *optional*, defaults to `"fp16"`):
            Mixed precision training.
        allow_tf32 (`bool`, *optional*, defaults to `True`):
            Allow `tf32` on Ampere GPUs.
        resume_from (`str`, *optional*, defaults to `""`):
            Path to resume training from a checkpoint.
        sample_num_steps (`int`, *optional*, defaults to `50`):
            Number of sampler inference steps.
        sample_eta (`float`, *optional*, defaults to `1.0`):
            Eta parameter for the DDIM sampler.
        sample_guidance_scale (`float`, *optional*, defaults to `5.0`):
            Classifier-free guidance weight.
        train_use_8bit_adam (`bool`, *optional*, defaults to `False`):
            Whether to use the 8bit Adam optimizer from `bitsandbytes`.
        train_learning_rate (`float`, *optional*, defaults to `1e-3`):
            Learning rate.
        train_adam_beta1 (`float`, *optional*, defaults to `0.9`):
            Beta1 for Adam optimizer.
        train_adam_beta2 (`float`, *optional*, defaults to `0.999`):
            Beta2 for Adam optimizer.
        train_adam_weight_decay (`float`, *optional*, defaults to `1e-4`):
            Weight decay for Adam optimizer.
        train_adam_epsilon (`float`, *optional*, defaults to `1e-8`):
            Epsilon value for Adam optimizer.
        train_gradient_accumulation_steps (`int`, *optional*, defaults to `1`):
            Number of gradient accumulation steps.
        train_max_grad_norm (`float`, *optional*, defaults to `1.0`):
            Maximum gradient norm for gradient clipping.
        negative_prompts (`Optional[str]`, *optional*, defaults to `None`):
            Comma-separated list of prompts to use as negative examples.
        truncated_backprop_rand (`bool`, *optional*, defaults to `True`):
            If `True`, randomized truncation to different diffusion timesteps is used.
        truncated_backprop_timestep (`int`, *optional*, defaults to `49`):
            Absolute timestep to which the gradients are backpropagated. Used only if `truncated_backprop_rand=False`.
        truncated_rand_backprop_minmax (`Tuple[int, int]`, *optional*, defaults to `(0, 50)`):
            Range of diffusion timesteps for randomized truncated backpropagation.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the final model to the Hub.
    """

    exp_name: str = os.path.basename(sys.argv[0])[: -len(".py")]
    run_name: str = ""
    seed: int = 0
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    log_image_freq: int = 1
    # the number of logging images will be log_image_iter * train_batch_size
    log_image_iter: int = 1
    tracker_kwargs: Dict[str, Any] = field(default_factory=dict)
    accelerator_kwargs: Dict[str, Any] = field(default_factory=dict)
    project_kwargs: Dict[str, Any] = field(default_factory=dict)
    tracker_project_name: str = "trl"
    logdir: str = "logs"
    num_epochs: int = 100
    save_freq: int = 50
    num_checkpoint_limit: int = 5
    mixed_precision: str = "fp16"
    allow_tf32: bool = True
    resume_from: str = ""
    sample_num_steps: int = 50
    reward_fn: str = 'aesthetic'
    grad_scale: float = 1
    loss_coeff: float = 0.01
    aesthetic_target: float = 10
    sample_eta: float = 1.0
    sample_guidance_scale: float = 5.0
    prompt_fn: str = 'simple_animals'
    use_custom_eval_prompts: bool = False
    custom_eval_prompts: List[str] = field(default_factory=list)
    backprop_strategy: str = 'fixed'    # gaussian, uniform, fixed, refl, sqdf, draft+
    backprop_kwargs = {'gaussian': {'mean': 42, 'std': 5}, 'uniform': {'min': 0, 'max': 50}, 'fixed': {'value': 49}, 'refl': {'min': 30, 'max': 50}, 'sqdf': {'min':0, 'max':49}, 'draft+': {'value': 49}, 'draftlv': {'value': 49}}
    
    # if backprop_strategy == 'fixed', backward_step in backprop_kwargs set to backward_step.
    backward_step: int = 0
    # ReFL
    refl_step_min: int = 30
    refl_step_max: int = 50

    # sqdf
    sqdf_gamma: float = 0.9  # reward decay
    sqdf_alpha: float = 0.01  # KL coefficient
    # random step min max range
    sqdf_step_min:int = 0 
    sqdf_step_max:int = 50
    # add consistency model
    use_consistency_model: bool = True
    # n-step sampling
    use_n_step_sampling: bool = False
    n_steps: int = 4
    # draftlv
    draftlv_loop: int = 2

    train_batch_size: int = 1
    train_use_8bit_adam: bool = False
    train_learning_rate: float = 1e-3
    train_adam_beta1: float = 0.9
    train_adam_beta2: float = 0.999
    train_adam_weight_decay: float = 1e-4
    train_adam_epsilon: float = 1e-8
    train_gradient_accumulation_steps: int = 1
    train_max_grad_norm: float = 1.0
    negative_prompts: Optional[str] = None

    push_to_hub: bool = False

    # Buffer
    use_buffer: bool = False
    buffer_size: int = 1024
    replay_ratio: int = 1
    buffer_method: str = 'random' # Supported: random, on-policy, reward+gamma_PER

    # Test mode
    test_mode: bool = False

    # LoRA
    lora_rank: int = 4

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

    def __post_init__(self):
        if self.log_with not in ["wandb", "tensorboard"]:
            warnings.warn(
                "Accelerator tracking only supports image logging if `log_with` is set to 'wandb' or 'tensorboard'."
            )

        if self.log_with == "wandb" and not is_torchvision_available():
            warnings.warn("Wandb image logging requires torchvision to be installed")

        if self.train_use_8bit_adam and not is_bitsandbytes_available():
            raise ImportError(
                "You need to install bitsandbytes to use 8bit Adam. "
                "You can install it with `pip install bitsandbytes`."
            )
        
        if self.backprop_strategy == 'fixed':
            self.backprop_kwargs['fixed']['value'] = self.backward_step
        elif self.backprop_strategy == 'refl':
            self.backprop_kwargs['refl']['min'] = self.refl_step_min
            self.backprop_kwargs['refl']['max'] = self.refl_step_max
        elif self.backprop_strategy == 'draftlv':
            self.backprop_kwargs['draftlv']['loop'] = self.draftlv_loop
        elif self.backprop_strategy == 'sqdf':
            self.backprop_kwargs['sqdf']['min'] = self.sqdf_step_min
            self.backprop_kwargs['sqdf']['max'] = self.sqdf_step_max

        if self.use_custom_eval_prompts:
            if self.prompt_fn == 'simple_animals':
                self.custom_eval_prompts = [
                    "whale", "squirrel", "butterfly", "rabbit",
                    "deer", "dog", "cat", 
                    "bird"
                ]
            elif self.prompt_fn == 'hps_v2_all':
                self.custom_eval_prompts = [
                    "A kangaroo wearing an orange hoodie and blue sunglasses stands in front of the Sydney Opera House holding a 'Welcome Friends' sign.",
                    "Portrait of goth girl in Warhammer armor.",
                    "An atom bomb explosion in Heaven, depicted in the oil on canvas masterpiece by Thomas Cole, currently trending on ArtStation.",
                    "A woman sitting on a bench with cars behind her.",
                    "A bird is speaking into a high-end microphone, wearing headphones in a recording studio.",
                    "A 3D rendering of a robot screaming at a death metal concert.",
                    "A painting of a girl standing on a mountain looking out at an approaching storm over the ocean, with wind blowing and ocean mist, surrounded by lightning.",
                    "Looking down on a stony surface shows a bowl with an orange in it and what looks like a large piece of red plastic.",
                ]

                # load eval dataset
                with open("./assets/hps_v2_all_eval.txt", "r", encoding="utf-8") as f:
                    self.metric_eval_prompts = [line.strip() for line in f if line.strip()]

            print(f"custom_eval_prompts: {self.custom_eval_prompts}")
            self.log_image_iter = math.ceil(len(self.custom_eval_prompts) / self.train_batch_size)
        
        self.now = datetime.now().strftime("%m%d_%H%M%S")


