import ipdb
st = ipdb.set_trace
import builtins
import time
from dataclasses import dataclass, field
import prompts as prompts_file
import numpy as np
from transformers import HfArgumentParser

from config.alignprop_config import AlignPropConfig
from alignprop_trainer import AlignPropTrainer
from sd_pipeline import DiffusionPipeline


@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use"}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})

def image_outputs_logger(image_pair_data, global_step, accelerate_logger):
    # Log all generated images
    result = {}
    images, prompts = [image_pair_data["images"], image_pair_data["prompts"]]
    for i, image in enumerate(images):
        prompt = prompts[i]
        result[f"{prompt}"] = image.unsqueeze(0).float()
    accelerate_logger.log_images(
        result,
        step=global_step,
    )

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, AlignPropConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    project_dir = f"{training_args.now}"
    
    training_args.project_kwargs = {
        "automatic_checkpoint_naming": False,
        "project_dir": f"checkpoints/{project_dir}",
    }
   
    prompt_fn = getattr(prompts_file, training_args.prompt_fn)
    
    pipeline = DiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
        use_consistency_model = training_args.use_consistency_model,
        lora_rank=training_args.lora_rank,
        backprop_strategy=training_args.backprop_strategy,
    )
    
    trainer = AlignPropTrainer(
        training_args,
        prompt_fn,
        pipeline, 
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()
    # Clean up buffer directory if used
    if training_args.use_buffer:
        import shutil
        shutil.rmtree(f"buffer/{training_args.now}", ignore_errors=True)