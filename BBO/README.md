# Soft Q-based Diffusion Finetuning for online Black-Box Optimization

## Installation
Create and activate a new environment with:

```bash
conda create -n sqdf-BBO python=3.10
conda activate sqdf-BBO
pip install -r requirements.txt
```

## Usage
We provides ready-to-use shell scripts.

### Finetuning with Aesthetic Reward
```bash
bash aesthetic.sh
```

## Acknowledgements
- https://github.com/huggingface/diffusers
- https://github.com/kvablack/ddpo-pytorch
- https://github.com/mihirp1998/AlignProp
- https://github.com/zhaoyl18/SEIKO.git