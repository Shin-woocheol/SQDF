# Soft Q-based Diffusion Finetuning for T2I

<!-- ## Abstract
Diffusion models excel at generating high-likelihood samples but often require alignment with downstream objectives. Existing fine-tuning methods for diffusion models significantly suffer from reward over-optimization, resulting in high-reward but unnatural samples and degraded diversity. To mitigate overoptimization, we propose $\textbf{Soft Q-based Diffusion Finetuning (SQDF)}$, a KL-regularized RL method for diffusion alignment that applies a reparameterized policy gradient of a training-free soft Q-function. SQDF is further enhanced with three innovations: a discount factor for proper credit assignment in the denoising process, the integration of consistency models to refine Q-function estimates, and the use of an off-policy replay buffer to improve mode coverage and manage the reward-diversity trade-off. Our experiments demonstrate that SQDF achieves superior target rewards while preserving diversity in text-to-image alignment. Furthermore, in online black-box optimization, SQDF attains high sample efficiency while maintaining naturalness and diversity. -->


## Installation
Create and activate a new environment with:

```bash
conda create -n sqdf python=3.10 -y
conda activate sqdf
pip install -r requirements.txt
```

## Usage
We provides ready-to-use shell scripts for different reward functions and baseline methods.

### Finetuning with Aesthetic Reward
```bash
bash aesthetic.sh
```

### Finetuning with HPSv2 Reward
```bash
bash hps.sh
```

### Other Baselines

Baseline finetuning commands are also included **within** the `aesthetic.sh` and `hps.sh` scripts.  
Simply **uncomment** the corresponding lines in the script to run different baseline methods.


## Acknowledgements
- https://github.com/huggingface/diffusers
- https://github.com/kvablack/ddpo-pytorch
- https://github.com/mihirp1998/AlignProp
- https://github.com/krafton-ai/DAS/tree/main