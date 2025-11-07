# Diffusion Fine-Tuning via Reparameterized Policy Gradient of the Soft Q-Function


Official implementation of **Diffusion Fine-Tuning via Reparameterized Policy Gradient of the Soft Q-Function (SQDF)**.

## Abstract
Diffusion models excel at generating high-likelihood samples but often require alignment with downstream objectives. Existing fine-tuning methods for diffusion models significantly suffer from reward over-optimization, resulting in high-reward but unnatural samples and degraded diversity. To mitigate overoptimization, we propose $\textbf{Soft Q-based Diffusion Finetuning (SQDF)}$, a KL-regularized RL method for diffusion alignment that applies a reparameterized policy gradient of a training-free soft Q-function. SQDF is further enhanced with three innovations: a discount factor for proper credit assignment in the denoising process, the integration of consistency models to refine Q-function estimates, and the use of an off-policy replay buffer to improve mode coverage and manage the reward-diversity trade-off. Our experiments demonstrate that SQDF achieves superior target rewards while preserving diversity in text-to-image alignment. Furthermore, in online black-box optimization, SQDF attains high sample efficiency while maintaining naturalness and diversity.

## Repository Structure

This repository contains the official codebase for both **Text-to-Image (T2I)** and **Online Black-Box Optimization (BBO)** experiments.  
Each subdirectory includes task-specific training and evaluation pipelines.