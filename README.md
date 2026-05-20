# Dare-MoE

Training, evaluation, and routing-analysis code for a two-stage deepfake detector built around Mixture-of-Experts vision transformers.

The repository contains:

- `MTS-MoE/`: the base expert model, a ViT-B/16 augmented with AuroRA LoRA-MoE and Multi-Transform Spectral MoE adapters.
- `MoE Finetuning/`: the DaRE-MoE stage, which loads multiple specialist experts, adds a Domain-Aware Router, and finetunes the fused detector with routing supervision.
- `routing_analysis/`: precomputed routing visualizations and JSON reports.
- `test/`: benchmark-style test folders organized for batch evaluation.
- `checkpoint/`: an example fine-tuned DaRE-MoE checkpoint committed with the repo.

## Method Overview

This codebase implements a two-step pipeline.

### Stage 1: Train specialist MTS-MoE experts

Each expert is a ViT backbone with two MoE adaptation paths:

- AuroRA LoRA-MoE inside attention.
- MTS-MoE adapters parallel to the MLP.

The MTS-MoE adapter contains five transform-specific experts:

- DFT
- DCT
- DWT-Haar
- DWT-DB2
- Spatial

These experts are intended to learn domain-specific forensic cues for a single source domain, for example a diffusion-model domain or a GAN domain.

### Stage 2: Finetune DaRE-MoE

The finetuning stage loads two pretrained MTS-MoE experts and adds:

- a lightweight Domain-Aware Router that predicts dense routing maps over the image,
- a shared classification head,
- Expert Knowledge Distillation (EKD),
- Domain-Aware Routing Supervision (DARS).

Training alternates between expert-specific batches and follows a 3-stage schedule:

1. warmup with BCE + EKD,
2. gradual DARS ramp-up,
3. full routing supervision.

## Repository Layout

```text
.
├── MTS-MoE/
│   ├── train.py
│   ├── test_mts_moe.py
│   ├── inspect_mts_moe.py
│   └── MTS_MoE.py
├── MoE Finetuning/
│   ├── train_moe_finetune.py
│   ├── test.py
│   ├── run_routing_analysis.py
│   ├── count_trainable_params.py
│   ├── visualize_routing_maps.py
│   ├── visualize_routing_advanced.py
│   └── models/moe_forensic.py
```



## Environment Setup

There is no packaged environment file in the repository, so setup is manual.

### Recommended environment

- Python 3.10 or newer
- PyTorch and torchvision
- `timm`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `opencv-python`
- `tqdm`

Example setup with `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision timm numpy matplotlib seaborn scikit-learn opencv-python tqdm
```

## Hardware Notes

- Training scripts are CUDA-oriented. They call `.cuda()` directly and should be treated as GPU training code.
- `MTS-MoE/test_mts_moe.py` includes MPS fallback and can run on Apple Silicon for inference.
- `MoE Finetuning/test.py` falls back to CPU if CUDA is unavailable, but the full finetuning workflow is still written around CUDA usage.

## Dataset Layout

All training and evaluation code expects torchvision `ImageFolder`-style folders with binary labels:

- `0_real/`
- `1_fake/`

Training datasets must look like this:

```text
dataset_root/
├── train/
│   ├── 0_real/
│   └── 1_fake/
└── val/
    ├── 0_real/
    └── 1_fake/
```

Evaluation datasets must look like this:

```text
dataset_root/
└── test/
    ├── 0_real/
    └── 1_fake/
```

Images are resized or center-cropped to `224 x 224` by the dataset loaders.

## Training Workflow

### 1. Train the domain experts with MTS-MoE

Run one training job per source domain. In the original workflow that means at least one DM expert and one GAN expert.

From the repository root:

```bash
cd MTS-MoE
python train.py \
  --data_dir /path/to/dm_dataset \
  --model_dir ../checkpoints/mts_moe_dm \
  --epochs 20 \
  --batch_size 32
```

Train a second expert for another domain:

```bash
cd MTS-MoE
python train.py \
  --data_dir /path/to/gan_dataset \
  --model_dir ../checkpoints/mts_moe_gan \
  --epochs 20 \
  --batch_size 32
```

Artifacts written by `MTS-MoE/train.py` include:

- `models_params_<epoch>.tar`
- `model_params_best_*.pkl`
- `training_history.json`
- `training_curves.png`
- `grad_norm_curves.png`
- `train.log`

Useful supporting commands:

```bash
cd MTS-MoE
python inspect_mts_moe.py --ckpt /path/to/model_params_best.pkl
```

### 2. Finetune DaRE-MoE with the pretrained experts

After you have one checkpoint per expert domain, run the finetuning stage:

```bash
cd "MoE Finetuning"
python train_moe_finetune.py \
  --model_dir ../checkpoints/moe_finetune \
  --data_dm_dir /path/to/dm_dataset \
  --data_gan_dir /path/to/gan_dataset \
  --expert_dm_path /path/to/dm_expert.pkl \
  --expert_gan_path /path/to/gan_expert.pkl \
  --epochs 20 \
  --batch_size 16
```

This stage writes:

- `models_params_<epoch>.tar`
- `model_params_best_*.pkl`
- `training_history.json`
- `training_curves.png`
- `train.log`

To inspect how many parameters are active during finetuning:

```bash
cd "MoE Finetuning"
python count_trainable_params.py \
  --expert_dm_path /path/to/dm_expert.pkl \
  --expert_gan_path /path/to/gan_expert.pkl
```

## Evaluation

### Evaluate an MTS-MoE expert on a single dataset

```bash
cd MTS-MoE
python test_mts_moe.py \
  --model_path /path/to/model_params_best.pkl \
  --test_data_path /path/to/dataset_root \
  --save_results
```

### Batch-evaluate an MTS-MoE expert across multiple datasets



```bash
cd MTS-MoE
python test_mts_moe.py \
  --model_path /path/to/model_params_best.pkl \
  --datasets_root ../test \
  --save_results
```

This produces per-dataset outputs plus a consolidated batch summary.

### Evaluate a DaRE-MoE checkpoint

If you want to try the committed example checkpoint against one of the bundled dataset roots:

```bash
cd "MoE Finetuning"
python test.py \
  --model_path ../checkpoint/model_params_best_-0.0180loss_0.9953auc_epoch008.pkl \
  --test_data_path ../test/freedom \
  --save_results
```

The DaRE-MoE test script expects `--test_data_path` to point to the dataset root that contains a `test/` split.

## Routing Analysis

The main post-hoc analysis entry point is `MoE Finetuning/run_routing_analysis.py`.

It performs:

1. checkpoint and dataset validation,
2. basic routing visualizations,
3. advanced difference-map and specialization analysis,
4. unified JSON report generation.

Example usage:

```bash
cd "MoE Finetuning"
python run_routing_analysis.py \
  --checkpoint_path ../checkpoint/model_params_best_-0.0180loss_0.9953auc_epoch008.pkl \
  --data_dm_dir /path/to/dm_dataset \
  --data_gan_dir /path/to/gan_dataset \
  --output_dir ../routing_analysis/custom_run \
  --max_samples 50
```

The analysis pipeline writes:

- `basic/individual/`: per-image routing figures
- `basic/comparison/`: domain comparison figure
- `basic/statistics/`: routing summary plots
- `advanced/difference_maps/`: GAN-vs-DM routing differences
- `advanced/realvsfake/`: real-vs-fake routing comparison
- `advanced/paper_figures/routing_grid.png`: paper-ready summary figure
- `advanced/specialization/specialization_report.json`: specialization metrics
- `analysis_report.json`: unified run metadata and summary metrics

## Important Script Defaults

Several scripts still contain dataset or checkpoint defaults from the original training environment, for example absolute paths under `/data3/...`.

You should usually override these explicitly on the command line:

- `--data_dir`
- `--data_dm_dir`
- `--data_gan_dir`
- `--expert_dm_path`
- `--expert_gan_path`
- `--model_dir`

## Practical Entry Points

If you are new to the repo, start here:

1. `MTS-MoE/train.py` for specialist expert training.
2. `MTS-MoE/test_mts_moe.py` for single- or multi-dataset evaluation.
3. `MoE Finetuning/train_moe_finetune.py` for the DaRE-MoE stage.
4. `MoE Finetuning/test.py` for final detector evaluation.
5. `MoE Finetuning/run_routing_analysis.py` for routing maps and specialization reports.

## Output Summary

At a high level, the repository covers four tasks:

- training specialist experts,
- finetuning a routed multi-expert detector,
- evaluating checkpoints on benchmark datasets,
- generating routing visualizations and quantitative routing reports.

That makes it suitable both for reproducing the training pipeline and for analyzing how the router specializes across diffusion and GAN domains.