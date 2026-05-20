## -*- coding: utf-8 -*-
"""
DaRE-MoE Finetuning Training Script

Domain-aware Routing Enhanced Mixture-of-Experts (DaRE-MoE) with:
  - Alternating expert training (freeze expert m when feeding D_m)
  - BCE detection loss
  - Expert Knowledge Distillation (EKD) loss for feature alignment
  - Domain-Aware Routing Supervision (DARS) loss with 3-stage schedule

3-Stage Training Schedule:
  Stage 1 (0–30%):   β = 0        (warmup — BCE + EKD only)
  Stage 2 (30–70%):  β ramps 0→β_max  (gradually introduce DARS)
  Stage 3 (70–100%): β = β_max    (all losses, constant weights)

Expert Model Architecture:
- VisionTransformer (ViT-B/16) from MTS_MoE.py
- 768-dimensional embeddings, 12 layers, 12 attention heads
- AuroRA LoRA-MoE and MTS-MoE adapter modules
- Shared routing (gating) across both experts
"""

import os, sys
sys.setrecursionlimit(15000)
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import logging
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add current directory to path for MTS_MoE import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from models.moe_forensic import ForensicMoE, load_expert_models


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def plot_training_curves(history, save_dir):
    """Plot and save training curves for DaRE-MoE finetuning."""
    epochs = history['epochs']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Training History (DaRE-MoE Finetuning)', fontsize=16, fontweight='bold')

    axes[0, 0].plot(epochs, history['train_loss'],     'b-', lw=2, label='Total Loss')
    axes[0, 0].plot(epochs, history['train_bce_loss'], 'r--', lw=2, label='BCE Loss')
    axes[0, 0].plot(epochs, history['train_ekd_loss'], 'g--', lw=2, label='EKD Loss')
    axes[0, 0].plot(epochs, history['train_dars_loss'], 'c--', lw=2, label='DARS Loss')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

    axes[0, 1].plot(epochs, history['val_acc'], 'r-', lw=2, label='Val Acc')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

    axes[0, 2].plot(epochs, history['val_auc'], 'g-', lw=2, label='Val AUC')
    axes[0, 2].set_xlabel('Epoch'); axes[0, 2].set_ylabel('AUC')
    axes[0, 2].set_title('Validation AUC-ROC', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3); axes[0, 2].legend()

    axes[1, 0].plot(epochs, history['val_loss'], 'm-', lw=2, label='Val Loss')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Validation Loss', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

    axes[1, 1].plot(epochs, history['beta'], 'k-', lw=2, label='β (DARS weight)')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('β')
    axes[1, 1].set_title('DARS Weight Schedule', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()

    axes[1, 2].plot(epochs, history['train_dars_loss'], 'c-', lw=2, label='DARS Loss')
    axes[1, 2].set_xlabel('Epoch'); axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_title('DARS Routing Loss', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3); axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_dir}")


def get_beta(epoch, total_epochs, beta_max, warmup_ratio=0.3, ramp_end=0.7):
    """
    3-stage DARS weight schedule.

    Stage 1 (0 – warmup_ratio):  β = 0
    Stage 2 (warmup_ratio – ramp_end):  β ramps linearly 0 → β_max
    Stage 3 (ramp_end – 1.0):  β = β_max
    """
    progress = epoch / total_epochs
    if progress < warmup_ratio:
        return 0.0
    elif progress < ramp_end:
        return beta_max * (progress - warmup_ratio) / (ramp_end - warmup_ratio)
    else:
        return beta_max


def train(args, model, optimizer, train_loaders, valid_loaders, scheduler,
          save_dir, num_experts):
    """
    Train DaRE-MoE with alternating expert batches and 3-stage DARS schedule.

    Each step draws a batch from one expert's loader, freezes that expert,
    and finetunes the other N-1 experts via BCE + EKD + DARS loss.

    Args:
        train_loaders: list of N DataLoaders (one per expert data source)
        valid_loaders: list of N DataLoaders (one per expert data source)
    """
    best_val_loss = float('inf')
    global_step    = 0

    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        'epochs': [],
        'train_loss': [], 'train_bce_loss': [], 'train_ekd_loss': [],
        'train_dars_loss': [], 'beta': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
    }

    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------ resume
    if args.resume > -1:
        ckpt_path = os.path.join(save_dir, f'models_params_{args.resume}.tar')
        checkpoint = torch.load(ckpt_path,
                                map_location=f'cuda:{torch.cuda.current_device()}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            for _ in range(args.resume + 1):
                scheduler.step()
        log.info(f"Resumed from epoch {args.resume}: {ckpt_path}")

    # ------------------------------------------------------------------ loop
    for epoch in range(args.resume + 1, args.epochs):

        # ---- 3-stage DARS schedule ----
        beta = get_beta(epoch, args.epochs, args.beta_max)
        stage = ('Stage 1 (warmup)' if beta == 0.0
                 else 'Stage 3 (full)' if beta >= args.beta_max
                 else 'Stage 2 (ramp-up)')

        # ------------------------------------------------------------ train
        print(f'\n[Epoch {epoch+1}/{args.epochs}] {stage}  β={beta:.4f}')
        model.train()

        epoch_total_loss = 0.0
        epoch_bce_loss   = 0.0
        epoch_ekd_loss   = 0.0
        epoch_dars_loss  = 0.0
        num_batches      = 0

        st_time = time.time()

        # Build iterators for each expert's training data
        iters   = [iter(loader) for loader in train_loaders]
        min_len = min(len(loader) for loader in train_loaders)
        total_batches = min_len * num_experts

        for batch_idx in range(total_batches):
            expert_type = batch_idx % num_experts

            try:
                images, labels = next(iters[expert_type])
            except StopIteration:
                break

            images = images.cuda()
            labels = labels.cuda()

            # Build domain labels: real → num_experts, fake → expert_type
            domain_labels = torch.full_like(labels, num_experts, dtype=torch.long)
            domain_labels[labels == 1] = expert_type

            # Freeze expert_m, unfreeze other N-1 experts
            model.unfreeze_experts_except(expert_type)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = model.compute_total_loss(
                    images, labels, expert_type,
                    domain_labels=domain_labels, beta=beta,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_total_loss += loss_dict['total'].item()
            epoch_bce_loss   += loss_dict['bce'].item()
            epoch_ekd_loss   += loss_dict['ekd'].item()
            epoch_dars_loss  += loss_dict['dars'].item()
            num_batches      += 1
            global_step      += 1

            if global_step % args.record_step == 0:
                period = time.time() - st_time
                log.info(
                    'Epoch [{:0>3}/{:0>3}] Iter [{:0>3}/{:0>3}] '
                    'Loss:{:.4f} (bce:{:.4f} ekd:{:.4f} dars:{:.4f}) '
                    'β:{:.4f} LR:{:.2e} '
                    'time:{}m{}s'.format(
                        epoch + 1, args.epochs,
                        batch_idx + 1, total_batches,
                        epoch_total_loss / num_batches,
                        epoch_bce_loss   / num_batches,
                        epoch_ekd_loss   / num_batches,
                        epoch_dars_loss  / num_batches,
                        beta,
                        optimizer.param_groups[0]['lr'],
                        int(period // 60), int(period % 60),
                    )
                )
                st_time = time.time()

        avg_total_loss = epoch_total_loss / max(num_batches, 1)
        avg_bce_loss   = epoch_bce_loss   / max(num_batches, 1)
        avg_ekd_loss   = epoch_ekd_loss   / max(num_batches, 1)
        avg_dars_loss  = epoch_dars_loss  / max(num_batches, 1)

        # ---------------------------------------------------------- validate
        print('Starting validation...')
        model.eval()
        all_probs  = []
        all_labels = []
        val_loss_total  = 0.0
        val_batches     = 0

        with torch.no_grad():
            for expert_type, val_loader in enumerate(valid_loaders):
                for images, labels in tqdm(val_loader, total=len(val_loader),
                                           ncols=70, leave=False, unit='batch'):
                    images = images.cuda()
                    labels = labels.cuda()

                    domain_labels = torch.full_like(labels, num_experts, dtype=torch.long)
                    domain_labels[labels == 1] = expert_type

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output = model(images)
                        v_loss, _ = model.compute_total_loss(
                            images, labels, expert_type,
                            domain_labels=domain_labels, beta=beta,
                        )

                    probs = torch.sigmoid(output).squeeze()
                    all_probs.extend(probs.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                    val_loss_total += v_loss.item()
                    val_batches    += 1

        all_probs_np  = np.array(all_probs)
        all_labels_np = np.array(all_labels)
        pred_labels   = (all_probs_np > 0.5).astype(int)
        accuracy      = (pred_labels == all_labels_np).mean()

        try:
            from sklearn.metrics import roc_auc_score
            auc_roc = roc_auc_score(all_labels_np, all_probs_np)
        except ValueError:
            auc_roc = 0.0

        avg_val_loss = val_loss_total / max(val_batches, 1)

        log.info(
            'Validation: Epoch [{:0>3}/{:0>3}] '
            'Loss:{:.4f} Acc:{:.2%} AUC:{:.4f}'.format(
                epoch + 1, args.epochs,
                avg_val_loss, accuracy, auc_roc,
            )
        )

        # -------------------------------------------------------- bookkeeping
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(float(avg_total_loss))
        history['train_bce_loss'].append(float(avg_bce_loss))
        history['train_ekd_loss'].append(float(avg_ekd_loss))
        history['train_dars_loss'].append(float(avg_dars_loss))
        history['beta'].append(float(beta))
        history['val_loss'].append(float(avg_val_loss))
        history['val_acc'].append(float(accuracy))
        history['val_auc'].append(float(auc_roc))

        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        # ------------------------------------------------------- save latest
        state = {
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, os.path.join(save_dir, f'models_params_{epoch}.tar'))

        # ----------------------------------------- save best (lowest val loss)
        if avg_val_loss < best_val_loss:
            for m in os.listdir(save_dir):
                if m.startswith('model_params_best'):
                    os.remove(os.path.join(save_dir, m))
            best_val_loss = avg_val_loss
            best_name = (
                f'model_params_best_{avg_val_loss:.4f}loss_'
                f'{auc_roc:.4f}auc_epoch{epoch+1:03d}.pkl'
            )
            torch.save(model.state_dict(), os.path.join(save_dir, best_name))
            log.info(f"  New best model saved: {best_name}")

        scheduler.step()

    plot_training_curves(history, save_dir)
    return history


# ============================================================== entry point ===
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='DaRE-MoE Finetuning with alternating training + DARS schedule'
    )

    parser.add_argument('--device',    '-dv', type=int, default=0)
    parser.add_argument('--model_dir', '-md', type=str,
                        default='checkpoints/moe_finetune')
    parser.add_argument('--resume',    '-rs', type=int, default=-1,
                        help='Epoch index to resume from (-1 = fresh start)')
    parser.add_argument('--epochs',          type=int,   default=20)
    parser.add_argument('--record_step',     type=int,   default=100)
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--num_workers',     type=int,   default=4)

    # Expert data directories -- each must contain train/ and val/ subfolders
    # with 0_real/ and 1_fake/ subdirectories (ImageFolder layout)
    parser.add_argument('--data_dm_dir', type=str,
                        default='/data3/law/data/FF++/dm',
                        help='Data directory for DM expert')
    parser.add_argument('--data_gan_dir', type=str,
                        default='/data3/law/data/FF++/gan',
                        help='Data directory for GAN expert')

    # Expert model checkpoint paths
    parser.add_argument('--expert_dm_path', type=str,
                        default='aurora dm [2,4,6,8] alpha4/trainmodel_params_best_0.9998auc1.0000epoch001.pkl',
                        help='Path to DM expert checkpoint')
    parser.add_argument('--expert_gan_path', type=str,
                        default='aurora gan [2,4,6,8] alpha4/trainmodel_params_best_1.0000auc1.0000epoch001.pkl',
                        help='Path to GAN expert checkpoint')

    # MoE model arguments
    parser.add_argument('--num_experts',  type=int,   default=2)
    parser.add_argument('--feature_dim',  type=int,   default=768)
    parser.add_argument('--lambda_ekd',   type=float, default=0.5,
                        help='Trade-off weight for EKD loss (alpha)')
    parser.add_argument('--margin',       type=float, default=0.7,
                        help='Margin for EKD loss')
    parser.add_argument('--beta_max',     type=float, default=0.2,
                        help='Maximum DARS weight (beta_max in 3-stage schedule)')

    # Optimizer
    parser.add_argument('--lr',           type=float, default=1e-4,
                        help='Learning rate for experts & classifier')
    parser.add_argument('--router_lr',    type=float, default=5e-5,
                        help='Learning rate for Domain-Aware Router')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()

    # ----------------------------------------------------------------- setup
    start_time = time.time()
    setup_seed(2024)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_dir = args.model_dir
    os.makedirs(save_dir, exist_ok=True)

    log_mode = 'a' if args.resume > -1 else 'w'
    logging.basicConfig(
        filename=os.path.join(save_dir, 'train.log'),
        filemode=log_mode,
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO,
    )
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())
    log.info(f'model dir: {args.model_dir}')
    log.info(f'args: {args}')

    # ------------------------------------------------------------------ data
    # Use dataset.py to create separate loaders for each expert's data.
    # Each data directory follows ImageFolder layout: train/{0_real,1_fake}
    data_dirs = [args.data_dm_dir, args.data_gan_dir]
    train_loaders = []
    valid_loaders = []
    for i, data_dir in enumerate(data_dirs):
        t_loader, v_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        train_loaders.append(t_loader)
        valid_loaders.append(v_loader)
        log.info(f'Expert {i}: {data_dir} -- '
                 f'train: {len(t_loader)} batches, '
                 f'val: {len(v_loader)} batches')

    # ------------------------------------------------ load expert checkpoints
    print('\n=== Loading Expert Models ===')
    expert_paths  = [args.expert_dm_path, args.expert_gan_path]
    expert_states = load_expert_models(expert_paths)
    log.info(f'Loaded {len(expert_states)} expert models')

    # --------------------------------------------------------- ForensicMoE
    model = ForensicMoE(
        backbone=None,
        expert_models=expert_states,
        num_experts=args.num_experts,
        feature_dim=args.feature_dim,
        freeze_backbone=True,
        lambda_ekd=args.lambda_ekd,
        margin=args.margin,
    )
    model = model.cuda()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    print(f'Total parameters:     {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}  '
          f'({100 * trainable_params / total_params:.2f}%)')

    # Separate param groups: router gets lower LR for stable routing
    router_params = list(model.router.parameters())
    router_ids = {id(p) for p in router_params}
    other_params = [p for p in model.parameters() if id(p) not in router_ids]

    optimizer = optim.AdamW([
        {'params': other_params,  'lr': args.lr},
        {'params': router_params, 'lr': args.router_lr},
    ], weight_decay=args.weight_decay)

    def warmup_lambda(epoch):
        warmup_epochs = 3
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 ** ((epoch - warmup_epochs) // 5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lambda,
    )

    # ---------------------------------------------------------------- train
    print('\nStarting training...')
    history  = train(args, model, optimizer, train_loaders, valid_loaders,
                     scheduler, save_dir, args.num_experts)
    duration = time.time() - start_time

    print('\n' + '=' * 60)
    print('Training Completed!')
    print('=' * 60)
    print(f'Total time:     {int(duration//3600)}h {int(duration%3600//60)}m')
    print(f'Model dir:      {args.model_dir}')
    print(f'lambda_ekd:     {args.lambda_ekd}')
    print(f'beta_max:       {args.beta_max}')
    print(f'margin:         {args.margin}')
    print(f'Expert LR:      {args.lr}')
    print(f'Router LR:      {args.router_lr}')
    print(f'\nFinal Metrics:')
    print(f'  Best Val Loss:     {min(history["val_loss"]):.4f}')
    print(f'  Best Val AUC:      {max(history["val_auc"]):.4f}')
    print(f'  Best Val Accuracy: {max(history["val_acc"]):.2%}')
    print(f'  Final Train Loss:  {history["train_loss"][-1]:.4f}')
    print(f'    (bce: {history["train_bce_loss"][-1]:.4f}, '
          f'ekd: {history["train_ekd_loss"][-1]:.4f}, '
          f'dars: {history["train_dars_loss"][-1]:.4f})')
    print('=' * 60)
