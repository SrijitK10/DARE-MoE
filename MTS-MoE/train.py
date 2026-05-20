## -*- coding: utf-8 -*-
import os, sys
sys.setrecursionlimit(15000)
import torch
import numpy as np
import random
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import logging
from tqdm import tqdm
from dataset import get_dataloaders
from utils import *
from MTS_MoE import *
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_train_mode(model):
    """
    Correctly set training mode for MTS_MoE.

    Calling model.train() recursively sets every submodule to train mode,
    which undoes the .eval() calls made in freeze_stages() on the frozen
    backbone blocks. This function re-applies the correct mode per component:

      - patch_embed, pos_drop          → eval()  (always frozen)
      - each Block                     → eval()  (frozen backbone)
      - block.attn.LoRA_MoE            → train() (trainable)
      - block.adapter_MoE              → train() (trainable)
      - block.norm1                    → train() (trainable per freeze_stages)
      - model.norm                     → eval()  (frozen final norm)
      - model.head                     → train() (trainable classifier)
    """
    model.train()
    model.patch_embed.eval()
    model.pos_drop.eval()

    for block in model.blocks:
        block.eval()
        # LoRA-MoE in attention (AuroRA QKV adaptation)
        if model.lora_topk > 0:
            block.attn.LoRA_MoE.train()
        # MTS-MoE parallel to MLP (frequency-domain adaptation)
        if model.adapter_topk > 0:
            block.adapter_MoE.train()
        # norm1 is trainable according to freeze_stages()
        block.norm1.train()

    model.norm.eval()   # final LayerNorm is frozen
    model.head.train()


def get_moe_grad_norms(model):
    """
    Return gradient norms separately for LoRA-MoE and adapter (MTS-MoE) params.
    Used to verify both sets of adapters are escaping their zero-init starting
    point and contributing to learning.

    Returns:
        lora_gnorm   (float): L2 grad norm across all LoRA_MoE parameters
        adapter_gnorm (float): L2 grad norm across all adapter_MoE parameters
    """
    lora_sq_sum    = 0.0
    adapter_sq_sum = 0.0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        sq = param.grad.data.norm(2).item() ** 2
        if 'LoRA' in name:
            lora_sq_sum += sq
        elif 'adapter' in name:
            adapter_sq_sum += sq

    return lora_sq_sum ** 0.5, adapter_sq_sum ** 0.5


def plot_training_curves(history, save_dir):
    """Plot and save all training curves including MoE loss."""
    epochs = history['epochs']

    # Main 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training History (MTS-MoE)', fontsize=16, fontweight='bold')

    axes[0, 0].plot(epochs, history['train_loss'],     'b-', lw=2, label='Total Loss')
    axes[0, 0].plot(epochs, history['train_cls_loss'], 'r--', lw=2, label='Cls Loss')
    axes[0, 0].plot(epochs, history['train_moe_loss'], 'g--', lw=2, label='MoE Loss')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

    axes[0, 1].plot(epochs, history['train_acc'], 'b-', lw=2, label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'],   'r-', lw=2, label='Val Acc')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

    axes[1, 0].plot(epochs, history['val_auc'], 'g-', lw=2, label='Val AUC')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_title('Validation AUC', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

    axes[1, 1].plot(epochs, history['val_eer'], 'm-', lw=2, label='Val EER')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('EER')
    axes[1, 1].set_title('Validation EER (Lower is Better)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Grad norm plot — separate figure to track LoRA vs adapter health
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['lora_grad_norms'],    'b-', lw=2, marker='o', label='LoRA-MoE ‖∇‖')
    ax.plot(epochs, history['adapter_grad_norms'], 'r-', lw=2, marker='s', label='MTS-MoE ‖∇‖')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gradient Norm')
    ax.set_title('Per-Epoch MoE Gradient Norms', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'grad_norm_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to {save_dir}")


def train(args, model, optimizer, train_loader, valid_loader, scheduler, save_dir):
    max_auc    = 0.0
    global_step = 0

    # Plain CrossEntropyLoss for the classification objective.
    # The MoE load-balancing loss (lora_loss * 200 + adapter_loss * 1) is
    # returned directly by model.forward() and added separately each step.
    criterion = nn.CrossEntropyLoss()

    # moe_loss_coef scales the combined MoE auxiliary loss relative to cls loss.
    # Default 0.1 keeps the routing loss from dominating early training when
    # the MoE gates are still random.
    moe_loss_coef = args.moe_loss_coef

    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        'epochs': [],
        'train_loss': [], 'train_cls_loss': [], 'train_moe_loss': [],
        'train_acc': [],
        'val_acc': [], 'val_auc': [], 'val_eer': [],
        'lora_grad_norms': [], 'adapter_grad_norms': [],
    }

    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------ resume
    if args.resume > -1:
        ckpt_path = os.path.join(save_dir, f'models_params_{args.resume}.tar')
        checkpoint = torch.load(ckpt_path,
                                map_location=f'cuda:{torch.cuda.current_device()}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log.info(f"Resumed from epoch {args.resume}: {ckpt_path}")

    # ------------------------------------------------------------------ loop
    for epoch in range(args.resume + 1, args.epochs):

        # ------------------------------------------------------------ train
        print(f'\n[Epoch {epoch+1}/{args.epochs}] Starting training...')

        # Re-apply correct train/eval modes every epoch.
        # model.train() alone would undo freeze_stages() .eval() calls.
        set_train_mode(model)

        epoch_total_loss = 0.0
        epoch_cls_loss   = 0.0
        epoch_moe_loss   = 0.0
        epoch_samples    = 0
        epoch_correct    = 0
        running_total    = 0
        running_correct  = 0
        lora_gnorm_accum    = 0.0
        adapter_gnorm_accum = 0.0
        grad_steps = 0

        st_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.cuda()
            labels = labels.cuda()

            # ------------------------------------------------------ forward
            # MTS_MoE model always returns (logits, moe_loss).
            # moe_loss = mean over blocks of (lora_loss*200 + adapter_loss*1),
            # which encourages balanced expert utilisation.
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs, moe_loss = model(inputs)
                cls_loss   = criterion(outputs, labels)
                total_loss = cls_loss + moe_loss_coef * moe_loss

            # --------------------------------------------- backward + update
            scaler.scale(total_loss).backward()

            # Unscale before clipping so clip threshold is in gradient units
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # Track grad norms for both MoE components
            lg, ag = get_moe_grad_norms(model)
            lora_gnorm_accum    += lg
            adapter_gnorm_accum += ag
            grad_steps += 1

            batch_size    = inputs.size(0)
            batch_correct = torch.sum(torch.argmax(outputs, 1) == labels).item()

            epoch_total_loss += total_loss.item()
            epoch_cls_loss   += cls_loss.item()
            epoch_moe_loss   += moe_loss.item()
            epoch_samples    += batch_size
            epoch_correct    += batch_correct
            running_total    += batch_size
            running_correct  += batch_correct
            global_step      += 1

            if global_step % args.record_step == 0:
                period       = time.time() - st_time
                lora_lr      = optimizer.param_groups[0]['lr']
                adapter_lr   = optimizer.param_groups[1]['lr']
                avg_lg       = lora_gnorm_accum    / max(grad_steps, 1)
                avg_ag       = adapter_gnorm_accum / max(grad_steps, 1)

                log.info(
                    'Epoch [{:0>3}/{:0>3}] Iter [{:0>3}/{:0>3}] '
                    'Loss:{:.4f} (cls:{:.4f} moe:{:.4f}) Acc:{:.2%} '
                    'LR(lora):{:.2e} LR(adapter):{:.2e} '
                    'LoRA‖∇‖:{:.4f} Adapter‖∇‖:{:.4f} '
                    'time:{}m{}s'.format(
                        epoch + 1, args.epochs,
                        i + 1, len(train_loader),
                        epoch_total_loss / (i + 1),
                        epoch_cls_loss   / (i + 1),
                        epoch_moe_loss   / (i + 1),
                        running_correct  / running_total,
                        lora_lr, adapter_lr,
                        avg_lg, avg_ag,
                        int(period // 60), int(period % 60),
                    )
                )
                st_time             = time.time()
                running_total       = 0
                running_correct     = 0
                lora_gnorm_accum    = 0.0
                adapter_gnorm_accum = 0.0
                grad_steps          = 0

        avg_total_loss = epoch_total_loss / len(train_loader)
        avg_cls_loss   = epoch_cls_loss   / len(train_loader)
        avg_moe_loss   = epoch_moe_loss   / len(train_loader)
        train_accuracy = epoch_correct    / epoch_samples
        mean_lora_gnorm    = lora_gnorm_accum    / max(grad_steps, 1)
        mean_adapter_gnorm = adapter_gnorm_accum / max(grad_steps, 1)

        # Warn if either MoE component has dead gradients
        for component, gnorm in [('LoRA-MoE', mean_lora_gnorm),
                                  ('MTS-MoE adapter', mean_adapter_gnorm)]:
            if gnorm < 1e-7:
                log.warning(
                    f"⚠️  Epoch {epoch+1}: {component} gradient norm is near-zero "
                    f"({gnorm:.2e}). Adapters may not be learning. "
                    "Consider increasing their learning rate."
                )

        # ---------------------------------------------------------- validate
        print('Starting validation...')
        model.eval()
        predictions = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, total=len(valid_loader),
                                       ncols=70, leave=False, unit='batch'):
                inputs = inputs.cuda()
                labels = labels.cuda()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    # moe_loss is discarded at validation — routing noise is
                    # disabled (model.eval()) so the loss would be uninformative
                    outputs, _ = model(inputs)
                outputs = F.softmax(outputs, dim=-1)
                predictions.extend(outputs[:, 1].cpu().tolist())
                labels_list.extend(labels.cpu().tolist())

        results = cal_metrics(labels_list, predictions, threshold=0.5)
        log.info(
            'Validation: Epoch [{:0>3}/{:0>3}] '
            'Acc:{:.2%} AUC:{:.4f} EER:{:.2%}'.format(
                epoch + 1, args.epochs,
                results.ACC, results.AUC, results.EER,
            )
        )

        # -------------------------------------------------------- bookkeeping
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(float(avg_total_loss))
        history['train_cls_loss'].append(float(avg_cls_loss))
        history['train_moe_loss'].append(float(avg_moe_loss))
        history['train_acc'].append(float(train_accuracy))
        history['val_acc'].append(float(results.ACC))
        history['val_auc'].append(float(results.AUC))
        history['val_eer'].append(float(results.EER))
        history['lora_grad_norms'].append(float(mean_lora_gnorm))
        history['adapter_grad_norms'].append(float(mean_adapter_gnorm))

        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        # ------------------------------------------------------- save latest
        state = {
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, os.path.join(save_dir, f'models_params_{epoch}.tar'))

        # ---------------------------------------------------- save best AUC
        if results.AUC > max_auc:
            for m in os.listdir(save_dir):
                if m.startswith('model_params_best'):
                    os.remove(os.path.join(save_dir, m))
            max_auc = results.AUC
            best_name = (
                f'model_params_best_{results.ACC:.4f}acc_'
                f'{results.AUC:.4f}auc_epoch{epoch+1:03d}.pkl'
            )
            torch.save(model.state_dict(), os.path.join(save_dir, best_name))
            log.info(f"  ✓ New best model saved: {best_name}")

        scheduler.step()

    plot_training_curves(history, save_dir)
    return history


# ============================================================== entry point ===
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train MTS-MoE + AuroRA LoRA-MoE ViT detector')

    parser.add_argument('--device',    '-dv', type=int, default=0)
    parser.add_argument('--model_dir', '-md', type=str, default='models/mts_moe')
    parser.add_argument('--resume',    '-rs', type=int, default=-1,
                        help='Epoch index to resume from (-1 = fresh start)')
    parser.add_argument('--epochs',          type=int,   default=20)
    parser.add_argument('--record_step',     type=int,   default=100)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--num_workers',     type=int,   default=16)
    parser.add_argument('--data_dir',        type=str,
                        default='/data3/law/data/FF++/c23')

    # ----------------------------------------------------------- LR per group
    # MTS_MoE has two independent adapter systems, each zero-initialised,
    # so both need a meaningfully higher LR than a single flat value would give.
    #
    #   lora_lr    — LoRA-MoE in attention  (AuroRA QKV, lora_B init=0)
    #   adapter_lr — MTS-MoE parallel to MLP (adapter_up init=0)
    #   head_lr    — classification head
    #   norm_lr    — trainable norm1 LayerNorms
    parser.add_argument('--lora_lr',    type=float, default=1e-4,
                        help='LR for LoRA-MoE params (lora_B starts at 0)')
    parser.add_argument('--adapter_lr', type=float, default=1e-4,
                        help='LR for MTS-MoE adapter params (adapter_up starts at 0)')
    parser.add_argument('--head_lr',    type=float, default=1e-4,
                        help='LR for classification head')
    parser.add_argument('--norm_lr',    type=float, default=1e-5,
                        help='LR for trainable LayerNorm (norm1) params')

    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout',      type=float, default=0.1)
    parser.add_argument('--num_classes',  type=int,   default=2)

    # ---------------------------------------------------- MoE topology args
    parser.add_argument('--lora_topk',    type=int, default=1,
                        help='Top-k experts per token for LoRA-MoE routing')
    parser.add_argument('--adapter_topk', type=int, default=2,
                        help='Top-k experts per image for MTS-MoE routing')

    # --------------------------------------------------- MoE auxiliary loss
    # moe_loss is already internally scaled as (lora*200 + adapter*1) inside
    # forward_features(). moe_loss_coef further scales the whole term relative
    # to classification loss. Start small (0.1) to prevent routing loss from
    # dominating before the adapters have warmed up.
    parser.add_argument('--moe_loss_coef', type=float, default=0.1,
                        help='Weight of MoE load-balancing loss vs cls loss')

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
    train_loader, valid_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ----------------------------------------------------------------- model
    model = vit_base_patch16_224_in21k(
        pretrained=True,
        num_classes=args.num_classes,
        drop_rate=args.dropout,
        lora_topk=args.lora_topk,
        adapter_topk=args.adapter_topk,
    )
    model = model.cuda()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters:     {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}  '
          f'({100 * trainable_params / total_params:.2f}%)')
    print(f'MoE topology: lora_topk={args.lora_topk}, adapter_topk={args.adapter_topk}')

    # ------------------------------------------------- param groups
    # Four groups matching the four trainable subsystems in freeze_stages():
    #
    #   Group 0 — LoRA_MoE params    → lora_lr    (QKV adapters, lora_B=0 init)
    #   Group 1 — adapter_MoE params → adapter_lr (MTS-MoE, adapter_up=0 init)
    #   Group 2 — head params        → head_lr
    #   Group 3 — norm1 params       → norm_lr
    lora_params    = []
    adapter_params = []
    head_params    = []
    norm_params    = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'LoRA' in name:
            lora_params.append(param)
        elif 'adapter' in name:
            adapter_params.append(param)
        elif 'head' in name:
            head_params.append(param)
        else:
            norm_params.append(param)   # norm1 weight/bias

    param_groups = [
        {'params': lora_params,    'lr': args.lora_lr,    'initial_lr': args.lora_lr,    'name': 'lora'},
        {'params': adapter_params, 'lr': args.adapter_lr, 'initial_lr': args.adapter_lr, 'name': 'adapter'},
        {'params': head_params,    'lr': args.head_lr,    'initial_lr': args.head_lr,    'name': 'head'},
        {'params': norm_params,    'lr': args.norm_lr,    'initial_lr': args.norm_lr,    'name': 'norm'},
    ]

    log.info(f'LoRA-MoE    param group: {sum(p.numel() for p in lora_params):,} params @ lr={args.lora_lr}')
    log.info(f'MTS-MoE     param group: {sum(p.numel() for p in adapter_params):,} params @ lr={args.adapter_lr}')
    log.info(f'Head        param group: {sum(p.numel() for p in head_params):,} params @ lr={args.head_lr}')
    log.info(f'Norm        param group: {sum(p.numel() for p in norm_params):,} params @ lr={args.norm_lr}')

    optimizer = optim.Adam(param_groups, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay)

    # Epoch-level warmup + step decay — same scheme as ViT_MoE training.
    # Each group's LR scales from its own initial_lr, preserving the
    # lora_lr : adapter_lr : head_lr ratio throughout training.
    def warmup_lambda(epoch):
        warmup_epochs = 3
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 ** ((epoch - warmup_epochs) // 5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lambda, last_epoch=args.resume,
    )

    # ---------------------------------------------------------------- train
    print('\nStarting training...')
    history  = train(args, model, optimizer, train_loader, valid_loader,
                     scheduler, save_dir)
    duration = time.time() - start_time

    print('\n' + '=' * 60)
    print('Training Completed!')
    print('=' * 60)
    print(f'Total time:     {int(duration//3600)}h {int(duration%3600//60)}m')
    print(f'Model dir:      {args.model_dir}')
    print(f'MoE topology:   lora_topk={args.lora_topk}, adapter_topk={args.adapter_topk}')
    print(f'LR (lora/adapter/head/norm): '
          f'{args.lora_lr}/{args.adapter_lr}/{args.head_lr}/{args.norm_lr}')
    print(f'MoE loss coef:  {args.moe_loss_coef}')
    print(f'\nFinal Metrics:')
    print(f'  Best Val AUC:      {max(history["val_auc"]):.4f}')
    print(f'  Best Val Accuracy: {max(history["val_acc"]):.2%}')
    print(f'  Lowest Val EER:    {min(history["val_eer"]):.2%}')
    print(f'  Final Train Loss:  {history["train_loss"][-1]:.4f}')
    print(f'    (cls: {history["train_cls_loss"][-1]:.4f}, '
          f'moe: {history["train_moe_loss"][-1]:.4f})')
    print('=' * 60)