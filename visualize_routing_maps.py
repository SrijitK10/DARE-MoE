"""
DaRE-MoE Routing Map Visualization

Generates, for a trained ForensicMoE checkpoint:
  1. Per-image 4-panel figures  (input | DM routing | GAN routing | diff)
  2. Domain comparison grid     (GAN-fake | DM-fake | Real)
  3. Routing statistics plots   (mean routing bar chart, accuracy, histograms)

Router output from DomainAwareRouter:
    routing_probs  (B, N, H', W')  where H'=W'=14 for 224-px input
    routing_agg    (B, N)          spatially averaged weights

Expert index convention (matches train_moe_finetune.py data_dirs order):
    Expert 0 = DM expert   (feeds data_dm_dir)
    Expert 1 = GAN expert  (feeds data_gan_dir)

Usage:
    python visualize_routing_maps.py --checkpoint_epoch 19
    python visualize_routing_maps.py --checkpoint_path checkpoints/moe_finetune/model_params_best_*.pkl \\
        --data_gan_dir /path/to/gan --data_dm_dir /path/to/dm
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import transform as dataset_transform
from models.moe_forensic import ForensicMoE
from routing_utils import RoutingVisualizationUtils, RoutingStatisticsComputer

# Expert 0 = DM, Expert 1 = GAN  (matches training data_dirs order)
EXPERT_NAMES = ['DM', 'GAN']


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, num_experts=2, feature_dim=768, device='cpu'):
    """
    Instantiate ForensicMoE and load weights from a checkpoint.

    Supports both:
      - raw state_dict (.pkl)
      - {'model_state_dict': ...} dict (.tar from train_moe_finetune.py)
    """
    expert_states = [{}] * num_experts
    model = ForensicMoE(
        backbone=None,
        expert_models=expert_states,
        num_experts=num_experts,
        feature_dim=feature_dim,
        freeze_backbone=False,
    )

    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = raw.get('model_state_dict', raw) if isinstance(raw, dict) else raw

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load] {len(missing)} missing key(s)")
    if unexpected:
        print(f"  [load] {len(unexpected)} unexpected key(s)")
    if not missing and not unexpected:
        print("  [load] All keys matched.")

    model = model.to(device)
    model.eval()
    return model


def _resolve_checkpoint(args):
    """Return the checkpoint path based on CLI arguments."""
    if args.checkpoint_path:
        return args.checkpoint_path
    if args.checkpoint_epoch >= 0:
        return os.path.join(args.model_dir,
                            f'models_params_{args.checkpoint_epoch}.tar')
    # Auto-select: prefer best .pkl, then latest .tar
    best = sorted(glob.glob(os.path.join(args.model_dir,
                                         'model_params_best*.pkl')))
    if best:
        return best[-1]
    tars = sorted(glob.glob(os.path.join(args.model_dir,
                                         'models_params_*.tar')))
    if tars:
        return tars[-1]
    return None


# ---------------------------------------------------------------------------
# Routing extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_routing_maps(model, dataloader, device, max_samples=50):
    """
    Runs inference and collects per-image routing data.

    Returns:
        images_list:        list of (C, H, W) cpu tensors
        labels_list:        list of int  (0=real, 1=fake)
        routing_probs_list: list of (N, H', W') numpy float32
        routing_agg_list:   list of (N,)        numpy float32
    """
    images_list, labels_list = [], []
    routing_probs_list, routing_agg_list = [], []
    count = 0

    for images, labels in tqdm(dataloader, desc='  Extracting', ncols=70):
        if count >= max_samples:
            break
        images = images.to(device)
        _, feat = model(images, return_all_features=True)

        r_probs = feat['routing_probs'].cpu().numpy()   # (B, N, H', W')
        r_agg   = feat['routing_agg'].cpu().numpy()     # (B, N)

        for b in range(images.size(0)):
            if count >= max_samples:
                break
            images_list.append(images[b].cpu())
            labels_list.append(int(labels[b].item()))
            routing_probs_list.append(r_probs[b])
            routing_agg_list.append(r_agg[b])
            count += 1

    return images_list, labels_list, routing_probs_list, routing_agg_list


# ---------------------------------------------------------------------------
# Figure 1: per-image 4-panel routing maps
# ---------------------------------------------------------------------------

def save_individual_routing_maps(images, labels, routing_probs, routing_agg,
                                  save_dir, num_experts=2):
    """
    For each image save a 4-panel figure:
        Panel 1: Input image
        Panel 2: DM expert routing overlay  (jet colormap)
        Panel 3: GAN expert routing overlay (jet colormap)
        Panel 4: Difference map GAN – DM    (RdYlBu diverging)

    Blue = DM-dominant, Red = GAN-dominant in the difference panel.
    """
    os.makedirs(save_dir, exist_ok=True)

    for idx, (img_t, lbl, r_probs, r_agg) in enumerate(
            zip(images, labels, routing_probs, routing_agg)):

        img_np = RoutingVisualizationUtils.normalize_image(img_t)
        H, W   = img_np.shape[:2]
        lbl_str = 'Real' if lbl == 0 else 'Fake'

        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
        fig.patch.set_facecolor('#1a1a1a')
        for ax in axes:
            ax.set_facecolor('#1a1a1a')

        # Panel 0 — input
        axes[0].imshow(img_np)
        axes[0].set_title(f'Input Image\n({lbl_str})',
                          color='white', fontsize=11)
        axes[0].axis('off')

        # Panels 1..num_experts — per-expert routing overlays
        for exp_idx in range(num_experts):
            exp_name = EXPERT_NAMES[exp_idx] if exp_idx < len(EXPERT_NAMES) \
                       else f'Expert {exp_idx}'
            r_map  = r_probs[exp_idx]               # (H', W')
            r_up   = RoutingVisualizationUtils.upsample_routing_map(r_map, H, W)
            overlay = RoutingVisualizationUtils.create_heatmap_overlay(img_np, r_up)
            axes[exp_idx + 1].imshow(overlay)
            axes[exp_idx + 1].set_title(
                f'{exp_name} Routing Map\n(mean={r_agg[exp_idx]:.3f})',
                color='white', fontsize=11)
            axes[exp_idx + 1].axis('off')

        # Panel 3 — difference (GAN – DM)  [only when num_experts == 2]
        if num_experts == 2:
            diff     = r_probs[1] - r_probs[0]          # GAN - DM in [-1, 1]
            diff_up  = RoutingVisualizationUtils.upsample_routing_map(diff, H, W)
            diff_rgb = RoutingVisualizationUtils.difference_map_to_rgb(diff_up)
            axes[3].imshow(diff_rgb)
            axes[3].set_title('Difference Map\n(GAN – DM)',
                              color='white', fontsize=11)
            axes[3].axis('off')
        else:
            axes[3].axis('off')

        plt.suptitle(
            f'DaRE-MoE Routing Analysis — Image {idx:03d} ({lbl_str})',
            color='white', fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out = os.path.join(save_dir, f'routing_{idx:03d}_{lbl_str.lower()}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()

    print(f"  Saved {len(images)} individual routing maps → {save_dir}")


# ---------------------------------------------------------------------------
# Figure 2: domain comparison (GAN-fake | DM-fake | Real)
# ---------------------------------------------------------------------------

def save_domain_comparison(gan_imgs, dm_imgs, real_imgs,
                            gan_routing, dm_routing, real_routing,
                            save_dir, num_experts=2):
    """
    3-column × (1 + num_experts) row grid:
        Columns: GAN-generated | DM-generated | Real
        Row 0:   Input image
        Row 1+:  Per-expert routing overlay

    Visually demonstrates domain-aware specialisation.
    """
    if not (gan_imgs and dm_imgs and real_imgs):
        print("  Skipping domain comparison: one or more domains have no samples.")
        return
    os.makedirs(save_dir, exist_ok=True)

    n_rows = 1 + num_experts
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.5 * n_cols, 5.5 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor('#111111')
    for row in axes:
        for ax in row:
            ax.set_facecolor('#111111')

    col_data = [
        ('GAN-generated',  gan_imgs[0],  gan_routing[0]),
        ('DM-generated',   dm_imgs[0],   dm_routing[0]),
        ('Real',           real_imgs[0], real_routing[0]),
    ]
    row_labels = ['Input Image'] + \
                 [f'{EXPERT_NAMES[i] if i < len(EXPERT_NAMES) else f"Expert {i}"} Routing'
                  for i in range(num_experts)]

    for col, (title, img_t, r_probs) in enumerate(col_data):
        img_np = RoutingVisualizationUtils.normalize_image(img_t)
        H, W   = img_np.shape[:2]

        # Row 0 — input
        axes[0, col].imshow(img_np)
        axes[0, col].set_title(title, color='white',
                                fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        # Rows 1..num_experts — routing overlays
        for exp_idx in range(num_experts):
            r_map  = r_probs[exp_idx]
            r_up   = RoutingVisualizationUtils.upsample_routing_map(r_map, H, W)
            overlay = RoutingVisualizationUtils.create_heatmap_overlay(img_np, r_up)
            axes[exp_idx + 1, col].imshow(overlay)
            axes[exp_idx + 1, col].axis('off')

    # Row labels on left column
    for r, rl in enumerate(row_labels):
        axes[r, 0].set_ylabel(rl, color='white', fontsize=10,
                              labelpad=6, rotation=90, va='center')

    plt.suptitle('Domain Comparison — DaRE-MoE Routing Specialisation',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])

    out = os.path.join(save_dir, 'domain_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Domain comparison → {out}")


# ---------------------------------------------------------------------------
# Figure 3: routing statistics
# ---------------------------------------------------------------------------

def save_routing_statistics(routing_agg_all, domain_labels_all,
                             save_dir, num_experts=2):
    """
    Three-panel statistics figure:
        Left:   Mean routing weight per expert per domain (bar chart)
        Middle: Routing accuracy per expert               (bar chart)
        Right:  GAN expert routing distribution by domain (histogram)
    """
    os.makedirs(save_dir, exist_ok=True)

    r_arr  = np.array(routing_agg_all)   # (N_total, num_experts)
    d_arr  = np.array(domain_labels_all) # (N_total,)

    domain_names = [EXPERT_NAMES[i] if i < len(EXPERT_NAMES)
                    else f'Expert {i}' for i in range(num_experts)] + ['Real']

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle('Routing Statistics — DaRE-MoE', fontsize=14, fontweight='bold')

    # --- Left: mean routing by domain ----------------------------------------
    x     = np.arange(len(domain_names))
    width = 0.35
    for exp_idx in range(num_experts):
        exp_name = EXPERT_NAMES[exp_idx] if exp_idx < len(EXPERT_NAMES) \
                   else f'Expert {exp_idx}'
        means = []
        for d_idx in range(num_experts + 1):
            mask = (d_arr == d_idx) if d_idx < num_experts \
                   else (d_arr == num_experts)
            means.append(float(r_arr[mask, exp_idx].mean()) if mask.sum() > 0 else 0.0)
        offset = (exp_idx - (num_experts - 1) / 2) * width
        axes[0].bar(x + offset, means, width, label=f'{exp_name} Expert')

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(domain_names)
    axes[0].set_ylabel('Mean Routing Weight')
    axes[0].set_title('Mean Routing Weight by Domain')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=1.0 / num_experts, linestyle='--', color='gray',
                    alpha=0.6, label='Uniform (1/N)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Middle: routing accuracy per expert ---------------------------------
    accs   = []
    colors = ['steelblue', 'salmon', 'mediumseagreen', 'gold']
    for exp_idx in range(num_experts):
        mask = (d_arr == exp_idx)
        if mask.sum() > 0:
            acc = (np.argmax(r_arr[mask], axis=1) == exp_idx).mean()
        else:
            acc = 0.0
        accs.append(float(acc))

    bar_labels = [EXPERT_NAMES[i] if i < len(EXPERT_NAMES) else f'Expert {i}'
                  for i in range(num_experts)]
    bars = axes[1].bar(bar_labels, accs,
                       color=colors[:num_experts], edgecolor='black')
    axes[1].set_ylabel('Routing Accuracy')
    axes[1].set_title('Routing Accuracy per Expert\nP(argmax(r̄) = correct domain)')
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=1.0 / num_experts, linestyle='--', color='gray',
                    alpha=0.7, label='Random baseline')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    for bar, acc in zip(bars, accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                     f'{acc:.1%}', ha='center', va='bottom', fontsize=11,
                     fontweight='bold')

    # --- Right: GAN expert routing distribution by domain --------------------
    hist_colors = ['royalblue', 'tomato', 'forestgreen']
    gan_idx = 1  # GAN expert is always index 1
    for d_idx, (d_name, h_color) in enumerate(zip(domain_names, hist_colors)):
        mask = (d_arr == d_idx) if d_idx < num_experts else (d_arr == num_experts)
        if mask.sum() > 0:
            axes[2].hist(r_arr[mask, gan_idx], bins=25, alpha=0.55,
                         label=f'{d_name} domain (n={mask.sum()})',
                         color=h_color, density=True)

    axes[2].set_xlabel('GAN Expert Routing Weight  r̄_GAN')
    axes[2].set_ylabel('Density')
    axes[2].set_title('GAN Expert Routing Distribution\n'
                      'GAN images → peak near 1,  DM/Real → near 0')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, 'routing_stats.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Routing statistics → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualize DaRE-MoE spatial routing maps after finetuning')

    # Checkpoint
    parser.add_argument('--checkpoint_epoch', '-e', type=int, default=-1,
                        help='Epoch index (uses models_params_{e}.tar)')
    parser.add_argument('--checkpoint_path', '-cp', type=str, default=None,
                        help='Direct checkpoint path (overrides --checkpoint_epoch)')
    parser.add_argument('--model_dir', type=str,
                        default='checkpoints/moe_finetune')

    # Data
    parser.add_argument('--data_gan_dir', type=str, default=None,
                        help='GAN domain data root (must contain test/ or val/). '
                             'Omit to skip GAN domain.')
    parser.add_argument('--data_dm_dir', type=str, default=None,
                        help='DM domain data root (must contain test/ or val/). '
                             'Omit to skip DM domain.')

    # Output
    parser.add_argument('--output_dir', '-o', type=str,
                        default='routing_analysis')

    # Model
    parser.add_argument('--num_experts',  type=int, default=2)
    parser.add_argument('--feature_dim',  type=int, default=768)

    # Run config
    parser.add_argument('--max_samples',  type=int, default=50,
                        help='Max samples per domain (for speed)')
    parser.add_argument('--batch_size',   type=int, default=8)
    parser.add_argument('--num_workers',  type=int, default=2)
    parser.add_argument('--device',       type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}'
                          if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Resolve checkpoint
    ckpt_path = _resolve_checkpoint(args)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        print(f"Error: checkpoint not found. "
              f"Tried: {ckpt_path}\nSet --checkpoint_path or --checkpoint_epoch.")
        return
    print(f"Checkpoint: {ckpt_path}")

    # Load model
    model = load_model(ckpt_path, args.num_experts, args.feature_dim, device)

    # Collect images and routing maps from both domains
    (all_images, all_routing_probs,
     all_routing_agg, all_domain_labels,
     gan_imgs, dm_imgs, real_imgs,
     gan_routing, dm_routing, real_routing) = _collect_all(model, device, args)

    if not all_images:
        print("Error: no images loaded. Check --data_gan_dir and --data_dm_dir.")
        return

    print(f"\nTotal collected: {len(all_images)} images")

    # 1. Individual maps (first 20)
    # Reconstruct binary labels: domain < num_experts → fake (1), else real (0)
    bin_labels_20 = [0 if d == args.num_experts else 1
                     for d in all_domain_labels[:20]]
    print("\n[1/3] Individual routing maps...")
    save_individual_routing_maps(
        all_images[:20], bin_labels_20,
        all_routing_probs[:20], all_routing_agg[:20],
        os.path.join(args.output_dir, 'individual'),
        args.num_experts,
    )

    # 2. Domain comparison
    print("\n[2/3] Domain comparison figure...")
    save_domain_comparison(
        gan_imgs, dm_imgs, real_imgs,
        gan_routing, dm_routing, real_routing,
        os.path.join(args.output_dir, 'comparison'),
        args.num_experts,
    )

    # 3. Statistics
    print("\n[3/3] Routing statistics...")
    save_routing_statistics(
        all_routing_agg, all_domain_labels,
        os.path.join(args.output_dir, 'statistics'),
        args.num_experts,
    )

    # Summary
    r_arr = np.array(all_routing_agg)
    d_arr = np.array(all_domain_labels)
    acc   = RoutingStatisticsComputer.compute_routing_accuracy(
        r_arr, d_arr, args.num_experts)
    stats = RoutingStatisticsComputer.compute_per_expert_stats(
        r_arr, d_arr, args.num_experts, EXPERT_NAMES)

    print(f"\n{'=' * 52}")
    print("ROUTING ANALYSIS SUMMARY")
    print(f"{'=' * 52}")
    print(f"  Total images:       {len(all_images)}")
    print(f"  Overall routing acc:{acc:.2%}")
    for name, s in stats.items():
        print(f"  {name}: domain_acc={s['routing_accuracy']:.2%}  "
              f"mean_r_domain={s['mean_routing_on_domain']:.3f}  "
              f"mean_r_real={s['mean_routing_on_real']:.3f}")
    print(f"  Output dir:         {args.output_dir}")
    print(f"{'=' * 52}")


def _build_label_map(class_to_idx):
    """
    Map ImageFolder class indices → semantic labels  (0 = real, 1 = fake).

    ImageFolder assigns indices alphabetically, so a single folder named
    '1_fake' or 'adm' gets index 0 — not 1.  We must derive fake/real from
    the *folder name*, not the numeric index.

    Priority:
      1. Keyword match against known fake / real vocabulary.
      2. Leading-digit convention  ('0_real' → 0, '1_fake' → 1).
      3. Fallback: treat unknown single-class dirs as fake (most common use-case).
    """
    _FAKE_KW = {
        'fake', 'forged', 'manipulated', 'generated', 'synthetic',
        'adm', 'ldm', 'ddpm', 'diffusion', 'dm',
        'gan', 'stylegan', 'progan', 'biggan', 'vqgan',
        'deepfake', 'df', 'f2f', 'fs', 'nt', 'faceswap',
    }
    _REAL_KW = {
        'real', 'genuine', 'original', 'authentic', 'pristine', 'youtube',
    }

    label_map = {}
    for cls_name, cls_idx in class_to_idx.items():
        name_lower = cls_name.lower()
        if any(kw in name_lower for kw in _FAKE_KW):
            label_map[cls_idx] = 1          # fake
        elif any(kw in name_lower for kw in _REAL_KW):
            label_map[cls_idx] = 0          # real
        elif cls_name[:1].isdigit():
            # e.g. '0_real' → 0,  '1_fake' → 1
            label_map[cls_idx] = int(cls_name[0]) % 2
        else:
            # Unknown name, single class dir → assume fake
            label_map[cls_idx] = 1

    return label_map


def _collect_all(model, device, args):
    """
    Collect routing data for all domains in one clean pass.

    Returns flat lists for statistics + typed lists for domain comparison.
    """
    all_images, all_routing_probs, all_routing_agg, all_domain_labels = \
        [], [], [], []
    gan_imgs,  dm_imgs,  real_imgs  = [], [], []
    gan_routing, dm_routing, real_routing = [], [], []

    for domain_idx, (domain_key, data_dir) in enumerate(
            [('DM', args.data_dm_dir), ('GAN', args.data_gan_dir)]):

        if not data_dir:
            print(f"  [SKIP] {domain_key}: no path provided (--data_{domain_key.lower()}_dir)")
            continue

        subdir = os.path.join(data_dir, 'test')
        if not os.path.isdir(subdir):
            subdir = os.path.join(data_dir, 'val')
        if not os.path.isdir(subdir):
            print(f"  [SKIP] {domain_key}: no test/ or val/ subfolder in {data_dir}")
            continue

        dataset = ImageFolder(subdir, transform=dataset_transform)

        # Build semantic label map from folder names so the code works
        # regardless of how many class folders are present.
        # ImageFolder sorts classes alphabetically: a lone '1_fake/' folder
        # gets index 0, which the old code wrongly treated as real.
        label_map = _build_label_map(dataset.class_to_idx)
        print(f"\n[{domain_key}] {len(dataset)} images from {subdir}")
        for cls_name, cls_idx in sorted(dataset.class_to_idx.items(),
                                        key=lambda x: x[1]):
            semantic = 'fake' if label_map[cls_idx] == 1 else 'real'
            print(f"  folder '{cls_name}' (ImageFolder idx {cls_idx}) → {semantic}")

        # shuffle=True so both real and fake images are collected
        # evenly within max_samples (alphabetical ordering would give only
        # one class when max_samples < total dataset size)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available())

        imgs, raw_labels, r_probs, r_agg = extract_routing_maps(
            model, loader, device, max_samples=args.max_samples)

        # Remap ImageFolder indices → semantic labels (0=real, 1=fake)
        labels = [label_map.get(lbl, lbl) for lbl in raw_labels]

        for img, lbl, rp, ra in zip(imgs, labels, r_probs, r_agg):
            all_images.append(img)
            all_routing_probs.append(rp)
            all_routing_agg.append(ra)
            # fake → domain index,  real → num_experts (sentinel)
            all_domain_labels.append(domain_idx if lbl == 1 else args.num_experts)

        # Typed splits for domain comparison figure
        if domain_key == 'GAN':
            gan_imgs    = [imgs[i] for i, l in enumerate(labels) if l == 1]
            gan_routing = [r_probs[i] for i, l in enumerate(labels) if l == 1]
            real_imgs   = [imgs[i] for i, l in enumerate(labels) if l == 0]
            real_routing = [r_probs[i] for i, l in enumerate(labels) if l == 0]
        elif domain_key == 'DM':
            dm_imgs    = [imgs[i] for i, l in enumerate(labels) if l == 1]
            dm_routing = [r_probs[i] for i, l in enumerate(labels) if l == 1]

    return (all_images, all_routing_probs, all_routing_agg, all_domain_labels,
            gan_imgs, dm_imgs, real_imgs,
            gan_routing, dm_routing, real_routing)


if __name__ == '__main__':
    main()
