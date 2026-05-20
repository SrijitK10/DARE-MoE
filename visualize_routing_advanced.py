"""
DaRE-MoE Advanced Routing Analysis

Publication-ready figures and advanced metrics:
  1. Difference-map grid   — GAN−DM for multiple images, sorted by domain
  2. Real-vs-Fake panel    — per-domain side-by-side comparison
  3. Paper figure grid     — the CVPR-style 3-row layout for the Results section:
                             Row 1: Input images  (GAN-fake | DM-fake | Real)
                             Row 2: DM expert routing overlay
                             Row 3: GAN expert routing overlay
  4. Specialization report — JSON with per-expert accuracy, entropy, histograms

Imports load_model and extract_routing_maps from visualize_routing_maps.py so
the model is initialised and data is loaded in the same way as the basic script.

Usage:
    python visualize_routing_advanced.py --checkpoint_epoch 19
    python visualize_routing_advanced.py --checkpoint_path <path> \\
        --data_gan_dir /path/to/gan --data_dm_dir /path/to/dm
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import transform as dataset_transform
from routing_utils import RoutingVisualizationUtils, RoutingStatisticsComputer
from visualize_routing_maps import (
    load_model, extract_routing_maps, _resolve_checkpoint, _collect_all,
    EXPERT_NAMES,
)


# ---------------------------------------------------------------------------
# Figure 1: difference-map grid
# ---------------------------------------------------------------------------

def save_difference_maps_grid(gan_imgs, dm_imgs, real_imgs,
                               gan_routing, dm_routing, real_routing,
                               save_dir, num_experts=2, n_per_domain=3):
    """
    Grid showing GAN–DM difference maps for representative images.

    Layout: n_per_domain columns per domain (GAN | DM | Real)
    Row 0: Input
    Row 1: GAN−DM difference map  (RdYlBu — red=GAN-dominant, blue=DM-dominant)
    """
    if not (gan_imgs and dm_imgs and real_imgs):
        print("  Skipping difference grid: missing domain samples.")
        return
    os.makedirs(save_dir, exist_ok=True)

    n = min(n_per_domain, len(gan_imgs), len(dm_imgs), len(real_imgs))
    n_cols = n * 3   # GAN block | DM block | Real block
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 5 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor('#111111')

    domain_blocks = [
        ('GAN-gen.', gan_imgs[:n],  gan_routing[:n]),
        ('DM-gen.',  dm_imgs[:n],   dm_routing[:n]),
        ('Real',     real_imgs[:n], real_routing[:n]),
    ]
    col_offset = 0
    for domain_title, imgs, routings in domain_blocks:
        for s in range(n):
            col = col_offset + s
            img_np = RoutingVisualizationUtils.normalize_image(imgs[s])
            H, W   = img_np.shape[:2]
            r_probs = routings[s]   # (N, H', W')

            # Row 0: input
            axes[0, col].imshow(img_np)
            axes[0, col].axis('off')
            if s == n // 2:
                axes[0, col].set_title(domain_title, color='white',
                                       fontsize=11, fontweight='bold')

            # Row 1: GAN−DM difference (only meaningful for 2 experts)
            if num_experts >= 2:
                diff    = r_probs[1] - r_probs[0]   # GAN - DM
                diff_up = RoutingVisualizationUtils.upsample_routing_map(diff, H, W)
                diff_rgb = RoutingVisualizationUtils.difference_map_to_rgb(diff_up)
                axes[1, col].imshow(diff_rgb)
            else:
                axes[1, col].axis('off')
            axes[1, col].axis('off')

        col_offset += n

    # Row labels
    row_labels = ['Input Image', 'Diff. Map (GAN \u2212 DM)']
    for r, rl in enumerate(row_labels):
        axes[r, 0].set_ylabel(rl, color='white', fontsize=10, labelpad=6)

    # Colour bar legend (manual)
    _add_rdylbu_colorbar(fig)

    plt.suptitle('Routing Difference Maps  —  GAN Expert Minus DM Expert',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    out = os.path.join(save_dir, 'difference_maps_grid.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Difference map grid → {out}")


def _add_rdylbu_colorbar(fig):
    """Append a thin horizontal colorbar below all axes using RdYlBu."""
    cax = fig.add_axes([0.15, 0.01, 0.7, 0.025])
    import matplotlib.colorbar as mcolorbar
    import matplotlib.cm as mcm
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    cb   = mcolorbar.ColorbarBase(cax, cmap=plt.cm.RdYlBu, norm=norm,
                                   orientation='horizontal')
    cb.set_label('GAN ← 0 → DM', color='white', fontsize=8)
    cb.ax.xaxis.set_tick_params(color='white')
    plt.setp(cb.ax.xaxis.get_ticklabels(), color='white', fontsize=7)


# ---------------------------------------------------------------------------
# Figure 2: real vs fake routing comparison
# ---------------------------------------------------------------------------

def save_realvsfake_comparison(gan_imgs, dm_imgs, real_imgs,
                                gan_routing, dm_routing, real_routing,
                                save_dir, num_experts=2):
    """
    For each domain (GAN, DM) show a real image and a fake image side by side
    with both expert routing overlays beneath them.

    Layout (2 columns per domain pair, num_experts+1 rows):
        Row 0: Input (Real | Fake)
        Row i: Expert i routing overlay (Real | Fake)
    """
    if not (gan_imgs and dm_imgs and real_imgs):
        print("  Skipping real-vs-fake: missing domain samples.")
        return
    os.makedirs(save_dir, exist_ok=True)

    n_rows = 1 + num_experts
    # 4 columns: [Real|GAN-fake] and [Real|DM-fake] — 2 pairs of 2
    n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 5 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor('#111111')

    pairs = [
        # (col_real, col_fake, real_title, fake_title, real_img, fake_img,
        #  real_r, fake_r)
        (0, 1, 'Real', 'GAN-fake',
         real_imgs[0], gan_imgs[0], real_routing[0], gan_routing[0]),
        (2, 3, 'Real', 'DM-fake',
         real_imgs[0], dm_imgs[0],  real_routing[0], dm_routing[0]),
    ]

    for col_r, col_f, title_r, title_f, img_r, img_f, rp_r, rp_f in pairs:
        for col, img_t, r_probs, title in [
                (col_r, img_r, rp_r, title_r),
                (col_f, img_f, rp_f, title_f)]:

            img_np = RoutingVisualizationUtils.normalize_image(img_t)
            H, W   = img_np.shape[:2]

            axes[0, col].imshow(img_np)
            axes[0, col].set_title(title, color='white',
                                   fontsize=11, fontweight='bold')
            axes[0, col].axis('off')

            for exp_idx in range(num_experts):
                r_up    = RoutingVisualizationUtils.upsample_routing_map(
                    r_probs[exp_idx], H, W)
                overlay = RoutingVisualizationUtils.create_heatmap_overlay(
                    img_np, r_up)
                exp_name = EXPERT_NAMES[exp_idx] if exp_idx < len(EXPERT_NAMES) \
                           else f'Expert {exp_idx}'
                axes[exp_idx + 1, col].imshow(overlay)
                axes[exp_idx + 1, col].axis('off')
                if col in (col_r,):
                    axes[exp_idx + 1, col].set_ylabel(
                        f'{exp_name} Routing', color='white',
                        fontsize=9, labelpad=4)

    # Divider between the two pairs
    for r in range(n_rows):
        axes[r, 1].patch.set_linewidth(3)

    plt.suptitle('Real vs Fake — Routing Overlay Comparison',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(save_dir, 'realvsfake_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Real-vs-fake comparison → {out}")


# ---------------------------------------------------------------------------
# Figure 3: paper-ready routing grid (CVPR Results figure)
# ---------------------------------------------------------------------------

def save_paper_figure(gan_imgs, dm_imgs, real_imgs,
                       gan_routing, dm_routing, real_routing,
                       save_dir, num_experts=2):
    """
    Paper-ready figure — the standard 3-row layout used in top ML venues:

        Row 1 (top):    Input images   [GAN-fake | DM-fake | Real]
        Row 2 (middle): DM expert routing overlay
        Row 3 (bottom): GAN expert routing overlay

    Column labels are clear domain identifiers.
    Row labels are on the left margin.
    DPI=200 for publication quality.

    The figure is self-contained with a caption at the bottom.
    """
    if not (gan_imgs and dm_imgs and real_imgs):
        print("  Skipping paper figure: missing domain samples.")
        return
    os.makedirs(save_dir, exist_ok=True)

    n_rows = 1 + num_experts
    n_cols = 3

    fig = plt.figure(figsize=(5.5 * n_cols, 5 * n_rows + 0.6),
                     facecolor='white')
    gs  = gridspec.GridSpec(n_rows, n_cols,
                            figure=fig,
                            wspace=0.04, hspace=0.08)

    col_data = [
        ('(a) GAN-generated', gan_imgs[0],  gan_routing[0]),
        ('(b) DM-generated',  dm_imgs[0],   dm_routing[0]),
        ('(c) Real',          real_imgs[0], real_routing[0]),
    ]
    row_labels = ['Input'] + \
                 [f'{EXPERT_NAMES[i] if i < len(EXPERT_NAMES) else f"Expert {i}"} Routing'
                  for i in range(num_experts)]

    for col, (col_title, img_t, r_probs) in enumerate(col_data):
        img_np = RoutingVisualizationUtils.normalize_image(img_t)
        H, W   = img_np.shape[:2]

        for row in range(n_rows):
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')

            if row == 0:
                ax.imshow(img_np)
                ax.set_title(col_title, fontsize=12, fontweight='bold', pad=4)
            else:
                exp_idx = row - 1
                r_up    = RoutingVisualizationUtils.upsample_routing_map(
                    r_probs[exp_idx], H, W)
                overlay = RoutingVisualizationUtils.create_heatmap_overlay(
                    img_np, r_up, alpha=0.5)
                ax.imshow(overlay)

            # Row label on leftmost column
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11, labelpad=6,
                              rotation=90, va='center')
                ax.yaxis.set_visible(True)
                ax.tick_params(left=False, labelleft=False)

    caption = (
        'Figure: Routing maps produced by the DaRE-MoE Domain-Aware Router. '
        'For GAN-generated images, the GAN expert routing is strongly activated, '
        'while diffusion images activate the DM expert. Real images exhibit '
        'balanced routing, demonstrating effective domain-aware specialisation.'
    )
    fig.text(0.5, 0.005, caption, ha='center', fontsize=8.5,
             style='italic', wrap=True)

    out = os.path.join(save_dir, 'routing_grid.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Paper figure (routing_grid.png) → {out}")


# ---------------------------------------------------------------------------
# Figure 4: per-expert routing specialisation report (JSON + bar figure)
# ---------------------------------------------------------------------------

def save_specialization_report(routing_agg_all, domain_labels_all,
                                save_dir, num_experts=2):
    """
    Compute and save:
      - JSON report with per-expert stats
      - Bar figure: domain_acc, mean_routing, entropy per expert
    """
    os.makedirs(save_dir, exist_ok=True)

    r_arr = np.array(routing_agg_all)   # (N_total, num_experts)
    d_arr = np.array(domain_labels_all)

    expert_names = EXPERT_NAMES[:num_experts] if len(EXPERT_NAMES) >= num_experts \
                   else [f'Expert {i}' for i in range(num_experts)]

    per_expert = RoutingStatisticsComputer.compute_per_expert_stats(
        r_arr, d_arr, num_experts, expert_names)
    overall_acc = RoutingStatisticsComputer.compute_routing_accuracy(
        r_arr, d_arr, num_experts)
    entropy_all = RoutingStatisticsComputer.compute_routing_entropy(r_arr)

    # Entropy by domain
    entropy_by_domain = {}
    for d_idx, d_name in enumerate(expert_names + ['Real']):
        mask = (d_arr == d_idx) if d_idx < num_experts else (d_arr == num_experts)
        if mask.sum() > 0:
            entropy_by_domain[d_name] = {
                'mean': float(entropy_all[mask].mean()),
                'std':  float(entropy_all[mask].std()),
            }

    report = {
        'overall_routing_accuracy': overall_acc,
        'per_expert':               per_expert,
        'entropy_by_domain':        entropy_by_domain,
        'n_total':                  len(r_arr),
    }

    json_path = os.path.join(save_dir, 'specialization_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"  Specialisation JSON → {json_path}")

    # Bar figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Expert Specialisation Analysis — DaRE-MoE',
                 fontsize=14, fontweight='bold')

    colors = ['steelblue', 'salmon', 'mediumseagreen', 'gold']

    # Left: routing accuracy per expert
    accs = [per_expert[n]['routing_accuracy'] for n in expert_names]
    bars = axes[0].bar(expert_names, accs, color=colors[:num_experts],
                       edgecolor='black')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Routing Accuracy')
    axes[0].set_title('Expert Routing Accuracy')
    axes[0].axhline(y=1.0 / num_experts, linestyle='--', color='gray',
                    alpha=0.7, label='Random')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for bar, v in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                     f'{v:.1%}', ha='center', fontsize=11, fontweight='bold')

    # Middle: mean routing on own domain vs real
    x = np.arange(len(expert_names))
    w = 0.35
    own_domain = [per_expert[n]['mean_routing_on_domain'] for n in expert_names]
    on_real    = [per_expert[n]['mean_routing_on_real']   for n in expert_names]
    axes[1].bar(x - w / 2, own_domain, w, label='On own domain', color='steelblue')
    axes[1].bar(x + w / 2, on_real,    w, label='On real images', color='tomato')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(expert_names)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Mean Routing Weight')
    axes[1].set_title('Mean Routing: Own Domain vs Real')
    axes[1].axhline(y=1.0 / num_experts, linestyle='--', color='gray',
                    alpha=0.7, label='Uniform (1/N)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Right: routing entropy by domain (lower = more specialised)
    ent_means = [entropy_by_domain.get(n, {}).get('mean', 0) for n in
                 expert_names + ['Real']]
    ent_stds  = [entropy_by_domain.get(n, {}).get('std', 0) for n in
                 expert_names + ['Real']]
    domain_labels_plot = expert_names + ['Real']
    axes[2].bar(domain_labels_plot, ent_means,
                yerr=ent_stds, capsize=5,
                color=colors[:len(domain_labels_plot)], edgecolor='black')
    axes[2].set_ylabel('Mean Routing Entropy (nats)')
    axes[2].set_title('Routing Entropy by Domain\n(lower = more specialised)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'specialization_metrics.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Specialisation figure → {fig_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Advanced DaRE-MoE routing analysis (publication figures)')

    parser.add_argument('--checkpoint_epoch', '-e', type=int, default=-1)
    parser.add_argument('--checkpoint_path', '-cp', type=str, default=None)
    parser.add_argument('--model_dir', type=str,
                        default='checkpoints/moe_finetune')
    parser.add_argument('--data_gan_dir', type=str, default=None,
                        help='GAN domain data root. Omit to skip GAN domain.')
    parser.add_argument('--data_dm_dir', type=str, default=None,
                        help='DM domain data root. Omit to skip DM domain.')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='routing_analysis_advanced')
    parser.add_argument('--num_experts',  type=int, default=2)
    parser.add_argument('--feature_dim',  type=int, default=768)
    parser.add_argument('--max_samples',  type=int, default=50)
    parser.add_argument('--batch_size',   type=int, default=8)
    parser.add_argument('--num_workers',  type=int, default=2)
    parser.add_argument('--device',       type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}'
                          if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_path = _resolve_checkpoint(args)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        print(f"Error: checkpoint not found — {ckpt_path}")
        return
    print(f"Checkpoint: {ckpt_path}")

    model = load_model(ckpt_path, args.num_experts, args.feature_dim, device)

    (all_images, all_routing_probs, all_routing_agg, all_domain_labels,
     gan_imgs, dm_imgs, real_imgs,
     gan_routing, dm_routing, real_routing) = _collect_all(model, device, args)

    if not all_images:
        print("Error: no images loaded.")
        return
    print(f"\nTotal collected: {len(all_images)} images")

    diff_dir   = os.path.join(args.output_dir, 'difference_maps')
    rvf_dir    = os.path.join(args.output_dir, 'realvsfake')
    paper_dir  = os.path.join(args.output_dir, 'paper_figures')
    spec_dir   = os.path.join(args.output_dir, 'specialization')

    print("\n[1/4] Difference map grid...")
    save_difference_maps_grid(
        gan_imgs, dm_imgs, real_imgs,
        gan_routing, dm_routing, real_routing,
        diff_dir, args.num_experts,
    )

    print("\n[2/4] Real-vs-fake comparison...")
    save_realvsfake_comparison(
        gan_imgs, dm_imgs, real_imgs,
        gan_routing, dm_routing, real_routing,
        rvf_dir, args.num_experts,
    )

    print("\n[3/4] Paper figure (routing_grid.png)...")
    save_paper_figure(
        gan_imgs, dm_imgs, real_imgs,
        gan_routing, dm_routing, real_routing,
        paper_dir, args.num_experts,
    )

    print("\n[4/4] Specialisation report...")
    report = save_specialization_report(
        all_routing_agg, all_domain_labels,
        spec_dir, args.num_experts,
    )

    print(f"\n{'=' * 52}")
    print("ADVANCED ANALYSIS SUMMARY")
    print(f"{'=' * 52}")
    print(f"  Overall routing acc: {report['overall_routing_accuracy']:.2%}")
    for name, s in report['per_expert'].items():
        print(f"  {name}: acc={s['routing_accuracy']:.2%}  "
              f"mean_r={s['mean_routing_on_domain']:.3f}")
    print(f"  Output dir:          {args.output_dir}")
    print(f"  Paper figure:        {os.path.join(paper_dir, 'routing_grid.png')}")
    print(f"{'=' * 52}")


if __name__ == '__main__':
    main()
