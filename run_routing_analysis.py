"""
DaRE-MoE Routing Analysis — Master Pipeline

Runs the full routing map analysis in one command after finetuning:
  1. Validates checkpoint and data directories
  2. Runs basic visualizations  (individual maps, domain comparison, statistics)
  3. Runs advanced analysis     (difference maps, paper figure, specialisation)
  4. Writes a unified JSON report with all routing metrics

This is the RECOMMENDED entry point (see FILE_INDEX.md).

Usage:
    # Minimal — uses defaults for data paths and latest checkpoint
    python run_routing_analysis.py --checkpoint_epoch 19

    # Full explicit invocation
    python run_routing_analysis.py \\
        --model_dir   checkpoints/moe_finetune \\
        --checkpoint_epoch 19 \\
        --data_gan_dir /path/to/gan \\
        --data_dm_dir  /path/to/dm \\
        --output_dir   routing_analysis_complete \\
        --max_samples  50

Outputs (under --output_dir):
    basic/individual/          Per-image 4-panel figures
    basic/comparison/          Domain comparison grid
    basic/statistics/          Routing stats bar charts + histograms
    advanced/difference_maps/  GAN−DM difference grids
    advanced/realvsfake/       Real vs Fake routing panels
    advanced/paper_figures/    routing_grid.png  (insert into paper)
    advanced/specialization/   specialization_report.json + bar figure
    analysis_report.json       Unified metadata + summary metrics
"""

import os
import sys
import glob
import json
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import building blocks from the other scripts
from visualize_routing_maps import (
    load_model, _resolve_checkpoint, _collect_all,
    save_individual_routing_maps, save_domain_comparison,
    save_routing_statistics, EXPERT_NAMES,
)
from visualize_routing_advanced import (
    save_difference_maps_grid, save_realvsfake_comparison,
    save_paper_figure, save_specialization_report,
)
from routing_utils import RoutingStatisticsComputer


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_data_dir(label, data_dir):
    """Return (ok: bool, resolved_subdir: str)."""
    for sub in ('test', 'val'):
        subdir = os.path.join(data_dir, sub)
        if os.path.isdir(subdir):
            return True, subdir
    return False, ''


def validate_inputs(args):
    """Validate checkpoint and data dirs; print clear error messages."""
    ok = True

    # Checkpoint
    ckpt = _resolve_checkpoint(args)
    if ckpt is None or not os.path.isfile(ckpt):
        print(f"[ERROR] Checkpoint not found.")
        if args.checkpoint_epoch >= 0:
            expected = os.path.join(args.model_dir,
                                    f'models_params_{args.checkpoint_epoch}.tar')
            print(f"        Expected: {expected}")
        else:
            print(f"        Searched in: {args.model_dir}")
            print(f"        Set --checkpoint_epoch or --checkpoint_path.")
        ok = False
    else:
        print(f"[OK]  Checkpoint:  {ckpt}")

    # Data dirs (both are optional — at least one must be provided)
    n_valid_dirs = 0
    for label, data_dir in [('GAN', args.data_gan_dir),
                             ('DM',  args.data_dm_dir)]:
        if not data_dir:
            print(f"[SKIP] {label}: --data_{label.lower()}_dir not provided.")
            continue
        found, subdir = _check_data_dir(label, data_dir)
        if not found:
            print(f"[WARN] {label}: no test/ or val/ subfolder in {data_dir}")
            print(f"       Figures that require {label} domain will be skipped.")
        else:
            print(f"[OK]  {label} data: {subdir}")
            n_valid_dirs += 1

    if n_valid_dirs == 0:
        print("[ERROR] No valid data directories found. "
              "Provide at least one of --data_gan_dir or --data_dm_dir.")
        ok = False
    elif n_valid_dirs == 1:
        print("[INFO] Only one domain provided — multi-domain figures "
              "(comparison, paper grid, difference maps) will be skipped.")

    return ok


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    t0 = time.time()

    # Output directories
    basic_dir    = os.path.join(args.output_dir, 'basic')
    advanced_dir = os.path.join(args.output_dir, 'advanced')
    os.makedirs(basic_dir,    exist_ok=True)
    os.makedirs(advanced_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("DaRE-MoE ROUTING ANALYSIS PIPELINE")
    print(f"{'=' * 60}")

    # Validate
    print("\n--- Input Validation ---")
    ok = validate_inputs(args)
    if not ok:
        print("\nPipeline aborted due to validation errors.")
        return None

    # Device
    device = torch.device(f'cuda:{args.device}'
                          if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model (once, shared by both stages)
    ckpt_path = _resolve_checkpoint(args)
    print(f"\n--- Loading Model ---")
    model = load_model(ckpt_path, args.num_experts, args.feature_dim, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Collect routing data (one pass, shared by all figures)
    print("\n--- Collecting Routing Data ---")
    (all_images, all_routing_probs, all_routing_agg, all_domain_labels,
     gan_imgs, dm_imgs, real_imgs,
     gan_routing, dm_routing, real_routing) = _collect_all(model, device, args)

    if not all_images:
        print("No images collected. Check data directories.")
        return None

    n_total = len(all_images)
    d_arr   = np.array(all_domain_labels)
    r_arr   = np.array(all_routing_agg)

    print(f"Collected: {n_total} images  "
          f"(DM-fake: {(d_arr==0).sum()}  "
          f"GAN-fake: {(d_arr==1).sum()}  "
          f"Real: {(d_arr==args.num_experts).sum()})")

    # ------------------------------------------------------------------ BASIC
    print("\n--- Stage 1: Basic Visualizations ---")

    bin_labels = [0 if d == args.num_experts else 1 for d in all_domain_labels]

    print("[1/3] Individual routing maps (first 20)...")
    save_individual_routing_maps(
        all_images[:20], bin_labels[:20],
        all_routing_probs[:20], all_routing_agg[:20],
        os.path.join(basic_dir, 'individual'),
        args.num_experts,
    )

    print("[2/3] Domain comparison...")
    save_domain_comparison(
        gan_imgs, dm_imgs, real_imgs,
        gan_routing, dm_routing, real_routing,
        os.path.join(basic_dir, 'comparison'),
        args.num_experts,
    )

    print("[3/3] Routing statistics...")
    save_routing_statistics(
        all_routing_agg, all_domain_labels,
        os.path.join(basic_dir, 'statistics'),
        args.num_experts,
    )

    # --------------------------------------------------------------- ADVANCED
    print("\n--- Stage 2: Advanced Analysis ---")

    print("[1/4] Difference map grid...")
    save_difference_maps_grid(
        gan_imgs, dm_imgs, real_imgs,
        gan_routing, dm_routing, real_routing,
        os.path.join(advanced_dir, 'difference_maps'),
        args.num_experts,
    )

    print("[2/4] Real-vs-fake comparison...")
    save_realvsfake_comparison(
        gan_imgs, dm_imgs, real_imgs,
        gan_routing, dm_routing, real_routing,
        os.path.join(advanced_dir, 'realvsfake'),
        args.num_experts,
    )

    print("[3/4] Paper figure (routing_grid.png)...")
    save_paper_figure(
        gan_imgs, dm_imgs, real_imgs,
        gan_routing, dm_routing, real_routing,
        os.path.join(advanced_dir, 'paper_figures'),
        args.num_experts,
    )

    print("[4/4] Specialisation report...")
    spec_report = save_specialization_report(
        all_routing_agg, all_domain_labels,
        os.path.join(advanced_dir, 'specialization'),
        args.num_experts,
    )

    # ---------------------------------------------------------- UNIFIED REPORT
    elapsed = time.time() - t0
    overall_acc = RoutingStatisticsComputer.compute_routing_accuracy(
        r_arr, d_arr, args.num_experts)

    report = {
        'checkpoint':        ckpt_path,
        'data_gan_dir':      args.data_gan_dir,
        'data_dm_dir':       args.data_dm_dir,
        'num_experts':       args.num_experts,
        'feature_dim':       args.feature_dim,
        'n_total_images':    int(n_total),
        'n_dm_fake':         int((d_arr == 0).sum()),
        'n_gan_fake':        int((d_arr == 1).sum()),
        'n_real':            int((d_arr == args.num_experts).sum()),
        'overall_routing_accuracy': float(overall_acc),
        'per_expert':        spec_report['per_expert'],
        'entropy_by_domain': spec_report['entropy_by_domain'],
        'elapsed_seconds':   round(elapsed, 1),
        'output_files': {
            'individual_maps':      os.path.join(basic_dir, 'individual'),
            'domain_comparison':    os.path.join(basic_dir, 'comparison',
                                                 'domain_comparison.png'),
            'routing_stats':        os.path.join(basic_dir, 'statistics',
                                                 'routing_stats.png'),
            'difference_maps':      os.path.join(advanced_dir, 'difference_maps',
                                                 'difference_maps_grid.png'),
            'realvsfake':           os.path.join(advanced_dir, 'realvsfake',
                                                 'realvsfake_comparison.png'),
            'paper_figure':         os.path.join(advanced_dir, 'paper_figures',
                                                 'routing_grid.png'),
            'specialization_json':  os.path.join(advanced_dir, 'specialization',
                                                 'specialization_report.json'),
        },
    }

    report_path = os.path.join(args.output_dir, 'analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    return report


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(report, output_dir):
    exp_names = EXPERT_NAMES

    print(f"\n{'=' * 60}")
    print("ROUTING ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Images analysed  : {report['n_total_images']}")
    print(f"    DM fakes       : {report['n_dm_fake']}")
    print(f"    GAN fakes      : {report['n_gan_fake']}")
    print(f"    Real           : {report['n_real']}")
    print(f"  Overall routing acc: {report['overall_routing_accuracy']:.2%}")
    for name, s in report['per_expert'].items():
        print(f"  {name:12s}: acc={s['routing_accuracy']:.2%}  "
              f"mean_r_domain={s['mean_routing_on_domain']:.3f}  "
              f"mean_r_real={s['mean_routing_on_real']:.3f}")
    print(f"  Elapsed          : {report['elapsed_seconds']:.1f}s")
    print(f"\n  Key outputs:")
    print(f"    Paper figure    : {report['output_files']['paper_figure']}")
    print(f"    Statistics      : {report['output_files']['routing_stats']}")
    print(f"    JSON report     : {os.path.join(output_dir, 'analysis_report.json')}")
    print(f"{'=' * 60}\n")

    # Guidance on what to put in the paper
    acc = report['overall_routing_accuracy']
    print("  Paper integration:")
    print(f"    Routing accuracy = {acc:.1%}  "
          f"({'strong' if acc > 0.75 else 'moderate'} specialisation)")
    print("    Use routing_grid.png in your Results / Ablation section.")
    print("    Cite routing_stats.png as evidence for DARS effectiveness.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='DaRE-MoE master routing analysis pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint
    parser.add_argument('--checkpoint_epoch', '-e', type=int, default=-1,
                        help='Epoch index (loads models_params_{e}.tar)')
    parser.add_argument('--checkpoint_path', '-cp', type=str, default=None,
                        help='Direct checkpoint path (overrides --checkpoint_epoch)')
    parser.add_argument('--model_dir', type=str,
                        default='checkpoints/moe_finetune',
                        help='Directory containing checkpoint files')

    # Data
    parser.add_argument('--data_gan_dir', type=str, default=None,
                        help='GAN domain data root (must have test/ or val/). '
                             'Omit to skip GAN domain.')
    parser.add_argument('--data_dm_dir', type=str, default=None,
                        help='DM domain data root (must have test/ or val/). '
                             'Omit to skip DM domain.')

    # Output
    parser.add_argument('--output_dir', '-o', type=str,
                        default='routing_analysis_complete',
                        help='Root directory for all outputs')

    # Model
    parser.add_argument('--num_experts',  type=int, default=2)
    parser.add_argument('--feature_dim',  type=int, default=768)

    # Run config
    parser.add_argument('--max_samples',  type=int, default=50,
                        help='Max samples per domain (trade-off speed vs coverage)')
    parser.add_argument('--batch_size',   type=int, default=8)
    parser.add_argument('--num_workers',  type=int, default=2)
    parser.add_argument('--device',       type=int, default=0,
                        help='CUDA device ID (0-indexed)')

    args = parser.parse_args()

    report = run_pipeline(args)
    if report is not None:
        print_summary(report, args.output_dir)


if __name__ == '__main__':
    main()
