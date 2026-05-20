#!/usr/bin/env python3
"""
Inspect MTS_MoE model architecture, layers, and trainable parameters.

Usage:
    python inspect_mts_moe.py                    # inspect architecture only
    python inspect_mts_moe.py --ckpt path.pkl    # also load checkpoint weights
    python inspect_mts_moe.py --verbose          # show every single parameter
"""

import sys
import argparse
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Load the model (exec-based to avoid timm import conflicts)
# ---------------------------------------------------------------------------

def load_mts_moe_class():
    import re
    code = open('MTS_MoE.py').read()
    code = code.replace('@register_model\n', '')
    code = re.sub(r"if __name__ == '__main__':.*", '', code, flags=re.DOTALL)
    ns = {}
    exec(compile(code, 'MTS_MoE.py', 'exec'), ns)
    return ns


def build_model(ns, num_classes=2):
    # Instantiate VisionTransformer directly to bypass build_model_with_cfg
    # which passes 'default_cfg' kwarg that the model's __init__ doesn't accept.
    VisionTransformer = ns['VisionTransformer']
    model = VisionTransformer(
        img_size=224, patch_size=16, in_chans=3,
        num_classes=num_classes, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=True,
        lora_topk=1, adapter_topk=1,
    )
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def human(n):
    """Format a parameter count as a human-readable string."""
    if n >= 1_000_000:
        return f"{n/1e6:.3f}M"
    if n >= 1_000:
        return f"{n/1e3:.1f}K"
    return str(n)


def param_counts(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def tag(requires_grad):
    return "[TRAIN]" if requires_grad else "[frozen]"


# ---------------------------------------------------------------------------
# Section 1 — High-level summary
# ---------------------------------------------------------------------------

def print_summary(model):
    total, trainable = param_counts(model)
    frozen = total - trainable

    print("=" * 72)
    print("  MTS-MoE MODEL SUMMARY")
    print("=" * 72)
    print(f"  {'Total parameters':<30} {human(total):>10}  ({total:,})")
    print(f"  {'Trainable parameters':<30} {human(trainable):>10}  ({trainable:,})")
    print(f"  {'Frozen parameters':<30} {human(frozen):>10}  ({frozen:,})")
    print(f"  {'Trainable ratio':<30} {100*trainable/total:>9.2f}%")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Section 2 — Per-component breakdown
# ---------------------------------------------------------------------------

COMPONENT_GROUPS = {
    "Classification head": lambda n: 'head' in n,
    "LayerNorm1 (trainable)": lambda n: 'norm1' in n and 'norm2' not in n,
    "AuroRA LoRA-MoE (Attention)": lambda n: 'LoRA' in n,
    "MTS-MoE adapters (Parallel-MLP)": lambda n: 'adapter' in n,
    "Frozen backbone": lambda n: True,   # catch-all — must be last
}


def print_component_breakdown(model):
    print()
    print("─" * 72)
    print("  COMPONENT BREAKDOWN")
    print("─" * 72)
    fmt = "  {:<38} {:>10}  {:>10}"
    print(fmt.format("Component", "Params", "Trainable"))
    print("  " + "─" * 68)

    accounted = set()

    for label, pred in COMPONENT_GROUPS.items():
        params = [(n, p) for n, p in model.named_parameters()
                  if pred(n) and n not in accounted]
        if not params:
            continue
        for n, _ in params:
            accounted.add(n)

        total = sum(p.numel() for _, p in params)
        train = sum(p.numel() for _, p in params if p.requires_grad)
        print(fmt.format(label, human(total), human(train)))

    print("─" * 72)


# ---------------------------------------------------------------------------
# Section 3 — Per-block summary (12 transformer blocks)
# ---------------------------------------------------------------------------

def print_block_table(model):
    print()
    print("─" * 72)
    print("  PER-BLOCK PARAMETER TABLE  (12 Transformer Blocks)")
    print("─" * 72)
    fmt = "  {:<6} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("Block", "Total", "LoRA-MoE", "MTS-MoE", "norm1", "mlp+attn"))
    print("  " + "─" * 68)

    for i, block in enumerate(model.blocks):
        t, _ = param_counts(block)

        lora, _ = param_counts(block.attn.LoRA_MoE) if hasattr(block.attn, 'LoRA_MoE') else (0, 0)
        mts, _  = param_counts(block.adapter_MoE)   if hasattr(block, 'adapter_MoE') else (0, 0)
        n1, _   = param_counts(block.norm1)
        rest    = t - lora - mts - n1

        print(fmt.format(f"[{i:02d}]",
                         human(t), human(lora), human(mts), human(n1), human(rest)))

    print("─" * 72)


# ---------------------------------------------------------------------------
# Section 4 — MTS-MoE expert detail (one block, representative)
# ---------------------------------------------------------------------------

def print_expert_detail(model):
    block = list(model.blocks)[0]
    if not hasattr(block, 'adapter_MoE'):
        print("  (no adapter_MoE found in block 0)")
        return

    mts = block.adapter_MoE
    print()
    print("─" * 72)
    print("  MTS-MoE EXPERT DETAIL  (Block 0, representative)")
    print("─" * 72)

    # Gating
    g_total = mts.w_gate.numel() + mts.w_noise.numel()
    print(f"  Gating   W_gate  {tuple(mts.w_gate.shape)}  →  {human(mts.w_gate.numel())}")
    print(f"           W_noise {tuple(mts.w_noise.shape)} →  {human(mts.w_noise.numel())}")
    print(f"           Gating total: {human(g_total)}")
    print()

    # Experts
    fmt = "  {:<4} {:<12} {:>8} {:>8} {:>8} {:>8}"
    print(fmt.format("Exp", "Transform", "down", "core", "up", "total"))
    print("  " + "─" * 60)

    for i, exp in enumerate(mts.adapter_experts):
        transform = exp.transform_type if hasattr(exp, 'transform_type') else f"expert_{i}"
        t, _ = param_counts(exp)
        d = exp.adapter_down.weight.numel() + exp.adapter_down.bias.numel()
        u = exp.adapter_up.weight.numel()   + exp.adapter_up.bias.numel()
        core = t - d - u
        print(fmt.format(f"[{i}]", transform, human(d), human(core), human(u), human(t)))

    print("─" * 72)


# ---------------------------------------------------------------------------
# Section 5 — AuroRA LoRA-MoE expert detail
# ---------------------------------------------------------------------------

def print_lora_detail(model):
    block = list(model.blocks)[0]
    if not hasattr(block.attn, 'LoRA_MoE'):
        print("  (no LoRA_MoE found)")
        return

    lora = block.attn.LoRA_MoE
    print()
    print("─" * 72)
    print("  AuroRA LoRA-MoE EXPERT DETAIL  (Block 0, representative)")
    print("─" * 72)

    fmt = "  {:<4} {:>6} {:>10} {:>10} {:>10} {:>12}"
    print(fmt.format("Exp", "Rank", "A (768→r)", "ANL (r→r)", "B (r→2304)", "Total"))
    print("  " + "─" * 60)

    for i in range(lora.num_experts):
        a   = lora.Lora_a_experts[i]
        b   = lora.Lora_b_experts[i]
        anl = lora.Lora_ab_experts[i] if lora.Lora_ab_experts else None

        rank     = a.weight.shape[0]
        a_cnt    = a.weight.numel()
        b_cnt    = b.weight.numel()
        anl_cnt  = sum(p.numel() for p in anl.parameters()) if anl else 0
        total    = a_cnt + b_cnt + anl_cnt

        print(fmt.format(f"[{i}]", rank,
                         human(a_cnt), human(anl_cnt) if anl_cnt else "-",
                         human(b_cnt), human(total)))

    g = human(lora.w_gate.numel() + lora.w_noise.numel())
    print(f"\n  Gating (W_gate + W_noise): {g}")
    print("─" * 72)


# ---------------------------------------------------------------------------
# Section 6 — Full named-parameter list (optional)
# ---------------------------------------------------------------------------

def print_all_parameters(model):
    print()
    print("─" * 72)
    print("  ALL NAMED PARAMETERS")
    print("─" * 72)
    fmt = "  {:<7} {:<55} {:>10}  {}"
    print(fmt.format("Status", "Name", "Shape", "Numel"))
    print("  " + "─" * 68)

    for name, param in model.named_parameters():
        shape_str = "×".join(str(d) for d in param.shape)
        print(fmt.format(tag(param.requires_grad), name[:54],
                         shape_str, human(param.numel())))

    print("─" * 72)


# ---------------------------------------------------------------------------
# Section 7 — Module tree (torch.nn modules, abbreviated)
# ---------------------------------------------------------------------------

def print_module_tree(model, max_depth=4):
    print()
    print("─" * 72)
    print("  MODULE TREE  (depth ≤ {})".format(max_depth))
    print("─" * 72)

    def _walk(module, prefix, depth):
        children = list(module.named_children())
        total, train = param_counts(module)

        if not children or depth >= max_depth:
            cls = module.__class__.__name__
            extra = ""
            # Show weight shape for Linear/Conv layers
            if hasattr(module, 'weight') and module.weight is not None:
                extra = f"  w={tuple(module.weight.shape)}"
            marker = "✓" if train > 0 else " "
            print(f"  {prefix}{marker} {cls}{extra}  [{human(total)}]")
            return

        cls = module.__class__.__name__
        marker = "✓" if train > 0 else " "
        print(f"  {prefix}{marker} {cls}  [{human(total)} total | {human(train)} trainable]")

        for i, (name, child) in enumerate(children):
            connector = "└─ " if i == len(children) - 1 else "├─ "
            indent    = "   " if i == len(children) - 1 else "│  "
            print(f"  {prefix}{connector}{name}")
            _walk(child, prefix + indent, depth + 1)

    _walk(model, "", 0)
    print("─" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect MTS-MoE model")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a .pkl checkpoint to load (optional)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every named parameter (long output)")
    parser.add_argument("--depth", type=int, default=4,
                        help="Max depth for module tree (default: 4)")
    parser.add_argument("--num-classes", type=int, default=2)
    args = parser.parse_args()

    print("\nLoading MTS_MoE class definitions ...")
    ns = load_mts_moe_class()
    print("Building model (pretrained=False) ...")
    model = build_model(ns, num_classes=args.num_classes)

    # Optionally load checkpoint
    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        try:
            sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
            if isinstance(sd, dict) and 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"  Missing keys : {len(missing)}")
            print(f"  Unexpected   : {len(unexpected)}")
        except Exception as e:
            print(f"  WARNING: could not load checkpoint — {e}")

    # ---- Print sections ----
    print_summary(model)
    print_component_breakdown(model)
    print_block_table(model)
    print_expert_detail(model)
    print_lora_detail(model)
    print_module_tree(model, max_depth=args.depth)

    if args.verbose:
        print_all_parameters(model)

    print()


if __name__ == '__main__':
    main()
