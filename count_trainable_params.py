"""
count_trainable_params.py

Counts and reports trainable parameters for the DaRE-MoE finetuning stage
(train_moe_finetune.py) using the actual expert checkpoints.

Usage:
    python count_trainable_params.py \
        --expert_dm_path  mts_moe_celeba_dm/model_params_best_1.0000acc_1.0000auc_epoch001.pkl \
        --expert_gan_path mts_moe_celeba_progan/model_params_best_0.9995acc_1.0000auc_epoch001.pkl

Defaults point to the checkpoints already present in this repo.
"""

import sys, os, warnings, argparse
warnings.filterwarnings("ignore")

# ── Compatibility patch for torch 2.1 + old transformers/timm ────────────────
import torch
if not hasattr(torch.utils._pytree, "register_pytree_node"):
    torch.utils._pytree.register_pytree_node = lambda *a, **kw: None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.moe_forensic import ForensicMoE, load_expert_models


# ─────────────────────────────────────────────────────────────────────────────
def _fmt(n):
    """Format an integer with commas and an M/K suffix for readability."""
    if n >= 1_000_000:
        return f"{n:>12,}  ({n/1e6:.2f} M)"
    if n >= 1_000:
        return f"{n:>12,}  ({n/1e3:.1f} K)"
    return f"{n:>12,}"


def count_params(module, only_trainable=False):
    return sum(p.numel() for p in module.parameters()
               if (p.requires_grad if only_trainable else True))


def expert_trainable_subset(expert):
    """
    Return trainable param count for one expert as loaded by ForensicMoE
    (i.e. after freeze_stages() inside VisionTransformer.__init__).
    All expert params are initially requires_grad=False inside ForensicMoE,
    but freeze_stages() inside the ViT itself marks LoRA / adapter / head /
    norm1 as requires_grad=True before ForensicMoE freezes everything again.
    We need the count that becomes active once unfreeze_experts_except() re-
    enables them.
    """
    # Temporarily re-enable all expert params to see the true frozen/unfrozen split
    states = {}
    for n, p in expert.named_parameters():
        states[n] = p.requires_grad

    # The ViT's freeze_stages() marks these as trainable; ForensicMoE then
    # forces them all to False.  Simulate what unfreeze_experts_except() does:
    # it calls `param.requires_grad = True` for every param of the expert.
    # Then the REAL trainable count is whatever freeze_stages() left as True.
    # We reproduce that by looking at name patterns (same logic as freeze_stages).
    trainable = sum(
        p.numel() for n, p in expert.named_parameters()
        if any(k in n for k in ("LoRA", "adapter", "head", "norm1"))
    )
    frozen = sum(
        p.numel() for n, p in expert.named_parameters()
        if not any(k in n for k in ("LoRA", "adapter", "head", "norm1"))
    )
    return trainable, frozen


def main():
    parser = argparse.ArgumentParser(description="Count DaRE-MoE finetuning trainable parameters")
    parser.add_argument(
        "--expert_dm_path", type=str,
        default="mts_moe_celeba_dm/model_params_best_1.0000acc_1.0000auc_epoch001.pkl",
    )
    parser.add_argument(
        "--expert_gan_path", type=str,
        default="mts_moe_celeba_progan/model_params_best_0.9995acc_1.0000auc_epoch001.pkl",
    )
    parser.add_argument("--num_experts",  type=int,   default=2)
    parser.add_argument("--feature_dim",  type=int,   default=768)
    parser.add_argument("--lambda_ekd",   type=float, default=0.5)
    parser.add_argument("--margin",       type=float, default=0.7)
    args = parser.parse_args()

    # ── Load expert checkpoints ───────────────────────────────────────────────
    print("\nLoading expert checkpoints …")
    expert_paths  = [args.expert_dm_path, args.expert_gan_path]
    expert_states = load_expert_models(expert_paths)
    print(f"  Loaded {len(expert_states)} expert(s)\n")

    # ── Build ForensicMoE (same call as train_moe_finetune.py) ───────────────
    model = ForensicMoE(
        backbone=None,
        expert_models=expert_states,
        num_experts=args.num_experts,
        feature_dim=args.feature_dim,
        freeze_backbone=True,
        lambda_ekd=args.lambda_ekd,
        margin=args.margin,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.  New components introduced by finetuning
    # ─────────────────────────────────────────────────────────────────────────
    router_total  = count_params(model.router)
    cls_total     = count_params(model.classifier)
    new_comp_total = router_total + cls_total

    # ─────────────────────────────────────────────────────────────────────────
    # 2.  Per-expert breakdown (using name-pattern matching from freeze_stages)
    # ─────────────────────────────────────────────────────────────────────────
    expert_trainable_list = []
    expert_frozen_list    = []
    for i, expert in enumerate(model.experts):
        t, f = expert_trainable_subset(expert)
        expert_trainable_list.append(t)
        expert_frozen_list.append(f)

    # All experts should be identical in architecture
    per_expert_trainable = expert_trainable_list[0]
    per_expert_frozen    = expert_frozen_list[0]
    per_expert_total     = per_expert_trainable + per_expert_frozen

    # Per-expert adapter breakdown
    def _sub(expert, keys):
        return sum(p.numel() for n, p in expert.named_parameters() if any(k in n for k in keys))

    lora_params    = _sub(model.experts[0], ("LoRA",))
    adapter_params = _sub(model.experts[0], ("adapter",))
    head_params    = _sub(model.experts[0], ("head",))
    norm1_params   = _sub(model.experts[0], ("norm1",))

    # ─────────────────────────────────────────────────────────────────────────
    # 3.  Whole-model totals
    # ─────────────────────────────────────────────────────────────────────────
    total_model     = count_params(model)
    # At init all experts are frozen by ForensicMoE → only router + classifier
    trainable_init  = count_params(model, only_trainable=True)

    # During a training step: unfreeze_experts_except(m) unfreezes N-1 experts
    # Each unfrozen expert contributes per_expert_trainable params
    n_unfrozen = args.num_experts - 1          # = 1 for the default 2-expert setup
    trainable_per_step = new_comp_total + n_unfrozen * per_expert_trainable

    # ─────────────────────────────────────────────────────────────────────────
    # Print report
    # ─────────────────────────────────────────────────────────────────────────
    W = 62
    print("=" * W)
    print("  DaRE-MoE Finetuning — Trainable Parameter Report")
    print("=" * W)

    print("\n[ New finetuning components  (always trainable) ]")
    print(f"  DomainAwareRouter (lightweight CNN)  {_fmt(router_total)}")
    # Per-layer router detail
    for name, m in model.router.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            n = count_params(m)
            print(f"    ├─ {name:<30s}  {_fmt(n)}")
    print(f"  Classifier  Linear(768 → 1)          {_fmt(cls_total)}")
    print(f"  ─── Subtotal (new components)        {_fmt(new_comp_total)}")

    print(f"\n[ Per-expert breakdown  (ViT-B/16 + MTS-MoE)  ×{args.num_experts} ]")
    print(f"  Total params per expert              {_fmt(per_expert_total)}")
    print(f"  Frozen backbone per expert           {_fmt(per_expert_frozen)}")
    print(f"  Trainable per expert (after unfreeze){_fmt(per_expert_trainable)}")
    print(f"    ├─ AuroRA LoRA-MoE                 {_fmt(lora_params)}")
    print(f"    ├─ MTS-MoE adapters                {_fmt(adapter_params)}")
    print(f"    ├─ Classification head             {_fmt(head_params)}")
    print(f"    └─ LayerNorm1 (norm1)              {_fmt(norm1_params)}")

    print(f"\n[ Whole-model totals ]")
    print(f"  Total params (all components)        {_fmt(total_model)}")
    print(f"  Permanently frozen backbone          {_fmt(args.num_experts * per_expert_frozen)}")
    print(f"  Trainable at init (router + cls)     {_fmt(trainable_init)}")

    print(f"\n[ Trainable params per training step ]")
    print(f"  Strategy: N−1 = {n_unfrozen} expert(s) unfrozen, 1 frozen per batch")
    print(f"  Router + Classifier                  {_fmt(new_comp_total)}")
    print(f"  Unfrozen expert(s) ×{n_unfrozen}               {_fmt(n_unfrozen * per_expert_trainable)}")
    print(f"  ══ TRAINABLE PER STEP                {_fmt(trainable_per_step)}")

    pct = 100 * trainable_per_step / total_model
    print(f"     ({pct:.2f}% of total model params)")

    print("=" * W)


if __name__ == "__main__":
    main()
