"""
Forensic-MoE: DaRE-MoE with Expert Knowledge Distillation (EKD)

Domain-aware Routing Enhanced Mixture-of-Experts (DaRE-MoE) with:
  - Domain-Aware Router producing dense spatial routing maps
  - Expert Knowledge Distillation (EKD) for feature alignment
  - Domain-Aware Routing Supervision (DARS) loss for routing specialisation
  - 3-stage training schedule: warmup → ramp-up → stabilisation

Uses MTS-MoE (Multi-Transform Spectral MoE) experts with shared routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DomainAwareRouter(nn.Module):
    """
    Domain-Aware Router producing dense spatial routing maps.

    Takes raw images and outputs per-expert spatial routing probabilities
    R(x) ∈ R^{H'×W'×num_experts}, then aggregates to image-level routing
    scores r̄_i(x) by spatial averaging.
    """

    def __init__(self, num_experts=2):
        super(DomainAwareRouter, self).__init__()
        self.num_experts = num_experts

        # Lightweight CNN: 224 → 56 → 28 → 14
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_experts, kernel_size=1),  # 14×14×num_experts
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image
        Returns:
            routing_probs: (B, num_experts, H', W') softmax routing maps
            routing_agg:   (B, num_experts) image-level aggregated routing
        """
        logits = self.conv_layers(x)                  # (B, num_experts, H', W')
        routing_probs = F.softmax(logits, dim=1)      # softmax over expert dim
        routing_agg = routing_probs.mean(dim=[2, 3])  # (B, num_experts)
        return routing_probs, routing_agg


class ForensicMoE(nn.Module):
    """
    Forensic MoE Architecture for Finetuning Stage
    
    Integrates N specialized experts with backbone for deep interaction
    Implements Expert Knowledge Distillation (EKD) paradigm
    """
    def __init__(self, backbone, expert_models, num_experts=2, feature_dim=768, 
                 freeze_backbone=True, lambda_ekd=1.0, margin=0.7):
        """
        Args:
            backbone: Shared backbone model (e.g., ViT encoder)
            expert_models: List of N pre-trained expert models (state_dicts)
            num_experts: Number of experts (N)
            feature_dim: Dimension of expert output features
            freeze_backbone: Whether to freeze backbone during finetuning
            lambda_ekd: Trade-off hyperparameter for EKD loss
            margin: Margin for EKD loss (default 0.7)
        """
        super(ForensicMoE, self).__init__()
        
        self.num_experts = num_experts
        self.feature_dim = feature_dim
        self.lambda_ekd = lambda_ekd
        self.margin = margin
        
        # Shared backbone
        self.backbone = backbone
        if backbone is not None and freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Initialize N experts from pre-trained models
        self.experts = nn.ModuleList()
        for expert_state in expert_models:
            expert = self._create_expert_from_state(expert_state)
            self.experts.append(expert)
        
        # Domain-Aware Router
        self.router = DomainAwareRouter(num_experts=num_experts)
        
        # Shared classification head (applied per-expert, fused via routing)
        self.classifier = nn.Linear(feature_dim, 1)
        
        # Share routing parameters across experts
        self._share_routing()
        
    def _create_expert_from_state(self, state_dict):
        """
        Create expert model from state dict
        Uses the VisionTransformer model from MTS_MoE.py (Multi-Transform Spectral MoE)
        """
        import sys
        import os
        # Add parent directory to path to import MTS_MoE
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from MTS_MoE import vit_base_patch16_224_in21k
        
        # Create expert model with same architecture as training
        expert = vit_base_patch16_224_in21k(
            pretrained=False,
            num_classes=2,
            lora_topk=1,
            adapter_topk=1
        )
        
        # Load weights from checkpoint
        expert.load_state_dict(state_dict, strict=False)
        
        # Freeze expert parameters (they shouldn't be updated during finetuning)
        for param in expert.parameters():
            param.requires_grad = False
        
        return expert
    
    def _share_routing(self):
        """
        Share the routing (gating) parameters across all experts.
        
        Both the LoRA-MoE (in attention) and MTS-MoE (adapter) layers use
        noisy top-k gating with w_gate and w_noise parameters. This method
        ties those parameters so all experts use identical routing decisions.
        """
        if self.num_experts < 2:
            return
        
        ref_expert = self.experts[0]
        for expert in self.experts[1:]:
            for ref_block, other_block in zip(ref_expert.blocks, expert.blocks):
                # Share LoRA-MoE routing in attention
                if hasattr(ref_block.attn, 'LoRA_MoE') and hasattr(other_block.attn, 'LoRA_MoE'):
                    other_block.attn.LoRA_MoE.w_gate = ref_block.attn.LoRA_MoE.w_gate
                    other_block.attn.LoRA_MoE.w_noise = ref_block.attn.LoRA_MoE.w_noise
                
                # Share MTS-MoE (adapter) routing
                if hasattr(ref_block, 'adapter_MoE') and hasattr(other_block, 'adapter_MoE'):
                    other_block.adapter_MoE.w_gate = ref_block.adapter_MoE.w_gate
                    other_block.adapter_MoE.w_noise = ref_block.adapter_MoE.w_noise

    def forward_single_expert(self, x, expert_idx):
        """
        Forward pass through single expert (Equation 2)
        
        Args:
            x: Input image (batch_size, C, H, W)
            expert_idx: Index of the expert to use
        
        Returns:
            feature: Expert trace feature f_i^m (batch_size, feature_dim)
        """
        # VisionTransformer from MTS_MoE returns (features, moe_loss)
        # We use forward_features to get the CLS token features
        expert_feature, _ = self.experts[expert_idx].forward_features(x)
        
        return expert_feature
    
    def forward(self, x, return_all_features=False):
        """
        Forward pass with Domain-Aware Router prediction fusion.

        1. Each expert produces CLS features  f_i
        2. Per-expert logits  p_i = classifier(f_i)
        3. Router produces spatial routing → image-level r̄_i
        4. Fused prediction  p̂ = Σ r̄_i · p_i

        Args:
            x: Input image (batch_size, C, H, W)
            return_all_features: Whether to return routing & feature details

        Returns:
            output: Fused binary classification logits (batch_size, 1)
        """
        # --- expert features & per-expert logits ---
        expert_features = []
        expert_logits = []
        for i in range(self.num_experts):
            f_i = self.forward_single_expert(x, i)
            p_i = self.classifier(f_i)  # (B, 1)
            expert_features.append(f_i)
            expert_logits.append(p_i)

        # --- Domain-Aware Router ---
        routing_probs, routing_agg = self.router(x)  # (B, N, H', W'), (B, N)

        # --- fuse expert predictions via routing weights ---
        stacked = torch.stack(expert_logits, dim=1)           # (B, N, 1)
        output = (routing_agg.unsqueeze(-1) * stacked).sum(dim=1)  # (B, 1)

        if return_all_features:
            return output, {
                'expert_features': expert_features,
                'expert_logits': expert_logits,
                'routing_probs': routing_probs,
                'routing_agg': routing_agg,
            }

        return output
    
    def compute_bce_loss(self, output, labels):
        """
        Binary Cross-Entropy Loss (Equation 4)
        
        Args:
            output: Model output logits (batch_size, 1)
            labels: Ground truth labels (batch_size,)
        
        Returns:
            loss: BCE loss scalar
        """
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output.squeeze(), labels.float())
        return loss
    
    def compute_ekd_loss(self, expert_features, labels, expert_type):
        """
        Expert Knowledge Distillation Loss (Equation 5, 6, 7)
        
        For real images: Pull in distance between anchor and other features (homogeneity)
        For fake images: Push away distance between anchor and other features (diversity)
        
        Args:
            expert_features: List of N feature tensors [f_1^m, ..., f_N^m]
            labels: Ground truth labels (batch_size,) - 0 for real, 1 for fake
            expert_type: Index m of the corresponding expert
        
        Returns:
            ekd_loss: EKD loss scalar
        """
        batch_size = labels.size(0)
        
        # Anchor feature: f_m^m (from corresponding expert)
        anchor_feature = expert_features[expert_type]  # (batch_size, feature_dim)
        
        # Separate real and fake samples
        real_mask = (labels == 0)
        fake_mask = (labels == 1)
        
        B_r = real_mask.sum().item()
        B_f = fake_mask.sum().item()
        
        if B_r == 0 and B_f == 0:
            return torch.tensor(0.0).to(labels.device)
        
        total_loss = 0.0
        num_terms = 0
        
        # Iterate over other experts (i ≠ m)
        for i in range(self.num_experts):
            if i == expert_type:
                continue
            
            other_feature = expert_features[i]
            
            # Compute cosine similarity for real samples (S_i^r)
            if B_r > 0:
                anchor_real = anchor_feature[real_mask]  # (B_r, feature_dim)
                other_real = other_feature[real_mask]
                
                # Average cosine similarity over real samples (Equation 6)
                sim_real = F.cosine_similarity(anchor_real, other_real, dim=1)
                S_i_r = sim_real.mean()
            else:
                S_i_r = torch.tensor(0.0).to(labels.device)
            
            # Compute cosine similarity for fake samples (S_i^f)
            if B_f > 0:
                anchor_fake = anchor_feature[fake_mask]  # (B_f, feature_dim)
                other_fake = other_feature[fake_mask]
                
                # Average cosine similarity over fake samples (Equation 7)
                sim_fake = F.cosine_similarity(anchor_fake, other_fake, dim=1)
                S_i_f = sim_fake.mean()
            else:
                S_i_f = torch.tensor(0.0).to(labels.device)
            
            # EKD loss term (Equation 5)
            # Pull in real features (maximize similarity -> minimize -S_i^r)
            # Push away fake features (minimize similarity with margin)
            loss_term = -S_i_r + torch.clamp(S_i_f - S_i_r + self.margin, min=0.0)
            
            total_loss += loss_term
            num_terms += 1
        
        # Average over N-1 experts
        ekd_loss = total_loss / max(num_terms, 1)
        
        return ekd_loss
    
    def compute_dars_loss(self, routing_agg, domain_labels):
        """
        Domain-Aware Routing Supervision (DARS) loss.

        Cross-entropy between image-level aggregated routing probabilities
        and domain-dependent target distributions.

        Domain label convention:
            0 .. num_experts-1  →  fake from that expert's domain
            num_experts         →  REAL image

        Target routing distributions:
            fake from domain i : one-hot on expert i
            real               : uniform  [1/N, ..., 1/N]
        """
        targets = torch.zeros_like(routing_agg)  # (B, N)

        for i in range(self.num_experts):
            mask = (domain_labels == i)
            targets[mask, i] = 1.0

        real_mask = (domain_labels == self.num_experts)
        targets[real_mask] = 1.0 / self.num_experts

        eps = 1e-7
        log_routing = torch.log(routing_agg + eps)
        loss = -(targets * log_routing).sum(dim=1).mean()
        return loss

    def compute_total_loss(self, x, labels, expert_type,
                           domain_labels=None, beta=0.0):
        """
        Total DaRE-MoE loss:  L = L_BCE + α·L_EKD + β·L_DARS

        Args:
            x: Input images (batch_size, C, H, W)
            labels: Ground truth binary labels (batch_size,)
            expert_type: Index m of the corresponding expert
            domain_labels: Domain labels (batch_size,) for DARS
                           (None to skip DARS)
            beta: Current DARS weight (follows 3-stage schedule)

        Returns:
            total_loss, loss_dict
        """
        output, features_dict = self.forward(x, return_all_features=True)
        expert_features = features_dict['expert_features']
        routing_agg = features_dict['routing_agg']

        bce_loss = self.compute_bce_loss(output, labels)
        ekd_loss = self.compute_ekd_loss(expert_features, labels, expert_type)

        dars_loss = torch.tensor(0.0, device=x.device)
        if domain_labels is not None and beta > 0:
            dars_loss = self.compute_dars_loss(routing_agg, domain_labels)

        total_loss = bce_loss + self.lambda_ekd * ekd_loss + beta * dars_loss

        loss_dict = {
            'total': total_loss,
            'bce': bce_loss,
            'ekd': ekd_loss,
            'dars': dars_loss,
        }

        return total_loss, loss_dict
    
    def freeze_expert(self, expert_idx):
        """Freeze the corresponding expert during training"""
        for param in self.experts[expert_idx].parameters():
            param.requires_grad = False
    
    def unfreeze_experts_except(self, expert_idx):
        """Unfreeze all experts except the corresponding one"""
        for i in range(self.num_experts):
            if i != expert_idx:
                for param in self.experts[i].parameters():
                    param.requires_grad = True
            else:
                self.freeze_expert(i)


def load_expert_models(expert_paths):
    """
    Load expert models from pickle files
    
    Args:
        expert_paths: List of paths to expert .pkl files
    
    Returns:
        expert_states: List of state_dicts
    """
    expert_states = []
    for path in expert_paths:
        state_dict = torch.load(path, map_location='cpu')
        expert_states.append(state_dict)
    return expert_states


if __name__ == '__main__':
    print("DaRE-MoE model architecture loaded successfully!")
    print("\nKey components:")
    print("1. DomainAwareRouter - spatial routing maps + DARS loss")
    print("2. ForensicMoE - DaRE-MoE architecture")
    print("3. EKD Loss (feature alignment)")
    print("4. DARS Loss (routing supervision)")
    print("5. Total Loss: L = BCE + α·EKD + β·DARS")
