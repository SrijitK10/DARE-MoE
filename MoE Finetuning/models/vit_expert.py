"""
ViT-Based Expert Model Architecture
Wrapper for loading and using pre-trained ViT-LoRA-MoE expert models
"""

import torch
import torch.nn as nn
import timm
from collections import OrderedDict


class LoRAMoEExpert(nn.Module):
    """
    Single LoRA-MoE Expert Model (ViT-based)
    This wraps your pre-trained expert models
    """
    def __init__(self, model_name='vit_base_patch16_224', num_classes=1, 
                 pretrained=False, feature_dim=768):
        super(LoRAMoEExpert, self).__init__()
        
        # Load ViT backbone using timm
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        self.feature_dim = feature_dim
        
        # Feature projection head (if needed)
        self.feature_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input images (batch_size, 3, 224, 224)
        
        Returns:
            features: (batch_size, feature_dim)
        """
        # Extract features from ViT
        features = self.vit(x)  # (batch, feature_dim)
        
        # Project features
        features = self.feature_head(features)
        
        return features
    
    def load_pretrained_weights(self, state_dict):
        """
        Load pre-trained weights from your .pkl file
        
        Args:
            state_dict: OrderedDict from torch.load(.pkl)
        """
        # Filter state dict to match model parameters
        model_dict = self.state_dict()
        
        # Map loaded keys to model keys
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # Handle potential key mismatches
            # Your state dict has keys like: 'blocks.0.attn.qkv.weight', etc.
            # We need to map these to the timm ViT model structure
            
            # Check if key exists in model
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                # Try to find matching key
                # For ViT models, timm uses different naming conventions
                new_k = self._map_key(k)
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    pretrained_dict[new_k] = v
        
        # Update model dict
        model_dict.update(pretrained_dict)
        
        # Load updated state dict
        self.load_state_dict(model_dict, strict=False)
        
        print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} parameters")
    
    def _map_key(self, key):
        """
        Map keys from your checkpoint format to timm model format
        
        This is a simple mapping - you may need to adjust based on your exact model
        """
        # Example mappings (adjust as needed):
        # 'patch_embed.proj.weight' -> 'vit.patch_embed.proj.weight'
        # 'blocks.0.norm1.weight' -> 'vit.blocks.0.norm1.weight'
        
        if key.startswith('patch_embed') or key.startswith('blocks') or \
           key.startswith('norm') or key.startswith('head'):
            return f'vit.{key}'
        
        return key


class ViTBackbone(nn.Module):
    """
    Shared ViT Backbone for all experts
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViTBackbone, self).__init__()
        
        # Load pre-trained ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # No classification head
        )
        
        # Freeze by default
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Input images (batch_size, 3, 224, 224)
        
        Returns:
            features: (batch_size, 768) for ViT-Base
        """
        return self.vit(x)


def create_expert_from_checkpoint(checkpoint_path, model_name='vit_base_patch16_224',
                                  feature_dim=768):
    """
    Create an expert model from a checkpoint file
    
    Args:
        checkpoint_path: Path to .pkl checkpoint
        model_name: Name of ViT model architecture
        feature_dim: Feature dimension
    
    Returns:
        expert: LoRAMoEExpert model with loaded weights
    """
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    expert = LoRAMoEExpert(
        model_name=model_name,
        num_classes=1,
        pretrained=False,  # We'll load from checkpoint
        feature_dim=feature_dim
    )
    
    # Load weights
    expert.load_pretrained_weights(state_dict)
    
    return expert


# Alternative: Direct ViT model loading without timm
class DirectViTExpert(nn.Module):
    """
    Direct ViT Expert that matches your checkpoint structure exactly
    Use this if timm model doesn't match your architecture
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 num_classes=1, feature_dim=768):
        super(DirectViTExpert, self).__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        num_patches = (img_size // patch_size) ** 2
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            ) for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward_features(self, x):
        """Extract features"""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # Return class token features
        return x[:, 0]
    
    def forward(self, x):
        """
        Args:
            x: Input images (batch_size, 3, 224, 224)
        
        Returns:
            features: (batch_size, feature_dim)
        """
        features = self.forward_features(x)
        return features
    
    def load_from_checkpoint(self, state_dict):
        """Load from your checkpoint"""
        self.load_state_dict(state_dict, strict=False)


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


if __name__ == '__main__':
    # Test loading experts
    print("Testing expert model loading...")
    
    # Option 1: Using timm
    print("\n1. Creating expert with timm...")
    expert_timm = LoRAMoEExpert(model_name='vit_base_patch16_224', feature_dim=768)
    print(f"   Model created. Parameters: {sum(p.numel() for p in expert_timm.parameters()):,}")
    
    # Option 2: Direct ViT
    print("\n2. Creating expert with direct ViT...")
    expert_direct = DirectViTExpert(feature_dim=768)
    print(f"   Model created. Parameters: {sum(p.numel() for p in expert_direct.parameters()):,}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    x = torch.randn(2, 3, 224, 224)
    features = expert_direct(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {features.shape}")
    
    print("\nExpert models ready!")
