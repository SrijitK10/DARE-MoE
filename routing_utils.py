"""
Routing Visualization Utilities for DaRE-MoE.

Reusable helpers for generating heatmap overlays, upsampling routing maps,
and computing routing metrics after finetuning.

Classes:
    RoutingVisualizationUtils  - image / heatmap manipulation
    RoutingStatisticsComputer  - routing accuracy and specialization metrics
"""

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class RoutingVisualizationUtils:
    """Core utility functions for routing map visualization."""

    @staticmethod
    def normalize_image(img_tensor):
        """
        Convert image tensor (C, H, W) in [0, 1] to uint8 numpy (H, W, 3).

        Args:
            img_tensor: torch.Tensor (C, H, W) or numpy (C, H, W) in [0, 1]

        Returns:
            uint8 numpy array (H, W, 3) in [0, 255]
        """
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.detach().cpu().numpy()
        else:
            img = np.array(img_tensor)
        # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def upsample_routing_map(routing_map, target_h, target_w):
        """
        Bilinearly upsample a routing map to the target image resolution.

        Args:
            routing_map: float numpy (H', W') in [0, 1]
            target_h, target_w: target spatial dimensions

        Returns:
            float numpy (target_h, target_w)
        """
        if routing_map.shape == (target_h, target_w):
            return routing_map.astype(np.float32)
        upsampled = cv2.resize(
            routing_map.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return upsampled

    @staticmethod
    def create_heatmap_overlay(image_np, routing_map, alpha=0.5,
                               colormap=cv2.COLORMAP_JET):
        """
        Blend a routing map as a coloured heatmap on top of the input image.

        Overlay = (1-alpha) * image + alpha * heatmap

        Args:
            image_np:    uint8 numpy (H, W, 3) RGB
            routing_map: float numpy (H', W') in [0, 1]  (upsampled internally)
            alpha:       blending factor (0.5 = equal blend, standard in CVPR)
            colormap:    cv2 colormap constant (default JET)

        Returns:
            uint8 numpy (H, W, 3) RGB overlay
        """
        H, W = image_np.shape[:2]
        r_up = RoutingVisualizationUtils.upsample_routing_map(routing_map, H, W)
        heat_u8 = np.uint8(np.clip(r_up * 255.0, 0, 255))
        heatmap_bgr = cv2.applyColorMap(heat_u8, colormap)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(image_bgr, 1.0 - alpha,
                                      heatmap_bgr, alpha, 0)
        return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def difference_map_to_rgb(diff_map):
        """
        Render a signed difference map (GAN – DM, range ~ [-1, 1]) as RGB
        using the RdYlBu diverging colormap.

        Red = GAN-dominant, Blue = DM-dominant, Yellow = balanced.

        Args:
            diff_map: float numpy (H, W) in [-1, 1]

        Returns:
            uint8 numpy (H, W, 3) RGB
        """
        cmap = plt.cm.get_cmap('RdYlBu')
        norm = (diff_map + 1.0) / 2.0          # [-1,1] -> [0,1]
        norm = np.clip(norm, 0.0, 1.0)
        rgba = cmap(norm)
        return (rgba[:, :, :3] * 255).astype(np.uint8)

    @staticmethod
    def normalize_to_01(arr):
        """Linearly rescale array to [0, 1]."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - mn) / (mx - mn)).astype(np.float32)


class RoutingStatisticsComputer:
    """Compute quantitative routing metrics described in Routing_Maps.md."""

    @staticmethod
    def compute_routing_entropy(routing_agg):
        """
        Entropy of per-image routing distribution — high entropy means balanced
        (good for real images), low entropy means specialised (good for fakes).

        Args:
            routing_agg: (B, N) numpy or torch tensor

        Returns:
            entropy: (B,) numpy float32
        """
        if isinstance(routing_agg, torch.Tensor):
            routing_agg = routing_agg.cpu().numpy()
        eps = 1e-8
        p = routing_agg + eps
        p = p / p.sum(axis=1, keepdims=True)
        return -(p * np.log(p)).sum(axis=1).astype(np.float32)

    @staticmethod
    def compute_routing_accuracy(routing_agg, domain_labels, num_experts):
        """
        Routing accuracy = P(argmax(r̄) == domain) over all fake images.

        Args:
            routing_agg:   (B, N) numpy or tensor
            domain_labels: (B,)  int — fake domain index 0..N-1,
                                       real = num_experts
            num_experts:   N

        Returns:
            float in [0, 1]
        """
        if isinstance(routing_agg, torch.Tensor):
            routing_agg = routing_agg.cpu().numpy()
        if isinstance(domain_labels, torch.Tensor):
            domain_labels = domain_labels.cpu().numpy()
        domain_labels = np.asarray(domain_labels)

        fake_mask = domain_labels < num_experts
        if fake_mask.sum() == 0:
            return 0.0

        r = routing_agg[fake_mask]
        d = domain_labels[fake_mask]
        predicted = np.argmax(r, axis=1)
        return float((predicted == d).mean())

    @staticmethod
    def compute_per_expert_stats(routing_agg, domain_labels, num_experts,
                                 expert_names=None):
        """
        For each expert i compute:
          - mean routing weight on its own domain (should be high for fakes)
          - routing accuracy on its domain
          - mean routing weight on real images (should be ~1/N)

        Args:
            routing_agg:   (B, N) numpy
            domain_labels: (B,)  int
            num_experts:   N
            expert_names:  list of str, default ['Expert 0', ...]

        Returns:
            dict[str, dict]  keyed by expert name
        """
        if expert_names is None:
            expert_names = [f'Expert {i}' for i in range(num_experts)]

        domain_labels = np.asarray(domain_labels)
        stats = {}
        for i in range(num_experts):
            name = expert_names[i]
            mask_domain = domain_labels == i
            mask_real   = domain_labels == num_experts

            if mask_domain.sum() > 0:
                mean_r_domain = float(routing_agg[mask_domain, i].mean())
                routing_acc   = float((np.argmax(routing_agg[mask_domain], axis=1) == i).mean())
            else:
                mean_r_domain = 0.0
                routing_acc   = 0.0

            mean_r_real = float(routing_agg[mask_real, i].mean()) \
                if mask_real.sum() > 0 else 0.0

            stats[name] = {
                'mean_routing_on_domain': mean_r_domain,
                'routing_accuracy':       routing_acc,
                'mean_routing_on_real':   mean_r_real,
                'n_domain_samples':       int(mask_domain.sum()),
                'n_real_samples':         int(mask_real.sum()),
            }
        return stats
