""" Vision Transformer (ViT) with Multi-Transform Spectral MoE (MTS-MoE) + AuroRA MoE

Replaces the Adapter_MoElayer from ViT_MoE.py with MTS-MoElayer, which uses
5 frequency/spatial experts gated via noisy top-k routing:
    - DFT:  full-spectrum complex filter + phase-aware MLP (GAN fingerprints)
    - DCT:  matrix-multiply DCT-II with learnable frequency filter (JPEG artifacts)
    - DWT-Haar: separable 1D Haar wavelet, sub-band weighting + shared MLP (sharp edges)
    - DWT-DB2:  separable 1D Daubechies-2 wavelet (smooth multi-resolution textures)
    - Spatial: depthwise 3x3 conv baseline (non-frequency fallback)

The LoRA-MoE in the attention block supports AuroRA mode (ANL activation between A and B).

Architecture per Block:
    x -> LayerNorm1 -> Attention (frozen QKV + AuroRA LoRA-MoE delta) -> + residual
    x -> LayerNorm2 -> MLP (frozen) --------------------------------------------------> +
    x -> LayerNorm2 -> MTS-MoElayer (DFT|DCT|DWT-Haar|DWT-DB2|Spatial experts) -------> + residual

Key design decisions:
    - Unified TransformAdapter class (one class, transform_type switch) — easy to extend
    - DCT via precomputed matrix multiply (D @ x @ D^T) — exact, numerically stable
    - DFT with phase-aware MLP: magnitude+phase concatenated → joint processing
    - DWT via separable 1D conv1d/conv_transpose1d — textbook approach, robust at any size
    - Spatial expert gives the router a non-frequency fallback option
    - All weights zero-initialized (near-identity start for stable training)
    - Gating is per-image (mean-pooled tokens), matching the original Adapter_MoElayer convention
"""

import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from torch.distributions.normal import Normal


_logger = logging.getLogger(__name__)


# =============================================================================
# Utility Modules
# =============================================================================

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# =============================================================================
# ANL: Activation Nonlinear Layer (B-spline based) for AuroRA
# =============================================================================

class ANL(nn.Module):
    """Activation Nonlinear Layer using B-splines for AuroRA.
    
    Replaces the identity/ReLU between LoRA's A and B matrices with a learnable
    activation function: output = tanh(W_base * tanh(x)) + spline(x).
    The spline branch uses cubic B-spline basis functions with learnable weights.
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        base_activation=nn.Tanh,
        grid_range=[-1, 1],
    ):
        super(ANL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, grid_size + spline_order)
        )

        self.scale_noise = scale_noise
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.base_weight)
        nn.init.uniform_(self.spline_weight, -self.scale_noise, self.scale_noise)

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

        for k in range(1, self.spline_order + 1):
            denom1 = grid[k:-1] - grid[:-k - 1]
            denom2 = grid[k + 1:] - grid[1:-k]
            denom1[denom1 == 0] = 1
            denom2[denom2 == 0] = 1

            term1 = ((x - grid[:-k - 1]) / denom1) * bases[:, :, :-1]
            term2 = ((grid[k + 1:] - x) / denom2) * bases[:, :, 1:]

            bases = term1 + term2

        return bases

    def forward(self, x):
        original_shape = x.shape
        x = x.contiguous().reshape(-1, self.in_features)

        base_output = self.base_activation(F.linear(self.base_activation(x), self.base_weight))

        bases = self.b_splines(x)
        spline_input = bases.sum(dim=1)

        spline_input = spline_input.to(dtype=self.spline_weight.dtype)

        spline_output = F.linear(spline_input, self.spline_weight)

        output = base_output + spline_output
        output = output.contiguous().reshape(*original_shape[:-1], self.out_features)
        return output


# =============================================================================
# Sparse Dispatcher for MoE routing
# =============================================================================

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    
    Creates input minibatches for experts and combines their outputs.
    Batch element b is sent to expert e iff gates[b, e] != 0.
    Inputs/outputs are 2D [batch, depth].
    """

    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=False):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)




# =============================================================================
# Unified Transform Adapter Expert
# =============================================================================

class TransformAdapter(nn.Module):
    """Unified frequency/spatial adapter expert.

    A single class that implements five different transform domains:
        'dft'      - Full-spectrum complex filter + phase-aware MLP
        'dct'      - Matrix-multiply DCT-II with learnable frequency filter
        'dwt_haar' - Separable 1D Haar wavelet + sub-band weighting + shared MLP
        'dwt_db2'  - Separable 1D Daubechies-2 wavelet + sub-band weighting + shared MLP
        'spatial'  - Depthwise 3x3 conv baseline (non-frequency fallback)

    All experts share the same interface: (B, N, dim) -> (B, N, dim)
    with down/up projection bottleneck and near-identity initialization.
    """

    def __init__(self, dim, adapter_dim, spatial_size=14, transform_type='dft'):
        super().__init__()
        self.dim = dim
        self.adapter_dim = adapter_dim
        self.spatial_size = spatial_size
        self.transform_type = transform_type

        # Shared bottleneck projection
        self.adapter_down = nn.Linear(dim, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, dim)

        if transform_type == 'dft':
            self._init_dft(adapter_dim, spatial_size)
        elif transform_type == 'dct':
            self._init_dct(adapter_dim, spatial_size)
        elif transform_type in ('dwt_haar', 'dwt_db2'):
            self._init_dwt(adapter_dim, spatial_size, transform_type)
        elif transform_type == 'spatial':
            self._init_spatial(adapter_dim)
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")

        self._init_near_identity()

    # ------------------------------------------------------------------ #
    #  Per-transform initialization helpers                                #
    # ------------------------------------------------------------------ #

    def _init_dft(self, d, S):
        """DFT expert: full complex filter + phase-aware MLP."""
        self.freq_filter_real = nn.Parameter(torch.zeros(1, d, S, S))
        self.freq_filter_imag = nn.Parameter(torch.zeros(1, d, S, S))
        # Phase-aware MLP: mag + phase concatenated -> joint processing
        self.spectral_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def _init_dct(self, d, S):
        """DCT expert: precomputed matrix + learnable filter + depthwise MLP."""
        D = self._build_dct_matrix(S)
        self.register_buffer('dct_matrix', D)
        self.freq_filter = nn.Parameter(torch.zeros(1, d, S, S))
        self.freq_mlp = nn.Sequential(
            nn.Conv2d(d, d, 1, groups=d, bias=True),
            nn.GELU(),
            nn.Conv2d(d, d, 1, groups=d, bias=True),
        )

    def _init_dwt(self, d, S, wavelet_type):
        """DWT expert: separable 1D wavelet filters + sub-band weights + shared MLP."""
        if wavelet_type == 'dwt_haar':
            lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
            hi = torch.tensor([-1.0, 1.0]) / math.sqrt(2)
        else:  # dwt_db2
            lo = torch.tensor([
                (1 + math.sqrt(3)) / (4 * math.sqrt(2)),
                (3 + math.sqrt(3)) / (4 * math.sqrt(2)),
                (3 - math.sqrt(3)) / (4 * math.sqrt(2)),
                (1 - math.sqrt(3)) / (4 * math.sqrt(2)),
            ])
            hi = torch.tensor([
                (1 - math.sqrt(3)) / (4 * math.sqrt(2)),
                -(3 - math.sqrt(3)) / (4 * math.sqrt(2)),
                (3 + math.sqrt(3)) / (4 * math.sqrt(2)),
                -(1 + math.sqrt(3)) / (4 * math.sqrt(2)),
            ])

        # Register as 1D conv filters: (1, 1, filter_len)
        self.register_buffer('lo_d', lo.reshape(1, 1, -1))
        self.register_buffer('hi_d', hi.reshape(1, 1, -1))
        self.register_buffer('lo_r', lo.flip(0).reshape(1, 1, -1))
        self.register_buffer('hi_r', hi.flip(0).reshape(1, 1, -1))

        # Learnable sub-band attention weights (softmax over 4 sub-bands)
        self.subband_weights = nn.Parameter(torch.zeros(4))

        # Shared MLP: processes concatenated 4 sub-bands
        self.subband_mlp = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def _init_spatial(self, d):
        """Spatial expert: depthwise 3x3 conv + pointwise MLP."""
        self.depth_conv = nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=True)
        self.spatial_mlp = nn.Sequential(
            nn.Conv2d(d, d, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(d, d, 1, bias=True),
        )

    def _init_near_identity(self):
        """Zero-init output layers so the adapter starts as a near-identity."""
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        if self.transform_type == 'dft':
            for m in self.spectral_mlp:
                if hasattr(m, 'weight'): nn.init.zeros_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
        elif self.transform_type == 'dct':
            for m in self.freq_mlp:
                if hasattr(m, 'weight'): nn.init.zeros_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
        elif self.transform_type in ('dwt_haar', 'dwt_db2'):
            for m in self.subband_mlp:
                if hasattr(m, 'weight'): nn.init.zeros_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
        elif self.transform_type == 'spatial':
            nn.init.zeros_(self.depth_conv.weight)
            nn.init.zeros_(self.depth_conv.bias)
            for m in self.spatial_mlp:
                if hasattr(m, 'weight'): nn.init.zeros_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    #  Static / shared helpers                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_dct_matrix(N):
        """Build orthonormal NxN DCT-II basis matrix.

        Row k of D is the k-th DCT basis vector:
            D[k, n] = alpha_k * cos(pi * (n + 0.5) * k / N)
        where alpha_0 = 1/sqrt(N), alpha_k = sqrt(2/N) for k > 0.
        """
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(N, dtype=torch.float32)
        D = torch.cos(math.pi * (n.unsqueeze(0) + 0.5) * k.unsqueeze(1) / N)
        D[0, :] *= 1.0 / math.sqrt(N)
        D[1:, :] *= math.sqrt(2.0 / N)
        return D

    @staticmethod
    def _dwt_1d(x, lo, hi):
        """Separable 1D DWT along last dimension via conv1d with stride 2.

        Args:
            x: (..., L) tensor
            lo: (1, 1, filter_len) low-pass decomposition filter
            hi: (1, 1, filter_len) high-pass decomposition filter
        Returns:
            lo_coeffs, hi_coeffs: each (..., ceil(L/2))
        """
        *batch_shape, L = x.shape
        x_flat = x.reshape(-1, 1, L)
        flen = lo.shape[-1]
        pad = flen - 1
        x_padded = F.pad(x_flat, (pad // 2, pad - pad // 2), mode='reflect')
        lo_out = F.conv1d(x_padded, lo, stride=2)
        hi_out = F.conv1d(x_padded, hi, stride=2)
        sub_len = lo_out.shape[-1]
        return (lo_out.reshape(*batch_shape, sub_len),
                hi_out.reshape(*batch_shape, sub_len))

    @staticmethod
    def _idwt_1d(lo_coeff, hi_coeff, lo_r, hi_r, target_len):
        """Separable 1D inverse DWT along last dimension via conv_transpose1d.

        Args:
            lo_coeff, hi_coeff: (..., sub_L)
            lo_r, hi_r: (1, 1, filter_len) reconstruction filters
            target_len: desired output length
        Returns:
            reconstructed: (..., target_len)
        """
        *batch_shape, sub_L = lo_coeff.shape
        lo_flat = lo_coeff.reshape(-1, 1, sub_L)
        hi_flat = hi_coeff.reshape(-1, 1, sub_L)
        lo_up = F.conv_transpose1d(lo_flat, lo_r, stride=2)
        hi_up = F.conv_transpose1d(hi_flat, hi_r, stride=2)
        result = lo_up + hi_up
        result = result[..., :target_len]
        return result.reshape(*batch_shape, target_len)

    def _dwt_2d(self, x):
        """2D DWT via separable 1D operations: rows then columns.

        Args:
            x: (B, C, H, W)
        Returns:
            (LL, LH, HL, HH): each (B, C, sub_H, sub_W)
        """
        B, C, H, W = x.shape

        # Row-wise DWT (along W dimension)
        x_rows = x.reshape(B * C * H, W)
        lo_w, hi_w = self._dwt_1d(x_rows, self.lo_d, self.hi_d)
        sub_w = lo_w.shape[-1]
        lo_w = lo_w.reshape(B, C, H, sub_w)
        hi_w = hi_w.reshape(B, C, H, sub_w)

        # Column-wise DWT (along H dimension) — transpose, apply, transpose back
        lo_w_t = lo_w.permute(0, 1, 3, 2).reshape(B * C * sub_w, H)
        hi_w_t = hi_w.permute(0, 1, 3, 2).reshape(B * C * sub_w, H)

        ll, lh = self._dwt_1d(lo_w_t, self.lo_d, self.hi_d)
        hl, hh = self._dwt_1d(hi_w_t, self.lo_d, self.hi_d)

        sub_h = ll.shape[-1]
        ll = ll.reshape(B, C, sub_w, sub_h).permute(0, 1, 3, 2)
        lh = lh.reshape(B, C, sub_w, sub_h).permute(0, 1, 3, 2)
        hl = hl.reshape(B, C, sub_w, sub_h).permute(0, 1, 3, 2)
        hh = hh.reshape(B, C, sub_w, sub_h).permute(0, 1, 3, 2)

        return ll, lh, hl, hh

    def _idwt_2d(self, ll, lh, hl, hh, target_h, target_w):
        """2D inverse DWT via separable 1D operations.

        Args:
            ll, lh, hl, hh: (B, C, sub_H, sub_W) sub-band coefficients
            target_h, target_w: desired output spatial size
        Returns:
            x: (B, C, target_H, target_W)
        """
        B, C, sub_h, sub_w = ll.shape

        # Column-wise iDWT (along H dimension): transpose, apply, transpose back
        ll_t = ll.permute(0, 1, 3, 2).reshape(B * C * sub_w, sub_h)
        lh_t = lh.permute(0, 1, 3, 2).reshape(B * C * sub_w, sub_h)
        hl_t = hl.permute(0, 1, 3, 2).reshape(B * C * sub_w, sub_h)
        hh_t = hh.permute(0, 1, 3, 2).reshape(B * C * sub_w, sub_h)

        lo_w = self._idwt_1d(ll_t, lh_t, self.lo_r, self.hi_r, target_h)
        hi_w = self._idwt_1d(hl_t, hh_t, self.lo_r, self.hi_r, target_h)

        lo_w = lo_w.reshape(B, C, sub_w, target_h).permute(0, 1, 3, 2)
        hi_w = hi_w.reshape(B, C, sub_w, target_h).permute(0, 1, 3, 2)

        # Row-wise iDWT (along W dimension)
        lo_w_flat = lo_w.reshape(B * C * target_h, sub_w)
        hi_w_flat = hi_w.reshape(B * C * target_h, sub_w)

        x = self._idwt_1d(lo_w_flat, hi_w_flat, self.lo_r, self.hi_r, target_w)
        x = x.reshape(B, C, target_h, target_w)

        return x

    # ------------------------------------------------------------------ #
    #  Forward branches                                                    #
    # ------------------------------------------------------------------ #

    def _forward_dft(self, x_patch):
        """DFT forward: FFT2 -> complex filter -> phase-aware MLP -> iFFT2."""
        # cuFFT requires float32 for non-power-of-2 sizes (e.g. 14×14 patches)
        input_dtype = x_patch.dtype
        if x_patch.dtype == torch.float16:
            x_patch = x_patch.float()
        x_fft = torch.fft.fft2(x_patch, norm='ortho')  # full complex spectrum

        # Apply learnable complex filter (residual: 1 + filter)
        filt = torch.complex(self.freq_filter_real, self.freq_filter_imag)
        x_filtered = x_fft * (1.0 + filt)

        # Phase-aware processing: extract magnitude and phase
        mag = x_filtered.abs()                                         # (B, C, H, W)
        phase = torch.atan2(x_filtered.imag, x_filtered.real + 1e-8)  # (B, C, H, W)

        # Concatenate mag + phase, process jointly to learn magnitude correction
        combined = torch.cat([mag, phase], dim=1)        # (B, 2C, H, W)
        combined = combined.permute(0, 2, 3, 1)          # (B, H, W, 2C)
        delta = self.spectral_mlp(combined)               # (B, H, W, C)
        delta = delta.permute(0, 3, 1, 2)                 # (B, C, H, W)

        # Modulate magnitude, reconstruct with original phase
        new_mag = mag + delta
        x_out = new_mag * torch.exp(1j * phase)
        x_spatial = torch.fft.ifft2(x_out, norm='ortho').real

        # Restore original dtype (e.g. float16 under AMP)
        if x_spatial.dtype != input_dtype:
            x_spatial = x_spatial.to(input_dtype)

        return x_spatial

    def _forward_dct(self, x_patch):
        """DCT forward: matrix-multiply DCT -> filter + MLP -> inverse DCT."""
        D = self.dct_matrix  # (S, S)

        # Forward DCT: D @ x @ D^T via einsum
        x_dct = torch.einsum('ij,bcjk,lk->bcil', D, x_patch, D)

        # Apply learnable frequency gate
        gate = torch.sigmoid(self.freq_filter)
        x_dct = x_dct * gate

        # Process in DCT domain
        x_dct = x_dct + self.freq_mlp(x_dct)

        # Inverse DCT: D^T @ x_dct @ D (D is orthonormal => D^-1 = D^T)
        x_spatial = torch.einsum('ji,bcjk,kl->bcil', D, x_dct, D)

        return x_spatial

    def _forward_dwt(self, x_patch):
        """DWT forward: separable 2D DWT -> sub-band weighting + shared MLP -> iDWT."""
        B, C, H, W = x_patch.shape

        # Forward 2D DWT
        ll, lh, hl, hh = self._dwt_2d(x_patch)
        subs = [ll, lh, hl, hh]

        # Learnable sub-band attention weights
        w = F.softmax(self.subband_weights, dim=0)
        weighted = [w[i] * subs[i] for i in range(4)]

        # Inverse 2D DWT with weighted sub-bands
        x_recon = self._idwt_2d(*weighted, target_h=H, target_w=W)

        # Process sub-bands through shared MLP for additional expressivity
        subs_up = [F.interpolate(s, size=(H, W), mode='bilinear', align_corners=False)
                   for s in subs]
        concat = torch.cat(subs_up, dim=1)                 # (B, 4C, H, W)
        concat = concat.permute(0, 2, 3, 1)                # (B, H, W, 4C)
        mlp_out = self.subband_mlp(concat)                  # (B, H, W, C)
        mlp_out = mlp_out.permute(0, 3, 1, 2)              # (B, C, H, W)

        return x_recon + mlp_out

    def _forward_spatial(self, x_patch):
        """Spatial forward: depthwise 3x3 conv + pointwise MLP (residual)."""
        return x_patch + self.spatial_mlp(self.depth_conv(x_patch))

    # ------------------------------------------------------------------ #
    #  Main forward                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, x):
        """
        Args:
            x: (B, N, dim) -- token features (N = 197 for ViT-B/16 with CLS)
        Returns:
            (B, N, dim) -- adapted features
        """
        B, N, C = x.shape
        S = self.spatial_size

        # Down-project
        x_down = self.adapter_down(x)

        # Separate CLS token
        x_cls = x_down[:, :1]                                                      # (B, 1, d)
        x_patch = x_down[:, 1:]                                                    # (B, S*S, d)
        x_patch = x_patch.reshape(B, S, S, self.adapter_dim).permute(0, 3, 1, 2)  # (B, d, S, S)

        # Apply transform-specific processing
        if self.transform_type == 'dft':
            x_patch = self._forward_dft(x_patch)
        elif self.transform_type == 'dct':
            x_patch = self._forward_dct(x_patch)
        elif self.transform_type in ('dwt_haar', 'dwt_db2'):
            x_patch = self._forward_dwt(x_patch)
        elif self.transform_type == 'spatial':
            x_patch = self._forward_spatial(x_patch)

        # Reshape back and up-project
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, S * S, self.adapter_dim)
        x_down = torch.cat([x_cls, x_patch], dim=1)
        x_up = self.adapter_up(x_down)

        return x_up


# =============================================================================
# MTS-MoE Layer: Multi-Transform Spectral Mixture of Experts
# =============================================================================

class MTS_MoElayer(nn.Module):
    """Multi-Transform Spectral Mixture of Experts layer.

    Drop-in replacement for Adapter_MoElayer.  Routes each image to the most
    relevant frequency/spatial expert(s) via noisy top-k gating.

    Default experts (5):
        Expert 0: DFT      -- full-spectrum complex filter + phase-aware MLP
        Expert 1: DCT      -- matrix-multiply DCT-II with learnable filter
        Expert 2: DWT-Haar -- separable 1D Haar wavelet + sub-band attention
        Expert 3: DWT-DB2  -- separable 1D Daubechies-2 wavelet + sub-band attention
        Expert 4: Spatial  -- depthwise 3x3 conv baseline (non-frequency fallback)

    All experts use the unified TransformAdapter class with a transform_type switch.

    Args:
        dim: input feature dimension (e.g., 768)
        adapter_dim: bottleneck dimension (default 8)
        noisy_gating: use noisy gating during training
        k: top-k experts per sample
        spatial_size: spatial resolution of patch grid (default 14 for 224/16)
        transform_types: tuple of transform type strings to use as experts
    """

    def __init__(self, dim=768, adapter_dim=8, noisy_gating=True, k=1,
                 spatial_size=14,
                 transform_types=('dft', 'dct', 'dwt_haar', 'dwt_db2', 'spatial')):
        super(MTS_MoElayer, self).__init__()
        self.noisy_gating = noisy_gating
        self.dim = dim
        self.k = k

        # Create experts -- one TransformAdapter per transform type
        self.adapter_experts = nn.ModuleList([
            TransformAdapter(dim, adapter_dim, spatial_size, t)
            for t in transform_types
        ])
        self.num_experts = len(self.adapter_experts)

        # Gating network
        self.w_gate = nn.Parameter(torch.zeros(dim, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(dim, self.num_experts), requires_grad=True)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        assert self.k <= self.num_experts

    def cv_squared(self, x):
        """Squared coefficient of variation -- load-balancing loss."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating. See: https://arxiv.org/abs/1701.06538."""
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, dtype=top_k_gates.dtype, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        """
        Args:
            x: (B, N, dim) -- token features (N=197 for ViT-B/16)
            loss_coef: scalar multiplier on load-balancing loss
        Returns:
            y: (B, N, dim) -- adapted features
            loss: scalar -- load-balancing loss
        """
        B, N, _ = x.shape
        x_global = torch.mean(x, dim=1, keepdim=False)  # (B, dim) -- image-level gating

        gates, load = self.noisy_top_k_gating(x_global, self.training)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()

        expert_outputs = []
        for i in range(self.num_experts):
            if len(expert_inputs[i]) == 0:
                continue
            expert_output = self.adapter_experts[i](expert_inputs[i])
            expert_output = expert_output.reshape(expert_output.size(0), N * self.dim)
            expert_outputs.append(expert_output)

        y = dispatcher.combine(expert_outputs)
        y = y.reshape(B, N, self.dim)

        return y, loss


# =============================================================================
# AuroRA LoRA-MoE Layer (in Attention)
# =============================================================================

class LoRA_MoElayer(nn.Module):
    """Sparsely-gated LoRA Mixture of Experts for QKV adaptation.
    
    Each expert is a LoRA pair (A, B) with a different rank.
    When lora_use_act=True (AuroRA mode), an ANL nonlinear activation is
    inserted between A and B matrices.
    
    Routing is per-token (each patch token independently selects its expert).
    """

      # def __init__(self, dim, lora_dim=[8, 16, 32, 48, 64, 96, 128], lora_alpha=None,
    #              noisy_gating=True, k=1, lora_use_act=False):
    def __init__(self, dim, lora_dim=[2,4,6,8], lora_alpha=None,
                 noisy_gating=True, k=1, lora_use_act=True):
        super(LoRA_MoElayer, self).__init__()

        self.noisy_gating = noisy_gating
        self.k = k
        self.lora_use_act = lora_use_act

        # Handle lora_alpha
        if lora_alpha is None:
            lora_alpha = lora_dim
        elif isinstance(lora_alpha, (int, float)):
            lora_alpha = [lora_alpha] * len(lora_dim)
        self.scaling = [alpha / r for alpha, r in zip(lora_alpha, lora_dim)]

        Lora_a_experts = nn.ModuleList()
        Lora_b_experts = nn.ModuleList()
        Lora_ab_experts = nn.ModuleList()
        for i, d in enumerate(lora_dim):
            Lora_a_experts.append(nn.Linear(dim, d, bias=False))
            nn.init.kaiming_uniform_(Lora_a_experts[i].weight, a=math.sqrt(5))
            Lora_b_experts.append(nn.Linear(d, dim * 3, bias=False))
            nn.init.zeros_(Lora_b_experts[i].weight)
            if lora_use_act:
                Lora_ab_experts.append(ANL(in_features=d, out_features=d))

        self.num_experts = len(Lora_a_experts)
        self.Lora_a_experts = Lora_a_experts
        self.Lora_b_experts = Lora_b_experts
        self.Lora_ab_experts = Lora_ab_experts if lora_use_act else None
        self.w_gate = nn.Parameter(torch.zeros(dim, len(Lora_a_experts)), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(dim, len(Lora_a_experts)), requires_grad=True)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, dtype=top_k_gates.dtype, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        B, N, C = x.shape
        x = x.reshape(B * N, C)
        gates, load = self.noisy_top_k_gating(x, self.training)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()

        expert_outputs = []
        for i in range(self.num_experts):
            if len(expert_inputs[i]) == 0:
                continue
            qkv_delta = F.linear(expert_inputs[i], self.Lora_a_experts[i].weight)
            if self.lora_use_act:
                # AuroRA: Apply ANL activation between A and B matrices
                qkv_delta = self.Lora_ab_experts[i](qkv_delta)
            qkv_delta = F.linear(qkv_delta, self.Lora_b_experts[i].weight)
            qkv_delta = qkv_delta * self.scaling[i]
            expert_outputs.append(qkv_delta)

        y = dispatcher.combine(expert_outputs)
        y = y.reshape(B, N, C * 3)
        return y, loss


# =============================================================================
# Attention & Block with AuroRA LoRA-MoE + MTS-MoE
# =============================================================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., lora_topk=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.LoRA_k = lora_topk
        if self.LoRA_k > 0:
            self.LoRA_MoE = LoRA_MoElayer(dim, k=self.LoRA_k)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.LoRA_k > 0:
            qkv_delta, lora_loss = self.LoRA_MoE(x)
            qkv_delta = qkv_delta.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_delta, k_delta, v_delta = qkv_delta.unbind(0)
            q, k, v = q + q_delta, k + k_delta, v + v_delta
        else:
            lora_loss = torch.zeros(1).to(x.device)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, lora_loss


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 lora_topk=1, adapter_topk=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop, lora_topk=lora_topk)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.adapter_k = adapter_topk
        if self.adapter_k > 0:
            # MTS-MoE replaces Adapter_MoElayer
            self.adapter_MoE = MTS_MoElayer(dim, adapter_dim=8, k=self.adapter_k)

    def forward(self, x):
        x1, lora_loss = self.attn(self.norm1(x))
        x = x + self.drop_path(x1)

        if self.adapter_k > 0:
            x_adapter, adapter_loss = self.drop_path(self.adapter_MoE(self.norm2(x)))
            x = x + x_adapter + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            adapter_loss = torch.zeros(1).to(x.device)

        return x, lora_loss, adapter_loss


# =============================================================================
# Vision Transformer with MTS-MoE + AuroRA LoRA-MoE
# =============================================================================

class VisionTransformer(nn.Module):
    """Vision Transformer with:
    - AuroRA LoRA-MoE in attention (QKV adaptation)
    - MTS-MoE parallel to MLP (frequency-domain adaptation)
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', lora_topk=1, adapter_topk=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer,
                lora_topk=lora_topk, adapter_topk=adapter_topk)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.lora_topk = lora_topk
        self.adapter_topk = adapter_topk

        self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.freeze_stages()

    def freeze_stages(self):
        """Freeze pretrained backbone, keep MoE adapters + head trainable."""
        self.pos_drop.eval()
        self.patch_embed.eval()

        for block in self.blocks:
            block.eval()
            if self.lora_topk > 0:
                block.attn.LoRA_MoE.train()
            if self.adapter_topk > 0:
                block.adapter_MoE.train()

        for name, param in self.named_parameters():
            if 'LoRA' not in name and 'adapter' not in name and 'head' not in name and 'norm1' not in name:
                param.requires_grad = False

        total_para_nums = 0
        LoRA_para_nums = 0
        adapter_para_nums = 0
        head_para_nums = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total_para_nums += param.numel()
                if 'LoRA' in name:
                    LoRA_para_nums += param.numel()
                elif 'head' in name:
                    head_para_nums += param.numel()
                elif 'adapter' in name:
                    adapter_para_nums += param.numel()

        print('parameters:', total_para_nums, 'LoRA', LoRA_para_nums, 'MTS-MoE', adapter_para_nums, 'head', head_para_nums)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.mask_token, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        lora_loss_list = []
        adapter_loss_list = []
        for block in self.blocks:
            x, cur_lora_loss, cur_adapter_loss = block(x)
            lora_loss_list.append(cur_lora_loss)
            adapter_loss_list.append(cur_adapter_loss)

        lora_loss = torch.mean(torch.stack(lora_loss_list))
        adapter_loss = torch.mean(torch.stack(adapter_loss_list))
        moe_loss = lora_loss * 200 + adapter_loss * 1
        x = self.norm(x)
        return self.pre_logits(x[:, 0]), moe_loss

    def forward(self, x):
        x, moe_loss = self.forward_features(x)
        x = self.head(x)
        return x, moe_loss


# =============================================================================
# Weight Initialization & Loading
# =============================================================================

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    out_dict = {}
    if 'model' in state_dict:
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


# =============================================================================
# Model Config & Registration
# =============================================================================

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch32_224': _cfg(url=''),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_huge_patch14_224': _cfg(url=''),
    'vit_giant_patch14_224': _cfg(url=''),
    'vit_gigantic_patch14_224': _cfg(url=''),
    # ImageNet-21K pretrained
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch8_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        num_classes=21843),
    # SAM
    'vit_base_patch32_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),
    # DeiT
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),
    # MIIL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear'),
}


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


# =============================================================================
# Registered Model Constructors
# =============================================================================

@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_tiny_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_small_patch32_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_small_patch32_384(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_small_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch8_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch16_sam_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=0, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_sam_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch32_sam_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, representation_size=0, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_sam_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_huge_patch14_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_giant_patch14_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_gigantic_patch14_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_gigantic_patch14_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_tiny_patch16_224_in21k(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_small_patch32_224_in21k(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_small_patch16_224_in21k(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch8_224_in21k(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('deit_tiny_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model

@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('deit_small_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model

@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model

@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model

@register_model
def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch16_224_miil(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, **model_kwargs)
    return model


if __name__ == '__main__':
    model = vit_base_patch16_224_in21k(pretrained=True, num_classes=2)
    x = torch.rand(32, 3, 224, 224)
    y, _ = model(x)
    print(y.shape)
