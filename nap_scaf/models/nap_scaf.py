"""NAP-SCAF model.

Similarity-Aware Feature Calibration for low-contrast and missing-modality
brain tumor segmentation.

Core modules:
    - NAP: Non-local Adaptive Prior via variational reconstruction.
    - CERA: Contrast-Enhanced Region Attention.
    - SWG: Similarity-Weighted Gating based on local normalized
      cross-correlation.

Input shape:  (B, 4, H, W), modalities ordered as FLAIR, T1, T1ce, T2.
Output shape: (B, num_classes, H, W), with BraTS labels
              0 background, 1 NCR/NET, 2 ED, 3 ET.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

Tensor = torch.Tensor


class ConvBNAct(nn.Sequential):
    """3x3 convolution followed by BatchNorm and LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        padding: Optional[int] = None,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


class VariationalEncoder(nn.Module):
    """Lightweight convolutional encoder that predicts latent mean and log-variance."""

    def __init__(self, in_channels: int = 4, latent_channels: int = 1024, pretrained: bool = False) -> None:
        super().__init__()
        # ``pretrained`` is kept for API compatibility. This cleaned implementation
        # avoids torchvision model downloads and uses a self-contained convolutional VAE.
        del pretrained
        widths = (64, 128, 256, 512, latent_channels)
        layers = []
        current = in_channels
        for width in widths:
            layers.append(ConvBNAct(current, width, stride=2))
            current = width
        self.encoder = nn.Sequential(*layers)
        self.to_mu = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feat = self.encoder(x)
        return self.to_mu(feat), self.to_logvar(feat)


class VariationalDecoder(nn.Module):
    """Convolutional decoder that reconstructs the four-modality input."""

    def __init__(self, out_channels: int = 4, latent_channels: int = 1024) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 512, kernel_size=2, stride=2),
            ConvBNAct(512, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            ConvBNAct(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            ConvBNAct(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ConvBNAct(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ConvBNAct(32, 32),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.decoder(z)


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Sample a latent tensor using the reparameterization trick."""

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class NonLocalAdaptivePrior(nn.Module):
    """NAP branch producing multi-scale anatomical prior features.

    The branch reconstructs an anatomy-consistent proxy and extracts a decoder-aligned
    prior pyramid from it. The reconstruction can be supervised by a VAE loss during
    training by calling ``forward(..., return_aux=True)`` in the main network.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        stages: int,
        latent_channels: int = 1024,
        pretrained_vae_encoder: bool = False,
        detach_reconstruction: bool = False,
    ) -> None:
        super().__init__()
        self.stages = stages
        self.detach_reconstruction = detach_reconstruction
        self.encoder = VariationalEncoder(in_channels, latent_channels, pretrained_vae_encoder)
        self.decoder = VariationalDecoder(in_channels, latent_channels)
        self.input_proj = ConvBNAct(in_channels, base_channels)

        enc_layers: List[nn.Module] = []
        channels = base_channels
        for _ in range(stages):
            enc_layers.append(ConvBNAct(channels, channels * 2, stride=2))
            channels *= 2
        self.encoders = nn.ModuleList(enc_layers)

        dec_layers: List[nn.Module] = []
        for _ in range(stages):
            dec_layers.append(
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2),
                        "merge": ConvBNAct(channels, channels // 2),
                    }
                )
            )
            channels //= 2
        self.decoders = nn.ModuleList(dec_layers)

    def forward(self, x: Tensor, return_aux: bool = False) -> Union[List[Tensor], Tuple[List[Tensor], Dict[str, Tensor]]]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar) if self.training else mu
        reconstruction = self.decoder(z)
        if reconstruction.shape[-2:] != x.shape[-2:]:
            reconstruction = F.interpolate(reconstruction, size=x.shape[-2:], mode="bilinear", align_corners=False)
        prior_seed = reconstruction.detach() if self.detach_reconstruction else reconstruction

        feats: List[Tensor] = []
        y = self.input_proj(prior_seed)
        for layer in self.encoders:
            y = layer(y)
            feats.append(y)

        y = feats[-1]
        prior_pyramid: List[Tensor] = []
        for i, block in enumerate(self.decoders):
            y = block["up"](y)
            if i < len(feats) - 1:
                skip = feats[-2 - i]
                if y.shape[-2:] != skip.shape[-2:]:
                    y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                y = block["merge"](torch.cat([y, skip], dim=1))
            prior_pyramid.append(y)

        if return_aux:
            return prior_pyramid, {"reconstruction": reconstruction, "mu": mu, "logvar": logvar}
        return prior_pyramid


class ChannelAttention(nn.Module):
    """Channel attention used in the lightweight transformer block."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.scale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim)
        self.pos_emb = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        q = rearrange(self.to_q(tokens), "b n (h d) -> b h d n", h=self.heads)
        k = rearrange(self.to_k(tokens), "b n (h d) -> b h d n", h=self.heads)
        v = rearrange(self.to_v(tokens), "b n (h d) -> b h d n", h=self.heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (k @ q.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0, 3, 1, 2).reshape(b, h * w, self.heads * self.dim_head)
        out = self.proj(out).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return out + self.pos_emb(x)


class FeedForward(nn.Module):
    """Convolutional feed-forward network."""

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FusionTransformer(nn.Module):
    """Transformer-style fusion block for modality-specific encoder features."""

    def __init__(self, channels: int, dim_head: int, heads: int) -> None:
        super().__init__()
        self.attention = ChannelAttention(channels, dim_head=dim_head, heads=heads)
        self.ffn = FeedForward(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x


class CERA(nn.Module):
    """Contrast-Enhanced Region Attention.

    Multi-rate depthwise convolutions capture weak boundary evidence, and a residual
    attention path reinforces low-contrast transitions without discarding the original
    feature map.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dilated = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, groups=channels, bias=False)
                for d in (1, 2, 3)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        multi = torch.cat([layer(x) for layer in self.dilated], dim=1)
        attention = self.fuse(multi)
        return x + x * attention


class LocalNCCGate(nn.Module):
    """Similarity-weighted gate using local normalized cross-correlation.

    High structural agreement produces a larger gate value, allowing reliable skip
    information to pass. Low agreement attenuates the skip transmission.
    """

    def __init__(self, decoder_channels: int, skip_channels: int, mid_channels: int, window_size: int = 5) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")
        self.window_size = window_size
        self.eps = 1e-6
        self.q = nn.Conv2d(decoder_channels, mid_channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(skip_channels, mid_channels, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def _local_mean(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, self.window_size, stride=1, padding=self.window_size // 2)

    def forward(self, decoder: Tensor, skip: Tensor) -> Tuple[Tensor, Tensor]:
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        q = self.q(decoder)
        k = self.k(skip)
        q_centered = q - self._local_mean(q)
        k_centered = k - self._local_mean(k)
        numerator = self._local_mean(q_centered * k_centered).sum(dim=1, keepdim=True)
        q_var = self._local_mean(q_centered.pow(2)).sum(dim=1, keepdim=True)
        k_var = self._local_mean(k_centered.pow(2)).sum(dim=1, keepdim=True)
        ncc = numerator / torch.sqrt(q_var * k_var + self.eps)
        ncc = ncc.clamp(-1.0, 1.0)
        gate = torch.sigmoid(self.scale * ncc + self.bias)
        return skip * gate, gate


class SimilarityCalibrationBlock(nn.Module):
    """CERA followed by local-NCC similarity-weighted gating."""

    def __init__(self, decoder_channels: int, skip_channels: int, mid_channels: int, window_size: int = 5) -> None:
        super().__init__()
        self.cera = CERA(skip_channels)
        self.swg = LocalNCCGate(decoder_channels, skip_channels, mid_channels, window_size)

    def forward(self, decoder: Tensor, skip: Tensor) -> Tuple[Tensor, Tensor]:
        enhanced_skip = self.cera(skip)
        return self.swg(decoder, enhanced_skip)


@dataclass
class NAPSCAFConfig:
    in_channels: int = 4
    num_classes: int = 4
    base_channels: int = 16
    stages: int = 4
    ncc_window: int = 5
    latent_channels: int = 1024
    pretrained_vae_encoder: bool = False
    detach_reconstruction: bool = False


class NAPSCAF(nn.Module):
    """Similarity-Aware Calibration Framework.

    The network uses modality-specific encoders, transformer fusion, a NAP branch,
    and skip-level CERA+SWG calibration.
    """

    def __init__(self, config: Optional[NAPSCAFConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = NAPSCAFConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either a config object or keyword arguments, not both.")
        self.config = config
        self.in_channels = config.in_channels
        self.num_classes = config.num_classes
        self.stages = config.stages

        self.nap = NonLocalAdaptivePrior(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            stages=config.stages,
            latent_channels=config.latent_channels,
            pretrained_vae_encoder=config.pretrained_vae_encoder,
            detach_reconstruction=config.detach_reconstruction,
        )

        self.modality_stems = nn.ModuleList(
            [ConvBNAct(1, config.base_channels) for _ in range(config.in_channels)]
        )

        self.modality_encoders = nn.ModuleList()
        for _ in range(config.in_channels):
            layers = nn.ModuleList()
            channels = config.base_channels
            for _ in range(config.stages):
                layers.append(ConvBNAct(channels, channels * 2, stride=2))
                channels *= 2
            self.modality_encoders.append(layers)

        self.fusion = nn.ModuleList()
        channels = config.base_channels
        for _ in range(config.stages):
            channels *= 2
            fused_channels = config.in_channels * channels
            heads = max(1, channels // config.base_channels)
            self.fusion.append(
                nn.Sequential(
                    FusionTransformer(fused_channels, dim_head=config.base_channels, heads=heads),
                    ConvBNAct(fused_channels, channels, kernel_size=1, padding=0),
                )
            )

        self.decoder = nn.ModuleList()
        self.prior_align = nn.ModuleList()
        self.calibration = nn.ModuleList()
        self.heads = nn.ModuleList()

        decoder_channels = channels
        for i in range(config.stages):
            out_channels = decoder_channels // 2
            self.decoder.append(
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose2d(decoder_channels, out_channels, kernel_size=2, stride=2),
                        "prior_merge": ConvBNAct(out_channels * 2, out_channels),
                        "skip_merge": ConvBNAct(out_channels * 2, out_channels),
                    }
                )
            )
            self.prior_align.append(ConvBNAct(out_channels, out_channels, kernel_size=1, padding=0))
            self.calibration.append(
                SimilarityCalibrationBlock(
                    decoder_channels=out_channels,
                    skip_channels=out_channels,
                    mid_channels=max(1, out_channels // 2),
                    window_size=config.ncc_window,
                )
            )
            self.heads.append(nn.Conv2d(out_channels, config.num_classes, kernel_size=1))
            decoder_channels = out_channels

    def _encode_modalities(self, x: Tensor) -> List[Tensor]:
        modality_inputs = torch.chunk(x, self.in_channels, dim=1)
        stacks: List[List[Tensor]] = []
        for idx, modality in enumerate(modality_inputs):
            feat = self.modality_stems[idx](modality)
            feats: List[Tensor] = []
            for layer in self.modality_encoders[idx]:
                feat = layer(feat)
                feats.append(feat)
            stacks.append(feats)

        fused: List[Tensor] = []
        for level in range(self.stages):
            level_feats = [stack[level] for stack in stacks]
            fused.append(self.fusion[level](torch.cat(level_feats, dim=1)))
        return fused

    def forward(self, x: Tensor, return_aux: bool = False) -> Union[Tensor, Dict[str, Tensor]]:
        input_size = x.shape[-2:]
        if return_aux:
            prior_feats, aux = self.nap(x, return_aux=True)
        else:
            prior_feats = self.nap(x, return_aux=False)
            aux = {}

        fused = self._encode_modalities(x)
        feat = fused[-1]
        gate_maps: List[Tensor] = []

        for i, block in enumerate(self.decoder):
            feat = block["up"](feat)
            prior = prior_feats[i]
            if prior.shape[-2:] != feat.shape[-2:]:
                prior = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            prior = self.prior_align[i](prior)
            feat = block["prior_merge"](torch.cat([feat, prior], dim=1))

            if i < self.stages - 1:
                skip = fused[-2 - i]
                if skip.shape[-2:] != feat.shape[-2:]:
                    skip = F.interpolate(skip, size=feat.shape[-2:], mode="bilinear", align_corners=False)
                calibrated_skip, gate = self.calibration[i](feat, skip)
                feat = block["skip_merge"](torch.cat([feat, calibrated_skip], dim=1))
                gate_maps.append(gate)
            else:
                # Final full-resolution level has no encoder feature at the same scale.
                gate_maps.append(torch.ones(feat.shape[0], 1, *feat.shape[-2:], device=feat.device, dtype=feat.dtype))

        logits = self.heads[-1](feat)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        if not return_aux:
            return logits
        aux.update({"logits": logits, "gates": gate_maps})
        return aux


def build_nap_scaf(**kwargs) -> NAPSCAF:
    """Factory function used by scripts."""

    return NAPSCAF(NAPSCAFConfig(**kwargs))


if __name__ == "__main__":
    model = build_nap_scaf(base_channels=8, stages=3, latent_channels=256)
    x = torch.randn(1, 4, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(tuple(y.shape))
