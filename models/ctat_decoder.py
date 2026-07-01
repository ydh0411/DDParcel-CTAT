# models/ctat_decoder.py
"""CTAT CNN decoder — token-to-spatial skip fusion + progressive upsampling to 256x256."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv3x3->BN->ReLU->Conv3x3->BN, with residual."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class DecoderStage(nn.Module):
    """Upsample x2 -> concat skip -> 1x1 fusion -> ConvBlocks."""
    def __init__(self, in_ch, skip_ch, out_ch, num_blocks=2):
        super().__init__()
        self.fusion = nn.Conv2d(in_ch + skip_ch, out_ch, 1)
        self.blocks = nn.ModuleList([ConvBlock(out_ch) for _ in range(num_blocks)])

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class CTATDecoder(nn.Module):
    """
    CNN decoder upsampling tokens from 4x4 bottleneck to 256x256 feature maps.
    Grid path: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256.
    Encoder stage dims: [96, 192, 384, 768, 1536]
    """
    def __init__(self, encoder_stage_dims, decoder_dim=96, num_modalities=4):
        super().__init__()
        self.M = num_modalities
        dims = encoder_stage_dims  # [dim0, dim1, dim2, dim3, dim4]

        # Skip projection: M*dim_i -> decoder_dim via 1x1 conv
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(num_modalities * dims[0], decoder_dim, 1),
            nn.Conv2d(num_modalities * dims[1], decoder_dim, 1),
            nn.Conv2d(num_modalities * dims[2], decoder_dim, 1),
            nn.Conv2d(num_modalities * dims[3], decoder_dim, 1),
        ])

        # 4 decoder stages: 4->8, 8->16, 16->32, 32->64
        self.stage3 = DecoderStage(num_modalities * dims[4], decoder_dim, decoder_dim)
        self.stage2 = DecoderStage(decoder_dim, decoder_dim, decoder_dim)
        self.stage1 = DecoderStage(decoder_dim, decoder_dim, decoder_dim)
        self.stage0 = DecoderStage(decoder_dim, decoder_dim, 64)

    def _skip2spatial(self, tokens, proj, H, W):
        """[B, M*N, C] -> [B, decoder_dim, H, W]"""
        B = tokens.shape[0]
        C = tokens.shape[-1]
        x = tokens.view(B, self.M, H, W, C)
        x = x.permute(0, 1, 4, 2, 3).reshape(B, self.M * C, H, W)
        return proj(x)

    def forward(self, bottleneck, skips, H_list, W_list, return_intermediates=False):
        """
        bottleneck: [B, M*H_bn*W_bn, dim4]
        skips: [skip0, skip1, skip2, skip3]
        H_list, W_list: spatial resolutions at each skip level
        return_intermediates: if True, return (final, [stage3_out, stage2_out, stage1_out])
        Returns: [B, 64, 256, 256] or tuple if return_intermediates
        """
        B = bottleneck.shape[0]
        M, dim4 = self.M, bottleneck.shape[-1]
        H_bn, W_bn = H_list[3] // 2, W_list[3] // 2

        # Bottleneck tokens -> spatial: [B, M*dim4, H_bn, W_bn]
        x = bottleneck.view(B, M, H_bn * W_bn, dim4).view(B, M, H_bn, W_bn, dim4)
        x = x.permute(0, 1, 4, 2, 3).reshape(B, M * dim4, H_bn, W_bn)

        # Convert skips to spatial (reverse order: deep->shallow)
        s3 = self._skip2spatial(skips[3], self.skip_projs[3], H_list[3], W_list[3])
        s2 = self._skip2spatial(skips[2], self.skip_projs[2], H_list[2], W_list[2])
        s1 = self._skip2spatial(skips[1], self.skip_projs[1], H_list[1], W_list[1])
        s0 = self._skip2spatial(skips[0], self.skip_projs[0], H_list[0], W_list[0])

        intermediates = []
        x = self.stage3(x, s3)   # 4x4 -> 8x8
        if return_intermediates:
            intermediates.append(x)
        x = self.stage2(x, s2)   # 8x8 -> 16x16
        if return_intermediates:
            intermediates.append(x)
        x = self.stage1(x, s1)   # 16x16 -> 32x32
        if return_intermediates:
            intermediates.append(x)
        x = self.stage0(x, s0)   # 32x32 -> 64x64

        # Final upsample: 64x64 -> 128x128 -> 256x256
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        if return_intermediates:
            return x, intermediates  # [B, 64, 256, 256], [stage3_8x8, stage2_16x16, stage1_32x32]
        return x
