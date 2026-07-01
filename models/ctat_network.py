# models/ctat_network.py
"""CTAT full network: encoder + decoder + deep supervision heads + classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .ctat_encoder import CTATEncoder
    from .ctat_decoder import CTATDecoder
except ImportError:
    from ctat_encoder import CTATEncoder
    from ctat_decoder import CTATDecoder


class CTAT(nn.Module):
    """
    Competitive Token Attention Transformer for brain parcellation.

    Input:  [B, 28, 256, 256]  (4 modalities x 7 thick slices)
    Output: [B, 82, 256, 256]  logits for 82 FreeSurfer classes

    Deep supervision: auxiliary classifier heads at decoder stages 1, 2, 3.
    Returns main logits + list of auxiliary logits for training.
    """
    def __init__(self, num_classes=82, in_channels=7, num_modalities=4, embed_dim=96,
                 num_heads=8, window_size=8, depths=[2,2,2,6], mlp_ratio=4,
                 alpha=2.0):
        super().__init__()
        self.encoder = CTATEncoder(in_channels, num_modalities, embed_dim,
                                   num_heads, window_size, depths, mlp_ratio, alpha)
        self.decoder = CTATDecoder(self.encoder.stage_dims, decoder_dim=embed_dim,
                                   num_modalities=num_modalities)

        # Main classifier
        self.classifier = nn.Conv2d(64, num_classes, 1)

        # Deep supervision heads (attached to decoder stage outputs)
        ds_ch = embed_dim
        self.ds_heads = nn.ModuleList([
            self._make_ds_head(ds_ch, num_classes),   # after stage3 (8x8)
            self._make_ds_head(ds_ch, num_classes),   # after stage2 (16x16)
            self._make_ds_head(ds_ch, num_classes),   # after stage1 (32x32)
        ])

    def _make_ds_head(self, in_ch, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, num_classes, 1),
        )

    def set_alpha(self, alpha):
        self.encoder.set_alpha(alpha)

    def forward(self, x, return_aux=True):
        """
        Args:
            x: [B, 28, 256, 256]
            return_aux: if True, return (main_logits, aux_logits_list)
        Returns:
            main_logits: [B, num_classes, 256, 256]
            aux_logits: list of [B, num_classes, 256, 256] (upsampled to match)
        """
        bottleneck, skips, H_list, W_list = self.encoder(x)

        feat, intermediates = self.decoder(bottleneck, skips, H_list, W_list,
                                           return_intermediates=True)
        # intermediates: [stage3_8x8, stage2_16x16, stage1_32x32]

        main_logits = self.classifier(feat)  # [B, 82, 256, 256]

        if return_aux:
            aux_logits_upsampled = []
            target_size = (256, 256)
            for i, aux_feat in enumerate(intermediates):
                aux_up = F.interpolate(self.ds_heads[i](aux_feat), size=target_size,
                                       mode='bilinear', align_corners=False)
                aux_logits_upsampled.append(aux_up)
            return main_logits, aux_logits_upsampled
        return main_logits
