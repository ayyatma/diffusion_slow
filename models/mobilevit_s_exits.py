import torch
import torch.nn as nn
import timm
from .exit_heads import LocalPerceptionHead, GlobalAggregationHead


class MobileViTSWithExits(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, exit_head_config=None):
        super().__init__()
        exit_head_config = exit_head_config or {}
        lph_hidden_mult = exit_head_config.get("lph_hidden_mult", 2)
        lph_dropout = exit_head_config.get("lph_dropout", 0.1)
        gah_hidden_dim = exit_head_config.get("gah_hidden_dim", 256)
        gah_dropout = exit_head_config.get("gah_dropout", 0.1)

        self.backbone = timm.create_model(
            "mobilevit_s",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        stage_channels = [32, 64, 96, 128, 160]

        self.exits = nn.ModuleList([
            LocalPerceptionHead(
                in_channels=stage_channels[0],
                num_classes=num_classes,
                hidden_mult=lph_hidden_mult,
                dropout=lph_dropout,
            ),
            LocalPerceptionHead(
                in_channels=stage_channels[1],
                num_classes=num_classes,
                hidden_mult=lph_hidden_mult,
                dropout=lph_dropout,
            ),
            GlobalAggregationHead(
                in_channels=stage_channels[2],
                num_classes=num_classes,
                hidden_dim=gah_hidden_dim,
                dropout=gah_dropout,
            ),
            GlobalAggregationHead(
                in_channels=stage_channels[3],
                num_classes=num_classes,
                hidden_dim=gah_hidden_dim,
                dropout=gah_dropout,
            ),
            GlobalAggregationHead(
                in_channels=stage_channels[4],
                num_classes=num_classes,
                hidden_dim=gah_hidden_dim,
                dropout=gah_dropout,
            ),
        ])

    def forward(self, x):
        """
        Forward pass returning all exit logits.
        """
        return self.get_all_exit_logits(x)

    def get_intermediate_features(self, x):
        """
        Returns stage outputs before MobileViT's final 1x1 conv.
        """
        features = []
        x = self.backbone.stem(x)
        for stage in self.backbone.stages:
            x = stage(x)
            features.append(x)
        return features

    def get_all_exit_logits(self, x):
        """
        Returns list of logits from early exits + final head.
        """
        features = []
        x = self.backbone.stem(x)
        for stage in self.backbone.stages:
            x = stage(x)
            features.append(x)

        logits = []
        for i, feat in enumerate(features):
            logits.append(self.exits[i](feat))

        final_features = self.backbone.final_conv(features[-1])
        logits.append(self.backbone.forward_head(final_features))
        return logits
