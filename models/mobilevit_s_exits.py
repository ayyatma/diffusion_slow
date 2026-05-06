import torch
import torch.nn as nn
import timm
from .exit_heads import LocalPerceptionHead, GlobalAggregationHead


class MobileViTSWithExits(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilevit_s",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        stage_channels = [32, 64, 96, 128, 160]

        self.exits = nn.ModuleList([
            LocalPerceptionHead(in_channels=stage_channels[0], num_classes=num_classes),
            LocalPerceptionHead(in_channels=stage_channels[1], num_classes=num_classes),
            GlobalAggregationHead(in_channels=stage_channels[2], num_classes=num_classes),
            GlobalAggregationHead(in_channels=stage_channels[3], num_classes=num_classes),
            GlobalAggregationHead(in_channels=stage_channels[4], num_classes=num_classes),
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
