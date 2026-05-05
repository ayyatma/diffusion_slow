import torch
import torch.nn as nn
import timm
from .exit_heads import LocalPerceptionHead, GlobalAggregationHead

class MobileViTSWithExits(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.feature_extractor = timm.create_model(
            'mobilevit_s',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        self.backbone = self.feature_extractor
        feature_channels = self.feature_extractor.feature_info.channels()

        self.exits = nn.ModuleList([
            LocalPerceptionHead(in_channels=feature_channels[0], num_classes=num_classes),
            LocalPerceptionHead(in_channels=feature_channels[1], num_classes=num_classes),
            GlobalAggregationHead(in_channels=feature_channels[2], num_classes=num_classes),
            GlobalAggregationHead(in_channels=feature_channels[3], num_classes=num_classes),
            GlobalAggregationHead(in_channels=feature_channels[4], num_classes=num_classes),
        ])

        self.final_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(feature_channels[-1], num_classes),
        )

    def forward(self, x):
        """
        Forward pass returning all exit logits.
        """
        return self.get_all_exit_logits(x)

    def get_intermediate_features(self, x):
        """
        Returns list of 5 intermediate feature maps from the backbone.
        """
        return self.feature_extractor(x)

    def get_all_exit_logits(self, x):
        """
        Returns list of logits from early exits + final head.
        """
        features = self.get_intermediate_features(x)
        logits = []
        for i, feat in enumerate(features):
            logits.append(self.exits[i](feat))
        logits.append(self.final_head(features[-1]))
        return logits
