import torch
import torch.nn as nn

class LocalPerceptionHead(nn.Module):
    def __init__(self, in_channels, num_classes, spatial_size=4, hidden_mult=2, dropout=0.1):
        super().__init__()
        hidden_channels = int(in_channels * hidden_mult)
        self.pool = nn.AdaptiveAvgPool2d(spatial_size)          # reduce spatial
        self.conv = nn.Conv2d(in_channels, in_channels, 
                              kernel_size=3, padding=1, 
                              groups=in_channels)               # depthwise conv
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.pw_bn = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.flat_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.act(self.dw_bn(self.conv(x)))
        x = self.dropout(self.act(self.pw_bn(self.pointwise(x))))
        x = self.flat_pool(x).flatten(1)
        return self.fc(x)

class GlobalAggregationHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)                     # global spatial average
        self.proj = nn.Linear(in_channels, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (B, C, H, W) from MobileViT block output
        x = self.pool(x).flatten(1)                             # (B, C)
        x = self.norm(x)
        x = self.act(self.proj(x))
        x = self.dropout(x)
        return self.fc(x)
