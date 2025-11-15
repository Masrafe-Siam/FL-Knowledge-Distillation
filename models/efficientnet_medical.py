import torch
import torch.nn as nn
from torchvision import models


class EfficientNetB3Medical(nn.Module):
    """
    EfficientNet-B3 adapted for 1-channel medical images.
    - Loads ImageNet weights (optional).
    - Converts first conv to accept 1-channel by averaging RGB weights.
    - Replaces classifier with an MLP head similar to your MobileNetV3 head.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.5):
        super(EfficientNetB3Medical, self).__init__()

        # Backbone
        if pretrained:
            self.backbone = models.efficientnet_b3(
                weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.efficientnet_b3(weights=None)

        # Adapt first conv to 1-channel (from 3-channel)
        first_conv = self.backbone.features[0][0]  # Conv2d in Conv2dNormActivation

        # New conv with 1 input channel
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        if pretrained:
            with torch.no_grad():
                # Average over RGB channels -> grayscale
                new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                if first_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)

        self.backbone.features[0][0] = new_conv

        # Replace classifier with custom MLP head
        # Original: classifier = Sequential(Dropout, Linear)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # so backbone(x) returns features

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W] (grayscale) expected.
        """
        features = self.backbone(x)         # [B, num_features]
        logits = self.classifier(features)  # [B, num_classes]
        return logits

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return backbone features before the MLP head.
        Useful for distillation / feature visualization.
        """
        return self.backbone(x)


class EfficientNetB4Medical(nn.Module):
    """
    EfficientNet-B4 adapted for 1-channel medical images.
    Same pattern as EfficientNetB3Medical but with B4 backbone.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.5):
        super(EfficientNetB4Medical, self).__init__()

        # Backbone
        if pretrained:
            self.backbone = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.efficientnet_b4(weights=None)

        # Adapt first conv to 1-channel
        first_conv = self.backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                if first_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)

        self.backbone.features[0][0] = new_conv

        # Custom MLP head
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
