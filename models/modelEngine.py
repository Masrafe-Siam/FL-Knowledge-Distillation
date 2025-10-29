import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HybridViTCNNMLP import HybridViTCNNMLP
from models.cnn import CustomCNN
from models.mobilenetv3 import MobileNetV3


def get_model(model_name: str, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.5):
    if model_name == 'HVTCMLP': #HybridViTCNNMLP
        model = HybridViTCNNMLP(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbones=False
        )
    elif model_name == 'cnn':
        model = CustomCNN(num_classes=num_classes)
    elif model_name == 'mobilenetv3':
        model = MobileNetV3(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )

    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    
    return model