import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HybridViTCNNMLP import HybridViTCNNMLP


def get_model(model_name: str, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.5):
    if model_name == 'HVTCMLP': #HybridViTCNNMLP
        model = HybridViTCNNMLP(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbones=False
        )
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    
    return model