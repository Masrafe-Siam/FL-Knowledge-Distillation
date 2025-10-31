import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import argparse
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import time
import json
from datetime import datetime
import threading
import sys
import subprocess #For running the distillation script

import numpy as np
import torch
import flwr as fl
from flwr.server.client_manager import SimpleClientManager, ClientProxy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.modelEngine import get_model

RESULTS_BASE_DIR = os.path.abspath("Result/FLResult")
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
logger = logging.getLogger("FL-Server")
GRPC_MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024

def get_init_parameters(model_name: str, num_classes: int) -> fl.common.Parameters:
    try:
        model = get_model(model_name, num_classes, pretrained=False)
        with torch.no_grad():
            parameters = [v.cpu().numpy() for _, v in model.state_dict().items()]
        logger.info(f"Initial model parameters loaded for: {model_name} (Classes: {num_classes})")
        return fl.common.ndarrays_to_parameters(parameters)
    except Exception as e:
        logger.error(f"Failed to get initial parameters: {e}", exc_info=True)
        return None

def fit_config(server_round: int, local_epochs: int) -> Dict[str, fl.common.Scalar]:
    config = { "server_round": server_round, "local_epochs": local_epochs, "learning_rate": 1e-3, }
    # (Your dynamic LR/epoch logic...)
    return config

def evaluate_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    return {"server_round": server_round}

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    if not metrics: return {}
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0: return {}
    aggregated_metrics: Dict[str, float] = {}
    for num_samples, client_metrics in metrics:
        weight = num_samples / total_samples
        for key, value in client_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + weight * float(value)
    return aggregated_metrics

class MedicalFLStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Optional[fl.common.Parameters] = None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        results_base_dir: str = RESULTS_BASE_DIR,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.model_name = model_name
        self.num_classes = num_classes

        # (History, best_f1, etc. definitions...)
        self.history = { "round": [], "train_loss": [], "train_accuracy": [], "train_f1": [],
                         "val_loss": [], "val_accuracy": [], "val_f1": [],
                         "test_loss": [], "test_accuracy": [], "test_f1": [],
                         "xai_del_auc_mean": [], "xai_del_auc_std": [], }
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        self.best_round = 0
        self.best_parameters: Optional[fl.common.Parameters] = None
        self.last_parameters: Optional[fl.common.Parameters] = None
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_base_dir = os.path.join(results_base_dir, f"fl_results_{model_name}_{ts}")
        os.makedirs(self.results_base_dir, exist_ok=True)
        self._save_strategy_config()
        logger.info(f"FL Strategy initialized. Results will be in: {self.results_base_dir}")

