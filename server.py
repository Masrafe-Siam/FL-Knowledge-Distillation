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
    # dynamic LR/epoch logic
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

        # History, best_f1, etc. definitions
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

    def _save_strategy_config(self):
        config = { "strategy": "MedicalFLStrategy", "model_name": self.model_name, "num_classes": self.num_classes, }
        # save config
        with open(os.path.join(self.results_base_dir, "strategy_config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
    def configure_fit(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager):
        # configure_fit logic
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        fit_ins = fl.common.FitIns(parameters, config)
        return [(c, fit_ins) for c in clients]
    
    def configure_evaluate(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager):
        # configure_evaluate logic
        if self.fraction_evaluate == 0.0: return []
        config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        sample_size, min_num = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        eval_ins = fl.common.EvaluateIns(parameters, config)
        return [(c, eval_ins) for c in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]):
        # aggregate_fit logic
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None: return None, {}
        self.last_parameters = aggregated_parameters
        summary = self._calculate_fit_metrics(results)
        # history appending
        self.history["round"].append(server_round)
        self.history["train_loss"].append(summary["train_loss_avg"])
        self.history["val_loss"].append(summary["val_loss_avg"])
        self.history["val_f1"].append(summary["val_f1_avg"])
        self.history["xai_del_auc_mean"].append(summary.get("xai_del_auc_mean_avg", np.nan))
        # best model check
        if summary["val_f1_avg"] > self.best_f1:
            self.best_f1 = summary["val_f1_avg"]
            self.best_round = server_round
            self.best_parameters = aggregated_parameters
            self.save_best_model()
            logger.info(f"🏆 New best model: round={self.best_round}, val_f1={self.best_f1:.4f}")
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]], failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]):
        # aggregate_evaluate logic
        if not results: return None, {}
        test = self._calculate_eval_metrics(results)
        return test["test_loss_avg"], test

    def _calculate_fit_metrics(self, results):
        # calculate fit logic
        total_examples = sum(fit_res.num_examples for _, fit_res in results) or 1
        metric_keys = ["train_loss", "train_accuracy", "train_f1", "val_loss", "val_accuracy", "val_f1",
                       "xai_del_auc_mean", "xai_del_auc_std"]
        weighted_sums = {key: 0.0 for key in metric_keys}
        for _, fit_res in results:
            weight = fit_res.num_examples / total_examples
            for key in metric_keys:
                val = fit_res.metrics.get(key, 0.0)
                if isinstance(val, (int, float, np.integer, np.floating)):
                    weighted_sums[key] += float(val) * weight
        return {
            "train_loss_avg": weighted_sums["train_loss"], "val_loss_avg": weighted_sums["val_loss"],
            "val_f1_avg": weighted_sums["val_f1"], "xai_del_auc_mean_avg": weighted_sums["xai_del_auc_mean"],
        } # abbreviated for clarity
    
    def _calculate_eval_metrics(self, results):
        # calculate eval logic
        total_examples = sum(eval_res.num_examples for _, eval_res in results) or 1
        weighted_loss = sum(float(eval_res.loss or 0.0) * (eval_res.num_examples / total_examples) for _, eval_res in results)
        weighted_f1 = sum(float(eval_res.metrics.get("f1_macro", 0.0)) * (eval_res.num_examples / total_examples) for _, eval_res in results)
        return {"test_loss_avg": weighted_loss, "test_f1_avg": weighted_f1} # (abbreviated)

    def save_best_model(self):
        # save_best_model logic
        if self.best_parameters is None: return
        try:
            model = get_model(self.model_name, num_classes=self.num_classes, pretrained=False)
            best_state_dict = OrderedDict()
            for (name, param), arr in zip(model.state_dict().items(), fl.common.parameters_to_ndarrays(self.best_parameters)):
                best_state_dict[name] = torch.as_tensor(arr, dtype=param.dtype)
            save_path = os.path.join(self.results_base_dir, f"best_model_round_{self.best_round}.pth")
            torch.save({"model_state_dict": best_state_dict}, save_path)
            logger.info(f"Best model saved successfully → {save_path}")
        except Exception as exc: logger.error(f"Failed to save best model: {exc}", exc_info=True)

    def save_last_model(self):
        # save_last_model logic
        if self.last_parameters is None: return
        try:
            model = get_model(self.model_name, num_classes=self.num_classes, pretrained=False)
            last_state_dict = OrderedDict()
            for (name, param), arr in zip(model.state_dict().items(), fl.common.parameters_to_ndarrays(self.last_parameters)):
                last_state_dict[name] = torch.as_tensor(arr, dtype=param.dtype)
            save_path = os.path.join(self.results_base_dir, "last_global_model.pth")
            torch.save({"model_state_dict": last_state_dict}, save_path)
            logger.info(f"Last global model saved successfully → {save_path}")
        except Exception as exc: logger.error(f"Failed to save last model: {exc}", exc_info=True)

    def save_final_results(self):
        # save_final_results logic
        try:
            with open(os.path.join(self.results_base_dir, "final_training_history.json"), "w") as f:
                json.dump(self.history, f, indent=2)
            self.plot_training_curves(save_suffix="_final")
        except Exception as e: logger.error(f"Failed to save final results: {e}", exc_info=True)

    def plot_training_curves(self, save_suffix: str = ""):
        # plot_training_curves logic
        if not self.history["round"]: return
        # (plotting logic...)
        plt.figure()
        plt.plot(self.history["round"], self.history["val_f1"], label="Val F1")
        plt.savefig(os.path.join(self.results_base_dir, f"f1_curve{save_suffix}.png"))
        plt.close()


# LoggingClientManager, start_waiting_heartbeat, create_server_strategy
class LoggingClientManager(SimpleClientManager):
    def register(self, client: ClientProxy) -> bool:
        ok = super().register(client)
        logger.info(f"Client connected: {client.cid} | Total={self.num_available()}")
        return ok

def start_waiting_heartbeat(cm: SimpleClientManager, target: int, interval_sec: float = 2.0):

    pass

def create_server_strategy(*, min_clients: int, fraction_fit: float, fraction_evaluate: float,
                           model_name: str, num_classes: int, local_epochs: int) -> MedicalFLStrategy:
    initial_parameters = get_init_parameters(model_name, num_classes)
    if initial_parameters is None:
        raise RuntimeError("Failed to initialize model parameters")
    return MedicalFLStrategy(
        model_name=model_name, num_classes=num_classes,
        fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_clients, min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=lambda r: fit_config(r, local_epochs),
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )


def main():
    parser = argparse.ArgumentParser("Federated Learning Server")
    # Standard FL Args 
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="Port")
    parser.add_argument("--rounds", type=int, default=3, help="FL rounds")
    parser.add_argument("--min-clients", type=int, default=1, help="Minimum clients per round")
    parser.add_argument("--fraction-fit", type=float, default=1.0)
    parser.add_argument("--fraction-evaluate", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="densenet121", help="Teacher model architecture")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes") # <-- CHANGED
    parser.add_argument("--local-epochs", type=int, default=6)
    
    # Distillation Pipeline Args 
    parser.add_argument("--run-distillation", action="store_true", 
                        help="Automatically run distillation after FL training completes")
    parser.add_argument("--student-model", type=str, default="mobilenetv3", 
                        help="Student model architecture for distillation")
    parser.add_argument("--distill-data-dir", type=str, default="Dataset", 
                        help="Path to the FULL dataset for distillation training")
    parser.add_argument("--distill-save-dir", type=str, default="distillation/saved_models", 
                        help="Where to save final student models")
    parser.add_argument("--distill-epochs", type=int, default=50, help="Epochs for distillation training")
    parser.add_argument("--distill-batch-size", type=int, default=32, help="Batch size for distillation")

    args = parser.parse_args()

    # (Logging setup...)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", force=True)
    logger.info("Starting Federated Learning Server (Teacher Training)")
    logger.info(f"Config: {vars(args)}")

    strategy = None # Define strategy in outer scope
    try:
        strategy = create_server_strategy(
            min_clients=args.min_clients,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            model_name=args.model,
            num_classes=args.num_classes,
            local_epochs=args.local_epochs,
        )
        client_manager = LoggingClientManager()
        server_cfg = fl.server.ServerConfig(num_rounds=args.rounds)
        server_addr = f"{args.host}:{args.port}"

        logger.info(f"Flower gRPC server starting at {server_addr} for {args.rounds} rounds...")
        
        fl.server.start_server(
            server_address=server_addr,
            config=server_cfg,
            strategy=strategy,
            client_manager=client_manager,
            grpc_max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"FL Server failed: {e}", exc_info=True)
    finally:
        # --- MODIFIED: Dynamic Distillation Trigger ---
        if strategy is None or not strategy.history["round"]:
             logger.warning("FL training did not complete. No results to save or distill.")
             return # Exit if FL failed

        try:
            # 1. Save all FL results first
            strategy.save_final_results()
            strategy.save_last_model()
            logger.info("\nFL training complete. Final results saved.")
            logger.info(f"Results: {strategy.results_base_dir}")
            logger.info(f"Best round (Teacher): {strategy.best_round} | Best Val F1: {strategy.best_f1:.4f}")

            # 2. Check if distillation is requested
            if args.run_distillation:
                logger.info("=" * 80)
                logger.info("🚀 STARTING DYNAMIC KNOWLEDGE DISTILLATION 🚀")
                logger.info("=" * 80)

                # 3. Find the best saved teacher model
                teacher_path = os.path.join(
                    strategy.results_base_dir,
                    f"best_model_round_{strategy.best_round}.pth"
                )
                if not os.path.exists(teacher_path) or strategy.best_round == 0:
                    # Fallback to last model if best wasn't saved
                    teacher_path = os.path.join(strategy.results_base_dir, "last_global_model.pth")

                if not os.path.exists(teacher_path):
                    logger.error("Could not find a trained teacher model to distill from. Skipping.")
                    return # Exit

                logger.info(f"Using Teacher Model: {args.model} from {teacher_path}")
                logger.info(f"Training Student Model: {args.student_model}")

                # 4. Prepare the command to call distillation/distill.py
                # Using sys.executable ensures we use the same Python (e.g., from venv)
                cmd = [
                    sys.executable,
                    "distillation/distill.py",
                    "--teacher-model", args.model,
                    "--teacher-path", teacher_path,
                    "--student-model", args.student_model,
                    "--data-dir", args.distill_data_dir,
                    "--save-dir", args.distill_save_dir,
                    "--num-classes", str(args.num_classes),
                    "--epochs", str(args.distill_epochs),
                    "--batch-size", str(args.distill_batch_size),
                ]
                
                logger.info(f"Running command: {' '.join(cmd)}")

                # 5. Run the distillation script as a subprocess
                # We stream the output directly to the console instead of capturing
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
                
                # Log output line by line as it comes
                for line in iter(process.stdout.readline, ''):
                    logger.info(f"[Distill] {line.strip()}")
                
                process.wait() # Wait for it to finish

                if process.returncode == 0:
                    logger.info("✅ Knowledge Distillation completed successfully.")
                else:
                    logger.error(f"❌ Knowledge Distillation FAILED with return code {process.returncode}.")

        except Exception as e:
            logger.error(f"Error during shutdown/distillation: {e}", exc_info=True)