import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None: 
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True  
        return self.early_stop
        
class ModelMetrics: 
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.class_names = ['Glioma_tumor', 'Healthy', 'Meningioma_tumor', 'Pituitary_tumor'][:num_classes]

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_probs: Optional[np.ndarray] = None) -> Dict:
        """Calculate accuracy, precision, recall, F1 score, and AUC"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }

        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name.lower()}_precision'] = precision[i] if i < len(precision) else 0.0
            metrics[f'{class_name.lower()}_recall'] = recall[i] if i < len(recall) else 0.0
            metrics[f'{class_name.lower()}_f1'] = f1[i] if i < len(f1) else 0.0
        
        # Medical-specific metrics
        if len(np.unique(y_true)) == 3: 
            cm = confusion_matrix(y_true, y_pred)
            for i, class_name in enumerate(self.class_names):
                if i < cm.shape[0]:
                    tp = cm[i, i]
                    fn = np.sum(cm[i, :]) - tp
                    fp = np.sum(cm[:, i]) - tp
                    tn = np.sum(cm) - (tp + fn + fp) 

                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                    metrics[f'{class_name.lower()}_sensitivity'] = sensitivity  
                    metrics[f'{class_name.lower()}_specificity'] = specificity
        
        # AUC-ROC if probabilities are provided
        if y_probs is not None:
            try:
                if self.num_classes == 2:
                    auc = roc_auc_score(y_true, y_probs[:, 1])
                    metrics['auc_roc'] = auc
                else:
                    auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
                    metrics['auc_roc_macro'] = auc
            except ValueError:
                logger.warning("Could not calculate AUC-ROC score")
        
        return metrics


class ModelTrainer: 
    """Main Training Engine for the model"""

    def __init__(self, model: nn.Module, device: torch.device, save_dir: str = 'checkpoints', log_dir: str = 'logs'):  # Fixed: was nn.modules
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.metrics_calculator = ModelMetrics() 
        self.writer = SummaryWriter(log_dir=log_dir)
        self.history = defaultdict(list)

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
    ) -> Dict:
        """Train for one epoch and return metrics (loss, acc, F1, AUC, etc.)."""

        self.model.train()
        running_loss = 0.0

        all_predictions = []
        all_labels = []
        all_probabilities = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, target)
            loss.backward()

            # gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # accumulate loss
            running_loss += loss.item()

            # predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.detach().cpu().numpy())

            # TensorBoard: per-batch loss
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar("Train/BatchLoss", loss.item(), global_step)

            # tqdm bar
            progress_bar.set_postfix({"Loss": loss.item()})

        # ---- aggregate epoch metrics ----
        avg_loss = running_loss / len(train_loader)

        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities),
        )
        metrics["loss"] = avg_loss  # add loss explicitly

        return metrics

    def validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        epoch: int,
    ) -> Dict:
        """Run validation for one epoch and return metrics (loss, acc, F1, AUC, etc.)."""

        self.model.eval()
        running_loss = 0.0

        all_predictions = []
        all_labels = []
        all_probabilities = []

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation")

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                loss = criterion(outputs, target)

                running_loss += loss.item()

                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.detach().cpu().numpy())

                progress_bar.set_postfix({"Loss": loss.item()})

        avg_loss = running_loss / len(val_loader)

        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities),
        )
        metrics["loss"] = avg_loss

        # TensorBoard: per-epoch val metrics
        self.writer.add_scalar("Val/Loss", metrics["loss"], epoch)
        self.writer.add_scalar("Val/Accuracy", metrics["accuracy"], epoch)
        if "f1_macro" in metrics:
            self.writer.add_scalar("Val/F1_macro", metrics["f1_macro"], epoch)
        if "auc_roc_macro" in metrics:
            self.writer.add_scalar("Val/AUC_ROC_macro", metrics["auc_roc_macro"], epoch)
        elif "auc_roc" in metrics:
            self.writer.add_scalar("Val/AUC_ROC", metrics["auc_roc"], epoch)

        return metrics


    # def train(self, train_loader: DataLoader, val_loader: DataLoader, 
    #       num_epochs: int, learning_rate: float = 0.001,
    #       weight_decay: float = 1e-4, class_weights: Optional[torch.Tensor] = None,
    #       use_scheduler: bool = True, patience: int = 5,
    #       criterion: Optional[nn.Module] = None,
    #       optimizer_name: str = 'adamw',
    #       scheduler_name: str = 'plateau') -> Dict:
        
    #     optimizer = get_optimizer(self.model, optimizer_name, learning_rate, weight_decay)

    #     if use_scheduler:
    #         scheduler = get_scheduler(optimizer, scheduler_name)

    #     #Loss Function
    #     if criterion is None:
    #         if class_weights is not None:
    #             criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
    #         else:
    #             criterion = nn.CrossEntropyLoss()

    #     best_val_loss = float('inf')
    #     best_model_state = None
    #     early_stopping = EarlyStopping(patience=patience, mode='min')

    #     logger.info(f"Starting training loop with optimizer={optimizer_name}, scheduler={scheduler_name if use_scheduler else 'None'}")

    #     total_loss = 0.0
    #     total_samples = 0
    #     correct_predictions = 0

    #     for epoch in range(num_epochs):
    #         self.model.train()
    #         epoch_loss = 0.0
    #         epoch_samples = 0
    #         epoch_correct = 0

    #         for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")):
    #             data, target = data.to(self.device), target.to(self.device)

    #             optimizer.zero_grad()
    #             outputs = self.model(data)
    #             loss = criterion(outputs, target)
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #             optimizer.step()

    #             epoch_loss += loss.item()
    #             epoch_samples += data.size(0)
    #             _, predicted = torch.max(outputs.data, 1)
    #             epoch_correct += (predicted == target).sum().item()

    #             global_step = epoch * len(train_loader) + batch_idx
    #             self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

    #         avg_epoch_loss = epoch_loss / len(train_loader)
    #         epoch_accuracy = epoch_correct / epoch_samples

    #         #Validation
    #         val_metrics = self.validate_epoch(val_loader, criterion, epoch)
    #         val_loss = val_metrics['loss']

    #         #Scheduler Step
    #         if use_scheduler:
    #             if scheduler_name.lower() == 'plateau':
    #                 scheduler.step(val_loss)
    #             else:
    #                 scheduler.step()

    #         logger.info(
    #             f"Epoch {epoch+1}/{num_epochs} | "
    #             f"Train Loss: {avg_epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f} | "
    #             f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f} | "
    #             f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    #         )

    #         self._log_epoch_metrics(epoch, {
    #             'loss': avg_epoch_loss,
    #             'accuracy': epoch_accuracy
    #         }, val_metrics)

    #         self.history['train_loss'].append(avg_epoch_loss)
    #         self.history['train_accuracy'].append(epoch_accuracy)
    #         self.history['val_loss'].append(val_loss)
    #         self.history['val_accuracy'].append(val_metrics['accuracy'])
    #         self.history['val_f1_macro'].append(val_metrics['f1_macro'])

    #         #Best Model Checkpoint
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             best_model_state = self.model.state_dict()
    #             self.save_checkpoint(epoch, {'loss': avg_epoch_loss, 'accuracy': epoch_accuracy}, val_metrics, is_best=True)
    #         elif early_stopping(val_loss):
    #             logger.info(f"Early stopping triggered at epoch {epoch+1}")
    #             break

    #         total_loss += epoch_loss
    #         total_samples += epoch_samples
    #         correct_predictions += epoch_correct

    #     if best_model_state is not None:
    #         self.model.load_state_dict(best_model_state)
    #         logger.info("Best model weights loaded.")

    #     logger.info("Training loop completed.")
    #     self.writer.close()

    #     self.plot_training_history(metrics=['loss'], save_path=os.path.join(self.save_dir, 'loss_curve.png'))
    #     self.plot_training_history(metrics=['accuracy'], save_path=os.path.join(self.save_dir, 'accuracy_curve.png'))

    #     return self.history

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        use_scheduler: bool = True,
        patience: int = 5,
        criterion: Optional[nn.Module] = None,
        optimizer_name: str = "adamw",
        scheduler_name: str = "plateau",
    ) -> Dict:
        """
        Full training loop:
        - uses train_epoch / validate_epoch (which compute AUC via y_probs)
        - tracks train/val loss, accuracy, F1, and AUC
        - early stopping on validation loss
        """

        # Optimizer & scheduler
        optimizer = get_optimizer(
            self.model,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = None
        if use_scheduler:
            scheduler = get_scheduler(optimizer, scheduler_name)

        # Loss function
        if criterion is None:
            if class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            else:
                criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_model_state = None
        early_stopping = EarlyStopping(patience=patience, mode="min")

        logger.info(
            f"Starting training loop with optimizer={optimizer_name}, "
            f"scheduler={scheduler_name if use_scheduler else 'None'}"
        )

        for epoch in range(num_epochs):
            # 1) TRAIN ONE EPOCH (returns metrics incl. AUC if possible) ----
            train_metrics = self.train_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
            )

            # 2) VALIDATION EPOCH (returns metrics incl. AUC if possible) ----
            val_metrics = self.validate_epoch(
                val_loader=val_loader,
                criterion=criterion,
                epoch=epoch,
            )

            # 3) Step scheduler ----
            if use_scheduler and scheduler is not None:
                if scheduler_name.lower() == "plateau":
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            # 4) Log to console (with LR) ----
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Also send all scalar metrics to TensorBoard
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)

            # 5) Update history (loss, acc, F1, AUC) ----
            # Train
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            if "f1_macro" in train_metrics:
                self.history["train_f1_macro"].append(train_metrics["f1_macro"])
            # Training AUC (binary or multi-class)
            if "auc_roc_macro" in train_metrics:
                self.history["train_auc_roc_macro"].append(train_metrics["auc_roc_macro"])
            elif "auc_roc" in train_metrics:
                self.history["train_auc_roc"].append(train_metrics["auc_roc"])

            # Val
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            if "f1_macro" in val_metrics:
                self.history["val_f1_macro"].append(val_metrics["f1_macro"])
            # Validation AUC
            if "auc_roc_macro" in val_metrics:
                self.history["val_auc_roc_macro"].append(val_metrics["auc_roc_macro"])
            elif "auc_roc" in val_metrics:
                self.history["val_auc_roc"].append(val_metrics["auc_roc"])

            # 6) Best model & early stopping (based on val loss) ----
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                # clone weights so they are not modified by later training
                best_model_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
                self.save_checkpoint(
                    epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    is_best=True,
                )
            elif early_stopping(val_metrics["loss"]):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # 7) Restore best model weights ----
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Best model weights loaded.")

        logger.info("Training loop completed.")
        self.writer.close()

        # 8) Plot curves (loss, acc, and AUC if available) ----
        self.plot_training_history(
            metrics=["loss"],
            save_path=os.path.join(self.save_dir, "loss_curve.png"),
        )
        self.plot_training_history(
            metrics=["accuracy"],
            save_path=os.path.join(self.save_dir, "accuracy_curve.png"),
        )

        # AUC curves (multi-class or binary)
        if (
            "train_auc_roc_macro" in self.history
            and "val_auc_roc_macro" in self.history
        ):
            self.plot_training_history(
                metrics=["auc_roc_macro"],
                save_path=os.path.join(self.save_dir, "auc_roc_macro_curve.png"),
            )
        elif "train_auc_roc" in self.history and "val_auc_roc" in self.history:
            self.plot_training_history(
                metrics=["auc_roc"],
                save_path=os.path.join(self.save_dir, "auc_roc_curve.png"),
            )

        return self.history



    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Comprehensive evaluation on test set"""
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc='Testing')
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().detach().numpy())
            
        # Calculate comprehensive metrics
        test_metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), 
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        self.plot_confusion_matrix(all_labels, all_predictions)
        
        self._generate_classification_report(test_metrics)
        
        return test_metrics
        

    def plot_confusion_matrix(self, y_true: List, y_pred: List, save_path: str = None):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.metrics_calculator.class_names,
                   yticklabels=self.metrics_calculator.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()

    def plot_training_history(self, metrics: List[str] = ['loss', 'accuracy'], save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in self.history and val_key in self.history:
                axes[i].plot(self.history[train_key], label=f'Train {metric}')
                axes[i].plot(self.history[val_key], label=f'Val {metric}')
                axes[i].set_title(f'{metric.capitalize()} History')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_history.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()


    def save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to tensorboard and console"""
        
        # Log to tensorboard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Log to console
        logger.info(f"Epoch {epoch+1}:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Val F1: {val_metrics['f1_macro']:.4f}")
        logger.info("val AUC-ROC: {:.4f}".format(val_metrics.get('auc_roc_macro', val_metrics.get('auc_roc', 0.0))))
    
    def _generate_classification_report(self, metrics: Dict):
        """Generate and save detailed classification report"""
        report = {
            'Overall Metrics': {
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Macro F1': f"{metrics['f1_macro']:.4f}",
                'Weighted F1': f"{metrics['f1_weighted']:.4f}",
            },
            'Per-Class Metrics': {}
        }
        
        for class_name in self.metrics_calculator.class_names:
            class_key = class_name.lower()
            report['Per-Class Metrics'][class_name] = {
                'Precision': f"{metrics.get(f'{class_key}_precision', 0):.4f}",
                'Recall': f"{metrics.get(f'{class_key}_recall', 0):.4f}",
                'F1-Score': f"{metrics.get(f'{class_key}_f1', 0):.4f}",
                'Sensitivity': f"{metrics.get(f'{class_key}_sensitivity', 0):.4f}",
                'Specificity': f"{metrics.get(f'{class_key}_specificity', 0):.4f}",
            }
        
        # Save report
        report_path = os.path.join(self.save_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Classification report saved: {report_path}")
        
        # Print summary
        logger.info("=== CLASSIFICATION REPORT ===")
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
        
        for class_name in self.metrics_calculator.class_names:
            class_key = class_name.lower()
            logger.info(f"{class_name}: F1={metrics.get(f'{class_key}_f1', 0):.4f}, "
                       f"Precision={metrics.get(f'{class_key}_precision', 0):.4f}, "
                       f"Recall={metrics.get(f'{class_key}_recall', 0):.4f}")
            

def get_optimizer(model: nn.Module, optimizer_name: str = 'adamw',
                 learning_rate: float = 0.001, weight_decay: float = 1e-4) -> optim.Optimizer:
    """Get optimizer for training"""
    if optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str = 'plateau'):
    """Get learning rate scheduler"""
    if scheduler_name.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    elif scheduler_name.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
