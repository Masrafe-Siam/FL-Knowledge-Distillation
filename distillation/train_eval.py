import os
import logging
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt


logger = logging.getLogger("KDTrainEval")
RESULTS_BASE_DIR = os.path.abspath("Result/Distillation")

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train_one_epoch_kd(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> Tuple[float, float]:
    """
    Train student for one epoch using knowledge distillation.
    Returns:
        avg_train_loss, train_accuracy
    """
    student_model.train()
    teacher_model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Training]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(images)

        student_logits = student_model(images)
        loss = criterion(student_logits, teacher_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # accuracy
        _, preds = torch.max(student_logits, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader)
    acc = 100.0 * correct / max(total, 1)

    return avg_loss, acc


def evaluate(
    student_model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
    epoch: int,
    num_epochs: int,
    split_name: str = "Val",
):
    """
    Evaluate student model on a data loader.
    Returns:
        avg_loss, accuracy, auc, y_true, y_pred
    """
    student_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_labels: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs} [{split_name}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = student_model(images)

            loss = F.cross_entropy(logits, labels, reduction="sum")
            val_loss += loss.item()

            # probabilities for AUC
            probs = F.softmax(logits, dim=1)

            _, preds = torch.max(logits, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())

    avg_loss = val_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)

    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)
    all_preds_tensor = torch.cat(all_preds, dim=0)

    # Compute multi-class AUC (macro, one-vs-rest)
    try:
        auc = roc_auc_score(
            all_labels_tensor.numpy(),
            all_probs_tensor.numpy(),
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        # This can happen if some classes are missing in val set for a given epoch
        auc = float("nan")
        logger.warning(
            "AUC could not be computed for this epoch (likely missing classes in this split)."
        )

    return avg_loss, acc, auc, all_labels_tensor, all_preds_tensor

def plot_confusion_matrix(
    cm,
    class_names,
    save_path: str,
    normalize: bool = True,
    title: str = "Confusion Matrix",
):
    """
    Save confusion matrix as a PNG heatmap.
    cm: 2D array (num_classes x num_classes)
    class_names: list of class labels (strings)
    """
    import numpy as np

    if normalize:
        cm = cm.astype("float")
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 
        cm = cm / row_sums

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # annotate cells
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix PNG to {save_path}")


def plot_curves(
    history: Dict[str, List[float]],
    save_dir: str,
    run_name: str,
) -> None:
    _ensure_dir(save_dir)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train vs Val Loss ({run_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    loss_path = os.path.join(save_dir, f"{run_name}_loss_curve.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved loss curve to {loss_path}")

    # Accuracy curves
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Train vs Val Accuracy ({run_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    acc_path = os.path.join(save_dir, f"{run_name}_acc_curve.png")
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved accuracy curve to {acc_path}")

    # AUC curve (validation only)
    if "val_auc" in history:
        plt.figure()
        plt.plot(epochs, history["val_auc"], label="Val AUC (macro)")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"Validation AUC ({run_name})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        auc_path = os.path.join(save_dir, f"{run_name}_auc_curve.png")
        plt.savefig(auc_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved AUC curve to {auc_path}")


def train_kd(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    save_dir: str,
    student_model_name: str,
    teacher_name: str,
    num_classes: int,
):
    """
    Full KD training loop with metrics, curves, confusion matrix, and AUC.
    Returns:
        history dict, best_model_path, confusion_matrix (for best epoch)
    """
    _ensure_dir(save_dir)

    run_name = f"student_{student_model_name}_from_teacher_{teacher_name}"

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_auc": [],
    }

    best_val_loss = float("inf")
    best_model_path = None
    best_cm = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch_kd(
            student_model,
            teacher_model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            num_epochs,
        )

        val_loss, val_acc, val_auc, y_true, y_pred = evaluate(
            student_model,
            val_loader,
            device,
            num_classes=num_classes,
            epoch=epoch,
            num_epochs=num_epochs,
            split_name="Val",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # Confusion matrix for this epoch
        cm = confusion_matrix(
            y_true.numpy(), y_pred.numpy(), labels=list(range(num_classes))
        )

        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val AUC: {val_auc:.4f}"
        )
        logger.info(f"Validation Confusion Matrix (epoch {epoch}):\n{cm}")

        # Scheduler (ReduceLROnPlateau expects a metric to minimize, here val_loss)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, f"{run_name}.pth")
            torch.save(student_model.state_dict(), best_model_path)
            best_cm = cm
            logger.info(f"*** New best student model saved to {best_model_path} ***")

        # After all epochs, plot curves and save confusion matrix
    metrics_dir = os.path.join(save_dir, "metrics")
    _ensure_dir(metrics_dir)
    plot_curves(history, metrics_dir, run_name)

    if best_cm is not None:
        # Save raw confusion matrix as text
        cm_txt_path = os.path.join(metrics_dir, f"{run_name}_confusion_matrix.txt")
        with open(cm_txt_path, "w") as f:
            f.write(str(best_cm))
        logger.info(f"Saved best confusion matrix to {cm_txt_path}")

        # Save normalized confusion matrix as PNG heatmap
        class_names = [str(i) for i in range(num_classes)]  # or pass real labels
        cm_png_path = os.path.join(metrics_dir, f"{run_name}_confusion_matrix.png")
        plot_confusion_matrix(
            best_cm,
            class_names=class_names,
            save_path=cm_png_path,
            normalize=True,
            title=f"Confusion Matrix ({run_name})",
        )

    return history, best_model_path, best_cm

