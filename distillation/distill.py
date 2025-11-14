import os
import argparse
import logging
import torch
import torch.optim as optim
import torch.nn as nn

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.modelEngine import get_model
from utils.dataloder import create_data_loaders
from distillation.loss import DistillationLoss
from distillation.train_eval import train_kd

# Root folder for all KD results
RESULTS_ROOT = os.path.abspath(os.path.join("Result", "Distillation"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("Distillation")


def load_checkpoint_flex(model: nn.Module, path: str, device: torch.device):
    """Load a checkpoint with some flexibility about key names / wrapping."""
    sd = torch.load(path, map_location=device)

    # If this is a wrapper dict, try common keys
    if isinstance(sd, dict):
        for k in ["model_state_dict", "state_dict", "weights", "model"]:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
        # Keep only tensor entries if it looks like a bigger dict
        if not all(isinstance(v, torch.Tensor) for v in sd.values()):
            sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)} or sd

    # Strip common prefixes like "module." or "model."
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("model."):
            k = k[6:]
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading teacher: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading teacher: {unexpected}")


def run_distillation(args):
    """Main distillation training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    teacher_student_name = f"{args.teacher_model}_{args.student_model}"
    RESULTS_BASE_DIR = os.path.join(RESULTS_ROOT, teacher_student_name)
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    logger.info(f"Results (models + metrics) will be saved under: {RESULTS_BASE_DIR}")

    # 2. Load data
    train_loader, val_loader, _ = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        num_workers=4,
    )
    logger.info(
        f"Data loaded: {len(train_loader.dataset)} training samples from {args.data_dir}"
    )

    # 3. Load teacher & student
    teacher_model = get_model(
        args.teacher_model,
        args.num_classes,
        pretrained=False,
    ).to(device)
    load_checkpoint_flex(teacher_model, args.teacher_path, device)
    teacher_model.eval()
    logger.info(
        f"Teacher model ({args.teacher_model}) loaded from {args.teacher_path}"
    )

    # Student (trainable)
    student_model = get_model(
        args.student_model,
        args.num_classes,
        pretrained=True,
    ).to(device)
    student_model.train()
    logger.info(f"Student model ({args.student_model}) created for distillation.")

    # 4. Setup KD training
    criterion = DistillationLoss(alpha=args.alpha, temperature=args.temperature)
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    teacher_name_for_logging = args.teacher_model  # short name for run_name

    # 5. Train with KD + metrics
    history, best_model_path, best_cm = train_kd(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_dir=RESULTS_BASE_DIR, 
        student_model_name=args.student_model,
        teacher_name=teacher_name_for_logging,
        num_classes=args.num_classes,
    )

    logger.info("Distillation training complete.")
    if best_model_path is not None:
        logger.info(f"Best student model saved at: {best_model_path}")
    else:
        logger.warning("Training finished, but no best model was saved (check logs).")

    if best_cm is not None:
        logger.info(f"Best validation confusion matrix:\n{best_cm}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training Script")
    parser.add_argument("--teacher-model", type=str, required=True, help="Teacher model name (e.g., densenet121)")
    parser.add_argument("--teacher-path", type=str, required=True, help="Path to the trained teacher model .pth file")
    parser.add_argument("--student-model", type=str, required=True, help="Student model name (e.g., mobilenetv3)")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the training/validation dataset")
    parser.add_argument("--save-dir", type=str, default="Result/Distillation", help="Directory to save student models")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes (e.g., 4 for brain tumor)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for hard label loss")
    parser.add_argument("--temperature", type=float, default=6.0, help="Temperature for softmax")
    
    args = parser.parse_args()
    run_distillation(args)