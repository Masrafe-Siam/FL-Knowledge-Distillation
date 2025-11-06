# import torch
# import torch.optim as optim
# from tqdm import tqdm

# from models.modelEngine import get_model
# from utils.dataloder import create_data_loaders
# from distillation.loss import DistillationLoss
# # from dataloder import create_data_loaders # (If you keep the typo)
# from xai_utils import load_checkpoint_flex # (Re-use your robust loader from the XAI script)

# # --- CONFIGURATION ---
# TEACHER_MODEL_NAME = "densenet121"
# TEACHER_MODEL_PATH = "Result/FLResult/.../best_global_model.pth" # <-- Path to your FL model
# STUDENT_MODEL_NAME = "mobilenetv3"
# NUM_CLASSES = 4
# DATA_DIR = "path/to/your/training_data" # This script needs data
# BATCH_SIZE = 32
# EPOCHS = 50
# LEARNING_RATE = 1e-3

# # (Paste the DistillationLoss class definition here or import it)
# # class DistillationLoss(nn.Module):
# #    ...

# # --- SETUP ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1. Load Data
# # You can use your existing dataloader. This script trains on the full dataset.
# train_loader, val_loader, _ = create_data_loaders(
#     data_dir=DATA_DIR,
#     batch_size=BATCH_SIZE,
#     train_split=0.8,
#     val_split=0.2, # No test set needed for distillation, just val
#     test_split=0.0
# )
# print(f"Data loaded: {len(train_loader.dataset)} training samples.")

# # 2. Load Models
# # Load Teacher and set to evaluation mode (no gradients)
# teacher_model = get_model(TEACHER_MODEL_NAME, NUM_CLASSES, pretrained=False).to(device)
# load_checkpoint_flex(teacher_model, TEACHER_MODEL_PATH, device)
# teacher_model.eval()
# print(f"Teacher model ({TEACHER_MODEL_NAME}) loaded.")

# # Load Student and set to training mode
# student_model = get_model(STUDENT_MODEL_NAME, NUM_CLASSES, pretrained=True).to(device) # Can use pretraining
# student_model.train()
# print(f"Student model ({STUDENT_MODEL_NAME}) loaded for training.")

# # 3. Setup Training
# criterion = DistillationLoss(alpha=0.1, temperature=6.0)
# optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# # --- TRAINING LOOP ---
# best_val_loss = float('inf')
# for epoch in range(EPOCHS):
#     student_model.train()
#     running_loss = 0.0
    
#     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
#     for images, labels in pbar:
#         images, labels = images.to(device), labels.to(device)

#         # Get Teacher predictions (no gradients needed)
#         with torch.no_grad():
#             teacher_logits = teacher_model(images)
            
#         # Get Student predictions (gradients ARE needed)
#         student_logits = student_model(images)
        
#         # Calculate distillation loss
#         loss = criterion(student_logits, teacher_logits, labels)
        
#         # Standard backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         pbar.set_postfix({"loss": loss.item()})
        
#     avg_train_loss = running_loss / len(train_loader)
    
#     # --- VALIDATION LOOP ---
#     student_model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = student_model(images)
#             # Validate on standard CE loss
#             val_loss += F.cross_entropy(outputs, labels, reduction='sum').item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     avg_val_loss = val_loss / len(val_loader.dataset)
#     val_accuracy = 100 * correct / total
    
#     print(f"Epoch {epoch+1} Summary: \n"
#           f"  Train Loss: {avg_train_loss:.4f} \n"
#           f"  Val Loss:   {avg_val_loss:.4f} \n"
#           f"  Val Acc:    {val_accuracy:.2f}% \n")
    
#     scheduler.step(avg_val_loss)
    
#     # Save the best student model
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         torch.save(student_model.state_dict(), f"{STUDENT_MODEL_NAME}_distilled.pth")
#         print(f"*** New best student model saved to {STUDENT_MODEL_NAME}_distilled.pth ***")

# print("Distillation training complete.")


# E:\Python\Research\KnowledgeDistilation\FL-Knowledge-Distillation\distillation\distill.py

import os
import argparse
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

# Adjust imports based on your project structure
# We assume 'models' and 'utils' are in the parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.modelEngine import get_model
from utils.dataloder import create_data_loaders # (Make sure your dataloder.py is updated for 4 classes!)
from distillation.loss import DistillationLoss
# from utils.xai_utils import load_checkpoint_flex # <-- THIS LINE IS NOW REMOVED

# --- A flexible checkpoint loader (from your XAI script) ---
# (This is the only definition the script needs)
def load_checkpoint_flex(model: nn.Module, path: str, device: torch.device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict):
        for k in ["model_state_dict", "state_dict", "weights", "model"]:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
        if not all(isinstance(v, torch.Tensor) for v in sd.values()):
            sd = {k:v for k,v in sd.items() if isinstance(v, torch.Tensor)} or sd
    new_sd = {}
    for k,v in sd.items():
        if k.startswith("module."): k = k[7:]
        if k.startswith("model."):  k = k[6:]
        new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:   print(f"Warning: Missing keys: {missing}")
    if unexpected:print(f"Warning: Unexpected keys: {unexpected}")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("Distillation")

def run_distillation(args):
    """Main distillation training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Data
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader, _ = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,
        num_workers=4
    )
    logger.info(f"Data loaded: {len(train_loader.dataset)} training samples from {args.data_dir}")

    # 2. Load Models
    # Load Teacher and set to evaluation mode
    teacher_model = get_model(args.teacher_model, args.num_classes, pretrained=False).to(device)
    load_checkpoint_flex(teacher_model, args.teacher_path, device)
    teacher_model.eval()
    logger.info(f"Teacher model ({args.teacher_model}) loaded from {args.teacher_path}")

    # Load Student and set to training mode
    student_model = get_model(args.student_model, args.num_classes, pretrained=True).to(device)
    student_model.train()
    logger.info(f"Student model ({args.student_model}) loaded for training.")

    # 3. Setup Training
    criterion = DistillationLoss(alpha=args.alpha, temperature=args.temperature)
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # --- TRAINING LOOP ---
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        student_model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = running_loss / len(train_loader)
        
        # --- VALIDATION LOOP ---
        student_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                val_loss += F.cross_entropy(outputs, labels, reduction='sum').item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        
        logger.info(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        scheduler.step(avg_val_loss)
        
        # Save the best student model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Create a dynamic save path
            teacher_name = os.path.splitext(os.path.basename(args.teacher_path))[0]
            save_path = os.path.join(args.save_dir, f"student_{args.student_model}_from_teacher_{teacher_name}.pth")
            torch.save(student_model.state_dict(), save_path)
            logger.info(f"*** New best student model saved to {save_path} ***")

    logger.info("Distillation training complete.")
    # Use the 'save_path' variable which holds the last best model path
    if 'save_path' in locals():
        logger.info(f"Final best student model is available at {save_path}")
    else:
        logger.warning("Training finished, but no model was saved (val loss may not have improved).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training Script")
    parser.add_argument("--teacher-model", type=str, required=True, help="Teacher model name (e.g., densenet121)")
    parser.add_argument("--teacher-path", type=str, required=True, help="Path to the trained teacher model .pth file")
    parser.add_argument("--student-model", type=str, required=True, help="Student model name (e.g., mobilenetv3)")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the training/validation dataset")
    parser.add_argument("--save-dir", type=str, default="distillation/saved_models", help="Directory to save student models")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes (e.g., 4 for brain tumor)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for hard label loss")
    parser.add_argument("--temperature", type=float, default=6.0, help="Temperature for softmax")
    
    args = parser.parse_args()
    run_distillation(args)