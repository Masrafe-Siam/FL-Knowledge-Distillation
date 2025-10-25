import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.1, temperature: float = 6.0):
        """
        Args:
            alpha (float): Weight for the standard Cross-Entropy loss (vs. distillation loss).
                           A common value is 0.1 (90% distillation, 10% standard).
            temperature (float): Softens the probabilities. Higher temp = softer probabilities.
                                 A common value is 4-10.
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.distillation_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.student_loss = nn.CrossEntropyLoss()

    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, hard_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_outputs (torch.Tensor): Logits from the student model [batch_size, num_classes].
            teacher_outputs (torch.Tensor): Logits from the teacher model [batch_size, num_classes].
            labels (torch.Tensor): Ground truth labels [batch_size].
        
        Returns:
            torch.Tensor: Combined distillation loss.
        """
        """
        Calculate the total loss.
        
        Args:
            student_logits: Raw output from the student model (B, num_classes)
            teacher_logits: Raw output from the teacher model (B, num_classes)
            hard_labels: The true class labels (B,)
        """
        
        # 1. Calculate the "soft" distillation loss (Student vs. Teacher)
        # Soften probabilities using temperature
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.log_softmax(teacher_logits / self.T, dim=1)
        
        # (self.T**2) scales the gradient back down, this is the standard way
        loss_distill = (self.T**2) * self.distillation_loss(soft_student, soft_teacher)

        # 2. Calculate the "hard" student loss (Student vs. True Labels)
        loss_student = self.student_loss(student_logits, hard_labels)

        # 3. Combine the two losses
        # (1 - alpha) for distillation, (alpha) for student
        total_loss = (1.0 - self.alpha) * loss_distill + self.alpha * loss_student
        
        return total_loss