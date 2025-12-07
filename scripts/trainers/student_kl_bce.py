"""
Student trainer with KL + BCE loss (Baseline for ablation study).
Loss: alpha * KL_loss + beta * BCE_loss
"""

import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from tqdm import tqdm

from components.evaluation import evaluate_model
from trainers.teacher_trainer import EarlyStopping


class StudentTrainer_KL_BCE:
    """Student trainer using KL divergence + BCE loss (hybrid)."""
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        temperature: float = 4.0,
        alpha: float = 0.7,  # KL weight
        beta: float = 0.3,   # BCE weight
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5
    ):
        """
        Args:
            student_model: Student model
            teacher_model: Frozen teacher model
            train_loader: Training dataloader (RGB only)
            val_loader: Validation dataloader (RGB only)
            device: Device to train on
            temperature: Temperature for softmax in KD
            alpha: Weight for KL loss
            beta: Weight for BCE loss
            lr: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
        """
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.teacher.eval()  # Freeze teacher
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=3,
            factor=0.5
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        
        # Training history
        self.train_history = []
        self.val_history = []
        self.best_val_f1 = 0.0
        self.best_model_state = None
    
    def kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence loss for knowledge distillation.
        
        For multi-label, we apply sigmoid instead of softmax and use binary KL.
        
        Args:
            student_logits: Student model outputs (B, 2)
            teacher_logits: Teacher model outputs (B, 2)
            
        Returns:
            KD loss
        """
        T = self.temperature
        
        # Apply temperature scaling and sigmoid
        student_soft = torch.sigmoid(student_logits / T)
        teacher_soft = torch.sigmoid(teacher_logits / T)
        
        # Binary KL divergence for multi-label
        # KL(P||Q) = P*log(P/Q) + (1-P)*log((1-P)/(1-Q))
        eps = 1e-7  # For numerical stability
        
        kl = teacher_soft * torch.log((teacher_soft + eps) / (student_soft + eps)) + \
             (1 - teacher_soft) * torch.log((1 - teacher_soft + eps) / (1 - student_soft + eps))
        
        # Return raw KL loss (T^2 scaling removed for binary classification)
        loss = kl.mean()
        
        return loss
    
    def train_epoch(self, teacher_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch using hybrid loss (KL + BCE).
        
        Args:
            teacher_loader: Dataloader with multimodal data for teacher
        """
        self.student.train()
        total_loss = 0.0
        total_kd_loss = 0.0
        total_hard_loss = 0.0
        num_batches = 0
        
        # Hard label criterion
        hard_criterion = nn.BCEWithLogitsLoss()
        
        pbar = tqdm(
            zip(self.train_loader, teacher_loader),
            desc='Training Student (KL+BCE)',
            total=len(self.train_loader)
        )
        
        for (rgb_student, labels_student), (rgb_teacher, ir_teacher, labels_teacher) in pbar:
            rgb_student = rgb_student.to(self.device)
            labels_student = labels_student.to(self.device)
            rgb_teacher = rgb_teacher.to(self.device)
            ir_teacher = ir_teacher.to(self.device)
            
            # Get student logits
            student_logits = self.student(rgb_student)
            
            # Get teacher logits (no grad)
            with torch.no_grad():
                teacher_logits = self.teacher(rgb_teacher, ir_teacher)
            
            # Compute hybrid loss
            kd_loss = self.kd_loss(student_logits, teacher_logits)
            hard_loss = hard_criterion(student_logits, labels_student)
            loss = self.alpha * kd_loss + self.beta * hard_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_kd_loss += kd_loss.item()
            total_hard_loss += hard_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item(),
                'kd': kd_loss.item(),
                'bce': hard_loss.item()
            })
        
        avg_loss = total_loss / num_batches
        return {
            'loss': avg_loss,
            'kd_loss': total_kd_loss / num_batches,
            'bce_loss': total_hard_loss / num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        metrics = evaluate_model(
            self.student,
            self.val_loader,
            self.device,
            mode='rgb_only',
            desc='Validating Student'
        )
        
        # Add loss (using ground truth BCE for validation monitoring)
        self.student.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for rgb, labels in self.val_loader:
                rgb, labels = rgb.to(self.device), labels.to(self.device)
                outputs = self.student(rgb)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
        
        metrics['loss'] = total_loss / num_batches
        return metrics
    
    def train(
        self,
        num_epochs: int,
        save_path: str,
        teacher_train_loader: DataLoader,
    ) -> Tuple[nn.Module, List[Dict], List[Dict]]:
        """
        Train student model with knowledge distillation.
        
        Args:
            num_epochs: Number of epochs
            save_path: Path to save best model
            teacher_train_loader: Multimodal dataloader for teacher soft targets
            
        Returns:
            Tuple of (best_model, train_history, val_history)
        """
        print(f"Training student model with KL+BCE for {num_epochs} epochs...")
        print(f"Loss weights: alpha={self.alpha} (KL), beta={self.beta} (BCE)")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(teacher_train_loader)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            val_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            self.val_history.append(val_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(KL: {train_metrics['kd_loss']:.4f}, BCE: {train_metrics['bce_loss']:.4f})")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1_macro']:.4f}, "
                  f"F1 Fire: {val_metrics['f1_fire']:.4f}, "
                  f"F1 Smoke: {val_metrics['f1_smoke']:.4f}")
            
            # Update scheduler
            self.scheduler.step(val_metrics['f1_macro'])
            
            # Save best model
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_model_state = copy.deepcopy(self.student.state_dict())
                torch.save(self.student.state_dict(), save_path)
                print(f"✓ Saved best student model (F1: {self.best_val_f1:.4f})")
            
            # Early stopping
            if self.early_stopping(val_metrics['f1_macro']):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        self.student.load_state_dict(self.best_model_state)
        
        return self.student, self.train_history, self.val_history
