"""
Student trainer with KL + Contrastive + L2 + BCE loss (Variant 2 for ablation study).
Loss: alpha * KL + beta * Contrastive + gamma * L2 + delta * BCE

Contrastive loss uses normalized features and computes pairwise distances across 
the batch, encouraging the student to preserve the relational structure learned by the teacher.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from tqdm import tqdm

from components.evaluation import evaluate_model
from trainers.teacher_trainer import EarlyStopping


class StudentTrainer_KL_Contrastive_L2_BCE:
    """Student trainer using KL + Contrastive + L2 + BCE loss."""
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        temperature: float = 4.0,
        alpha: float = 0.4,  # KL weight
        beta: float = 0.2,   # Contrastive weight
        gamma: float = 0.2,  # L2 weight
        delta: float = 0.2,  # BCE weight
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
            beta: Weight for contrastive loss
            gamma: Weight for L2 feature matching loss
            delta: Weight for BCE loss
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
        self.gamma = gamma
        self.delta = delta
        
        # Automatically detect feature dimensions by running dummy batch
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, 224, 224).to(device)
            dummy_ir = torch.randn(1, 3, 224, 224).to(device)
            student_feats = self.student.get_features(dummy_rgb)
            teacher_feats = self.teacher.get_features(dummy_rgb, dummy_ir)
            student_dim = student_feats.shape[1]
            teacher_dim = teacher_feats.shape[1]
        
        print(f"Feature dimensions detected: Student={student_dim}, Teacher={teacher_dim}")
        
        # Feature projection layer (student → teacher for feature matching)
        self.feature_projection = nn.Linear(student_dim, teacher_dim).to(device)
        
        # Optimizer and scheduler (include projection layer parameters)
        self.optimizer = AdamW(
            list(self.student.parameters()) + list(self.feature_projection.parameters()),
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
        """Compute KL divergence loss for knowledge distillation."""
        T = self.temperature
        
        # Apply temperature scaling and sigmoid
        student_soft = torch.sigmoid(student_logits / T)
        teacher_soft = torch.sigmoid(teacher_logits / T)
        
        # Binary KL divergence for multi-label
        eps = 1e-7
        kl = teacher_soft * torch.log((teacher_soft + eps) / (student_soft + eps)) + \
             (1 - teacher_soft) * torch.log((1 - teacher_soft + eps) / (1 - student_soft + eps))
        
        # Return raw KL loss (T^2 scaling removed for binary classification)
        loss = kl.mean()
        return loss
    
    def contrastive_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss based on pairwise distances.
        
        Encourages student to preserve the relational structure learned by teacher.
        Uses normalized features and computes pairwise distances across the batch.
        Projects student features to match teacher dimensionality first.
        Feature dimensions are automatically detected during initialization.
        
        Args:
            student_features: Student feature representations (B, student_dim)
            teacher_features: Teacher feature representations (B, teacher_dim)
            
        Returns:
            Contrastive loss
        """
        # Project student features to match teacher dimensionality
        student_projected = self.feature_projection(student_features)
        
        # Normalize features
        student_norm = F.normalize(student_projected, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        
        # Compute pairwise distance matrices
        # D_student[i,j] = ||student_i - student_j||^2
        student_distances = torch.cdist(student_norm, student_norm, p=2).pow(2)
        teacher_distances = torch.cdist(teacher_norm, teacher_norm, p=2).pow(2)
        
        # MSE between distance matrices (preserve relational structure)
        loss = F.mse_loss(student_distances, teacher_distances)
        
        return loss
    
    def l2_feature_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L2 loss between student and teacher features.
        
        Projects student features to match teacher feature dimensionality before computing MSE.
        Feature dimensions are automatically detected during initialization.
        
        Args:
            student_features: Student feature representations (B, student_dim)
            teacher_features: Teacher feature representations (B, teacher_dim)
            
        Returns:
            L2 feature matching loss
        """
        # Project student features to match teacher dimensionality
        student_projected = self.feature_projection(student_features)
        
        # MSE loss between projected student and teacher features
        loss = F.mse_loss(student_projected, teacher_features)
        return loss
    
    def train_epoch(self, teacher_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch using quadruple loss (KL + Contrastive + L2 + BCE).
        
        Args:
            teacher_loader: Dataloader with multimodal data for teacher
        """
        self.student.train()
        total_loss = 0.0
        total_kd_loss = 0.0
        total_contrastive_loss = 0.0
        total_l2_loss = 0.0
        total_hard_loss = 0.0
        num_batches = 0
        
        # Hard label criterion
        hard_criterion = nn.BCEWithLogitsLoss()
        
        pbar = tqdm(
            zip(self.train_loader, teacher_loader),
            desc='Training Student (KL+Cont+L2+BCE)',
            total=len(self.train_loader)
        )
        
        for (rgb_student, labels_student), (rgb_teacher, ir_teacher, labels_teacher) in pbar:
            rgb_student = rgb_student.to(self.device)
            labels_student = labels_student.to(self.device)
            rgb_teacher = rgb_teacher.to(self.device)
            ir_teacher = ir_teacher.to(self.device)
            
            # Get student logits and features
            student_logits = self.student(rgb_student)
            student_features = self.student.get_features(rgb_student)
            
            # Get teacher logits and features (no grad)
            with torch.no_grad():
                teacher_logits = self.teacher(rgb_teacher, ir_teacher)
                teacher_features = self.teacher.get_features(rgb_teacher, ir_teacher)
            
            # Compute quadruple loss
            kd_loss = self.kd_loss(student_logits, teacher_logits)
            contrastive_loss = self.contrastive_loss(student_features, teacher_features)
            l2_loss = self.l2_feature_loss(student_features, teacher_features)
            hard_loss = hard_criterion(student_logits, labels_student)
            
            loss = (self.alpha * kd_loss + 
                   self.beta * contrastive_loss + 
                   self.gamma * l2_loss + 
                   self.delta * hard_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_kd_loss += kd_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_l2_loss += l2_loss.item()
            total_hard_loss += hard_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item(),
                'kl': kd_loss.item(),
                'cont': contrastive_loss.item(),
                'l2': l2_loss.item(),
                'bce': hard_loss.item()
            })
        
        avg_loss = total_loss / num_batches
        return {
            'loss': avg_loss,
            'kd_loss': total_kd_loss / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches,
            'l2_loss': total_l2_loss / num_batches,
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
        print(f"Training student model with KL+Contrastive+L2+BCE for {num_epochs} epochs...")
        print(f"Loss weights: alpha={self.alpha} (KL), beta={self.beta} (Contrastive), "
              f"gamma={self.gamma} (L2), delta={self.delta} (BCE)")
        
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
                  f"(KL: {train_metrics['kd_loss']:.4f}, "
                  f"Cont: {train_metrics['contrastive_loss']:.4f}, "
                  f"L2: {train_metrics['l2_loss']:.4f}, "
                  f"BCE: {train_metrics['bce_loss']:.4f})")
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
