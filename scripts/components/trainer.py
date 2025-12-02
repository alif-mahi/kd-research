"""
Training logic for teacher and student models.
Includes BCEWithLogitsLoss for teacher and KL divergence for student (knowledge distillation).
"""

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np

from evaluation import evaluate_model


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""
    
    def __init__(self, patience: int = 10, mode: str = 'max', delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            mode: 'max' for metrics to maximize (F1), 'min' for metrics to minimize (loss)
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > (self.best_score + self.delta)
        else:
            improved = score < (self.best_score - self.delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class TeacherTrainer:
    """Trainer for multimodal teacher model using BCEWithLogitsLoss."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 10
    ):
        """
        Args:
            model: Teacher model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=5,
            factor=0.5,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        
        # Training history
        self.train_history = []
        self.val_history = []
        self.best_val_f1 = 0.0
        self.best_model_state = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for rgb, ir, labels in pbar:
            rgb, ir, labels = rgb.to(self.device), ir.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(rgb, ir)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        metrics = evaluate_model(
            self.model,
            self.val_loader,
            self.device,
            mode='multimodal',
            desc='Validating'
        )
        
        # Add loss
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for rgb, ir, labels in self.val_loader:
                rgb, ir, labels = rgb.to(self.device), ir.to(self.device), labels.to(self.device)
                outputs = self.model(rgb, ir)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
        
        metrics['loss'] = total_loss / num_batches
        return metrics
    
    def train(self, num_epochs: int, save_path: str) -> Tuple[nn.Module, List[Dict], List[Dict]]:
        """
        Train teacher model.
        
        Args:
            num_epochs: Number of epochs
            save_path: Path to save best model
            
        Returns:
            Tuple of (best_model, train_history, val_history)
        """
        print(f"Training teacher model for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            val_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            self.val_history.append(val_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}")
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
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ Saved best model (F1: {self.best_val_f1:.4f})")
            
            # Early stopping
            if self.early_stopping(val_metrics['f1_macro']):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        return self.model, self.train_history, self.val_history


class StudentTrainer:
    """Trainer for student model using knowledge distillation with KL divergence."""
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        temperature: float = 4.0,
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
            factor=0.5,
            verbose=True
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
        
        # Scale by T^2 as per Hinton et al.
        loss = kl.mean() * (T ** 2)
        
        return loss
    
    def train_epoch(self, teacher_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            teacher_loader: Dataloader with multimodal data for teacher
        """
        self.student.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            zip(self.train_loader, teacher_loader),
            desc='Training Student',
            total=len(self.train_loader)
        )
        
        for (rgb_student, labels_student), (rgb_teacher, ir_teacher, labels_teacher) in pbar:
            rgb_student = rgb_student.to(self.device)
            rgb_teacher = rgb_teacher.to(self.device)
            ir_teacher = ir_teacher.to(self.device)
            
            # Get student logits
            student_logits = self.student(rgb_student)
            
            # Get teacher logits (no grad)
            with torch.no_grad():
                teacher_logits = self.teacher(rgb_teacher, ir_teacher)
            
            # Compute KD loss
            loss = self.kd_loss(student_logits, teacher_logits)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
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
        print(f"Training student model with KD for {num_epochs} epochs...")
        
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
            print(f"Train Loss: {train_metrics['loss']:.4f}")
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
