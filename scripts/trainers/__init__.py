"""
Training modules for teacher and student models with different loss configurations.
"""

from trainers.teacher_trainer import TeacherTrainer, EarlyStopping
from trainers.student_kl_bce import StudentTrainer_KL_BCE
from trainers.student_kl_l2_bce import StudentTrainer_KL_L2_BCE
from trainers.student_kl_contrastive_l2_bce import StudentTrainer_KL_Contrastive_L2_BCE

__all__ = [
    'TeacherTrainer',
    'EarlyStopping',
    'StudentTrainer_KL_BCE',
    'StudentTrainer_KL_L2_BCE',
    'StudentTrainer_KL_Contrastive_L2_BCE'
]
