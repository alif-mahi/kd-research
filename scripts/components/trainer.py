"""
Training logic for teacher and student models.
DEPRECATED: This module is kept for backward compatibility.
New code should import from trainers/ module instead.

Imports:
    from trainers.teacher_trainer import TeacherTrainer, EarlyStopping
    from trainers.student_kl_bce import StudentTrainer_KL_BCE
"""

# Backward compatibility imports
from trainers.teacher_trainer import TeacherTrainer, EarlyStopping
from trainers.student_kl_bce import StudentTrainer_KL_BCE

# Alias for backward compatibility
StudentTrainer = StudentTrainer_KL_BCE

__all__ = [
    'TeacherTrainer',
    'EarlyStopping',
    'StudentTrainer',
    'StudentTrainer_KL_BCE'
]
