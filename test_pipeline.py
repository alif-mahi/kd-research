"""

class PipelineTester:
    """Comprehensive pipeline validation."""
    
    def __init__(self, csv_path: str, base_path: str):
        self.csv_path = csv_path
        self.base_path = base_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4  # Small batch for testing
        
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log with formatting."""
        symbols = {"INFO": "ℹ", "PASS": "✓", "FAIL": "✗", "WARN": "⚠"}
        symbol = symbols.get(level, "•")
        print(f"{symbol} {message}")
    
    def test(self, name: str, func):
        """Run a single test."""
        try:
            print(f"\n{'='*70}")
            self.log(f"Testing: {name}", "INFO")
            print(f"{'='*70}")
            func()
            self.tests_passed += 1
            self.log(f"PASSED: {name}", "PASS")
            return True
        except Exception as e:
            self.tests_failed += 1
            self.errors.append((name, str(e)))
            self.log(f"FAILED: {name}", "FAIL")
            self.log(f"Error: {str(e)}", "FAIL")
            import traceback
            traceback.print_exc()
            return False
    
    def test_data_loading(self):
            base_path=self.base_path,
            mode='multimodal'
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for testing
        )
        
        # Test batch loading
        rgb, ir, labels = next(iter(train_loader))
        assert rgb.shape == (self.batch_size, 3, 224, 224), f"RGB shape mismatch: {rgb.shape}"
        assert ir.shape == (self.batch_size, 3, 224, 224), f"IR shape mismatch: {ir.shape}"
        assert labels.shape == (self.batch_size, 2), f"Labels shape mismatch: {labels.shape}"
        
        self.log(f"Batch shapes - RGB: {rgb.shape}, IR: {ir.shape}, Labels: {labels.shape}", "INFO")
        
        # Store for later tests
        self.train_loader = train_loader
        self.sample_batch = (rgb.to(self.device), ir.to(self.device), labels.to(self.device))
    
    def test_teacher_model(self, name: str, model_class, expected_feature_dim: int):
        """Test a teacher model architecture."""
        self.log(f"Initializing {name}...")
        model = model_class(num_classes=2).to(self.device)
        model.eval()
        
        rgb, ir, labels = self.sample_batch
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(rgb, ir)
            features = model.get_features(rgb, ir)
        
        assert outputs.shape == (self.batch_size, 2), f"Output shape mismatch: {outputs.shape}"
        assert features.shape[0] == self.batch_size, f"Feature batch size mismatch: {features.shape}"
        assert features.shape[1] == expected_feature_dim, \
            f"Feature dim mismatch: expected {expected_feature_dim}, got {features.shape[1]}"
        
        self.log(f"Output shape: {outputs.shape}, Feature shape: {features.shape}", "INFO")
        self.log(f"Feature dimension: {features.shape[1]} (expected: {expected_feature_dim})", "INFO")
        
        # Store for student tests
        return model
    
    def test_student_model(self):
        """Test student model architecture."""
        self.log("Initializing Student Model...")
        model = LightweightCNN(num_classes=2).to(self.device)
        model.eval()
        
        rgb, _, labels = self.sample_batch
        
        # Test forward pass (RGB only)
        with torch.no_grad():
            outputs = model(rgb)
            features = model.get_features(rgb)
        
        assert outputs.shape == (self.batch_size, 2), f"Output shape mismatch: {outputs.shape}"
        
        self.log(f"Output shape: {outputs.shape}, Feature shape: {features.shape}", "INFO")
        self.log(f"Student feature dimension: {features.shape[1]}", "INFO")
        
        # Store for later
        self.student_model = model
        return model
    
    def test_student_trainer(self, name: str, trainer_class, teacher_model):
        """Test a student trainer with actual forward/backward pass."""
        self.log(f"Initializing {name}...")
        
        # Create fresh student model
        student = LightweightCNN(num_classes=2).to(self.device)
        
        # Create minimal dataloaders
        rgb_dataset = FLAME2Dataset(
            dataframe=pd.read_csv(self.csv_path).head(8),  # Just 8 samples
            base_path=self.base_path,
            mode='rgb_only'
        )
        multimodal_dataset = FLAME2Dataset(
            dataframe=pd.read_csv(self.csv_path).head(8),
            base_path=self.base_path,
            mode='multimodal'
        )
        
        rgb_loader = DataLoader(rgb_dataset, batch_size=self.batch_size, num_workers=0)
        mm_loader = DataLoader(multimodal_dataset, batch_size=self.batch_size, num_workers=0)
        
        # Initialize trainer
        trainer = trainer_class(
            student_model=student,
            teacher_model=teacher_model,
            train_loader=rgb_loader,
            val_loader=rgb_loader,
            device=self.device,
            patience=999
        )
        
        # Test one training iteration
        self.log("Testing training iteration...", "INFO")
        student.train()
        
        (rgb_student, labels_student), (rgb_teacher, ir_teacher, labels_teacher) = \
            next(zip(rgb_loader, mm_loader))
        
        rgb_student = rgb_student.to(self.device)
        labels_student = labels_student.to(self.device)
        rgb_teacher = rgb_teacher.to(self.device)
        ir_teacher = ir_teacher.to(self.device)
        
        # Get predictions
        student_logits = student(rgb_student)
        student_features = student.get_features(rgb_student)
        
        with torch.no_grad():
            teacher_logits = teacher_model(rgb_teacher, ir_teacher)
            teacher_features = teacher_model.get_features(rgb_teacher, ir_teacher)
        
        self.log(f"Student features: {student_features.shape}, Teacher features: {teacher_features.shape}", "INFO")
        
        # Test loss computation
        if hasattr(trainer, 'feature_projection'):
            self.log(f"Feature projection layer: {student_features.shape[1]} → {teacher_features.shape[1]}", "INFO")
            projected = trainer.feature_projection(student_features)
            self.log(f"Projected student features: {projected.shape}", "INFO")
            assert projected.shape == teacher_features.shape, \
                f"Projection failed: {projected.shape} != {teacher_features.shape}"
        
        # Compute losses
        kd_loss = trainer.kd_loss(student_logits, teacher_logits)
        self.log(f"KD Loss: {kd_loss.item():.4f}", "INFO")
        
        if hasattr(trainer, 'l2_feature_loss'):
            l2_loss = trainer.l2_feature_loss(student_features, teacher_features)
            self.log(f"L2 Loss: {l2_loss.item():.4f}", "INFO")
        
            first_teacher_name = list(teacher_models.keys())[0]
            first_teacher = teacher_models[first_teacher_name]
            
            self.log(f"Using {first_teacher_name} for student trainer tests", "INFO")
            
            self.test("Student Trainer: KL+BCE",
                     lambda: self.test_student_trainer("KL+BCE", StudentTrainer_KL_BCE, first_teacher))
            
            self.test("Student Trainer: KL+L2+BCE",
                     lambda: self.test_student_trainer("KL+L2+BCE", StudentTrainer_KL_L2_BCE, first_teacher))
            
            self.test("Student Trainer: KL+Contrastive+L2+BCE",
                     lambda: self.test_student_trainer("KL+Contrastive+L2+BCE", 
                                                      StudentTrainer_KL_Contrastive_L2_BCE, first_teacher))
        
        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"✓ Passed: {self.tests_passed}")
        print(f"✗ Failed: {self.tests_failed}")
        print(f"Total: {self.tests_passed + self.tests_failed}")
        print("="*70)
        
        if self.errors:
            print("\nFAILED TESTS:")
            for name, error in self.errors:
                print(f"  ✗ {name}")
                print(f"    Error: {error}")
        
        if self.tests_failed == 0:
            print("\n🎉 ALL TESTS PASSED! Pipeline is ready for ablation study.")
            return True
        else:
            print(f"\n⚠ {self.tests_failed} test(s) failed. Please fix errors before running ablation study.")
            return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate KD pipeline before running ablation study")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--base_path", required=True, help="Base path for images")
    
    args = parser.parse_args()
    
    tester = PipelineTester(args.csv, args.base_path)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)
