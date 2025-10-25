import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from model_selection import comprehensive_model_selection
from feature_engineering import FeatureEngineer
from data_preprocessing import DiseaseDataPreprocessor

class DiseaseClassifier:
    def __init__(self):
        self.model = None
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """Train multiple models and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                class_weight='balanced',  # Works for multi-class
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'  # Works for multi-class
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=42
                # No class weighting for XGBoost in multi-class
            )
        }
        
        best_model = None
        best_accuracy = 0
        best_model_name = ""
        
        for name, model in models.items():
            print(f"\nTraining {name}")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            # Use the model directly (no calibration)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            training_time = time.time() - start_time
            
            print(f"{name} - Accuracy: {accuracy:.4f}, Time: {training_time:.2f}s")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
        
        print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        self.model = best_model
        return best_model, best_accuracy, X_test, y_test

class AdvancedDiseaseClassifier:
    def __init__(self):
        self.best_model = None
        self.feature_engineer = None
        self.model_selector = None
    
    def train_advanced_pipeline(self, use_simple_training=False):
        """Advanced training pipeline with comprehensive model selection or simple training"""
        print("=== ADVANCED DISEASE CLASSIFIER TRAINING PIPELINE ===")
        
        # 1. Preprocess data
        print("\nStep 1: Data Preprocessing")
        preprocessor = DiseaseDataPreprocessor()
        preprocessor.load_and_combine_data()
        X_binary, y, symptoms = preprocessor.preprocess_training_data()
        
        # 2. Feature engineering
        print("\nStep 2: Feature Engineering")
        self.feature_engineer = FeatureEngineer()
        self.feature_engineer.load_symptom_severity()
        X_final, y_encoded = self.feature_engineer.prepare_features_for_training(X_binary, y)
        
        # 3. Model training
        print("\nStep 3: Model Training")
        if use_simple_training:
            print("Using simple training")
            classifier = DiseaseClassifier()
            best_model, best_accuracy, X_test, y_test = classifier.train_models(X_final, y_encoded)
            self.best_model = best_model
            results = {'best_model': best_model, 'test_accuracy': best_accuracy}
            self.model_selector = None
        else:
            print("Using comprehensive model selection")
            selector, best_name, best_results = comprehensive_model_selection(X_final, y_encoded)
            self.best_model = best_results['best_model']
            self.model_selector = selector
            results = best_results
        
        # 4. Save everything
        print("\nStep 4: Saving Models and Encoders")
        self.save_complete_pipeline()
        
        print("\nAdvanced training pipeline completed successfully!")
        return self.best_model, results
    
    def save_complete_pipeline(self, path='models/'):
        """Save complete pipeline including feature engineer and model"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save best model
        joblib.dump(self.best_model, f'{path}/best_disease_classifier.pkl')
        
        # Save feature engineer
        joblib.dump(self.feature_engineer, f'{path}/feature_engineer.pkl')
        
        # Save disease encoder
        joblib.dump(self.feature_engineer.disease_encoder, f'{path}/disease_encoder.pkl')
        
        # Save symptom severity map
        joblib.dump(self.feature_engineer.symptom_severity_map, f'{path}/symptom_severity_map.pkl')
        
        print("Complete pipeline saved successfully!")
    
    def load_complete_pipeline(self, path='models/'):
        """Load complete pipeline"""
        self.best_model = joblib.load(f'{path}/best_disease_classifier.pkl')
        self.feature_engineer = joblib.load(f'{path}/feature_engineer.pkl')
        print("Complete pipeline loaded successfully!")

def train_advanced_model(use_simple_training=False):
    """Main function to train advanced model"""
    classifier = AdvancedDiseaseClassifier()
    best_model, results = classifier.train_advanced_pipeline(use_simple_training=use_simple_training)
    return classifier, best_model, results

if __name__ == "__main__":
    # Allow switching between simple and comprehensive training
    classifier, best_model, results = train_advanced_model(use_simple_training=False)