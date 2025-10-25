import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ModelSelector:
    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
    def initialize_models(self):
        """Initialize multiple models with enhanced parameters"""
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state, 
                    class_weight='balanced'  # Handle class imbalance
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]  # Tune class weights
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state, 
                    eval_metric='logloss',
                    scale_pos_weight=1  # Handle class imbalance
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'scale_pos_weight': [1, 2, 5]  # Tune class weights
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }
    
    def evaluate_baseline_models(self, X, y, test_size=0.2):
        """Evaluate all models with basic train-test split"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print("Evaluating Baseline Models (Train-Test Split)")
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print("=" * 80)
        
        baseline_results = {}
        
        for name, model_info in self.models.items():
            print(f"\nTraining {name}")
            start_time = time.time()
            
            model = model_info['model']
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            training_time = time.time() - start_time
            
            baseline_results[name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'model': model
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, Time: {training_time:.2f}s")
            
            # Cross-validation for more robust evaluation
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='accuracy')
            print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return baseline_results
    
    def perform_grid_search(self, X, y, test_size=0.2):
        """Perform GridSearchCV for hyperparameter tuning"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print("\nPerforming Hyperparameter Tuning with GridSearchCV")
        print("=" * 80)
        
        tuned_models = {}
        
        for name, model_info in self.models.items():
            print(f"\nTuning {name}")
            start_time = time.time()
            
            # Use fewer CV folds for grid search to save time
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=3,  # Fewer folds for speed
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Best model from grid search
            best_model = grid_search.best_estimator_
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            tuned_models[name] = {
                'best_model': best_model,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_accuracy': accuracy,
                'training_time': training_time,
                'grid_search': grid_search
            }
            
            print(f"{name} - Best Params: {grid_search.best_params_}")
            print(f"{name} - Best CV Score: {grid_search.best_score_:.4f}")
            print(f"{name} - Test Accuracy: {accuracy:.4f}")
            print(f"{name} - Tuning Time: {training_time:.2f}s")
        
        return tuned_models
    
    def select_best_model(self, tuned_results):
        """Select the best model based on test accuracy"""
        best_model_name = None
        best_accuracy = 0
        
        for name, results in tuned_results.items():
            if results['test_accuracy'] > best_accuracy:
                best_accuracy = results['test_accuracy']
                best_model_name = name
        
        self.best_model = tuned_results[best_model_name]['best_model']
        self.best_score = best_accuracy
        
        print(f"\nBEST MODEL: {best_model_name}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"Best Parameters: {tuned_results[best_model_name]['best_params']}")
        
        return best_model_name, tuned_results[best_model_name]
    
    def plot_model_comparison(self, baseline_results, tuned_results):
        """Plot comparison of model performances"""
        models = list(baseline_results.keys())
        baseline_acc = [baseline_results[m]['accuracy'] for m in models]
        tuned_acc = [tuned_results[m]['test_accuracy'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.bar(x - width/2, baseline_acc, width, label='Baseline', alpha=0.7)
        bars2 = ax.bar(x + width/2, tuned_acc, width, label='Tuned', alpha=0.7)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison: Baseline vs Tuned')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_best_model(self, path='models/best_disease_classifier.pkl'):
        """Save the best tuned model"""
        import os
        os.makedirs('models', exist_ok=True)
        
        if self.best_model is not None:
            joblib.dump(self.best_model, path)
            print(f"Best model saved to {path}")
        else:
            print("No best model to save!")

def comprehensive_model_selection(X, y):
    """Complete model selection pipeline"""
    print("=== COMPREHENSIVE MODEL SELECTION PIPELINE ===")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print("=" * 80)
    
    # Initialize model selector
    selector = ModelSelector(cv_folds=5)
    selector.initialize_models()
    
    # Step 1: Baseline evaluation
    print("\n1. BASELINE MODEL EVALUATION")
    baseline_results = selector.evaluate_baseline_models(X, y)
    
    # Step 2: Hyperparameter tuning
    print("\n2. HYPERPARAMETER TUNING")
    tuned_results = selector.perform_grid_search(X, y)
    
    # Step 3: Select best model
    print("\n3. MODEL SELECTION")
    best_name, best_results = selector.select_best_model(tuned_results)
    
    # Step 4: Plot comparison
    print("\n4. VISUALIZATION")
    selector.plot_model_comparison(baseline_results, tuned_results)
    
    # Step 5: Save best model
    print("\n5. SAVING RESULTS")
    selector.save_best_model()
    
    # Print detailed comparison
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<20} {'Baseline Acc':<12} {'Tuned Acc':<12} {'Improvement':<12}")
    print("-" * 80)
    
    for name in baseline_results.keys():
        base_acc = baseline_results[name]['accuracy']
        tuned_acc = tuned_results[name]['test_accuracy']
        improvement = tuned_acc - base_acc
        print(f"{name:<20} {base_acc:<12.4f} {tuned_acc:<12.4f} {improvement:+.4f}")
    
    return selector, best_name, best_results

if __name__ == "__main__":
    # Test with sample data (you'll integrate with your actual data)
    from data_preprocessing import DiseaseDataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Load and preprocess data
    preprocessor = DiseaseDataPreprocessor()
    preprocessor.load_and_combine_data()
    X_binary, y, symptoms = preprocessor.preprocess_training_data()
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    feature_engineer.load_symptom_severity()
    X_final, y_encoded = feature_engineer.prepare_features_for_training(X_binary, y)
    
    # Run comprehensive model selection
    selector, best_name, best_results = comprehensive_model_selection(X_final, y_encoded)