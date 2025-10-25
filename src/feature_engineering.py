import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

class FeatureEngineer:
    def __init__(self):
        self.disease_encoder = LabelEncoder()
        self.symptom_severity_map = None
        self.feature_columns = None
        
    def load_symptom_severity(self):
        """Load symptom severity weights"""
        severity_df = pd.read_csv('data/raw/Symptom-severity.csv')
        self.symptom_severity_map = dict(zip(severity_df['Symptom'], severity_df['weight']))
        return self.symptom_severity_map
    
    def encode_diseases(self, diseases):
        """Encode disease names to numerical labels"""
        encoded_diseases = self.disease_encoder.fit_transform(diseases)
        print(f"Encoded {len(self.disease_encoder.classes_)} diseases")
        return encoded_diseases
    
    def create_weighted_features(self, X_binary, symptom_columns):
        """Create weighted features based on symptom severity"""
        weighted_X = X_binary.copy()
        
        # Apply MUCH higher severity weights (10x multiplier for strong signal)
        for i, symptom in enumerate(symptom_columns):
            if symptom in self.symptom_severity_map:
                # Original weight multiplied by 10 for much stronger signal
                weight = self.symptom_severity_map[symptom] * 10
                weighted_X.iloc[:, i] = weighted_X.iloc[:, i] * weight
        
        print("Created ENHANCED weighted features (10x severity multiplier)")
        return weighted_X
    
    def prepare_features_for_training(self, X_binary, y, use_weighting=True):
        """Prepare final features for model training"""
        symptom_columns = X_binary.columns.tolist()
        
        if use_weighting and self.symptom_severity_map:
            X_final = self.create_weighted_features(X_binary, symptom_columns)
        else:
            X_final = X_binary
        
        # Encode diseases
        y_encoded = self.encode_diseases(y)
        
        self.feature_columns = symptom_columns
        
        print(f"Final feature matrix shape: {X_final.shape}")
        print(f"Target shape: {y_encoded.shape}")
        
        return X_final, y_encoded
    
    def save_encoders(self, path='models/'):
        """Save encoders for future use"""
        import os
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.disease_encoder, f'{path}/disease_encoder.pkl')
        joblib.dump(self.symptom_severity_map, f'{path}/symptom_severity_map.pkl')
        
        print("Encoders saved successfully!")