import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DiseaseDataPreprocessor:
    def __init__(self):
        self.symptom_encoder = LabelEncoder()
        self.disease_encoder = LabelEncoder()
        self.all_symptoms = None
        
    def load_and_combine_data(self):
        """Load and combine all relevant data"""
        self.training_data = pd.read_csv('data/raw/Training.csv')
        self.symptoms_data = pd.read_csv('data/raw/symtoms_df.csv')
        self.symptom_severity = pd.read_csv('data/raw/Symptom-severity.csv')
        return self.training_data.shape
    
    def explore_training_data(self):
        """Explore the training data structure"""
        print("Training Data Info:")
        print(f"Shape: {self.training_data.shape}")
        print(f"Number of unique diseases: {self.training_data['prognosis'].nunique()}")
        print(f"Sample diseases: {self.training_data['prognosis'].unique()[:10]}")
        
        # Check symptom values
        symptom_columns = self.training_data.columns[:-1]  # All except prognosis
        print(f"\nNumber of symptom columns: {len(symptom_columns)}")
        print(f"First 10 symptoms: {list(symptom_columns[:10])}")
        
    def preprocess_training_data(self):
        """Preprocess the training data for model training"""
        # Extract features (symptoms) and target (disease)
        X = self.training_data.iloc[:, :-1]  # All columns except last (symptoms)
        y = self.training_data['prognosis']   # Disease column
        
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Check unique values in symptom columns
        unique_values_per_column = X.nunique()
        print(f"\nUnique values per symptom column:")
        print(unique_values_per_column.head(15))
        
        # Get all unique symptoms from column names
        self.all_symptoms = X.columns.tolist()
        print(f"\nTotal unique symptoms (from columns): {len(self.all_symptoms)}")
        
        # Convert symptom presence to binary (1/0)
        # Assuming symptoms are already encoded as 1/0 or similar
        X_binary = (X != 0).astype(int)  # Convert any non-zero to 1
        
        print(f"\nSample of binary encoded data:")
        print(X_binary.iloc[0:3, 0:10])  # First 3 rows, first 10 symptoms
        
        return X_binary, y, self.all_symptoms
    
    def create_symptom_mapping(self):
        """Create mapping between symptom names and their severities"""
        symptom_severity = pd.read_csv('data/raw/Symptom-severity.csv')
        print(f"\nSymptom Severity Mapping:")
        print(symptom_severity.head(10))
        return dict(zip(symptom_severity['Symptom'], symptom_severity['weight']))

if __name__ == "__main__":
    preprocessor = DiseaseDataPreprocessor()
    preprocessor.load_and_combine_data()
    preprocessor.explore_training_data()
    X, y, symptoms = preprocessor.preprocess_training_data()
    severity_map = preprocessor.create_symptom_mapping()