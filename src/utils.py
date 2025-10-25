import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load dataset from a CSV ffile"""
    data_files = {
        'symptoms': 'data/raw/symtoms_df.csv',
        'training': 'data/raw/Training.csv',
        'symptom_severity': 'data/raw/Symptom-severity.csv',
        'description': 'data/raw/description.csv',
        'precautions': 'data/raw/precautions_df.csv',
        'medications': 'data/raw/medications.csv',
        'diets': 'data/raw/diets.csv',
        'workout': 'data/raw/workout_df.csv'
    }

    data = {}

    for name, path in data_files.items():
        try:
            data[name] = pd.read_csv(path)
            print(f"Loaded {name}: {data[name].shape}")

        except Exception as e:
            print(f"Faild to load {name}: {e}")

    return data \
    
def explore_dataframe(df, name):
    """Basic exploration of a dataframe"""
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)