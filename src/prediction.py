import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator

class DiseasePredictor(BaseEstimator):
    def __init__(self, model_path='models/disease_classifier.pkl', 
                 encoder_path='models/disease_encoder.pkl',
                 severity_path='models/symptom_severity_map.pkl'):
        
        # Load trained model and encoders
        self.model = joblib.load(model_path)
        self.disease_encoder = joblib.load(encoder_path)
        self.symptom_severity_map = joblib.load(severity_path)
        
        # Load additional data for disease details
        self.description_df = pd.read_csv('data/raw/description.csv')
        self.precautions_df = pd.read_csv('data/raw/precautions_df.csv')
        self.medications_df = pd.read_csv('data/raw/medications.csv')
        self.diets_df = pd.read_csv('data/raw/diets.csv')
        self.workout_df = pd.read_csv('data/raw/workout_df.csv')
        
        # Get feature names from training
        self.symptom_names = list(self.symptom_severity_map.keys())
        
        print("Disease Predictor initialized successfully!")
        print(f"Loaded {len(self.symptom_names)} symptoms and {len(self.disease_encoder.classes_)} diseases")
    
    def symptoms_to_vector(self, input_symptoms):
        """Convert symptoms to feature vector with ENHANCED weighting"""
        symptom_vector = np.zeros(len(self.symptom_names))
        
        for symptom in input_symptoms:
            clean_symptom = symptom.strip().lower().replace('_', ' ').replace('-', ' ')
            
            # Find matching symptom
            for i, known_symptom in enumerate(self.symptom_names):
                known_clean = known_symptom.strip().lower().replace('_', ' ').replace('-', ' ')
                
                if clean_symptom == known_clean or clean_symptom in known_clean:
                    # Apply ENHANCED weight (original severity × 10)
                    original_weight = self.symptom_severity_map.get(known_symptom, 1)
                    enhanced_weight = original_weight * 10  # 10x multiplier
                    symptom_vector[i] = enhanced_weight
                    print(f"Matched '{symptom}' to '{known_symptom}' (weight: {enhanced_weight})")
                    break
        
        return symptom_vector.reshape(1, -1)
    
    def predict_top_diseases(self, input_symptoms, top_n=5):
        """
        Predict top N diseases based on input symptoms
        
        Args:
            input_symptoms (list): List of symptom strings
            top_n (int): Number of top diseases to return
        
        Returns:
            dict: Complete prediction results
        """
        print(f"\nPredicting diseases for symptoms: {input_symptoms}")
        
        # Convert symptoms to feature vector
        symptom_vector = self.symptoms_to_vector(input_symptoms)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(symptom_vector)[0]
        
        # Get top N disease indices and probabilities
        top_n_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_n_probabilities = probabilities[top_n_indices]
        
        # Get disease names
        disease_names = self.disease_encoder.inverse_transform(top_n_indices)
        
        # Prepare results
        predictions = []
        total_confidence = sum(top_n_probabilities)
        
        for i, (disease, prob) in enumerate(zip(disease_names, top_n_probabilities)):
            # Get additional disease information
            disease_info = self.get_disease_details(disease)
            
            prediction = {
                'rank': i + 1,
                'disease': disease,
                'probability': round(prob, 4),
                'confidence': f"{(prob/total_confidence)*100:.1f}%",
                'description': disease_info.get('description', 'No description available'),
                'precautions': disease_info.get('precautions', []),
                'medications': disease_info.get('medications', []),
                'diet': disease_info.get('diet', []),
                'workout': disease_info.get('workout', [])
            }
            predictions.append(prediction)
        
        return {
            'input_symptoms': input_symptoms,
            'total_symptoms_matched': np.sum(symptom_vector > 0),
            'predictions': predictions
        }
    
    def get_disease_details(self, disease_name):
        """Get detailed information about a disease"""
        details = {}
        
        # Clean disease name for matching
        clean_disease = disease_name.strip().lower()
        
        # Get description
        desc_match = self.description_df[
            self.description_df['Disease'].str.strip().str.lower() == clean_disease
        ]
        details['description'] = desc_match['Description'].values[0] if not desc_match.empty else "Description not available"
        
        # Get precautions (up to 4)
        prec_match = self.precautions_df[
            self.precautions_df['Disease'].str.strip().str.lower() == clean_disease
        ]
        if not prec_match.empty:
            details['precautions'] = [prec for prec in prec_match.iloc[0, 1:5] if pd.notna(prec) and prec != '']
        else:
            details['precautions'] = ["No specific precautions available"]
        
        # Get medications
        med_match = self.medications_df[
            self.medications_df['Disease'].str.strip().str.lower() == clean_disease
        ]
        if not med_match.empty:
            details['medications'] = [med for med in med_match.iloc[0, 1:5] if pd.notna(med) and med != '']
        else:
            details['medications'] = ["Consult doctor for medications"]
        
        # Get diet recommendations
        diet_match = self.diets_df[
            self.diets_df['Disease'].str.strip().str.lower() == clean_disease
        ]
        if not diet_match.empty:
            details['diet'] = [diet for diet in diet_match.iloc[0, 1:5] if pd.notna(diet) and diet != '']
        else:
            details['diet'] = ["Maintain balanced diet"]
        
        # Get workout recommendations
        workout_match = self.workout_df[
            self.workout_df['Disease'].str.strip().str.lower() == clean_disease
        ]
        if not workout_match.empty:
            details['workout'] = [workout for workout in workout_match.iloc[0, 1:5] if pd.notna(workout) and workout != '']
        else:
            details['workout'] = ["Regular light exercise recommended"]
        
        return details

class AdvancedDiseasePredictor(DiseasePredictor):
    def __init__(self, model_path='models/best_disease_classifier.pkl', 
                 feature_engineer_path='models/feature_engineer.pkl'):
        
        # Load complete pipeline
        self.best_model = joblib.load(model_path)
        self.feature_engineer = joblib.load(feature_engineer_path)
        self.disease_encoder = self.feature_engineer.disease_encoder
        self.symptom_severity_map = self.feature_engineer.symptom_severity_map
        
        # Load additional data for disease details
        self.description_df = pd.read_csv('data/raw/description.csv')
        self.precautions_df = pd.read_csv('data/raw/precautions_df.csv')
        self.medications_df = pd.read_csv('data/raw/medications.csv')
        self.diets_df = pd.read_csv('data/raw/diets.csv')
        self.workout_df = pd.read_csv('data/raw/workout_df.csv')
        
        # Get feature names
        self.symptom_names = list(self.symptom_severity_map.keys())
        
        # Set model to best_model for compatibility with parent class methods
        self.model = self.best_model
        
        print("Advanced Disease Predictor initialized!")
        print(f"Using: {type(self.best_model).__name__}")
        print(f"{len(self.symptom_names)} symptoms, {len(self.disease_encoder.classes_)} diseases")

def test_prediction():
    """Test the prediction system with sample symptoms"""
    predictor = AdvancedDiseasePredictor()
    
    # Test cases
    test_cases = [
        ['itching', 'skin_rash', 'nodal_skin_eruptions'],
        ['cough', 'fever', 'breathlessness'],
        ['headache', 'dizziness', 'blurred_vision'],
        ['joint_pain', 'fatigue', 'muscle_pain']
    ]
    
    for i, symptoms in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST CASE {i}: {symptoms}")
        print(f"{'='*50}")
        
        results = predictor.predict_top_diseases(symptoms, top_n=3)
        
        print(f"Input Symptoms: {results['input_symptoms']}")
        print(f"Matched {results['total_symptoms_matched']} symptoms")
        print(f"\nTOP PREDICTIONS:")
        
        for pred in results['predictions']:
            print(f"\n#{pred['rank']}: {pred['disease']} ({pred['confidence']})")
            print(f"   {pred['description'][:100]}...")
            print(f"   Precautions: {', '.join(pred['precautions'][:2])}")
            print(f"   Medications: {', '.join(pred['medications'][:2])}")

if __name__ == "__main__":
    test_prediction()