from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction import DiseasePredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize predictor
predictor = DiseasePredictor()

@app.route('/')
def home():
    return jsonify({
        "message": "Disease Diagnostic API",
        "endpoints": {
            "/predict": "POST - Predict diseases from symptoms",
            "/symptoms": "GET - Get all available symptoms",
            "/diseases": "GET - Get all possible diseases"
        }
    })

@app.route('/predict', methods=['POST'])
def predict_diseases():
    """Predict diseases from symptoms"""
    try:
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({"error": "Please provide 'symptoms' array in request body"}), 400
        
        symptoms = data['symptoms']
        top_n = data.get('top_n', 5)
        
        if not isinstance(symptoms, list) or len(symptoms) == 0:
            return jsonify({"error": "Symptoms must be a non-empty array"}), 400
        
        # Get predictions
        results = predictor.predict_top_diseases(symptoms, top_n=top_n)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Get all available symptoms"""
    return jsonify({
        "symptoms": predictor.symptom_names,
        "count": len(predictor.symptom_names)
    })

@app.route('/diseases', methods=['GET'])
def get_diseases():
    """Get all possible diseases"""
    diseases = list(predictor.disease_encoder.classes_)
    return jsonify({
        "diseases": diseases,
        "count": len(diseases)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"})

if __name__ == '__main__':
    print("🚀 Starting Disease Diagnostic API...")
    print("📊 Available symptoms:", len(predictor.symptom_names))
    print("🏥 Available diseases:", len(predictor.disease_encoder.classes_))
    app.run(debug=True, host='0.0.0.0', port=5000)