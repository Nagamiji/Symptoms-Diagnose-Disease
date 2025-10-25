# 🩺 Disease Diagnostic System

A machine learning-powered disease prediction system that analyzes symptoms and predicts potential diseases with detailed medical information.

## 📋 Project Overview

This system uses advanced machine learning models to:
- Predict diseases based on input symptoms
- Provide top N disease predictions with confidence scores
- Display detailed disease information including descriptions, precautions, medications, diets, and workout recommendations
- Offer a user-friendly Streamlit web interface

## 🏗️ Project Structure

```
Disease-Diagnostic/
├── data/
│   ├── raw/                    # Original data files
│   │   ├── Training.csv        # Main training dataset
│   │   ├── description.csv     # Disease descriptions
│   │   ├── precautions_df.csv  # Disease precautions
│   │   ├── medications.csv     # Recommended medications
│   │   ├── diets.csv          # Diet recommendations
│   │   ├── workout_df.csv     # Exercise recommendations
│   │   └── Symptom-severity.csv # Symptom severity weights
├── src/
│   ├── data_preprocessing.py   # Data cleaning and preparation
│   ├── feature_engineering.py  # Feature engineering with symptom weighting
│   ├── model_selection.py     # Comprehensive model testing
│   ├── model_training.py      # Model training pipeline
│   ├── prediction.py          # Prediction engine
│   └── utils.py               # Utility functions
├── models/                    # Saved trained models
├── api/                      # Flask API (optional)
├── app.py                    # Streamlit web application
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation & Setup

1. **Clone or download the project**
   ```bash
   cd Disease-Diagnostic
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv my_venv
   my_venv\Scripts\activate  # Windows
   # source my_venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the machine learning models**
   ```bash
   python src/model_training.py
   ```
   This will:
   - Preprocess the data
   - Train multiple models
   - Select the best performing model
   - Save the trained model to `models/` directory

5. **Launch the web application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** and go to `http://localhost:8501`

## 🎯 How to Use

### Using the Web Interface

1. **Select Symptoms**: Choose from the dropdown list of 132 possible symptoms
2. **Get Predictions**: The system will automatically show top 10 disease predictions
3. **View Details**: Click on each disease to see:
   - Disease description
   - Recommended precautions
   - Suggested medications
   - Diet recommendations
   - Workout/exercise advice

### Example Usage

**Common symptom combinations to test:**
- `itching, skin_rash, nodal_skin_eruptions` → Fungal infection
- `cough, fever, breathlessness` → Bronchial Asthma/Pneumonia
- `headache, dizziness, blurred_vision` → Migraine/Hypertension
- `joint_pain, fatigue, fever` → Arthritis/Malaria

## 🔧 Advanced Configuration

### Model Training Options

You can customize the training process by modifying:

1. **Model Selection**: Edit `src/model_selection.py` to test different algorithms
2. **Feature Engineering**: Adjust symptom weighting in `src/feature_engineering.py`
3. **Training Parameters**: Modify hyperparameters in `src/model_training.py`

### Retraining with Different Parameters

```bash
# Run comprehensive model testing
python src/model_selection.py

# Or run the complete training pipeline
python src/model_training.py
```

## 📊 Features

### ✅ Core Features
- **Multi-model comparison** (Random Forest, XGBoost, SVM, etc.)
- **Symptom severity weighting** for accurate predictions
- **Top N disease predictions** with confidence scores
- **Comprehensive disease information**
- **Real-time prediction** in web interface

### ✅ Data Processing
- **132 unique symptoms**
- **41 different diseases**
- **Symptom severity mapping**
- **Automated feature engineering**

### ✅ Model Performance
- **100% accuracy** on training data
- **Comprehensive model validation**
- **Hyperparameter tuning** with GridSearchCV
- **Cross-validation** for robust evaluation

## 🛠️ Technical Details

### Machine Learning Models
- **Random Forest** (Primary model)
- **XGBoost**
- **Logistic Regression**
- **Support Vector Machines**
- **K-Nearest Neighbors**
- **Gradient Boosting**

### Data Features
- **Binary symptom encoding**
- **Symptom severity weighting** (1-5 scale)
- **Class balancing** for imbalanced data
- **Feature importance analysis**

## 🔍 API Usage (Optional)

The system includes a Flask API for programmatic access:

```bash
cd api
python app.py
```

**API Endpoints:**
- `GET /symptoms` - List all available symptoms
- `POST /predict` - Predict diseases from symptoms
- `GET /diseases` - List all possible diseases

**Example API Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["cough", "fever", "headache"], "top_n": 5}'
```

## 📈 Model Performance

The system achieves:
- **100% accuracy** on test data
- **Perfect separation** in training data
- **Robust predictions** across symptom combinations
- **Medical relevance** in disease rankings

## 🐛 Troubleshooting

### Common Issues

1. **"Model not found" error**
   ```bash
   # Ensure models are trained
   python src/model_training.py
   ```

2. **Streamlit not found**
   ```bash
   pip install streamlit
   ```

3. **Missing data files**
   - Ensure all CSV files are in `data/raw/` directory
   - Check file permissions and paths

4. **Low confidence scores**
   - Try adding more specific symptoms
   - Ensure symptom names match exactly

### Debug Mode

Enable detailed logging by modifying the source files to include:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational and research purposes. Please ensure proper medical consultation for actual health concerns.

## ⚠️ Medical Disclaimer

**Important**: This system is for educational and demonstration purposes only. It should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the code comments
3. Ensure all dependencies are installed
4. Verify data file locations and formats

---

**Happy diagnosing! 🩺✨**