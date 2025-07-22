from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    with open('model/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please ensure random_forest_model.pkl exists in the model folder.")
    model = None

# Feature names and their descriptions
FEATURE_INFO = {
    'age': {'name': 'Age', 'type': 'number', 'min': 20, 'max': 100, 'description': 'Age in years'},
    'sex': {'name': 'Sex', 'type': 'select', 'options': [{'value': 1, 'label': 'Male'}, {'value': 0, 'label': 'Female'}], 'description': 'Biological sex'},
    'chest_pain_type': {'name': 'Chest Pain Type', 'type': 'select', 'options': [
        {'value': 0, 'label': 'Typical Angina'},
        {'value': 1, 'label': 'Atypical Angina'},
        {'value': 2, 'label': 'Non-Anginal Pain'},
        {'value': 3, 'label': 'Asymptomatic'}
    ], 'description': 'Type of chest pain experienced'},
    'resting_bp': {'name': 'Resting Blood Pressure', 'type': 'number', 'min': 80, 'max': 200, 'description': 'Resting blood pressure (mm Hg)'},
    'cholesterol': {'name': 'Cholesterol', 'type': 'number', 'min': 100, 'max': 400, 'description': 'Serum cholesterol (mg/dl)'},
    'fasting_blood_sugar': {'name': 'Fasting Blood Sugar', 'type': 'select', 'options': [
        {'value': 0, 'label': '≤ 120 mg/dl'},
        {'value': 1, 'label': '> 120 mg/dl'}
    ], 'description': 'Fasting blood sugar level'},
    'resting_ecg': {'name': 'Resting ECG', 'type': 'select', 'options': [
        {'value': 0, 'label': 'Normal'},
        {'value': 1, 'label': 'ST-T Wave Abnormality'},
        {'value': 2, 'label': 'Left Ventricular Hypertrophy'}
    ], 'description': 'Resting electrocardiographic results'},
    'max_heart_rate': {'name': 'Maximum Heart Rate', 'type': 'number', 'min': 60, 'max': 220, 'description': 'Maximum heart rate achieved'},
    'exercise_angina': {'name': 'Exercise Induced Angina', 'type': 'select', 'options': [
        {'value': 0, 'label': 'No'},
        {'value': 1, 'label': 'Yes'}
    ], 'description': 'Exercise induced angina'},
    'oldpeak': {'name': 'ST Depression', 'type': 'number', 'min': 0, 'max': 7, 'step': 0.1, 'description': 'ST depression induced by exercise'},
    'st_slope': {'name': 'ST Slope', 'type': 'select', 'options': [
        {'value': 0, 'label': 'Upsloping'},
        {'value': 1, 'label': 'Flat'},
        {'value': 2, 'label': 'Downsloping'}
    ], 'description': 'Slope of peak exercise ST segment'}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/features')
def get_features():
    """Return feature information for the frontend"""
    return jsonify(FEATURE_INFO)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        
        # Extract features in the correct order
        features = [
            float(data.get('age', 0)),
            float(data.get('sex', 0)),
            float(data.get('chest_pain_type', 0)),
            float(data.get('resting_bp', 0)),
            float(data.get('cholesterol', 0)),
            float(data.get('fasting_blood_sugar', 0)),
            float(data.get('resting_ecg', 0)),
            float(data.get('max_heart_rate', 0)),
            float(data.get('exercise_angina', 0)),
            float(data.get('oldpeak', 0)),
            float(data.get('st_slope', 0))
        ]
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Get probability for positive class (heart disease)
        heart_disease_prob = probability[1] if len(probability) > 1 else probability[0]
        
        result = {
            'prediction': int(prediction),
            'probability': float(heart_disease_prob),
            'risk_level': get_risk_level(heart_disease_prob),
            'message': get_prediction_message(prediction, heart_disease_prob)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.3:
        return 'Low'
    elif probability < 0.7:
        return 'Moderate'
    else:
        return 'High'

def get_prediction_message(prediction, probability):
    """Generate user-friendly message"""
    if prediction == 1:
        return f"The model indicates a {probability*100:.1f}% probability of heart disease. Please consult with a healthcare professional for proper evaluation."
    else:
        return f"The model indicates a {(1-probability)*100:.1f}% probability of no heart disease. However, regular check-ups are always recommended."

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)