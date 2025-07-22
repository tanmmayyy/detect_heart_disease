from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib  # <-- IMPORT JOBLIB
import numpy as np
import os

# Initialize Flask App
app = Flask(__name__)
# Enable CORS for all routes, allowing frontend to make API calls
CORS(app) 

# --- MODEL LOADING ---
# Load the trained model from the pickle file
try:
    # --- FIXED: Use joblib.load to match how the model was saved ---
    model = joblib.load('model/random_forest_model.pkl')
    
    # --- VALIDATE THE LOADED MODEL ---
    # Check if the loaded object has 'predict' and 'predict_proba' methods
    if not (hasattr(model, 'predict') and hasattr(model, 'predict_proba')):
        print("❌ ERROR: The loaded .pkl file is not a valid scikit-learn model.")
        print("Please ensure you are saving the model object itself.")
        model = None
    else:
        print("✅ Model loaded and validated successfully using joblib!")

except FileNotFoundError:
    print("❌ Model file not found. Please ensure 'random_forest_model.pkl' exists in the 'model/' folder.")
    model = None
except Exception as e:
    print(f"❌ An error occurred while loading the model: {e}")
    model = None

# --- FEATURE INFORMATION ---
# This dictionary defines the structure and details of the input features.
FEATURE_INFO = {
    'age': {'name': 'Age', 'type': 'number', 'min': 20, 'max': 100, 'placeholder': 'e.g., 52'},
    'sex': {'name': 'Sex', 'type': 'select', 'options': [{'value': 1, 'label': 'Male'}, {'value': 0, 'label': 'Female'}]},
    'cp': {'name': 'Chest Pain Type', 'type': 'select', 'options': [
        {'value': 0, 'label': 'Typical Angina'},
        {'value': 1, 'label': 'Atypical Angina'},
        {'value': 2, 'label': 'Non-Anginal Pain'},
        {'value': 3, 'label': 'Asymptomatic'}
    ]},
    'trestbps': {'name': 'Resting Blood Pressure', 'type': 'number', 'min': 80, 'max': 200, 'placeholder': 'mm Hg, e.g., 120'},
    'chol': {'name': 'Cholesterol', 'type': 'number', 'min': 100, 'max': 600, 'placeholder': 'mg/dl, e.g., 210'},
    'fbs': {'name': 'Fasting Blood Sugar > 120 mg/dl', 'type': 'select', 'options': [
        {'value': 1, 'label': 'Yes'},
        {'value': 0, 'label': 'No'}
    ]},
    'restecg': {'name': 'Resting ECG', 'type': 'select', 'options': [
        {'value': 0, 'label': 'Normal'},
        {'value': 1, 'label': 'ST-T Wave Abnormality'},
        {'value': 2, 'label': 'Left Ventricular Hypertrophy'}
    ]},
    'thalach': {'name': 'Maximum Heart Rate', 'type': 'number', 'min': 60, 'max': 220, 'placeholder': 'e.g., 150'},
    'exang': {'name': 'Exercise Induced Angina', 'type': 'select', 'options': [
        {'value': 1, 'label': 'Yes'},
        {'value': 0, 'label': 'No'}
    ]},
    'oldpeak': {'name': 'ST Depression', 'type': 'number', 'min': 0, 'max': 7, 'step': 0.1, 'placeholder': 'e.g., 1.8'},
    'slope': {'name': 'ST Slope', 'type': 'select', 'options': [
        {'value': 0, 'label': 'Upsloping'},
        {'value': 1, 'label': 'Flat'},
        {'value': 2, 'label': 'Downsloping'}
    ]}
}

# --- HELPER FUNCTIONS ---
def get_risk_level(probability):
    """Determine a user-friendly risk level based on prediction probability."""
    if probability < 0.3:
        return 'Low'
    elif probability < 0.7:
        return 'Moderate'
    else:
        return 'High'

def get_prediction_message(prediction, probability):
    """Generate a user-friendly message based on the prediction outcome."""
    prob_percent = probability * 100
    if prediction == 1:
        return f"The model indicates a {prob_percent:.1f}% probability of heart disease. Please consult with a healthcare professional for proper evaluation and advice."
    else:
        return f"The model indicates a low probability ({prob_percent:.1f}%) of heart disease. Maintaining a healthy lifestyle and regular check-ups are always recommended."

# --- API ROUTES ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/features', methods=['GET'])
def get_features():
    """Provides the feature information to the frontend to build the form."""
    return jsonify(FEATURE_INFO)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Receives patient data, makes a prediction, and returns the result."""
    if model is None:
        return jsonify({'error': 'Model not loaded or is invalid. Please check server logs for details.'}), 500
        
    try:
        data = request.json
        print(f"📊 Received prediction request: {data}")
        
        required_fields = list(FEATURE_INFO.keys())
        
        missing_fields = [field for field in required_fields if field not in data or data[field] == '']
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        features = [float(data[field]) for field in required_fields]
        
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        probability = model.predict_proba(features_array)[0]
        heart_disease_prob = probability[1]
        
        result = {
            'prediction': int(prediction),
            'probability': float(heart_disease_prob),
            'risk_level': get_risk_level(heart_disease_prob),
            'message': get_prediction_message(prediction, heart_disease_prob)
        }
        
        print(f"✅ Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 400

@app.route('/api/health')
def health_check():
    """A simple health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
