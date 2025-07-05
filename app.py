from flask import Flask, jsonify, request
import pandas as pd
from joblib import load
from datetime import datetime
from flask_cors import CORS
import logging
from typing import Dict, Any, Union
import os
import sklearn
import numpy as np

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log scikit-learn version
logger.info(f"Using scikit-learn version: {sklearn.__version__}")

# Initialize model and metadata
model = None
FEATURE_NAMES = [
    'Age', 'Location', 'ChronicalCondition', 'PreviousPregnancyComplication',
    'GestationAge', 'Gravidity', 'Parity', 'AntenatalVisit', 'Systolic',
    'Dystolic', 'PulseRate', 'SpecificComplication', 'DeliveryMode',
    'StaffConductedDelivery'
]
CATEGORICAL_COLS = [
    'Location', 'ChronicalCondition', 'PreviousPregnancyComplication',
    'SpecificComplication', 'DeliveryMode', 'StaffConductedDelivery'
]
NUMERICAL_COLS = [
    'Age', 'GestationAge', 'Gravidity', 'Parity', 'AntenatalVisit',
    'Systolic', 'Dystolic', 'PulseRate'
]

def load_model_with_fallback(filepath):
    """Attempt to load model with various fallback strategies"""
    try:
        loaded_data = load(filepath)
        
        # Case 1: Full model data dictionary
        if isinstance(loaded_data, dict):
            model = loaded_data.get('model')
            feature_names = loaded_data.get('feature_names', FEATURE_NAMES)
            categorical_cols = loaded_data.get('categorical_cols', CATEGORICAL_COLS)
            numerical_cols = loaded_data.get('numerical_cols', NUMERICAL_COLS)
            return model, feature_names, categorical_cols, numerical_cols
        
        # Case 2: Just the model object
        elif hasattr(loaded_data, 'predict'):
            logger.warning("Loaded file contains only model object, using default feature names")
            return loaded_data, FEATURE_NAMES, CATEGORICAL_COLS, NUMERICAL_COLS
        
        # Case 3: Unknown format
        else:
            raise ValueError("Unknown model file format")
    
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

# Load the model
try:
    model, FEATURE_NAMES, CATEGORICAL_COLS, NUMERICAL_COLS = load_model_with_fallback(
        'model/maternal_risk_predictor(1).pkl'
    )
    logger.info("Model and metadata loaded successfully")
    
    # Test the model with dummy data
    test_input = pd.DataFrame([{col: 0 for col in FEATURE_NAMES}])
    try:
        model.predict(test_input)
        logger.info("Model test prediction successful")
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
        raise

except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
    raise

# Feature name mapping from API input to model expected names
FEATURE_MAPPING = {
    'age': 'Age',
    'location': 'Location',
    'chronicCondition': 'ChronicalCondition',
    'previousPregnancyComplication': 'PreviousPregnancyComplication',
    'gestationAge': 'GestationAge',
    'gravidity': 'Gravidity',
    'parity': 'Parity',
    'antenatalVisit': 'AntenatalVisit',
    'systolic': 'Systolic',
    'diastolic': 'Dystolic',
    'pulseRate': 'PulseRate',
    'specificComplication': 'SpecificComplication',
    'deliveryMode': 'DeliveryMode',
    'staffConductedDelivery': 'StaffConductedDelivery'
}

# Recommendations
RECOMMENDATIONS = {
    "High": [
        "Immediate consultation with obstetric specialist required",
        "Increased frequency of antenatal visits",
        "Continuous fetal monitoring recommended",
        "Consider hospitalization for close observation",
        "Strict blood pressure monitoring",
        "Bed rest may be advised",
        "Emergency contact numbers provided"
    ],
    "Low": [
        "Continue with regular antenatal check-ups",
        "Maintain balanced diet with adequate protein and iron",
        "Moderate exercise recommended",
        "Monitor blood pressure weekly",
        "Attend all prenatal education classes",
        "Maintain hydration and proper rest",
        "Report any unusual symptoms immediately"
    ]
}

def validate_input(data: Dict[str, Any]) -> Union[None, Dict[str, str]]:
    """Validate input data against model requirements"""
    required_fields = set(FEATURE_MAPPING.keys())
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        return {'error': f'Missing required fields: {", ".join(missing_fields)}'}
    
    type_errors = []
    numeric_fields = ['age', 'gestationAge', 'gravidity', 'parity', 'antenatalVisit', 
                    'systolic', 'diastolic', 'pulseRate']
    
    for field in numeric_fields:
        try:
            float(data[field])
        except ValueError:
            type_errors.append(f"{field} must be a number")
    
    # Value range validation
    if 'age' in data and (float(data['age']) < 10 or float(data['age']) > 60):
        type_errors.append("Age must be between 10 and 60 years")
    if 'gestationAge' in data and (float(data['gestationAge']) < 0 or float(data['gestationAge']) > 45):
        type_errors.append("Gestation age must be between 0 and 45 weeks")
    if 'systolic' in data and (float(data['systolic']) < 50 or float(data['systolic']) > 300):
        type_errors.append("Systolic BP must be between 50 and 300 mmHg")
    if 'diastolic' in data and (float(data['diastolic']) < 30 or float(data['diastolic']) > 200):
        type_errors.append("Diastolic BP must be between 30 and 200 mmHg")
    if 'pulseRate' in data and (float(data['pulseRate']) < 30 or float(data['pulseRate']) > 220):
        type_errors.append("Pulse rate must be between 30 and 220 bpm")
    
    if type_errors:
        return {'error': " | ".join(type_errors)}
    
    return None

def prepare_input_data(api_data: Dict[str, Any]) -> pd.DataFrame:
    """Convert API input to model-ready DataFrame with correct feature names"""
    mapped_data = {
        FEATURE_MAPPING[k]: v 
        for k, v in api_data.items() 
        if k in FEATURE_MAPPING
    }
    
    # Ensure correct data types
    for field in NUMERICAL_COLS:
        if field in mapped_data:
            mapped_data[field] = float(mapped_data[field])
    
    # Create DataFrame with columns in correct order
    input_df = pd.DataFrame([mapped_data], columns=FEATURE_NAMES)
    
    return input_df

@app.route('/')
def home():
    return jsonify({
        "status": "active",
        "service": "Maternal Health Risk Prediction API",
        "model_loaded": model is not None,
        "expected_features": FEATURE_NAMES,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        
        # Get and validate input
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400
        
        validation_error = validate_input(data)
        if validation_error:
            logger.error(f"Validation error: {validation_error['error']}")
            return jsonify(validation_error), 400
        
        # Prepare input data
        input_df = prepare_input_data(data)
        logger.info(f"Input data prepared: {input_df.to_dict()}")
        
        # Make prediction
        predicted_class = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of High Risk
        
        # Prepare response
        status = "High" if predicted_class == 1 else "Low"
        response = {
            "patientId": data.get('patientId', ''),
            "patientName": data.get('patientName', ''),
            "riskLevel": status,
            "probability": round(float(prediction_proba), 4),
            "recommendations": RECOMMENDATIONS[status],
            "timestamp": datetime.now().isoformat(),
            "inputFeatures": input_df.iloc[0].to_dict()
        }
        
        logger.info(f"Prediction successful: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": "An internal server error occurred",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)