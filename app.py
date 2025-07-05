from flask import Flask, jsonify, request
import pandas as pd
from joblib import load
from datetime import datetime
from flask_cors import CORS
import logging
from typing import Dict, Any, Union
import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Using scikit-learn version: {sklearn.__version__}")

# Initialize model and metadata
model = None
encoder = None
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
        
        if isinstance(loaded_data, dict):
            model = loaded_data.get('model')
            encoder = loaded_data.get('encoder')
            feature_names = loaded_data.get('feature_names', FEATURE_NAMES)
            categorical_cols = loaded_data.get('categorical_cols', CATEGORICAL_COLS)
            numerical_cols = loaded_data.get('numerical_cols', NUMERICAL_COLS)
            return model, encoder, feature_names, categorical_cols, numerical_cols
        elif hasattr(loaded_data, 'predict'):
            logger.warning("Loaded file contains only model object, using default feature names")
            return loaded_data, None, FEATURE_NAMES, CATEGORICAL_COLS, NUMERICAL_COLS
        else:
            raise ValueError("Unknown model file format")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

# Load the model
try:
    model, encoder, FEATURE_NAMES, CATEGORICAL_COLS, NUMERICAL_COLS = load_model_with_fallback(
        'model/maternal_risk_predictor(1).pkl'
    )
    logger.info("Model and metadata loaded successfully")
    
    # Create proper test data with correct values
    test_data = {
        'Age': 25,
        'Location': 'Urban',
        'ChronicalCondition': 'No',
        'PreviousPregnancyComplication': 'No',
        'GestationAge': 38,
        'Gravidity': 2,
        'Parity': 1,
        'AntenatalVisit': 4,
        'Systolic': 120,
        'Dystolic': 80,
        'PulseRate': 70,
        'SpecificComplication': 'No',
        'DeliveryMode': 'Spontaneous Vertex Delivery',
        'StaffConductedDelivery': 'Skilled'
    }
    
    # Prepare test data
    test_df = pd.DataFrame([test_data])
    
    if encoder:
        # If we have an encoder, use it to transform categorical features
        encoded_categorical = encoder.transform(test_df[CATEGORICAL_COLS])
        encoded_df = pd.DataFrame(encoded_categorical, 
                                columns=encoder.get_feature_names_out(CATEGORICAL_COLS))
        numerical_df = test_df[NUMERICAL_COLS]
        test_df = pd.concat([numerical_df, encoded_df], axis=1)
    
    try:
        prediction = model.predict(test_df)
        logger.info(f"Model test successful. Prediction: {prediction}")
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
    """Convert API input to model-ready DataFrame"""
    mapped_data = {
        FEATURE_MAPPING[k]: v 
        for k, v in api_data.items() 
        if k in FEATURE_MAPPING
    }
    
    # Create DataFrame with raw values
    input_df = pd.DataFrame([mapped_data])
    
    if encoder:
        # One-hot encode categorical features
        encoded_categorical = encoder.transform(input_df[CATEGORICAL_COLS])
        encoded_df = pd.DataFrame(encoded_categorical, 
                                columns=encoder.get_feature_names_out(CATEGORICAL_COLS))
        numerical_df = input_df[NUMERICAL_COLS]
        input_df = pd.concat([numerical_df, encoded_df], axis=1)
    
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
        
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400
        
        validation_error = validate_input(data)
        if validation_error:
            logger.error(f"Validation error: {validation_error['error']}")
            return jsonify(validation_error), 400
        
        input_df = prepare_input_data(data)
        logger.info(f"Input data prepared with columns: {input_df.columns.tolist()}")
        
        predicted_class = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of High Risk
        
        status = "High" if predicted_class == 1 else "Low"
        response = {
            "patientId": data.get('patientId', ''),
            "patientName": data.get('patientName', ''),
            "riskLevel": status,
            "probability": round(float(prediction_proba), 4),
            "recommendations": RECOMMENDATIONS[status],
            "timestamp": datetime.now().isoformat(),
            "inputFeatures": data  # Return original input features
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