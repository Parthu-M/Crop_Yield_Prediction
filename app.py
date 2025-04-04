from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import warnings
from decimal import Decimal, InvalidOperation

app = Flask(__name__)

# Configuration
ARTIFACTS_DIR = Path("model_artifacts")
DATA_DIR = Path("data")

# Suppress warnings
warnings.filterwarnings('ignore')

# Load preprocessing artifacts
try:
    with open(ARTIFACTS_DIR / 'preprocessing_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    scaler = artifacts['scaler']
    label_encoders = artifacts['label_encoders']
    numerical_cols = artifacts['numerical_cols']
    categorical_cols = artifacts['categorical_cols']
    bool_cols = artifacts['bool_cols']
    feature_columns = artifacts['feature_names']
except Exception as e:
    print(f"Error loading preprocessing artifacts: {str(e)}")
    raise e

# Load TabNet model (primary model)
try:
    tabnet_model = TabNetRegressor()
    tabnet_model.load_model(DATA_DIR / "crop_yield_tabnet.pth.zip")
    print("Successfully loaded TabNet model")
except Exception as e:
    print(f"Error loading TabNet: {str(e)}")
    tabnet_model = None

# Load other models as fallback
fallback_models = {}
model_files = {
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl"
}

for name, filename in model_files.items():
    try:
        with open(ARTIFACTS_DIR / filename, 'rb') as f:
            fallback_models[name] = pickle.load(f)
    except Exception as e:
        print(f"Error loading {name}: {str(e)}")

def parse_decimal_input(value_str):
    """Parse decimal input with flexible precision"""
    try:
        return float(Decimal(value_str))
    except (ValueError, InvalidOperation):
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data with decimal parsing
        rainfall = parse_decimal_input(request.form['rainfall'])
        temperature = parse_decimal_input(request.form['temperature'])
        
        if rainfall is None or temperature is None:
            return render_template('index.html', 
                                regions=label_encoders['Region'].classes_,
                                soil_types=label_encoders['Soil_Type'].classes_,
                                crops=label_encoders['Crop'].classes_,
                                weather_conditions=label_encoders['Weather_Condition'].classes_,
                                error="Please enter valid numbers for rainfall and temperature")
        
        input_data = {
            "Region": request.form['region'],
            "Soil_Type": request.form['soil_type'],
            "Crop": request.form['crop'],
            "Rainfall_mm": rainfall,
            "Temperature_Celsius": temperature,
            "Fertilizer_Used": request.form.get('fertilizer') == 'on',
            "Irrigation_Used": request.form.get('irrigation') == 'on',
            "Weather_Condition": request.form['weather'],
            "Days_to_Harvest": int(request.form['days_to_harvest'])
        }
        
        # Make prediction
        prediction, fallback_predictions = predict_yield(input_data)
        
        return render_template('result.html', 
                            prediction=prediction,
                            fallback_predictions=fallback_predictions,
                            input_data=input_data)
    
    # For GET request, show the form
    return render_template('index.html', 
                         regions=label_encoders['Region'].classes_,
                         soil_types=label_encoders['Soil_Type'].classes_,
                         crops=label_encoders['Crop'].classes_,
                         weather_conditions=label_encoders['Weather_Condition'].classes_,
                         error=None)

def predict_yield(input_data):
    """Make predictions prioritizing TabNet with fallback to other models"""
    try:
        # Prepare input data
        df_input = pd.DataFrame([input_data])
        
        # Process categorical features
        for col in categorical_cols:
            if input_data[col] in label_encoders[col].classes_:
                df_input[col] = label_encoders[col].transform([input_data[col]])
            else:
                df_input[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])
        
        # Process numerical features
        df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
        
        # Process boolean features
        for col in bool_cols:
            df_input[col] = df_input[col].astype(int)
        
        # Ensure correct feature columns and order
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)
        
        X_numpy = df_input.values
        
        # Primary prediction with TabNet
        prediction = None
        if tabnet_model is not None:
            try:
                tabnet_pred = tabnet_model.predict(X_numpy)[0][0]
                prediction = {
                    "model": "TabNet",
                    "value": float(tabnet_pred),  # Keep full precision
                    "unit": "tons/ha"
                }
            except Exception as e:
                print(f"TabNet prediction failed: {str(e)}")
        
        # Fallback predictions if TabNet fails
        fallback_predictions = {}
        if prediction is None:
            for name, model in fallback_models.items():
                try:
                    pred = model.predict(X_numpy)[0]
                    fallback_predictions[name] = {
                        "value": float(pred),  # Keep full precision
                        "unit": "tons/ha"
                    }
                except Exception as e:
                    print(f"{name} prediction failed: {str(e)}")
            
            # If we have fallback predictions, use the first one as primary
            if fallback_predictions:
                first_model = next(iter(fallback_predictions))
                prediction = {
                    "model": first_model,
                    "value": fallback_predictions[first_model]["value"],
                    "unit": "tons/ha"
                }
                del fallback_predictions[first_model]
        
        return prediction, fallback_predictions
    
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return None, {}

if __name__ == '__main__':
    app.run(debug=True)