import os
import pickle
import pandas as pd
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load all models and preprocessing objects
def load_artifacts():
    artifacts = {
        'scaler': None,
        'label_encoders': {},
        'models': {}
    }

    try:
        # Load scaler
        with open("data/scaler.pkl", "rb") as f:
            artifacts['scaler'] = pickle.load(f)

        # Load label encoders
        with open("data/label_encoders.pkl", "rb") as f:
            artifacts['label_encoders'] = pickle.load(f)

        # Load models
        model_names = ["Linear", "Ridge", "GradientBoosting", "LightGBM", "CatBoost","XGBoost"]
        for name in model_names:
            with open(f"models/{name}.pkl", "rb") as f:
                artifacts['models'][name] = pickle.load(f)

    except Exception as e:
        print(f"Error loading artifacts: {e}")

    return artifacts

artifacts = load_artifacts()

# Home route: shows HTML form and predictions
@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    error = None

    if request.method == "POST":
        if not all(artifacts.values()):
            error = "Models or preprocessors not loaded correctly"
            return render_template("index.html", prediction=None, error=error)

        try:
            # Get form inputs
            form_data = {
                "Region": request.form["Region"],
                "Soil_Type": request.form["Soil_Type"],
                "Crop": request.form["Crop"],
                "Rainfall_mm": float(request.form["Rainfall_mm"]),
                "Temperature_Celsius": float(request.form["Temperature_Celsius"]),
                "Fertilizer_Used": int(request.form.get("Fertilizer_Used") == "on"),
                "Irrigation_Used": int(request.form.get("Irrigation_Used") == "on"),
                "Weather_Condition": request.form["Weather_Condition"],
                "Days_to_Harvest": float(request.form["Days_to_Harvest"])
            }

            df = pd.DataFrame([form_data])

            # Encode categorical values
            for col in ["Region", "Soil_Type", "Crop", "Weather_Condition"]:
                df[col] = artifacts['label_encoders'][col].transform(df[col])

            # Scale numerical features
            num_cols = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
            df[num_cols] = artifacts['scaler'].transform(df[num_cols])

            # Predict using all models
            predictions = {
                name: round(float(model.predict(df)[0]), 2)
                for name, model in artifacts['models'].items()
            }

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=predictions, error=error)

# JSON API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        required_fields = [
            "Region", "Soil_Type", "Crop", "Rainfall_mm",
            "Temperature_Celsius", "Fertilizer_Used",
            "Irrigation_Used", "Weather_Condition", "Days_to_Harvest"
        ]

        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        df = pd.DataFrame([data])

        # Label encoding
        for col in ["Region", "Soil_Type", "Crop", "Weather_Condition"]:
            df[col] = artifacts['label_encoders'][col].transform(df[col])

        # Scaling
        df[["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]] = artifacts['scaler'].transform(
            df[["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]]
        )

        predictions = {
            name: round(float(model.predict(df)[0]), 2)
            for name, model in artifacts['models'].items()
        }

        return jsonify({"success": True, "predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Batch CSV prediction endpoint
print("Models loaded:", list(artifacts['models'].keys()))

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            # TODO: Add preprocessing and prediction logic
            return jsonify({"success": True, "message": "Batch file received. Processing logic not yet implemented."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type"}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

if __name__ == "__main__":
    app.run(debug=True)
