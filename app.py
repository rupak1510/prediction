from flask import Flask, request, jsonify , send_from_directory
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)
# Load  model
import joblib

model = joblib.load('random_forest_model_2.pkl')
print("Model type:", type(model))

# Area to one-hot encoding mapping
area_columns = ['area_AEP', 'area_COMED', 'area_DAYTON', 'area_DEOK', 'area_DOM', 'area_DUQ']
area_mapping = {
    'AEP': [1, 0, 0, 0, 0, 0],
    'COMED': [0, 1, 0, 0, 0, 0],
    'DAYTON': [0, 0, 1, 0, 0, 0],
    'DEOK': [0, 0, 0, 1, 0, 0],
    'DOM': [0, 0, 0, 0, 1, 0],
    'DUQ': [0, 0, 0, 0, 0, 1]
}

from flask import render_template

@app.route('/')
def serve_index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    area = data['area']  
    date = data['date']  
    time = data['time']  

    # Validate area
    if area not in area_mapping:
        return jsonify({'error': 'Invalid area'}), 400

    # Combine date and time into a datetime string
    datetime_str = f"{date} {time}"
    dt = pd.to_datetime(datetime_str)

    # Extract features as per training
    hours = dt.hour
    day = dt.day
    month = dt.month
    week = dt.isocalendar().week
    year = dt.year - 2000  # Subtract 2000 as per training

    
    area_encoded = area_mapping[area] 

    
    # Order: area_AEP, area_COMED, area_DAYTON, area_DEOK, area_DOM, area_DUQ, hours, day, month, week, year
    input_data = [day, month,year, week, hours] +area_encoded

    # Run prediction
    try:
        prediction = model.predict([input_data])[0]  # Assuming single value output
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)