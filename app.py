from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS, cross_origin

# Load the models
forest_model = joblib.load('forest_regression_model.pkl')
encoder = joblib.load('forest_regression_encoder.pkl')
scaler = joblib.load('forest_regression_scaler.pkl')

app = Flask(__name__)
CORS(app)

# Function to predict new data
def predict_new_data(new_data):
    # Replace '-' with -1 in numeric columns
    numeric_cols = ['Length ', 'Width ', 'Diameter ', 'TriSide', 'Height']
    new_data[numeric_cols] = new_data[numeric_cols].replace('-', -1).astype(float)

    # Fill missing values in new data based on the shape
    new_data['Length '] = new_data.apply(lambda row: -1 if row['Shape'] in ['Triangle', 'Round'] else row['Length '], axis=1)
    new_data['Width '] = new_data.apply(lambda row: -1 if row['Shape'] in ['Triangle', 'Round'] else row['Width '], axis=1)
    new_data['Diameter '] = new_data.apply(lambda row: row['Diameter '] if row['Shape'] == 'Round' else -1, axis=1)
    new_data['TriSide'] = new_data.apply(lambda row: row['TriSide'] if row['Shape'] == 'Triangle' else -1, axis=1)

    # Preprocess new data
    encoded_new_data = encoder.transform(new_data[['Shape', 'Type']])
    scaled_new_data = scaler.transform(new_data[['Length ', 'Width ', 'Diameter ', 'TriSide', 'Height']])
    processed_new_data = np.concatenate([encoded_new_data, scaled_new_data], axis=1)

    return forest_model.predict(processed_new_data)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    shape = data["Shape"]
    type_ = data["Type"]
    length = data["Length"]
    width = data["Width"]
    diameter = data["Diameter"]
    tri_side = data["TriSide"]
    height = data["Height"]

    user_data = {
        'Shape': shape,
        'Type': type_,
        'Length ': float(length) if length else -1,
        'Width ': float(width) if width else -1,
        'Diameter ': float(diameter) if diameter else -1,
        'TriSide': float(tri_side) if tri_side else -1,
        'Height': float(height) if height else -1
    }
    new_data = pd.DataFrame([user_data])

    predictions_new = predict_new_data(new_data)
    predicted_values = predictions_new.tolist()[0]

    # Assuming the additional calculations are correct and relevant
    new_data['Soak_Time'] = new_data['Height'].astype(int)
    new_data['TopTemp'] = float(predicted_values[0])
    new_data['BotTemp'] = float(predicted_values[1])
    new_data['PreHeat'] = float(predicted_values[2])
    new_data['Cut'] = float(predicted_values[3])
    new_data['LUP_Curing'] = float(predicted_values[4])
    new_data['Bot_Curing'] = float(predicted_values[5])
    new_data['LUP_cm'] = float(predicted_values[6])
    new_data['LUP_sec'] = new_data['LUP_cm'] / 0.5
    new_data['CT'] = new_data['TopTemp'] - new_data['BotTemp']
    new_data['ULT'] = new_data['PreHeat'] / (new_data['Cut'] + 1)
    new_data['RT'] = new_data['LUP_Curing'] + new_data['Bot_Curing'] - new_data['LUP_cm']
    print(jsonify(new_data.to_dict('records')[0]))

    return jsonify(new_data.to_dict('records')[0])

if __name__ == '__main__':
     app.run()