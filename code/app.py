from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('./randomForest.joblib')
data = [50,70,5,300,5,4,3,0,2,0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        param1 = data['pct']
        param2 = data['probability']
        param3 = data['available']
        param4 = data['distance']
        param5 = data['price']
        param6 = data['stars']
        param7 = data['construction']
        param8 = data['events']
        param9 = data['congestion']
        param10 = data['hazards']

        data = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10] 
        data = process_input(data)
        prediction = model.predict(data)

        response = {
            'prediction': prediction.tolist()
        }

        return jsonify(response)

    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {e}"}), 400

def process_input(data):
    std = np.array([  5.78894742,   4.60483703, 146.83262808, 238.75573511,
         1.02875985,   1.41925462,  11.94819279,   1.        ,
         2.478264  ,   1.        ])
    mean = np.array([ 38.65624111,  84.24936025,  83.19277794, 645.22064259,
         2.11094987,   0.70699405,   9.38470287,   0.        ,
         0.75604208,   0.        ])
    
    data = (data - mean) / std
    df = pd.DataFrame([data])
    return df

if __name__ == '__main__':
    app.run(debug=True, port = 5001)