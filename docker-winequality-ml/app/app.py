# Import Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

# API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():
    if rf:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print("Predicting")
            prediction = list(rf.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    port = 5000
    rf = joblib.load("../model/RandomForest.pkl") # Load "RandomForest.pkl"
    print ('Model loaded')
    model_columns = joblib.load("../model/model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(host='0.0.0.0', port=port, debug=True)
