from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

# Load the models, scaler, and LabelEncoder
rf_model = joblib.load('rf_model.joblib')
dl_model = tf.keras.models.load_model('dl_model.h5')
scaler = joblib.load('scaler.joblib')
le = joblib.load('label_encoder.joblib')

# Load the dataset
data = pd.read_csv('symbipredict_2022.csv')

# Load the unique disease names
disease_names = le.classes_

@app.route('/')
def home():
    return render_template('index.html', column_names=data.columns.tolist()[:-1])

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    data = request.form.to_dict(flat=False)
    features = [1 if x[0] == 'Yes' else 0 for x in data.values()]
    features = np.array(features).reshape(1, -1)
    
    # Scale the input using the loaded scaler
    scaled_features = scaler.transform(features)
    
    # Make predictions
    rf_pred_index = rf_model.predict(scaled_features)[0]
    dl_pred_index = np.argmax(dl_model.predict(scaled_features), axis=-1)[0]
    
    # Decode the predicted indices to disease names
    rf_pred = le.inverse_transform([rf_pred_index])[0]
    dl_pred = le.inverse_transform([dl_pred_index])[0]
    
    # Return the predictions as JSON
    return jsonify({
        'rf_prediction': rf_pred,
        'dl_prediction': dl_pred
    })

##if __name__ == '__main__':
##    app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
