from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the models, scaler, and LabelEncoder
rf_model = joblib.load('rf_model.joblib')
dl_model = tf.keras.models.load_model('dl_model.h5')
scaler = joblib.load('scaler.joblib')
le = joblib.load('label_encoder.joblib')

@app.route('/')
def home():
    return "Welcome to the Disease Prediction API!"

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

# Handler function for serverless deployment
def handler(event, context):
    return app(event, context)

# Uncomment below if running locally (not needed for serverless)
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
