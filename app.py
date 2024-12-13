from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the models, scaler, and LabelEncoder
try:
    rf_model = joblib.load('rf_model.joblib')
    dl_model = tf.keras.models.load_model('dl_model.h5')
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Define the column names from the dataset
column_names = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain",
    "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings",
    "weight_loss", "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
    "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain",
    "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes",
    "acute_liver_failure", "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
    "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure",
    "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements",
    "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness", "cramps",
    "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid",
    "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips",
    "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints",
    "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side",
    "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases",
    "internal_itching", "toxic_look_(typhos)", "depression", "irritability", "muscle_pain", "altered_sensorium",
    "red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic_patches", "watering_from_eyes",
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration",
    "visual_disturbances", "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "stomach_bleeding",
    "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf",
    "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling", "silver_like_dusting",
    "small_dents_in_nails", "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
]

@app.route('/')
def home():
    return render_template('index.html', column_names=column_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = request.form.to_dict(flat=False)
        print("Received Data:", data)

        # Convert data to binary format (Yes/No)
        features = []
        for column in column_names:
            value = data.get(column, ['No'])[0]
            features.append(1 if value == 'Yes' else 0)

        # Convert to numpy array
        features = np.array(features).reshape(1, -1)
        print("Features:", features)

        # Scale the input using the loaded scaler
        scaled_features = scaler.transform(features)
        print("Scaled Features:", scaled_features)

        # Make predictions
        rf_pred_index = rf_model.predict(scaled_features)[0]
        dl_pred_index = np.argmax(dl_model.predict(scaled_features), axis=-1)[0]

        # Decode the predicted indices to disease names
        rf_pred = le.inverse_transform([rf_pred_index])[0]
        dl_pred = le.inverse_transform([dl_pred_index])[0]

        print("RF Prediction:", rf_pred)
        print("DL Prediction:", dl_pred)

        # Return the predictions as JSON
        return jsonify({
            'rf_prediction': rf_pred,
            'dl_prediction': dl_pred
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
