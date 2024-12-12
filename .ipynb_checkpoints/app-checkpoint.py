{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbdc8264",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify, render_template\n",
    "import joblib\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import shap\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load the models and scaler\n",
    "rf_model = joblib.load('rf_model.joblib')\n",
    "dl_model = tf.keras.models.load_model('dl_model.h5')\n",
    "scaler = joblib.load('scaler.joblib')\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    # Get input data from the form\n",
    "    features = [float(x) for x in request.form.values()]\n",
    "    features = np.array(features).reshape(1, -1)\n",
    "    \n",
    "    # Scale the input\n",
    "    scaled_features = scaler.transform(features)\n",
    "    \n",
    "    # Make predictions\n",
    "    rf_pred = rf_model.predict(scaled_features)[0]\n",
    "    dl_pred = np.argmax(dl_model.predict(scaled_features), axis=-1)[0]\n",
    "    \n",
    "    # Generate SHAP explanations\n",
    "    explainer_rf = shap.TreeExplainer(rf_model)\n",
    "    shap_values_rf = explainer_rf.shap_values(scaled_features)\n",
    "    \n",
    "    background = shap.sample(scaled_features, 100)\n",
    "    explainer_dl = shap.KernelExplainer(dl_model.predict, background)\n",
    "    shap_values_dl = explainer_dl.shap_values(scaled_features)\n",
    "    \n",
    "    # Convert SHAP values to list for JSON serialization\n",
    "    shap_values_rf = [sv.tolist() for sv in shap_values_rf]\n",
    "    shap_values_dl = [sv.tolist() for sv in shap_values_dl]\n",
    "    \n",
    "    return jsonify({\n",
    "        'rf_prediction': rf_pred,\n",
    "        'dl_prediction': dl_pred,\n",
    "        'rf_explanation': shap_values_rf,\n",
    "        'dl_explanation': shap_values_dl\n",
    "    })\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
