import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Load Model & Normalizer ---
try:
    with open("Model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("Normalizer.pkl", "rb") as f:
        normalizer = pickle.load(f)

    # Exact feature names extracted from Normalizer.pkl
    MODEL_FEATURES = [
        'SeniorCitizen',
        'tenure_trim',
        'MonthlyCharges_trim',
        'TotalCharges_mode_trim',
        'gender_Male',
        'Partner_Yes',
        'Dependents_Yes',
        'PhoneService_Yes',
        'MultipleLines_No phone service',
        'MultipleLines_Yes',
        'InternetService_Fiber optic',
        'InternetService_No',
        'OnlineSecurity_No internet service',
        'OnlineSecurity_Yes',
        'OnlineBackup_No internet service',
        'OnlineBackup_Yes',
        'DeviceProtection_No internet service',
        'DeviceProtection_Yes',
        'TechSupport_No internet service',
        'TechSupport_Yes',
        'StreamingTV_No internet service',
        'StreamingTV_Yes',
        'StreamingMovies_No internet service',
        'StreamingMovies_Yes',
        'PaperlessBilling_Yes',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check',
        'Telecom_Partner_BSNL',
        'Telecom_Partner_Jio',
        'Telecom_Partner_VI-!dea',
        'Contract_od'
    ]

    print(f"✅ Model loaded. Expecting {len(MODEL_FEATURES)} features.")

except Exception as e:
    print(f"❌ Error loading model files: {e}")
    exit()


# --- Telecom Partner mapping (mirrors training logic in main.py) ---
# PaymentMethod -> Telecom_Partner
TELECOM_PARTNER_MAP = {
    'Mailed check': 'Airtel',
    'Bank transfer (automatic)': 'VI-!dea',
    'Credit card (automatic)': 'BSNL',
    'Electronic check': 'Jio'
}

# Contract ordinal mapping (OrdinalEncoder: Month-to-month=0, One year=1, Two year=2)
CONTRACT_ORDINAL_MAP = {
    'Month-to-month': 0.0,
    'One year': 1.0,
    'Two year': 2.0
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # --- Initialize feature DataFrame with zeros ---
        input_df = pd.DataFrame(0.0, index=[0], columns=MODEL_FEATURES)

        # --- Numerical features ---
        tenure = float(data.get('tenure', 0))
        monthly_charges = float(data.get('MonthlyCharges', 0))

        # Auto-calculate TotalCharges if missing or zero
        total_charges = data.get('TotalCharges', '')
        if total_charges == '' or float(total_charges) == 0:
            total_charges = tenure * monthly_charges
        else:
            total_charges = float(total_charges)

        input_df['SeniorCitizen']         = int(data.get('SeniorCitizen', 0))
        input_df['tenure_trim']           = tenure
        input_df['MonthlyCharges_trim']   = monthly_charges
        input_df['TotalCharges_mode_trim'] = total_charges

        # --- Binary / One-Hot features ---

        # Gender
        input_df['gender_Male'] = 1 if data.get('gender') == 'Male' else 0

        # Partner
        input_df['Partner_Yes'] = 1 if data.get('Partner') == 'Yes' else 0

        # Dependents
        input_df['Dependents_Yes'] = 1 if data.get('Dependents') == 'Yes' else 0

        # PhoneService
        input_df['PhoneService_Yes'] = 1 if data.get('PhoneService') == 'Yes' else 0

        # PaperlessBilling
        input_df['PaperlessBilling_Yes'] = 1 if data.get('PaperlessBilling') == 'Yes' else 0

        # MultipleLines
        multiple_lines = data.get('MultipleLines', '')
        if multiple_lines == 'No phone service':
            input_df['MultipleLines_No phone service'] = 1
        elif multiple_lines == 'Yes':
            input_df['MultipleLines_Yes'] = 1

        # InternetService
        internet_service = data.get('InternetService', '')
        if internet_service == 'Fiber optic':
            input_df['InternetService_Fiber optic'] = 1
        elif internet_service == 'No':
            input_df['InternetService_No'] = 1

        # Internet-dependent services (OnlineSecurity, OnlineBackup, DeviceProtection,
        #                               TechSupport, StreamingTV, StreamingMovies)
        internet_services = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        for svc in internet_services:
            val = data.get(svc, '')
            if val == 'Yes':
                input_df[f'{svc}_Yes'] = 1
            elif val == 'No internet service':
                input_df[f'{svc}_No internet service'] = 1

        # PaymentMethod
        payment_method = data.get('PaymentMethod', '')
        if payment_method == 'Credit card (automatic)':
            input_df['PaymentMethod_Credit card (automatic)'] = 1
        elif payment_method == 'Electronic check':
            input_df['PaymentMethod_Electronic check'] = 1
        elif payment_method == 'Mailed check':
            input_df['PaymentMethod_Mailed check'] = 1
        # 'Bank transfer (automatic)' is the dropped baseline — all zeros

        # Telecom_Partner (derived from PaymentMethod, same logic as training)
        telecom_partner = TELECOM_PARTNER_MAP.get(payment_method, '')
        if telecom_partner == 'BSNL':
            input_df['Telecom_Partner_BSNL'] = 1
        elif telecom_partner == 'Jio':
            input_df['Telecom_Partner_Jio'] = 1
        elif telecom_partner == 'VI-!dea':
            input_df['Telecom_Partner_VI-!dea'] = 1
        # 'Airtel' is the dropped baseline — all zeros

        # Contract (Ordinal encoded: Month-to-month=0, One year=1, Two year=2)
        contract = data.get('Contract', 'Month-to-month')
        input_df['Contract_od'] = CONTRACT_ORDINAL_MAP.get(contract, 0.0)

        # --- Scale & Predict ---
        scaled_features = normalizer.transform(input_df)
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]

        churn_prob = round(float(probabilities[1]) * 100, 2)
        result = "Yes" if prediction == 1 else "No"

        # Debug log
        print(f"\n--- Prediction ---")
        print(f"Tenure: {tenure}, Monthly: {monthly_charges}, Total: {total_charges}")
        print(f"Telecom Partner: {telecom_partner}, Contract: {contract}")
        print(f"Churn Probability: {churn_prob}%  →  {result}")

        return jsonify({
            'churn': result,
            'probability': churn_prob
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)