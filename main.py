#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from flask import Flask, request, jsonify

# Step 1: Load and preprocess the data
data = pd.read_csv("LOAN_UPDATED.csv")

# Example preprocessing (you may need to adjust this based on your dataset)
# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
X = data.drop(['loan_id',' loan_status'], axis=1)  # Replace 'Loan_Status' with your target column
y = data[' loan_status']  # Replace 'Loan_Status' with your target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)
def predict():
    model.predict()


# Step 3: Save the model
joblib.dump(model, 'loan_prediction_model.pkl')

# Step 4: Create a Flask application
app = Flask(__name__)

# Load the model
model = joblib.load('loan_prediction_model.pkl')

# Flask endpoint with error handling and preprocessing
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    try:
        # Create a DataFrame with properly structured input
        input_data = pd.DataFrame([{
            'no_of_dependents': data.get('no_of_dependents', 0),
            'education': data.get('education', ''),
            'self_employed': data.get('self_employed', ''),
            'income_annum': data.get('income_annum', 0),
            'loan_amount': data.get('loan_amount', 0),
            'loan_term': data.get('loan_term', 0),
            'cibil_score': data.get('cibil_score', 0),
            'residential_assets_value': data.get('residential_assets_value', 0),
            'commercial_assets_value': data.get('commercial_assets_value', 0),
            'luxury_assets_value': data.get('luxury_assets_value', 0),
            'bank_asset_value': data.get('bank_asset_value', 0),
        }])

        # Handle categorical feature encoding
        if 'education' in label_encoders:
            input_data['education'] = label_encoders['education'].transform([input_data['education'][0]])

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 400

    # Make prediction
    try:
        prediction = model.predict(input_data)
    except Exception as e:
        return jsonify({'error': f"Prediction error: {e}"}), 500

    return jsonify({'prediction': prediction[0]})

    
    # Make prediction
    prediction = model.predict(input_data)
    
    return jsonify({'prediction':prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# In[ ]:




