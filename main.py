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

# Step 3: Save the model
joblib.dump(model, 'loan_prediction_model.pkl')

# Step 4: Create a Flask application
app = Flask(__name__)

# Load the model
model = joblib.load('loan_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Convert input data to DataFrame
    input_data = pd.DataFrame(data[' no_of_dependents', ' education', ' self_employed', ' income_annum', ' loan_amount', ' loan_term', ' cibil_score',' residential_assets_value', ' commercial_assets_value',' luxury_assets_value', ' bank_asset_value'], index=[0])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return jsonify({'prediction':prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# In[ ]:




