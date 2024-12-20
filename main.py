import pandas as pd
import pickle
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
data = pd.read_csv('nenene.csv')

# Separate features and target variable
X = data.drop(['loan_id' ,' loan_status'], axis=1)
y = data[' loan_status']
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save the model to a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Initialize the Flask app
app = Flask(__name__)

# Load the model when the app starts
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Flask endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON request
        data = request.get_json(force=True)
        #validate required field 
        required_fields=['dependents','education','self_employed','annual_income','loan_amount','loan_term','cibil_score','residential_assets','commercial_assets','luxury_assets','bank_assets']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"},400)
    


        # Create a DataFrame for model input
        input_data = pd.DataFrame([{
            'dependents': data.get('dependents', 0),
            'education': data.get('education', ''),
            'self_employed': data.get('self_employed', ''),
            'annual_income': data.get('annual_income', 0),
            'loan_amount': data.get('loan_amount', 0),
            'loan_term': data.get('loan_term', 0),
            'cibil_score': data.get('cibil_score', 0),
            'residential_assets': data.get('residential_assets', 0),
            'commercial_assets': data.get('commercial_assets', 0),
            'luxury_assets': data.get('luxury_assets', 0),
            'bank_assets': data.get('bank_assets', 0),
        }])

        

        # Make prediction
        prediction = model.predict(input_data)

        # Return the prediction as JSON
        return jsonify({'prediction': float(prediction[0])}),200

    except Exception as e:
        return jsonify({"error": f"An error is occured: {str(e)}"}), 500
    
    

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
