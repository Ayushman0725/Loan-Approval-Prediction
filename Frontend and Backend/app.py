# Author : Arnav A Verma

from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('loan_approval_prediction.joblib')

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    # Extract the input features from the form data
    features = [
        data['name'],
        int(data['dependents']),
        float(data['annualIncome']),
        float(data['loanAmount']),
        int(data['loanTerm']),
        int(data['cibilScore']),
        float(data['residentialAssets']),
        float(data['commercialAssets']),
        float(data['luxuryAssets']),
        float(data['bankAssets']),
        1 if data['educationStatus'] == 'graduate' else 0,
        1 if data['selfEmployed'] == 'yes' else 0
    ]
    # Reshape the input to the model's expected format
    features = [features[1:]]  # Exclude the 'name' field
    # Predict the loan approval status
    prediction = model.predict(features)[0]
    # Convert prediction to approval status
    approval_status = 'approved' if prediction == 1 else 'rejected'
    return jsonify({'approval_status': approval_status})

if __name__ == '__main__':
    app.run(debug=True)
