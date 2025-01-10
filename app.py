from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and training columns
model = joblib.load('loan_prediction_model.pkl')
with open('training_columns.pkl', 'rb') as file:
    training_columns = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    user_input = {
        'Gender': request.form['Gender'],
        'Married': request.form['Married'],
        'Dependents': request.form['Dependents'],
        'Education': request.form['Education'],
        'Self_Employed': request.form['Self_Employed'],
        'ApplicantIncome': float(request.form['ApplicantIncome'])/ 10,
        'CoapplicantIncome': float(request.form['CoapplicantIncome'])/ 10,
        'LoanAmount': float(request.form['LoanAmount'])/ 1000,
        'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
        'Credit_History': float(request.form['Credit_History']),
        'Property_Area': request.form['Property_Area']
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([user_input])

    # One-hot encode categorical variables
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Align with training columns
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    result = "Approved" if prediction[0] == 1 else "Not Approved"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
