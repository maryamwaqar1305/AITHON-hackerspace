
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

app = Flask(__name__)

# Placeholder for the AI model
fraud_model = RandomForestClassifier()

# Placeholder data for training
# In a real-world scenario, this would be replaced with real transaction data
def train_model():
    # Simulate training data: [amount, location_code, time_code, is_fraud]
    data = np.array([
        [100, 1, 1, 0],
        [1000, 2, 3, 1],
        [150, 1, 2, 0],
        [2000, 3, 1, 1],
        [50, 2, 2, 0],
        [2500, 3, 3, 1],
    ])
    X, y = data[:, :-1], data[:, -1]
    fraud_model.fit(X, y)

# Train the model
train_model()

@app.route('/')
def index():
    return render_template('payment_form.html')

@app.route('/process_payment', methods=['POST'])
def process_payment():
    card_number = request.form['card_number']
    expiry_date = request.form['expiry_date']
    cvv = request.form['cvv']
    amount = float(request.form['amount'])
    location_code = int(request.form['location_code'])  # Example: Map location to code
    time_code = int(request.form['time_code'])  # Example: Map time to code

    # Simulate transaction processing
    success = np.random.rand() > 0.1  # 90% success rate

    # Fraud detection
    is_fraud = fraud_model.predict([[amount, location_code, time_code]])[0]

    if is_fraud:
        return jsonify({"status": "failure", "message": "Transaction flagged as fraudulent"})
    elif success:
        return jsonify({"status": "success", "message": "Transaction processed successfully"})
    else:
        return jsonify({"status": "failure", "message": "Transaction failed"})

@app.route('/transaction_insights', methods=['GET'])
def transaction_insights():
    # Placeholder for transaction data
    transactions = pd.DataFrame({
        "amount": [100, 1000, 150, 2000, 50, 2500],
        "location_code": [1, 2, 1, 3, 2, 3],
        "time_code": [1, 3, 2, 1, 2, 3],
    })

    insights = {
        "average_amount": transactions['amount'].mean(),
        "common_transaction_time": transactions['time_code'].mode()[0],
        "high_risk_location": transactions.groupby('location_code').size().idxmax(),
    }

    return jsonify(insights)

if __name__ == '__main__':
    app.run(debug=True)

