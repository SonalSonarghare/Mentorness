import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('gradient_boosting_model.pkl')

# Define categories for each categorical feature
categories = {
    'Vehicle_Type': ['Bus', 'Car', 'Motorcycle', 'SUV', 'Sedan', 'Truck', 'Van'],
    'Lane_Type': ['Regular', 'Express'],
    'Geographical_Location': ['13.059816123454882, 77.77068662374292',
                              '13.042660878688794, 77.47580097259879',
                              '12.84197701525119, 77.67547528176169',
                              '12.936687032945434, 77.53113977439017',
                              '13.21331620748757, 77.55413526894684'],
    'Vehicle_Plate_Number': [],
    'Vehicle_Dimensions': ['Large', 'Medium', 'Small'],
    'FastagID': [],
    'TollBoothID': ['B-102', 'A-101', 'C-103', 'D-106', 'D-105', 'D-104']
}

# Function to preprocess input data
def preprocess_data(data):
    processed_data = []
    for feature, cat in categories.items():
        if cat:  # If predefined categories exist
            encoder = LabelEncoder()
            encoder.fit(cat)
            encoded_value = encoder.transform([data[feature]])[0]
        else:  # If free text or undefined categories
            encoded_value = hash(data[feature]) % 1000  # Example: Hash and modulo to keep within a range
        processed_data.append(encoded_value)
    
    numerical_features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed', 'Hour', 'DayOfWeek', 'Month']
    for feature in numerical_features:
        processed_data.append(float(data[feature]))
    
    return np.array([processed_data])

# Route to render HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the request
    try:
        features = preprocess_data(data)
        prediction = model.predict(features)
        fraud_indicator = int(prediction[0])
        message = 'Transaction is not fraudulent' if fraud_indicator == 0 else 'Transaction is fraudulent'
        return jsonify({'Fraud_indicator': fraud_indicator, 'message': message})
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Error making prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)
