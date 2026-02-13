from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and data
try:
    data = pd.read_csv('Cleaned_data.csv')
    pipe = pickle.load(open('RidgeModel.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model or data: {e}")
    data = pd.DataFrame({'location': []}) # Fallback if file not found
    pipe = None

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        sqft = float(request.form.get('total_sqft'))

        if not location or not bhk or not bath or not sqft:
             return "Please fill all the fields", 400

        # Create input DataFrame expected by the pipeline
        input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        
        # Predict
        prediction = pipe.predict(input_data)[0]
        
        # Currency formatting for Lakhs
        return str(np.round(prediction, 2))
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)