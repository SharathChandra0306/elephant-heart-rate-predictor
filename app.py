# !pip install numpy
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('models/model.pkl', 'rb'))

# Load the scaler and the original DataFrame structure (assumed to be saved during training)
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
X_columns = pickle.load(open('models/X_columns.pkl', 'rb'))  # This should be the structure of the training DataFrame

@app.route('/')
def home():
    return render_templates('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    form_data = request.form
    
    # Convert form data to a DataFrame
    data = {
        'Location': form_data['location'],
        'Behavior': form_data['behavior'],
        'Duration': float(form_data['duration']),
        'Food_Type': form_data['Food_type'],
        'Weather_Condition': form_data['Weather_condition'],
        'Is_Sleeping': form_data['Is_sleeping']
    }
    data_df = pd.DataFrame([data])
    
    # Preprocess the data similarly to the training phase
    data_df = pd.get_dummies(data_df, columns=['Location', 'Behavior', 'Food_Type', 'Weather_Condition', 'Is_Sleeping'])
    
    # Align the DataFrame to match the training DataFrame structure
    data_df = data_df.reindex(columns=X_columns, fill_value=0)
    
    # Standardize the data
    data_scaled = scaler.transform(data_df)
    
    # Make prediction
    prediction = model.predict(data_scaled)
    
    # Render the result in the HTML template
    return render_template('index.html', prediction_text=f'Predicted Heart Rate: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug = True ,host='0.0.0.0',port=5000)
