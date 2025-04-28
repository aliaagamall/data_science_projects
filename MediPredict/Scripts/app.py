from flask import Flask, request, render_template
import pandas as pd
from model import MedicalEntrancePredictor
from utils import encode_ordinal

app = Flask(__name__)

# Load predictor
predictor = MedicalEntrancePredictor()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'Caste': [request.form['Caste']],
        'coaching': [request.form['coaching']],
        'medium': [request.form['medium']],
        'Father_occupation': [request.form['Father_occupation']],
        'Mother_occupation': [request.form['Mother_occupation']],
        'Class_ten_education': [request.form['Class_ten_education']],
        'Class_X_Grade': [request.form['Class_X_Grade']],
        'Class_XII_Grade': [request.form['Class_XII_Grade']]
    }
    input_data = pd.DataFrame(data)
    
    # Make prediction
    prediction = predictor.predict(input_data)[0]
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':

    app.run(debug=True)