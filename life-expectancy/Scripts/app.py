from flask import Flask, request, render_template
from deploy_model import LifeExpectancyPredictor

app=Flask(__name__,template_folder='templates')

# Initialize the predictor
try:
    predictor = LifeExpectancyPredictor()
except Exception as e:
    print(f"Error initializing predictor: {str(e)}")
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            input_data = {
                'status': request.form['status'],
                'adult_mortality': float(request.form['adult_mortality']),
                'infant_deaths': int(request.form['infant_deaths']),
                'alcohol': float(request.form['alcohol']),
                'percentage_expenditure': float(request.form['percentage_expenditure']),
                'hepatitis_b': float(request.form['hepatitis_b']),
                'measles': int(request.form['measles']),
                'bmi': float(request.form['bmi']),
                'under-five_deaths': int(request.form['under-five_deaths']),
                'polio': float(request.form['polio']),
                'total_expenditure': float(request.form['total_expenditure']),
                'diphtheria': float(request.form['diphtheria']),
                'hiv/aids': float(request.form['hiv/aids']),
                'gdp': float(request.form['gdp']),
                'population': int(request.form['population']),
                'thinness_1-19_years': float(request.form['thinness_1-19_years']),
                'thinness_5-9_years': float(request.form['thinness_5-9_years']),
                'income_composition_of_resources': float(request.form['income_composition_of_resources']),
                'schooling': float(request.form['schooling'])
            }

            # Make prediction
            prediction = predictor.predict(input_data)
            return render_template('result.html', prediction=prediction[0])

        except Exception as e:
            error_message = f"Error: {str(e)}"
            return render_template('index.html', error=error_message)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)