import pandas as pd
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

class LifeExpectancyPredictor:
    """A class to load the trained pipeline model and make life expectancy predictions."""

    def __init__(self, model_path="Models/best_model.pkl", label_encoder_path="Models/status_encoder.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
        
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)

        self.expected_features = [
            'status', 'adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure',
            'hepatitis_b', 'measles', 'bmi', 'under-five_deaths', 'polio', 'total_expenditure',
            'diphtheria', 'hiv/aids', 'gdp', 'population', 'thinness_1-19_years',
            'thinness_5-9_years', 'income_composition_of_resources', 'schooling'
        ]

    def validate_and_preprocess(self, input_data):
        """Ensure input is valid and preprocess the status column."""
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_data = pd.DataFrame(input_data)
        elif not isinstance(input_data, pd.DataFrame):
            raise TypeError("Input must be a dict, list of dicts, or a pandas DataFrame.")

        # Check for missing columns
        missing_cols = [col for col in self.expected_features if col not in input_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for missing values
        if input_data[self.expected_features].isnull().values.any():
            null_cols = input_data[self.expected_features].isnull().sum()
            null_cols = null_cols[null_cols > 0].index.tolist()
            raise ValueError(f"Missing values in the following columns: {null_cols}")

        # Encode the 'status' column
        try:
            input_data['status'] = self.label_encoder.transform(input_data['status'])
        except ValueError as e:
            raise ValueError("Invalid 'status' value. Must be 'Developed' or 'Developing'.")

        return input_data[self.expected_features]

    def predict(self, input_data):
        """Make life expectancy predictions for the input data."""
        try:
            processed_data = self.validate_and_preprocess(input_data)
            prediction = self.model.predict(processed_data)
            return prediction
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

if __name__ == "__main__":
    predictor = LifeExpectancyPredictor()

    sample_input = {
        'status': 'Developing',
        'adult_mortality': 175.0,
        'infant_deaths': 36,
        'alcohol': 4.46,
        'percentage_expenditure': 685.0,
        'hepatitis_b': 77.0,
        'measles': 2730,
        'bmi': 30.0,
        'under-five_deaths': 50,
        'polio': 81.0,
        'total_expenditure': 8.0,
        'diphtheria': 81.0,
        'hiv/aids': 0.1,
        'gdp': 5392.0,
        'population': 13580000,
        'thinness_1-19_years': 5.0,
        'thinness_5-9_years': 5.0,
        'income_composition_of_resources': 0.6,
        'schooling': 12.0
    }

    try:
        prediction = predictor.predict(sample_input)
        print(f"Predicted Life Expectancy: {prediction[0]:.2f} years")
    except Exception as e:
        print(f"Error: {e}")