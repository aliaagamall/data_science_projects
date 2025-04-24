import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

class LifeExpectancyPredictor:
    """A class to load the trained model and make life expectancy predictions."""
    
    def __init__(self, model_path=r"Models\best_model.pkl", label_encoder_path=r"Models/status_encoder.pkl"):
        """Initialize the predictor by loading the model and label encoder."""
        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(label_encoder_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model or encoder file not found: {e}")
        
        # Define expected features (based on training data)
        self.expected_features = [
            'status', 'adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure',
            'hepatitis_b', 'measles', 'bmi', 'under-five_deaths', 'polio', 'total_expenditure',
            'diphtheria', 'hiv/aids', 'gdp', 'population', 'thinness_1-19_years',
            'thinness_5-9_years', 'income_composition_of_resources', 'schooling'
        ]
        
        # Define numerical columns for preprocessing
        self.numerical_cols = [
            'adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure',
            'hepatitis_b', 'measles', 'bmi', 'under-five_deaths', 'polio', 'total_expenditure',
            'diphtheria', 'hiv/aids', 'gdp', 'population', 'thinness_1-19_years',
            'thinness_5-9_years', 'income_composition_of_resources', 'schooling'
        ]
        
        # Define categorical columns
        self.categorical_cols = ['status']

    def preprocess_input(self, input_data):
        """Preprocess input data to match training data format."""
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data], columns=self.expected_features)
        
        # Validate input columns
        missing_cols = [col for col in self.expected_features if col not in input_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle missing values
        for col in self.numerical_cols:
            if input_data[col].isna().any():
                input_data[col].fillna(input_data[col].median(), inplace=True)
        
        # Encode categorical variable 'status'
        try:
            input_data['status'] = self.label_encoder.transform(input_data['status'])
        except ValueError as e:
            raise ValueError("Invalid 'status' value. Must be 'Developed' or 'Developing'.")
        
        # Ensure correct feature order
        input_data = input_data[self.expected_features]
        
        return input_data.values

    def predict(self, input_data):
        """Make life expectancy predictions for the input data."""
        try:
            processed_data = self.preprocess_input(input_data)
            predictions = self.model.predict(processed_data)
            return predictions
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    # Example usage
    predictor = LifeExpectancyPredictor()
    
    # Sample input data
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
    
    # Make prediction
    try:
        prediction = predictor.predict(sample_input)
        print(f"Predicted Life Expectancy: {prediction[0]:.2f} years")
    except Exception as e:
        print(f"Error: {str(e)}")