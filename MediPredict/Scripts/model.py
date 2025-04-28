import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

class MedicalEntrancePredictor:
    """A class to load the trained model and make predictions for medical entrance exam performance."""
    
    def __init__(self, model_path=r"Models\best_model.pkl", preprocessor_path=r"Models\preprocessor.joblib"):
        """Initialize the predictor by loading the model and preprocessor."""
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model or preprocessor file not found: {e}")
        
        # Define expected features 
        self.expected_features = [
            'Caste', 'coaching', 'medium', 'Father_occupation', 'Mother_occupation',
            'Class_ten_education', 'Class_X_Grade', 'Class_XII_Grade'
        ]
        
        # Define nominal features (for one-hot encoding in preprocessor)
        self.nominal_features = [
            'Caste', 'coaching', 'medium', 'Father_occupation', 'Mother_occupation', 'Class_ten_education'
        ]
        
        # Define ordinal features (for ordinal encoding in preprocessor)
        self.ordinal_features = ['Class_X_Grade', 'Class_XII_Grade']
        
        # Define the ordinal mapping (same as used in training)
        self.grade_order = {'Average': 0, 'Good': 1, 'Vg': 2, 'Excellent': 3}

    def preprocess_input(self, input_data):
        """Preprocess input data to match training data format."""
        # Convert input to DataFrame if it's a dictionary
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data], columns=self.expected_features)
        
        # Validate input columns
        missing_cols = [col for col in self.expected_features if col not in input_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure correct data types and handle missing values
        for col in self.nominal_features:
            input_data[col] = input_data[col].astype(str)
            if input_data[col].isna().any():
                raise ValueError(f"Missing values in {col}. All nominal features must have values.")
        
        for col in self.ordinal_features:
            input_data[col] = input_data[col].astype(str)
            if input_data[col].isna().any():
                raise ValueError(f"Missing values in {col}. All ordinal features must have values.")
            # Validate ordinal values
            valid_grades = list(self.grade_order.keys())
            invalid_grades = input_data[~input_data[col].isin(valid_grades)][col]
            if not invalid_grades.empty:
                raise ValueError(f"Invalid values in {col}: {invalid_grades.tolist()}. Must be one of {valid_grades}.")
        
        # Apply the preprocessor (which handles one-hot encoding for nominal features and ordinal encoding)
        try:
            processed_data = self.preprocessor.transform(input_data)
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {str(e)}")
        
        return processed_data

    def predict(self, input_data):
        """Make performance predictions for the input data."""
        try:
            processed_data = self.preprocess_input(input_data)
            predictions = self.model.predict(processed_data)
            # Map numerical predictions back to labels
            inverse_grade_mapping = {0: 'Average', 1: 'Good', 2: 'Vg', 3: 'Excellent'}
            predicted_labels = [inverse_grade_mapping[pred] for pred in predictions]
            return predicted_labels
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

if __name__ == "__main__":

    # Example usage
    predictor = MedicalEntrancePredictor()
    
    # Sample input data
    sample_input = {
        'Caste': 'General',
        'coaching': 'Outside Assam',
        'medium': 'English',
        'Father_occupation': 'Doctor',
        'Mother_occupation': 'College_teacher',
        'Class_ten_education': 'CBSE',
        'Class_X_Grade': 'Excellent',
        'Class_XII_Grade': 'Excellent'
    }
    
    # Make prediction
    try:
        prediction = predictor.predict(sample_input)
        print(f"Predicted Performance: {prediction[0]}")
    except Exception as e:
        print(f"Error: {str(e)}")
