# model.py

import joblib
import pandas as pd
import numpy as np
from config import MODEL_PATH, SCALER_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES

class ModelHandler:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def preprocess(self, input_df):
        df = input_df.copy()

        # Log transforms
        df["Age"] = np.log1p(df["Age"])
        df["Balance"] = np.log1p(df["Balance"])

        # Scale numeric
        df_scaled = pd.DataFrame(self.scaler.transform(df[NUMERIC_FEATURES]), columns=NUMERIC_FEATURES)

        # One-hot encode categorical
        df_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True)

        # Combine
        df_final = pd.concat([df_scaled, df_cat], axis=1)

        # Align with model's expected features
        model_features = self.model.feature_names_in_
        df_final = df_final.reindex(columns=model_features, fill_value=0)

        return df_final

    def predict(self, input_df):
        processed = self.preprocess(input_df)
        pred = self.model.predict(processed)[0]
        label = "Exited" if pred == 1 else "Stayed"
        return label
