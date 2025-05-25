import streamlit as st
import pandas as pd
from model import ModelHandler

# Set page config with custom CSS
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Custom CSS with transparent title background, slightly transparent form, borders, and shadows
st.markdown(
    """
    <style>
    .stApp {
        background-color: #00000;  /* Light gray background */
        color: #fffffff;  /* Dark gray text for readability */
    }
    .stHeader {
        color: #425078;  /* Dark blue for the header */
        text-align: center;
        background-color: transparent;  /* Fully transparent background for title */
        padding: 10px;
        border: 2px solid #425078;  /* Medium blue border */
        border-radius: 8px;
        
    }
    .stForm {
        background-color: #1E3A8A;  /* Dark blue form background */
        opacity: 0.9;  /* Slightly transparent form */
        padding: 20px;
        border-radius: 10px;
        color: #E5E7EB;  /* Light gray text inside form */
        border: 2px solid #fffffffe;  /* Medium blue border */
        box-shadow: 0 4px 8px #425078;  /* Subtle shadow for depth */
    }
    .stButton>button {
        background-color: #3B82F6;  /* Medium blue for buttons */
        color: #FFFFFF;  /* White text on buttons */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563EB;  /* Darker blue on hover */
        color: #FFFFFF;
    }
    .stSuccess {
        color: #10B981;  /* Green for success messages */
        font-weight: bold;
    }
    hr {
        border-color: #3B82F6;  /* Blue divider */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <h2 class='stHeader'>💼 Customer Churn Prediction App</h2>
    <p style='text-align: center; color: #fffffff;'>Predict whether a customer will stay or leave the bank.</p>
    <hr>
""", unsafe_allow_html=True)

with st.form("churn_form"):
    st.subheader("Customer Info")

    CreditScore = st.number_input("Credit Score", min_value=300, max_value=1000, value=650)
    Age = st.number_input("Age", min_value=18, max_value=100, value=40)
    Tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
    Balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=75000.0)
    EstimatedSalary = st.number_input("Estimated Salary", min_value=1000.0, max_value=200000.0, value=60000.0)
    Point_Earned = st.number_input("Points Earned", min_value=0, max_value=1000, value=200)

    Geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=0)
    Gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
    HasCrCard = st.selectbox("Has Credit Card", [0, 1], index=1)
    IsActiveMember = st.selectbox("Is Active Member", [0, 1], index=1)
    Complain = st.selectbox("Has Complaint", [0, 1], index=0)
    Satisfaction_Score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
    Card_Type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"], index=3)

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    input_data = pd.DataFrame([{
        "CreditScore": CreditScore,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "EstimatedSalary": EstimatedSalary,
        "Point Earned": Point_Earned,
        "Geography": Geography,
        "Gender": Gender,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "Complain": Complain,
        "Satisfaction Score": Satisfaction_Score,
        "Card Type": Card_Type,
    }])

    model_handler = ModelHandler()
    prediction = model_handler.predict(input_data)

    st.success(f"The model predicts: **{prediction}**")