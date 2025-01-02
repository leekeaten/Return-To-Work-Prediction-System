import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
from streamlit_shap import st_shap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from PIL import Image

st.set_page_config(layout="wide")
# Load the saved model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define app title
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="padding:0;">Return to Work Prediction for Patients</h1>
        <h1 style="padding:0;">after Cardiac Rehabilitation</h1>
        <br>
        <br>
    </div>
    """,
    unsafe_allow_html=True
)


col1, col2 = st.columns([1,1])

with col1:
    # Demographical Data
    st.markdown("### **🧍Demographical**")
    age = st.slider('Age', 18, 80, 50)
    health_funding = st.selectbox('Health Funding', ['Self funded', 'Semi-Funded', 'Fully Funded'])

    # Medical History
    st.markdown("### **🏥Medical History**")
    risk_factors_hypertension = st.radio('Risk Factors: Hypertension', ['Yes', 'No'])
    total_risk_factors = st.number_input('Total Risk Factors', min_value=0, max_value=3, value=1)
    anxiety_scores = st.number_input('Anxiety Scores', min_value=0, max_value=10, value=5)
    depression_scores = st.number_input('Depression Scores', min_value=0, max_value=10, value=5)

with col2: 
  # CR Status
  st.markdown("### **💪CR Status**")
  duration_between_ward_enrollment = st.number_input('Duration Between Ward Enrollment (days)', min_value=5, max_value=200, value=100)
  duration_cr = st.number_input('Duration of CR (days)', min_value=0, max_value=300, value=100)
  exercise_frequency_sessions_week = st.number_input('Exercise Sessions (per week)', min_value=0, max_value=7, value=3)
  # Pre-CR Data
  st.markdown("### **⏳Pre-CR**")
  pre_rtw = st.radio('Pre RTW', ['Yes', 'No'])

# Map user inputs to the model's expected format
# Encode categorical inputs
health_funding_self_funded = 1 if health_funding == 'Self funded' else 0
risk_factors_hypertension = 1 if risk_factors_hypertension == 'Yes' else 0
pre_rtw = 1 if pre_rtw == 'Yes' else 0

# Prepare the feature vector for the model
input_features = [
    exercise_frequency_sessions_week,
    anxiety_scores,
    depression_scores,
    pre_rtw,
    age,
    risk_factors_hypertension,
    total_risk_factors,
    duration_between_ward_enrollment,
    duration_cr,
    health_funding_self_funded,
]

# Convert to a 2D array for model input
input_data = [input_features]

# Scale the input data using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# Create a DataFrame for the scaled data (for SHAP compatibility)
input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=[
    'Exercise_frequency_sessions_week',
    'Anxiety_Scores',
    'Depression_Scores',
    'Pre_RTW',
    'Age',
    'Risk_Factors_Hypertension',
    'Total_Risk_Factors',
    'Duration_Between_Ward_Enrollment',
    'Duration_CR',
    'Health_funding_Self funded',
])

# Create a mapping dictionary - Rename columns to shorter names
column_name_mapping = {
    'Exercise_frequency_sessions_week': 'Ex_Sessions/Week',
    'Anxiety_Scores': 'Anxiety',
    'Depression_Scores': 'Depression',
    'Pre_RTW': 'Pre_RTW',
    'Age': 'Age',
    'Risk_Factors_Hypertension': 'HTN',
    'Total_Risk_Factors': 'Total_Risks',
    'Duration_Between_Ward_Enrollment': 'Ward_Duration',
    'Duration_CR': 'CR_Duration',
    'Health_funding_Self funded': 'Self_Funded',
}

# Inverse transform the scaled input data to get original values
original_values = scaler.inverse_transform(input_data_scaled)
original_values_df = pd.DataFrame(original_values, columns=input_data_scaled_df.columns)

# Rename the columns in the original input DataFrame
shortened_columns_df = original_values_df.rename(columns=column_name_mapping)

col1, col2, col3 = st.columns([0.1, 10, 0.1])
with col2:

  st.markdown("""
        <style>
        div.stButton > button:first-child {
            width: 100%;
            max-width: 100%;
            height: 50px;
            font-size: 18px;
        }
        </style>""", unsafe_allow_html=True)
   
  # Button to make predictions
  predictBtnClick = st.button('Predict')

if predictBtnClick:
    # Make prediction using the loaded model
    prediction_probabilities = model.predict_proba(input_data_scaled)[0]
    prediction = model.predict(input_data_scaled)[0]  # 0 or 1

    # Display the results
    st.markdown("### **Prediction Results**")
    if prediction == 1:
        st.success(f"The patient is likely to return to work after cardiac rehabilitation.")
    else:
        st.warning(f"The patient is unlikely to return to work after cardiac rehabilitation.")
    
    # Display probabilities
    st.markdown(f"**Probability of Return to Work:** {prediction_probabilities[1]:.2f}")
    st.markdown(f"**Probability of Not Returning to Work:** {prediction_probabilities[0]:.2f}")

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_data_scaled_df)

    # Display SHAP Force Plot for the prediction
    st.write("### SHAP Force Plot for the Prediction")

    # Force plot for class 1 (Return to Work)
    force_plot = shap.force_plot(
        explainer.expected_value[1], # Baseline value for class 1 (Return to Work)
        shap_values[0][:, 1],  # SHAP values for class 1
        shortened_columns_df.iloc[0],  # Original input values (non-scaled)
        # original_values_df.iloc[0],
    )
       
    st_shap(force_plot, height=200)

    shap.save_html("force_plot.html", force_plot)

    with st.expander("SHAP EXPLANATION USER GUIDE (Click to expand)"):
        st.markdown("""
            ### SHAP Force Plot Explanation:
            1. **Arrows**: Each arrow represents a piece of patient information. The direction of the arrow shows if that information makes the prediction go up or down.
            2. **Length of Arrows**: The longer the arrow, the bigger its impact on the prediction.
            3. **Color of Arrows**: Red arrows increase the prediction, and blue arrows decrease it.
            4. **End Point**: This is the final prediction after considering all the patient information.
            5. **Base Value**: This is the starting point of the plot. It’s what the model would predict if it didn’t know anything else.
        """)

    # Waterfall plot
    st.write("### SHAP Waterfall Plot for the Prediction")

    water_lot = shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0][:, 1],  
            base_values=explainer.expected_value[1], 
            data=original_values_df.iloc[0] 
        )
    )
    st_shap(water_lot, height=550, width=1000)






















    






