import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
try:
    with open("ols_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'ols_model.pkl' not found. Please upload the file.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Marks Prediction App", layout="centered", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        font-family: Arial, sans-serif;
    }
    .main-title {
        color: #4CAF50;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    .prediction-result {
        text-align: center;
        font-size: 1.5em;
        color: #ff5722;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 0.9em;
        color: #999;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-title'>📚 Marks Prediction App</div>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Input Features")
number_courses = st.sidebar.number_input("Number of Courses", min_value=1, max_value=10, value=3)
time_study = st.sidebar.number_input("Time Studied (in hours)", min_value=0.0, max_value=10.0, value=4.5, step=0.1)

# Predict button
if st.sidebar.button("Predict Marks"):
    try:
        # Create input data for the model
        input_data = pd.DataFrame({'number_courses': [number_courses], 'time_study': [time_study]})
        
        # Predict using the model
        prediction = model.predict(input_data).item()
        
        # Display the result
        st.markdown(f"<div class='prediction-result'>🎓 Predicted Marks: <b>{prediction:.2f}</b></div>", unsafe_allow_html=True)
        
        # Visualization
        st.subheader("Prediction Visualization")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Predicted Marks"], [prediction], color="#4CAF50", alpha=0.8)
        ax.set_ylabel("Marks")
        ax.set_title("Predicted Marks Visualization")
        ax.set_ylim(0, 100)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("<div class='footer'>Created with ❤ using Streamlit</div>", unsafe_allow_html=True)
