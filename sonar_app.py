import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('sonar_model.pkl')

st.title("🔍 SONAR Rock vs Mine Prediction")

st.markdown("""
This app predicts whether the given sonar signals belong to a **Rock** or a **Mine**.
You can:
- Enter 60 comma-separated values manually, or
- Upload a CSV file with the same format
""")

# Manual Input
st.subheader("📌 Manual Input")
user_input = st.text_area("Enter 60 comma-separated values:")

if st.button("Predict from Manual Input"):
    try:
        data = np.array([float(i) for i in user_input.strip().split(",")])
        if len(data) != 60:
            st.error("Please enter exactly 60 values.")
        else:
            result = model.predict([data])[0]
            st.success(f"Prediction: {'🪨 Rock' if result == 'R' else '💣 Mine'}")
    except:
        st.error("Invalid input. Please check your numbers.")

# File Upload
st.subheader("📁 Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", df.head())
        if df.shape[1] != 60:
            st.error("CSV must have exactly 60 columns.")
        else:
            preds = model.predict(df)
            df['Prediction'] = ['Rock' if p == 'R' else 'Mine' for p in preds]
            st.success("Predictions added to the uploaded file:")
            st.write(df)
    except Exception as e:
        st.error(f"Error processing file: {e}")
