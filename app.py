import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="📚",
    layout="centered"
)

# Title
st.title("📚 Student Marks Predictor")
st.write("Predict student marks using Machine Learning")

# Sidebar
st.sidebar.header("About")
st.sidebar.write(
    "This app uses FastAPI + Machine Learning + Streamlit"
)
hours = st.slider(
    "Select Study Hours",
    min_value=0,
    max_value=12,
    value=5
)

# Predict button
if st.button("Predict Marks"):

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"hours": hours}
    )

    result = response.json()

    predicted_marks = result["predicted_marks"]

    # Show prediction
    st.success(
        f"Predicted Marks: {predicted_marks:.2f}"
    )
    df = pd.DataFrame({
        "Category": ["Study Hours", "Predicted Marks"],
        "Values": [hours, predicted_marks]
    })

    # Matplotlib chart
    fig, ax = plt.subplots()

    ax.bar(df["Category"], df["Values"])

    ax.set_title("Study Hours vs Predicted Marks")

    st.pyplot(fig)

# Footer
st.markdown("---")
st.write("Built using FastAPI, Streamlit, and Scikit-learn 🚀")