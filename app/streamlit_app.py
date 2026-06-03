import streamlit as st
import pandas as pd


st.title("Student Dropout Prediction")
st.markdown("""
            This application predicts whether a student is likely to drop out based on various features. 
            The model used for prediction is a Random Forest Classifier trained on a dataset of student information.
            """)
# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('../data/students.csv')
    return data


st.columns(3)
data = load_data()
st.metric("Total Students: ", len(data))
st.metric("Dropped Out: ", data['dropout'].sum())
st.metric("Number of Features: ", len(data.columns) - 1)

st.columns(3)
st.subheader("Model Performance Metrics")
st.metric("AUC Score: ", "0.9632")
st.metric("Accuracy: ", "91%")
st.metric("Dropout Recall: ", "84%")