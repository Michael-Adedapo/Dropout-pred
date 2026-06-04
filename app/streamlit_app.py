from pandas import col

import streamlit as st
import pandas as pd
from joblib import load


st.title("Student Dropout Prediction")
st.markdown("""
            This application predicts whether a Nigerian university student is at risk of dropping out,
            based on academic performance, attendance, socioeconomic factors, and the impact of ASUU strike disruptions. 
            The underlying model is a Random Forest Classifier trained on 2,546 student records.          
            """)
# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('data/students.csv')
    return data


data = load_data()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Students: ", len(data))
with col2:
    st.metric("Dropped Out: ", data['Dropout'].sum())
with col3:
    st.metric("Number of Features: ", len(data.columns) - 1)

col4, col5, col6 = st.columns(3)
st.subheader("Model Performance Metrics")
with col4:
    st.metric("AUC Score: ", "0.9632")
with col5:
    st.metric("Accuracy: ", "91%")
with col6:
    st.metric("Dropout Recall: ", "84%")

st.info("use sidebar for navigation")