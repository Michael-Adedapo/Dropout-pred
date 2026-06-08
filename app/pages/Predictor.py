from joblib import load
import streamlit as st
from streamlit_app import load_data
import pandas as pd

st.set_page_config(
    page_title="Dropout Predictor",
    initial_sidebar_state="collapsed"  
)

model = load('models/best_dropout_model.pkl')
feature_cols = load('models/feature_columns.pkl')
label_enc = load('models/label_encoders.pkl')
st.title("DROPOUT PREDICTOR")
st.markdown(
    "Input the student details in the fields below."
)

data = load_data()

state_options = sorted(data['State_of_origin'].dropna().unique())
trend_options = ["Stable", "Improving", "Declining"]
fee_options = ["Paid", "Owing", "Partial"]
income_options = ["Low", "Middle", "High"]
education_options = ["No formal", "Primary", "Secondary", "Tertiary"]

with st.form("student_input_form"):
    st.subheader("Student Details")
    col1, col2 = st.columns(2)

    with col1:
        student_id = st.text_input("Student ID")
        age = st.number_input("Age", min_value=15, max_value=50, value=18, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        state_of_origin = st.selectbox("State of Origin", state_options)
        school_location = st.selectbox("School Location", ["Urban", "Rural"])
        distance_from_home_km = st.number_input("Distance from Home (km)", min_value=0, max_value=500, value=10, step=1)
        jamb_score = st.number_input("JAMB Score", min_value=0, max_value=400, value=200, step=1)
        o_level_credits = st.number_input("O-Level Credits", min_value=0, max_value=15, value=5, step=1)
        current_cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=5.0, value=2.5, step=0.01, format="%.2f")

    with col2:
        cgpa_trend = st.selectbox("CGPA Trend", trend_options)
        attendance_rate = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1, format="%.1f")
        fee_payment_status = st.selectbox("Fee Payment Status", fee_options)
        has_scholarship = st.selectbox("Has Scholarship", ["No", "Yes"])
        works_part_time = st.selectbox("Works Part-time", ["No", "Yes"])
        family_income_bracket = st.selectbox("Family Income Bracket", income_options)
        parent_education_level = st.selectbox("Parent Education Level", education_options)
        failed_courses_count = st.number_input("Failed Courses Count", min_value=0, max_value=20, value=0, step=1)
        asuu_strike_semesters = st.number_input("ASUU Strike Semesters", min_value=0, max_value=8, value=0, step=1)
        mental_health_support = st.selectbox("Mental Health Support", ["No", "Yes"])
        course_load = st.number_input("Course Load", min_value=0, max_value=40, value=20, step=1)

    submitted = st.form_submit_button("Predict Dropout Risk")

if submitted:
    student_details = {
        "Student_ID": student_id,
        "Age": age,
        "Gender": gender,
        "State_of_origin": state_of_origin,
        "School_location": school_location,
        "Distance_from_home_km": distance_from_home_km,
        "Jamb_score": jamb_score,
        "O_level_credits": o_level_credits,
        "Current_cgpa": current_cgpa,
        "Cgpa_trend": cgpa_trend,
        "Attendance_rate": attendance_rate,
        "Fee_payment_status": fee_payment_status,
        "Has_scholarship": 1 if has_scholarship == "Yes" else 0,
        "Works_part_time": 1 if works_part_time == "Yes" else 0,
        "Family_income_bracket": family_income_bracket,
        "Parent_education_level": parent_education_level,
        "Failed_courses_count": failed_courses_count,
        "Asuu_strike_semesters": asuu_strike_semesters,
        "Mental_health_support": 1 if mental_health_support == "Yes" else 0,
        "Course_load": course_load,
    }
    
    student_data = pd.DataFrame([student_details])[feature_cols]

    for col, encoder in label_enc.items():
        student_data[col] = encoder.transform(student_data[col])

    prediction = model.predict(student_data)[0]
    prediction_proba = model.predict_proba(student_data)[0][1]


    if prediction == 1:
        st.error(f"⚠️ High Dropout Risk — {prediction_proba * 100:.1f}% probability")
    else:
        st.success(f"✅ Low Dropout Risk — {prediction_proba * 100:.1f}% dropout probability")