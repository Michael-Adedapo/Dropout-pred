from operator import mod

from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Model Insights",
    initial_sidebar_state="collapsed"  
)
st.title("Model Insights")

model = load('models/best_dropout_model.pkl')
feature_cols = load(r'models\feature_columns.pkl')

importance = model.feature_importances_
importance_series = pd.Series(importance, index=feature_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
importance_series.plot(kind="barh", ax=ax)
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Features')
ax.set_title('Model Feature Importances')
fig.tight_layout()

st.pyplot(fig)

st.markdown(
    """### What the Model Learned

- **Current CGPA** is the single strongest predictor of dropout risk, 
accounting for over 28% of the model's decisions. Students with 
consistently low GPAs are significantly more likely to leave before graduation.

- **Failed Courses Count** is the second most important signal. Repeated 
course failures compound academic pressure and are a strong early warning 
sign that a student is struggling to keep up.

- **Attendance Rate** confirms that physical disengagement precedes dropout. 
Students who stop showing up to class are already on a path toward 
leaving the institution.

- **Age** suggests that older students face competing responsibilities such 
as work, family, and financial obligations that make sustained academic 
commitment more difficult.

- **Jamb Score and Parent Education Level** indicate that a student's 
academic foundation and home environment play a meaningful role. Students 
entering university underprepared or without educated family support are at 
higher risk.

- **ASUU Strike Semesters** captures a uniquely Nigerian risk factor.
Prolonged industrial strikes disrupt academic momentum and disproportionately 
affect students from lower-income backgrounds who cannot sustain themselves 
during extended university closures.

- **Has Scholarship** acts as a protective factor. Students with financial 
support are less likely to drop out, reinforcing that economic stability 
is as important as academic performance in keeping students enrolled."""
)

model_results = {
    'Model': ['Logistic Regression', 'Decision Tree', 'XGBoost', 'Random Forest'],
    'CV AUC Score': [0.9801, 0.9509, 0.9849, 0.9872],  
    'Selected': ['No', 'No', 'No', 'Yes']
}

results_df = pd.DataFrame(model_results)

st.dataframe(results_df, use_container_width=True, hide_index=True)
