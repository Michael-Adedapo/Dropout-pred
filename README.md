# 🎓 Nigerian Student Dropout Risk Predictor

A machine learning web application that predicts whether a Nigerian university student is at risk of dropping out, based on academic performance, socioeconomic factors, and Nigerian-specific challenges such as ASUU strike disruptions.

Built as a portfolio project using Python, scikit-learn, and Streamlit.

---

## Overview

Student dropout is a significant challenge in Nigerian universities, driven by a combination of academic struggles, financial pressure, and systemic disruptions like ASUU strikes. This project builds a Random Forest classifier trained on a synthetic dataset of 2,546 Nigerian student records to identify at-risk students early.

The application takes 20 student features as input and returns a dropout risk prediction along with a probability score, giving university advisors an actionable early warning signal.

---

## Project Structure

```
├── app/
│   ├── streamlit_app.py        ← Home page (Overview)
│   └── pages/
│       ├── Overview.py         ← Dataset summary and model metrics
│       ├── Predictor.py        ← Student risk prediction form
│       └── Insights.py         ← Feature importance and model comparison
├── data/
│   └── students.csv            ← Synthetic student dataset
├── models/
│   ├── best_dropout_model.pkl  ← Trained Random Forest model
│   ├── label_encoders.pkl      ← Fitted LabelEncoders for categorical features
│   └── feature_columns.pkl     ← Ordered feature column list
├── notebooks/
│   ├── eda.ipynb               ← Exploratory data analysis
│   └── modelling.ipynb         ← Preprocessing, training, and evaluation
├── .streamlit/
│   └── config.toml             ← App theme configuration
├── requirements.txt
└── README.md
```

---

## Features

The model uses 20 input features grouped across four categories:

**Academic**
- Current CGPA
- CGPA Trend (Improving / Stable / Declining)
- Attendance Rate
- Failed Courses Count
- JAMB Score
- O-Level Credits
- Course Load

**Socioeconomic**
- Fee Payment Status
- Has Scholarship
- Family Income Bracket
- Works Part-Time
- Parent Education Level

**Personal**
- Age
- Gender
- State of Origin
- Distance from Home (km)
- Mental Health Support

**Nigerian-Specific**
- ASUU Strike Semesters
- School Location (Urban / Rural)

---

## Model Performance

| Metric | Score |
|---|---|
| AUC Score | 0.9632 |
| Accuracy | 91% |
| Dropout Recall | 84% |
| Dropout Precision | 78% |
| Dropout F1-Score | 0.81 |

The model was selected via 5-fold cross-validation scored on ROC-AUC, comparing Logistic Regression, Decision Tree, XGBoost, and Random Forest. Random Forest achieved the highest cross-validation AUC of 0.9872.

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/Michael-Adedapo/Dropout-pred.git
cd nigerian-dropout-predictor
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app/streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Usage

- **Overview page** — view dataset statistics and model performance metrics
- **Predictor page** — fill in a student's details and click *Predict Dropout Risk* to get a prediction and probability score
- **Insights page** — explore feature importances, model interpretations, and the model comparison table

---

## Technologies Used

| Library | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | Model training, preprocessing, evaluation |
| XGBoost | Gradient boosting model (comparison) |
| imbalanced-learn | SMOTE for class imbalance |
| Streamlit | Web application framework |
| pandas | Data manipulation |
| NumPy | Numerical operations and data generation |
| matplotlib / seaborn | Data visualisation |
| joblib | Model serialisation |

---

## Key Design Decisions

- **Synthetic data with strong feature signal** — data was generated using `np.where()` conditional logic tied to a risk score, ensuring meaningful correlations between features and the dropout target (top correlation: Current CGPA at −0.70)
- **LabelEncoder over OneHotEncoder** — chosen because the primary model is tree-based, which handles ordinal-style encoding well without dimensionality expansion
- **SMOTE for class imbalance** — applied after train/test split to avoid data leakage
- **ROC-AUC as evaluation metric** — more appropriate than accuracy for imbalanced datasets

---

## Author

Built by Dapo — Data Science Portfolio Project
