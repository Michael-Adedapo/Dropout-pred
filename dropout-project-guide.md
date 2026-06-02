# 🎓 Nigerian Student Dropout Risk Predictor — Project Guide

> This guide walks you through building the project yourself, step by step.
> Code is intentionally not provided — use this as your roadmap.
> Paste specific errors into Claude when you get stuck.

---

## 🗂️ Project Structure to Create

```
student-dropout-predictor/
│
├── data/
│   ├── generate_data.py        # You will write this
│   └── students.csv            # Output of generate_data.py
│
├── notebooks/
│   └── dropout_prediction.ipynb
│
├── models/
│   └── (saved model files go here after training)
│
├── app/
│   └── streamlit_app.py        # You will write this last
│
├── reports/
│   └── figures/                # Your saved charts go here
│
├── requirements.txt
└── README.md
```

Create this structure manually or with `mkdir` in your terminal before starting.

---

## ✅ Phase 1 — Environment Setup

### Step 1.1 — Install Python
Make sure you have Python 3.10 or newer.
Check your version by running `python --version` in your terminal.

### Step 1.2 — Create a virtual environment
Research: "how to create a Python virtual environment"
This keeps your project's packages separate from your system Python.
The commands differ slightly between Windows and Mac/Linux.

### Step 1.3 — Create your requirements.txt
Create a `requirements.txt` file in your project root and add these libraries, one per line:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- streamlit
- joblib
- jupyter
- imbalanced-learn

Then install them all at once by running:
`pip install -r requirements.txt`

### Step 1.4 — Verify your setup
Open a Python shell and try importing each library.
If any import fails, pip install it individually.

---

## ✅ Phase 2 — Data Generation

Since we don't have a real Nigerian university dataset (for privacy reasons),
you will write a script to generate a **realistic synthetic one**.

### Step 2.1 — Understand what your dataset needs
Your CSV should have one row per student and these columns:

| Column | Type | Notes |
|---|---|---|
| student_id | string | e.g. STU0001 |
| age | integer | 16–30 |
| gender | category | Male / Female |
| state_of_origin | category | Any Nigerian state |
| school_location | category | Urban / Rural |
| distance_from_home_km | integer | 5–900 |
| jamb_score | integer | 140–360 |
| o_level_credits | integer | 3–9 |
| current_cgpa | float | 1.0–5.0 |
| cgpa_trend | category | Improving / Stable / Declining |
| attendance_rate | float | 10–100 |
| fee_payment_status | category | Paid / Partial / Owing |
| has_scholarship | binary | 0 or 1 |
| works_part_time | binary | 0 or 1 |
| family_income_bracket | category | Low / Middle / High |
| parent_education_level | category | No Formal / Primary / Secondary / Tertiary |
| failed_courses_count | integer | 0–5 |
| asuu_strike_semesters | integer | 0–3 |
| mental_health_support | binary | 0 or 1 |
| course_load | integer | 12–25 |
| dropout | binary | 0 = enrolled, 1 = dropped out ← TARGET |

### Step 2.2 — Write generate_data.py

Open `data/generate_data.py` and write a script that does the following:

1. **Import** numpy and pandas
2. **Set a random seed** (e.g. `np.random.seed(42)`) so your results are reproducible
3. **Set N = 2000** (number of students to generate)
4. **Generate each column** using numpy's random functions:
   - Use `np.random.randint()` for integer ranges
   - Use `np.random.choice()` for categorical columns (you can pass a `p=` argument to control how common each value is — e.g. make "Paid" more common than "Owing")
   - Use `np.random.normal()` for values like attendance rate, then clip them to valid range
5. **Create the dropout target** using a scoring approach:
   - Start with a score of 0 for each student
   - Add points for risk factors: low CGPA, "Owing" fees, high failed courses, declining trend
   - Subtract points for protective factors: scholarship, high family income, tertiary-educated parents
   - Add random noise so the model isn't perfect
   - Convert the final score to a probability using the sigmoid function, then randomly assign 0/1 based on that probability
   - Aim for roughly 25–30% dropout rate overall
6. **Build a DataFrame** from all your arrays
7. **Save to CSV** using `df.to_csv("data/students.csv", index=False)`
8. **Print a summary** so you can verify: total rows, dropout rate, and `df.head()`

> 💡 Tip: `np.clip(array, min, max)` is useful for keeping values in valid ranges.
> 💡 Tip: The sigmoid function is `1 / (1 + np.exp(-x))` — use it to convert a score to a 0–1 probability.

Run the script and confirm `students.csv` appears in your data folder.

---

## ✅ Phase 3 — Exploratory Data Analysis (EDA)

Open a new Jupyter notebook: `notebooks/dropout_prediction.ipynb`

All your modelling work happens in this notebook. Work through it section by section.

### Step 3.1 — Load your data
- Import pandas, numpy, matplotlib, seaborn
- Load `students.csv` with `pd.read_csv()`
- Check shape, dtypes, and missing values
- Print the dropout rate as a percentage

### Step 3.2 — Visualise class balance
- Create a pie chart or bar chart showing how many students dropped out vs stayed enrolled
- This tells you if your classes are imbalanced (they will be — ~70/30)

### Step 3.3 — Explore key relationships
Create at least 4 charts exploring the relationship between features and dropout:

**Suggested charts:**
1. Distribution of CGPA — split by dropout status (use a histogram with `hue="dropout"`)
2. Dropout rate by fee payment status (bar chart)
3. Dropout rate by family income bracket (bar chart)
4. Dropout rate by number of ASUU strike semesters (bar chart — this is your Nigerian angle!)
5. Attendance rate vs CGPA — scatter plot coloured by dropout

### Step 3.4 — Correlation heatmap
- Select only numeric columns
- Use `df.corr()` and plot with `sns.heatmap()`
- Look for which features correlate most with the `dropout` column
- Write 2–3 observations as markdown cells in your notebook

> 💡 Save each figure to `reports/figures/` using `plt.savefig()` before `plt.show()`

---

## ✅ Phase 4 — Feature Engineering & Preprocessing

### Step 4.1 — Drop the ID column
`student_id` is just a label — drop it before modelling.

### Step 4.2 — Encode categorical features
Machine learning models need numbers, not strings.
Use `sklearn.preprocessing.LabelEncoder` to convert each categorical column.
Store each encoder in a dictionary so you can reuse them in the Streamlit app later.

Columns to encode:
gender, state_of_origin, school_location, cgpa_trend,
fee_payment_status, family_income_bracket, parent_education_level

### Step 4.3 — Split features and target
- `X` = all columns except `dropout`
- `y` = the `dropout` column

### Step 4.4 — Train/test split
Use `sklearn.model_selection.train_test_split`
- Use `test_size=0.2` (80% train, 20% test)
- Use `stratify=y` to ensure both splits have the same dropout ratio
- Use `random_state=42`

### Step 4.5 — Handle class imbalance with SMOTE
Because only ~28% of students dropped out, your model could score 72% accuracy by just
predicting "no dropout" every time. SMOTE fixes this by synthetically generating more
examples of the minority class (dropouts) in the training set.

- Import `SMOTE` from `imblearn.over_sampling`
- Apply SMOTE **only on the training set** (never on test data)
- Print the class distribution before and after to confirm it's now balanced

---

## ✅ Phase 5 — Model Training & Comparison

### Step 5.1 — Define your models
Create a dictionary of 4 models to compare:
1. **Logistic Regression** — your baseline (simple, interpretable)
2. **Decision Tree** — easy to visualise and explain
3. **Random Forest** — ensemble, usually stronger
4. **XGBoost** — gradient boosting, typically best performer

### Step 5.2 — Cross-validation comparison
For each model:
- Use `cross_val_score` with `cv=5` (5-fold cross-validation)
- Score on `roc_auc` (better than accuracy for imbalanced problems)
- Print the mean and standard deviation of scores

### Step 5.3 — Plot cross-validation results
Create a boxplot showing all models' ROC-AUC score distributions side by side.
This is a key chart for your portfolio — it shows you evaluated models properly.

---

## ✅ Phase 6 — Model Evaluation

### Step 6.1 — Train best model on full training set
Train XGBoost (or whichever performed best) on your SMOTE-resampled training data.

### Step 6.2 — Evaluate on the test set
Generate predictions and predicted probabilities, then calculate:
- **Classification Report** — precision, recall, F1 per class
- **ROC-AUC Score** — how well the model ranks dropouts vs non-dropouts
- **Confusion Matrix** — actual vs predicted in a grid

> 💡 Pay special attention to **recall for the dropout class (class 1)**.
> Missing a student at risk (false negative) is worse than a false alarm.

### Step 6.3 — Plot the ROC Curve
- Use `roc_curve` and `roc_auc_score` from sklearn.metrics
- Plot the curve and shade the area under it
- Add a diagonal dashed line representing a random classifier

### Step 6.4 — Plot the Confusion Matrix
Use `ConfusionMatrixDisplay.from_predictions()` for a clean visual.

---

## ✅ Phase 7 — Feature Importance

### Step 7.1 — Extract importances
XGBoost (and Random Forest) have a `.feature_importances_` attribute.
Create a pandas Series mapping feature names to their importance scores.

### Step 7.2 — Plot a horizontal bar chart
Sort by importance and plot. This chart answers:
**"What actually predicts dropout in a Nigerian university?"**

Write a markdown cell in your notebook interpreting the top 5 features.
This narrative is what makes your project academically interesting.

---

## ✅ Phase 8 — Save the Model

Use `joblib` to save three things to your `models/` folder:
1. The trained XGBoost model
2. The dictionary of label encoders
3. The list of feature column names (in the correct order)

You will need all three when loading the model inside the Streamlit app.

---

## ✅ Phase 9 — Streamlit App

This is the final layer that makes your project visually impressive.

### Step 9.1 — Understand Streamlit basics
Before writing the app, spend 20–30 minutes on the Streamlit docs:
👉 https://docs.streamlit.io/get-started

Key components you'll use:
- `st.title()`, `st.subheader()`, `st.markdown()` — text
- `st.slider()`, `st.selectbox()`, `st.number_input()` — input widgets
- `st.button()` — trigger prediction
- `st.success()`, `st.warning()`, `st.error()` — coloured result boxes
- `st.progress()` — progress bar showing risk %

### Step 9.2 — App structure to build

Your app should do the following in order:

1. **Load the model** (use `@st.cache_resource` so it loads only once)
2. **Show a title and description**
3. **Create input widgets** for every feature in your model
   - Use two columns (`st.columns(2)`) to keep it compact
   - Match widget types to data types: sliders for numbers, selectbox for categories
4. **Add a "Predict" button**
5. **On button click:**
   a. Collect all widget values into a dictionary
   b. Convert to a DataFrame
   c. Encode categorical columns using your saved label encoders
   d. Reorder columns to match training order (critical — use your saved column list)
   e. Call `model.predict_proba()` and extract the probability for class 1
   f. Multiply by 100 for a percentage
6. **Display the result** with three risk tiers:
   - 🟢 Below 40% → Low Risk
   - 🟡 40–69% → Moderate Risk
   - 🔴 70%+ → High Risk
   - Show a `st.progress()` bar with the percentage

### Step 9.3 — Run the app
From your project root:
```
streamlit run app/streamlit_app.py
```

---

## ✅ Phase 10 — Portfolio Packaging

### Step 10.1 — Write your README.md
A good README is as important as the code. Include:
- Problem statement (1 paragraph, Nigeria-specific)
- Why this matters (real impact)
- Dataset description (synthetic, why, how it was generated)
- Features table
- Models compared and results table (fill in after training)
- How to run (install → generate data → notebook → app)
- What you'd do next (real data, API, etc.)

### Step 10.2 — Push to GitHub
- Create a repo on github.com
- Write a `.gitignore` that excludes: `__pycache__/`, `*.pkl`, `venv/`, `.ipynb_checkpoints/`
- Commit your work in logical stages, not all at once
- Pin the repo on your GitHub profile

### Step 10.3 — Screenshots for portfolio
Take screenshots of:
- A key EDA chart (ASUU strike impact is unique — use that one)
- The feature importance chart
- The Streamlit app with a High Risk result
- The confusion matrix

These go in your README as images and in any portfolio site you have.

---

## 🐛 When You Get Stuck

Come back to Claude with the **exact error message** and the **specific section** you're on.

Common places people get stuck:
- SMOTE import error → check imbalanced-learn is installed
- Label encoding mismatch between notebook and app → you need to save and reload the same encoders
- Streamlit column order error → always reindex your DataFrame to `feature_columns` before predicting
- XGBoost warning about `use_label_encoder` → safe to ignore, or pass `eval_metric="logloss"`

---

## 📚 Recommended Reading (in order)

1. [Pandas getting started](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)
2. [Scikit-learn user guide — supervised learning](https://scikit-learn.org/stable/supervised_learning.html)
3. [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/)
4. [Streamlit docs](https://docs.streamlit.io)
5. [Imbalanced-learn: SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
