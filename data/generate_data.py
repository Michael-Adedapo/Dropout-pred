import os
import numpy as np
import pandas as pd

np.random.seed(42)
N = 2546
NIGERIAN_STATES = [
    "Lagos", "Abuja", "Kano", "Rivers", "Oyo", "Enugu", "Kaduna",
    "Anambra", "Delta", "Imo", "Benue", "Kwara", "Ogun", "Osun",
    "Plateau", "Edo", "Cross River", "Akwa Ibom", "Kogi", "Niger"
]

def generate_data(n):
    student_id = [f"STU{str(i).zfill(4)}" for i in range(1, n + 1)]
    gender = np.random.choice(["Male", "Female"], n, p=[0.55, 0.45])
    state_of_origin = np.random.choice(NIGERIAN_STATES, n)
    course_load = np.random.randint(12, 25, n)

    # ==========================================
    # 1. LATENT RISK ENGINE (The Root Cause)
    # ==========================================
    # We generate a standardized underlying risk profile for each student
    latent_risk = np.random.normal(0, 1.0, n)

    # ==========================================
    # 2. GENERATING WEAK FEATURES AS A FUNCTION OF RISK
    # ==========================================
    
    # Has_scholarship: High latent risk students should rarely have scholarships
    # We map higher risk to a much lower probability of securing a scholarship
    scholarship_prob = 1 / (1 + np.exp(1.5 * latent_risk + 1.2)) 
    has_scholarship = (np.random.uniform(0, 1, n) < scholarship_prob).astype(int)

    # Distance from home: High-risk students skew further away from campus support structures
    # Base distance + an added exponential kick proportional to their latent risk profile
    distance_base = np.random.randint(5, 120, n)
    distance_kick = np.clip(np.exp(latent_risk + 2.5), 0, 780).astype(int)
    distance_from_home_km = np.clip(distance_base + distance_kick, 5, 900)

    # Age: Mature students balancing outer structural pressures have higher latent risk profiles
    age_base = np.random.randint(16, 22, n)
    age_boost = np.clip((latent_risk + 1.5) * 3, 0, 9).astype(int)
    age = np.clip(age_base + age_boost, 16, 30)

    # O_level_credits: Strong entry criteria heavily combats baseline risk profiles
    # Higher risk correlates with meeting just the bare minimum credits (e.g., 5 credits)
    o_level_base = np.random.randint(5, 9, n)
    o_level_penalty = np.clip((latent_risk * 0.8).astype(int), 0, 2)
    o_level_credits = np.clip(o_level_base - o_level_penalty, 3, 9)

    # ==========================================
    # 3. GENERATING CORE STRUCTURAL FEATURES
    # ==========================================
    
    family_income_bracket = np.where(latent_risk > 0.5, 
                                     np.random.choice(["Low", "Middle", "High"], n, p=[0.60, 0.35, 0.05]),
                                     np.random.choice(["Low", "Middle", "High"], n, p=[0.10, 0.45, 0.45]))
    
    fee_payment_status = np.where(family_income_bracket == "Low",
                                  np.random.choice(["Paid", "Partial", "Owing"], n, p=[0.15, 0.40, 0.45]),
                                  np.where(family_income_bracket == "Middle",
                                           np.random.choice(["Paid", "Partial", "Owing"], n, p=[0.60, 0.30, 0.10]),
                                           np.random.choice(["Paid", "Partial", "Owing"], n, p=[0.90, 0.08, 0.02])))

    works_parttime = np.where(latent_risk > 0.2, 
                              np.random.choice([1, 0], n, p=[0.35, 0.65]), 
                              np.random.choice([1, 0], n, p=[0.05, 0.95]))

    school_location = np.where(latent_risk > 0.6, 
                               np.random.choice(["Urban", "Rural"], n, p=[0.40, 0.60]), 
                               np.random.choice(["Urban", "Rural"], n, p=[0.75, 0.25]))

    parent_edu_p = {"Low": [0.35, 0.45, 0.18, 0.02], "Middle": [0.05, 0.15, 0.55, 0.25], "High": [0.01, 0.03, 0.16, 0.80]}
    parent_education_level = np.array([np.random.choice(["No formal", "Primary", "Secondary", "Tertiary"], p=parent_edu_p[inc]) for inc in family_income_bracket])

    # Academic Performance Vectoring
    jamb_score = np.clip(np.random.randint(180, 340, n) - (latent_risk * 30).astype(int), 140, 370)
    attendance_rate = np.clip(np.random.normal(82, 10, n) - (latent_risk * 12), 10, 100).round(1)
    asuu_strike_semesters = np.clip(np.random.randint(0, 3, n) + np.where(latent_risk > 0.4, 1, 0), 0, 3)

    # Core Academic Calculations
    base_cgpa = (((jamb_score - 140) / 230) * 3.3) + 1.3
    gpa_modifier = np.where(attendance_rate < 65, -0.8, 0.0) + np.where(works_parttime == 1, -0.3, 0.0)
    current_cgpa = np.clip(base_cgpa + gpa_modifier - (latent_risk * 0.4), 1.0, 5.0).round(2)

    failed_course_count = np.where(current_cgpa < 2.0, np.random.randint(2, 5, n),
                                   np.where(current_cgpa < 3.0, np.random.randint(0, 3, n), 0))
    
    cgpa_trend = np.where(latent_risk > 0.3, 
                          np.random.choice(["Improving", "Stable", "Declining"], n, p=[0.10, 0.35, 0.55]),
                          np.random.choice(["Improving", "Stable", "Declining"], n, p=[0.45, 0.40, 0.15]))
    
    mental_health_support = np.random.choice([1, 0], n, p=[0.15, 0.85])

    # ==========================================
    # 4. TARGET SEPARATION AND VOLUME CALIBRATION
    # ==========================================
    # Re-evaluating the mathematical log-odds framework. 
    # To fix the high dropout rate, the static intercept bias is set back significantly (-3.2).
    log_odds = -5.5
    
    # Combine feature vectors smoothly into structural outcome probabilities
    log_odds += latent_risk * 1.5
    log_odds += (3.5 - current_cgpa) * 1.2
    log_odds += failed_course_count * 0.5
    log_odds += np.where(fee_payment_status == "Owing", 1.2, 0.0)
    log_odds += np.where(has_scholarship == 0, 0.5, -0.8)
    log_odds += np.where(works_parttime == 1, 0.6, 0.0)
    log_odds += (age - 20) * 0.15
    log_odds += (distance_from_home_km / 200) * 0.3
    log_odds += (5 - o_level_credits) * 0.2

    # Map Log-odds cleanly through a standard sigmoid scale
    dropout_prob = 1 / (1 + np.exp(-log_odds))
    dropout = (np.random.uniform(0, 1, n) < dropout_prob).astype(int)

    df = pd.DataFrame({
        "Student_ID": student_id, "Age": age, "Gender": gender, "State_of_origin": state_of_origin,
        "School_location": school_location, "Distance_from_home_km": distance_from_home_km,
        "Jamb_score": jamb_score, "O_level_credits": o_level_credits, "Current_cgpa": current_cgpa,
        "Cgpa_trend": cgpa_trend, "Attendance_rate": attendance_rate, "Fee_payment_status": fee_payment_status,
        "Has_scholarship": has_scholarship, "Works_part_time": works_parttime, "Family_income_bracket": family_income_bracket,
        "Parent_education_level": parent_education_level, "Failed_courses_count": failed_course_count,
        "Asuu_strike_semesters": asuu_strike_semesters, "Mental_health_support": mental_health_support,
        "Course_load": course_load, "Dropout": dropout
    })

    return df

if __name__ == "__main__":
    df = generate_data(N)
    
    print(f"Shape: {df.shape}")
    print(f"Overall Dropout rate: {df['Dropout'].mean():.1%}\n")
    
    # Statistical Correlation Evaluation
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['Dropout'].sort_values(ascending=False)
    print("--- Top Correlated Features to Dropout Target ---")
    print(correlations)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/students.csv", index=False)
    print("\nDataset successfully regenerated and saved to data/students.csv")