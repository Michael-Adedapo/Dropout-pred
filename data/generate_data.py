# A script to generate mock students data
import pandas as pd
import numpy as np
np.random.seed(42)
N = 2546
NIGERIAN_STATES = [
    "Lagos", "Abuja", "Kano", "Rivers", "Oyo", "Enugu", "Kaduna",
    "Anambra", "Delta", "Imo", "Benue", "Kwara", "Ogun", "Osun",
    "Plateau", "Edo", "Cross River", "Akwa Ibom", "Kogi", "Niger"
]

def generate_data (n):
    #Student Id column
    student_id = [f"STU{str(i).zfill(4)}" for i in range(1, n + 1)]

    # Attendance rate  column
    attendance_rate = np.clip(np.random.normal(80, 15, n), 10, 100).round(1)

    # Integer columns
    age = np.random.randint(16, 31, n)
    jamb_score = np.random.randint(140, 370, n)
    o_level_credits = np.random.randint(3, 9, n)
    distance_from_home_km = np.random.randint(5, 900, n)
    failed_course_count = np.random.randint(0, 5, n)
    asuu_strike_semesters = np.random.randint(0, 4, n)
    course_load = np.random.randint(12, 25, n)


    # Category columns
    gender = np.random.choice(["Male", "Female"], n, p=[0.55, 0.45])
    state_of_origin = np.random.choice(NIGERIAN_STATES, n)
    school_location = np.random.choice(["Urban", "Rural"], n, p=[0.6, 0.4])
    fee_payment_status = np.random.choice(["Paid", "Partial", "Owing"], n, p=[0.45, 0.35, 0.2])
    family_income_bracket = np.random.choice(["Low", "Middle", "High"], n, p=[0.25, 0.40, 0.35])
    parent_education_level = np.random.choice(
        ["No formal", "Primary", "Secondary", "Tetiarary"], n, p=[0.10, 0.15, 0.35, 0.40])
    cgpa_trend = np.random.choice(["Improving", "Stable", "Declining"], n, p=[0.25, 0.45, 0.30])
    

    # Binary columns
    has_scholarship = np.random.choice([1, 0], n, p=[0.15, 0.85])
    works_parttime = np.random.choice([1, 0], n, p=[0.13, 0.87])
    mental_health_support = np.random.choice([1, 0], n, p=[0.18, 0.82])


    # CGPA calculation
    base_cgpa = (((jamb_score - 140) / (370-140) * 4) + 1.0)
    current_cgpa = np.clip(base_cgpa, 1.0, 5.0).round(2)
    # Apply adjustments using np.where
    current_cgpa = np.where(attendance_rate > 70, current_cgpa + 0.2, current_cgpa)
    current_cgpa = np.where(has_scholarship == 1, current_cgpa + 0.3, current_cgpa)
    current_cgpa = np.where(mental_health_support == 1, current_cgpa + 0.1, current_cgpa)

    #Clip at the end to keep values within 1.0–5.0
    current_cgpa = np.clip(current_cgpa, 1.0, 5.0).round(2)

    #Dropout column
    dropout = np.zeros(n)
    # Inputing risk factors
    dropout_score = np.zeros(n)

    dropout_score = np.where(current_cgpa < 2, dropout_score + 1.0, dropout_score)
    dropout_score = np.where(fee_payment_status == "Owing", dropout_score + 1.5, dropout_score)
    dropout_score = np.where(fee_payment_status == "Partial", dropout_score + 0.5, dropout_score)
    dropout_score = np.where(fee_payment_status == "Paid", dropout_score - 1.0, dropout_score)
    dropout_score = np.where(has_scholarship == 0, dropout_score + 0.3, dropout_score)
    dropout_score = np.where(failed_course_count > 2, dropout_score + 0.6, dropout_score)
    dropout_score = np.where(mental_health_support == 0, dropout_score + 0.5, dropout_score)
    dropout_score = np.where((cgpa_trend == "Declining") & (current_cgpa < 2) , dropout_score + 1.0, dropout_score)
    dropout_score = np.where(attendance_rate < 40, dropout_score + 1.0, dropout_score)
    dropout_score = np.where(asuu_strike_semesters > 2, dropout_score + 1.0, dropout_score)
    dropout_score = np.where(distance_from_home_km > 40, dropout_score + 1.0, dropout_score)
    dropout_score = np.where(works_parttime ==  0, dropout_score + 1.0, dropout_score)
    dropout_score = np.where(family_income_bracket == "Low", dropout_score + 1.0, dropout_score)
    dropout_score = np.where((parent_education_level == "No formal") | (parent_education_level == "Primary"), dropout_score + 1.0, dropout_score)
    dropout_score = np.where(school_location == "Rural", dropout_score + 1.0, dropout_score)

    # Convert score to probability using sigmoid
    dropout_prob = 1 / (1 + np.exp(-0.6 * (dropout_score - 6.0)))

    # Randomly assign 0 or 1 based on that probability
    dropout = (np.random.uniform(0, 1, n) < dropout_prob).astype(int)


    df = pd.DataFrame(
        {
            "Student_ID" : student_id,
            "Age" : age,
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
            "Has_scholarship": has_scholarship,
            "Works_part_time": works_parttime,
            "Family_income_bracket": family_income_bracket,
            "Parent_education_level": parent_education_level,
            "Failed_courses_count": failed_course_count,
            "Asuu_strike_semesters": asuu_strike_semesters,
            "Mental_health_support": mental_health_support,
            "Course_load": course_load,
            "Dropout": dropout
        }
    )

    return df


if __name__ == "__main__":
    df = generate_data(N)
    
    # Preview
    print(f"Shape: {df.shape}")
    print(f"Dropout rate: {df['Dropout'].mean():.1%}")
    print(df.head())
    
    # Save to CSV
    df.to_csv("data/students.csv", index=False)
    print("Dataset saved to data/students.csv")