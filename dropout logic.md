# Initial Dropout logic

dropout logic
Factors that can affect
current_cgpa
cgpa_trend
fee_payment_status
failed_courses_count
mental_health_support


relationships
current_cgpa -> base_gpa, jamb_score, attendance_rate, has_scholarship, mental_health_support

dropout -> current_cgpa, fee_payment_status, has_scholarship, failed_courses_count, mental_health_status, cgpa_trend
 
dropout_score
if current_Cgpa < 2 then dropout_score = 1.0
else dropout_score += 1.0

if fee_payment_status == "Paid" then dropout_score -= 1.0
elif fee_payment_status == "Partial" then dropout_score += 0.0
elif fee_payment_status == "Owing" then dropout_score += 1.0

if has_scholarship == 0 then dropout_score += 0.3
else dropout_score -= 0.0

if failed_course_count > 2 then dropout_score += 0.6
else dropout_score += 0.0

if mental_health_status == 0 then dropout_score += 0.5
else dropout_score -= 1.0

if cgpa_trend == "declining" && current_cgpa < 2 then dropout_score += 1.0
else dropout_score += 0.0

if attendance_rate < 40 then dropout_score += 1.0
else dropout_score += 0.0

if asuu_strike_semesters > 2 then dropout_score += 1.0
else dropout_score += 0.0

if distance_from_home_km > 50 then dropout_score += 1.0
else dropout_score += 0.0

if works_part_time == 1 then dropout_score += 1.0
else dropout_score += 0.0

if family_income_bracket == "Low" then dropout_score += 1.0
else dropout_score += 0.0

if parent_education_level == "No formal" // parent_education_level == "Primary" then dropout_score += 1.0
else dropout_score += 0.0

if school_location == "Rural" then dropout_score += 1.0
else dropout_score += 0.0




 # REVISED DROPOUT RISK LOGIC

*Latent Risk Engine (The Root Cause)*

To ensure authentic data cohesion and realistic feature correlations, 
every student is assigned a baseline Latent Risk Coefficient (Z).
Z ~ Standard Normal Distribution (Mean = 0, Standard Deviation = 1.0)

All behavioral, academic, and financial features are generated 
directly as a function of this latent risk score.


### Feature Interdependence & Dependencies

1. Financial & Socio-Economic Status
   - Family Income Bracket: High latent risk (Z > 0.5) skews to "Low". Low latent risk skews to "Middle" / "High".
   - Fee Payment Status: Inherited directly from Family Income Bracket.
        * Low Income   -> 45% Owing, 40% Partial, 15% Paid
        * Middle Income -> 10% Owing, 30% Partial, 60% Paid
        * High Income   ->  2% Owing,  8% Partial, 90% Paid
   - Has Scholarship: Derived via logit probability mapped against risk.
        * P(Scholarship) = 1 / (1 + exp(1.5 * Z + 1.2))

2. Geographic & Demographic Metrics
   - Distance From Home (km): Base distance (5-120 km) + Exponential risk penalty.
        * distance = Clip(Base + exp(Z + 2.5), 5, 900)
   - Age: Base age (16-21) + Step-scaled maturity modifier.
        * age = Clip(Base + (Z + 1.5) * 3, 16, 30)

3. Academic Performance Vectoring
   - JAMB Score: Base score (180-340) shifted downward by latent risk.
        * jamb_score = Clip(Base - (30 * Z), 140, 370)
   - Attendance Rate: Skewed negatively by high risk.
        * attendance_rate = Clip(Normal(82, 10) - (12 * Z), 10, 100)
   - Current CGPA: Base scaled from entry metrics + Dynamic operational penalties.
        * cgpa_base = ((jamb_score - 140) / 230 * 3.3) + 1.3
        * penalty = (-0.8 if attendance < 65) + (-0.3 if works_part_time == 1)
        * current_cgpa = Clip(cgpa_base + penalty - (0.4 * Z), 1.0, 5.0)
   - Failed Courses Count: Tightly coupled to final academic performance.
        * if current_cgpa < 2.0 then failed_courses = randint(2, 5)
        * elif current_cgpa < 3.0 then failed_courses = randint(0, 3)
        * else failed_courses = 0



# LOG-ODDS LOGIC & PROBABILITY CALIBRATION


To perfectly bound the institutional dropout velocity to ~22%, 
the structural baseline intercept is anchored heavily at -4.5.

Initialize Log-Odds:
log_odds = -4.5

### Apply Continuous Risk Weights 
log_odds += latent_risk * 1.5
log_odds += (3.5 - current_cgpa) * 1.2
log_odds += failed_courses_count * 0.5
log_odds += (age - 20) * 0.15
log_odds += (distance_from_home_km / 200) * 0.3
log_odds += (5 - o_level_credits) * 0.2

### Apply Categorical Risk Deltas
if fee_payment_status == "Owing" then log_odds += 1.2
elif fee_payment_status == "Partial" then log_odds += 0.6
else log_odds += 0.0

if has_scholarship == 0 then log_odds += 0.5
else log_odds -= 0.8

if works_part_time == 1 then log_odds += 0.6
else log_odds += 0.0

if cgpa_trend == "Declining" then log_odds += 0.7
else log_odds += 0.0

if school_location == "Rural" then log_odds += 0.4
else log_odds += 0.0

if mental_health_support == 0 then log_odds += 0.6
else log_odds -= 0.4

if parent_education_level in ["No formal", "Primary"] then log_odds += 0.5
else log_odds += 0.0

log_odds += asuu_strike_semesters * 0.4


### Probability Transformation & Selection

Convert structural log-odds to target probability using Sigmoid:
dropout_probability = 1 / (1 + exp(-log_odds))

Resolve definitive binary class assignment via a Uniform distribution threshold:
if Uniform(0, 1) < dropout_probability then dropout = 1
else dropout = 0