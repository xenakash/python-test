import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = "employee_attrition_model.pkl"

# ---------------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Could not find '{MODEL_PATH}'. Make sure the .pkl file is in the "
            "same repo/folder as app.py."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


pipeline = load_model()
classifier = pipeline.named_steps["classifier"]
preprocessor = pipeline.named_steps["preprocessor"]


# ---------------------------------------------------------------------------
# Pull the exact categories the model was trained on, straight from the
# fitted OneHotEncoder. This guarantees the dropdown options always match
# what the pipeline expects, even if the underlying dataset changes.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_categorical_options():
    cat_cols = ["Gender", "Marital_Status", "Department", "Job_Role", "Overtime"]
    fallback = {
        "Gender": ["Male", "Female"],
        "Marital_Status": ["Single", "Married", "Divorced"],
        "Department": ["Finance", "HR", "Marketing", "Sales", "IT"],
        "Job_Role": ["Manager", "Assistant", "Analyst", "Executive"],
        "Overtime": ["Yes", "No"],
    }
    try:
        ohe = preprocessor.named_transformers_["cat"]
        options = {col: list(cats) for col, cats in zip(cat_cols, ohe.categories_)}
        return options
    except Exception:
        return fallback


cat_options = get_categorical_options()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Employee Attrition Prediction")
st.markdown(
    "Enter an employee's details below to predict the likelihood they will "
    "leave the company. This app uses a machine learning pipeline trained on "
    "historical HR data (see the accompanying notebook for full methodology)."
)
st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("attrition_form"):

    st.subheader("👤 Personal Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=65, value=40)
    with c2:
        gender = st.selectbox("Gender", cat_options["Gender"])
    with c3:
        marital_status = st.selectbox("Marital Status", cat_options["Marital_Status"])

    st.subheader("💼 Job Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        department = st.selectbox("Department", cat_options["Department"])
    with c2:
        job_role = st.selectbox("Job Role", cat_options["Job_Role"])
    with c3:
        job_level = st.slider("Job Level", 1, 5, 3)

    c1, c2, c3 = st.columns(3)
    with c1:
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=15)
    with c2:
        years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=7)
    with c3:
        years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=4)

    c1, c2, c3 = st.columns(3)
    with c1:
        num_companies = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
    with c2:
        distance_from_home = st.number_input("Distance From Home (km)", min_value=0, max_value=100, value=25)
    with c3:
        overtime = st.selectbox("Overtime", cat_options["Overtime"])

    st.subheader("💰 Compensation & Workload")
    c1, c2, c3 = st.columns(3)
    with c1:
        monthly_income = st.number_input("Monthly Income", min_value=0, max_value=50000, value=11400, step=100)
    with c2:
        hourly_rate = st.number_input("Hourly Rate", min_value=0, max_value=200, value=57)
    with c3:
        avg_hours_week = st.number_input("Average Hours Worked / Week", min_value=0, max_value=100, value=44)

    c1, c2, c3 = st.columns(3)
    with c1:
        project_count = st.number_input("Project Count", min_value=0, max_value=20, value=5)
    with c2:
        training_hours = st.number_input("Training Hours Last Year", min_value=0, max_value=200, value=50)
    with c3:
        absenteeism = st.number_input("Absenteeism (days)", min_value=0, max_value=60, value=9)

    st.subheader("⭐ Satisfaction & Performance (scale 1 = low – 4/5 = high)")
    c1, c2, c3 = st.columns(3)
    with c1:
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 2)
    with c2:
        job_satisfaction = st.slider("Job Satisfaction", 1, 5, 3)
    with c3:
        performance_rating = st.slider("Performance Rating", 1, 4, 2)

    c1, c2, c3 = st.columns(3)
    with c1:
        work_env_satisfaction = st.slider("Work Environment Satisfaction", 1, 4, 2)
    with c2:
        relationship_manager = st.slider("Relationship with Manager", 1, 4, 2)
    with c3:
        job_involvement = st.slider("Job Involvement", 1, 4, 2)

    submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Marital_Status": marital_status,
        "Department": department,
        "Job_Role": job_role,
        "Job_Level": job_level,
        "Monthly_Income": monthly_income,
        "Hourly_Rate": hourly_rate,
        "Years_at_Company": years_at_company,
        "Years_in_Current_Role": years_in_role,
        "Years_Since_Last_Promotion": years_since_promotion,
        "Work_Life_Balance": work_life_balance,
        "Job_Satisfaction": job_satisfaction,
        "Performance_Rating": performance_rating,
        "Training_Hours_Last_Year": training_hours,
        "Overtime": overtime,
        "Project_Count": project_count,
        "Average_Hours_Worked_Per_Week": avg_hours_week,
        "Absenteeism": absenteeism,
        "Work_Environment_Satisfaction": work_env_satisfaction,
        "Relationship_with_Manager": relationship_manager,
        "Job_Involvement": job_involvement,
        "Distance_From_Home": distance_from_home,
        "Number_of_Companies_Worked": num_companies,
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]  # P(Attrition = Yes)

    st.divider()
    st.subheader("Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction == 1:
            st.error("### ⚠️ Likely to Leave")
        else:
            st.success("### ✅ Likely to Stay")
        st.metric("Attrition Probability", f"{probability * 100:.1f}%")

    with res_col2:
        st.progress(min(max(probability, 0.0), 1.0))
        if probability >= 0.7:
            st.warning("High risk of attrition. Consider proactive retention measures such as "
                       "a compensation review, workload adjustment, or a manager check-in.")
        elif probability >= 0.4:
            st.info("Moderate risk of attrition. Worth monitoring engagement and satisfaction "
                    "over the coming months.")
        else:
            st.info("Low risk of attrition based on current inputs.")

    # -----------------------------------------------------------------
    # Top drivers for this model (global feature importance, if available)
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("What Drives This Model's Predictions")

    try:
        feature_names = preprocessor.get_feature_names_out()
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        else:
            importances = None

        if importances is not None:
            fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            fi_df = fi_df.sort_values("Importance", ascending=False).head(10)
            fi_df["Feature"] = fi_df["Feature"].str.replace(r"^(num__|cat__)", "", regex=True)
            st.bar_chart(fi_df.set_index("Feature"))
            st.caption("Top 10 features the model relies on most, across all predictions "
                       "(not specific to this individual employee).")
    except Exception:
        st.caption("Feature importance is not available for this model type.")

st.divider()
st.caption(
    "Model trained and exported via the accompanying `Employee_Attrition.ipynb` notebook. "
    "Predictions are estimates based on historical patterns and should support, not replace, HR judgment."
)
