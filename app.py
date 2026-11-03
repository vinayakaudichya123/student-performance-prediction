
import streamlit as st
import pandas as pd
import os
import joblib

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.write("AI-powered student performance analysis dashboard")

st.markdown("---")

model_path = "student_performance_model.pkl"

if not os.path.exists(model_path):
    st.warning(
        "Trained model file nahi mili. "
        "Apni trained model file ko app.py ke folder mein rakho."
    )
    st.info(
        "Model file ka exact naam apne training code se check karo."
    )
    st.stop()

model = joblib.load(model_path)

st.subheader("📝 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 15, 60, 20)
    department = st.selectbox(
        "Department", ["CSE", "AIDS", "IT", "ECE", "AI&DS"]
    )
    city = st.text_input("City", "Udaipur")
    attendance = st.slider("Attendance (%)", 0, 100, 80)

with col2:
    study_hours = st.slider("Study Hours Per Day", 0.0, 24.0, 4.0)
    assignment_score = st.slider("Assignment Score", 0.0, 100.0, 75.0)
    internal_marks = st.slider("Internal Marks", 0.0, 100.0, 70.0)
    family_income = st.number_input(
        "Family Income", min_value=0, value=300000
    )
    backlogs = st.number_input(
        "Number of Backlogs", min_value=0, max_value=30, value=0
    )

if st.button("🔮 Predict Student Performance", type="primary"):
    student = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Department": department,
        "City": city,
        "Attendance_Percent": attendance,
        "Study_Hours_Per_Day": study_hours,
        "Assignment_Score": assignment_score,
        "Internal_Marks": internal_marks,
        "Family_Income": family_income,
        "Backlogs": backlogs
    }])

    try:
        prediction = model.predict(student)[0]
        st.success(f"Predicted Performance: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(student)[0]
            classes = model.classes_

            results = pd.DataFrame({
                "Performance Level": classes,
                "Model Score (%)": (probabilities * 100).round(2)
            }).sort_values("Model Score (%)", ascending=False)

            st.subheader("📊 Prediction Scores")
            st.bar_chart(
                results.set_index("Performance Level")["Model Score (%)"]
            )
            st.dataframe(results, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info(
            "Check karo ki model training mein wahi feature columns "
            "aur preprocessing use hui thi."
        )
