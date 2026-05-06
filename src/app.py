import streamlit as st
import numpy as np
import joblib
import json
import os
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Heart Disease Prediction System", layout="wide")

# ---------------- FILES ----------------
USER_DB = "users.json"
HISTORY_DB = "history.json"
PRED_DB = "predictions.json"
ADMIN_USER = "admin"

# ---------------- UTILS ----------------
def load_json(file, default):
    if not os.path.exists(file):
        return default
    try:
        with open(file, "r") as f:
            return json.load(f)
    except:
        return default

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

users = load_json(USER_DB, {})
history = load_json(HISTORY_DB, {"logins": [], "total_logins": 0})
predictions = load_json(PRED_DB, [])

# ---------------- AUTH ----------------
if not st.session_state.get("logged_in", False):

    st.title("Heart Disease Prediction System")

    menu = st.radio("", ["Login", "Register"], horizontal=True)

    if menu == "Register":
        st.subheader("Create Account")
        name = st.text_input("Full Name")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Register"):
            if username in users:
                st.error("User already exists")
            else:
                users[username] = {"name": name, "password": password}
                save_json(USER_DB, users)
                st.success("Registered successfully")

    elif menu == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in users and users[username]["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["user"] = username

                history["total_logins"] += 1
                history["logins"].append({
                    "user": username,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                save_json(HISTORY_DB, history)

                st.rerun()
            else:
                st.error("Invalid credentials")

# ---------------- MAIN ----------------
else:

    # TOP BAR
    col1, col2, col3 = st.columns([6, 2, 1])
    col1.title("Dashboard")
    col2.write(f"👤 {st.session_state['user']}")

    if col3.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

    # MENU
    if st.session_state["user"] == ADMIN_USER:
        menu = st.selectbox("Navigation", ["Dashboard", "Predict", "My History", "Analytics", "Admin"])
    else:
        menu = st.selectbox("Navigation", ["Dashboard", "Predict", "My History"])

    st.markdown("---")

    # ---------------- DASHBOARD ----------------
    if menu == "Dashboard":

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Users", len(users))
        c2.metric("Total Logins", history["total_logins"])
        c3.metric("Current User", st.session_state["user"])

        st.info("Go to Predict page to check heart disease risk.")

    # ---------------- PREDICT ----------------
    elif menu == "Predict":

        st.title("Patient Risk Prediction")

        model = joblib.load("models/xgb_model.pkl")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Patient Name")
            age = st.number_input("Age", 20, 80, 50)
            height = st.number_input("Height (cm)", 140, 210, 165)
            weight = st.number_input("Weight (kg)", 40, 150, 70)
            ap_hi = st.number_input("Systolic BP", 80, 200, 120)
            ap_lo = st.number_input("Diastolic BP", 40, 120, 80)

        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "High"])
            gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "High"])
            smoke = st.selectbox("Smoking", ["No", "Yes"])
            alco = st.selectbox("Alcohol", ["No", "Yes"])
            active = st.selectbox("Physical Activity", ["No", "Yes"])

        # Encoding
        gender = 1 if gender == "Female" else 2
        cholesterol = ["Normal", "Above Normal", "High"].index(cholesterol) + 1
        gluc = ["Normal", "Above Normal", "High"].index(gluc) + 1
        smoke = 1 if smoke == "Yes" else 0
        alco = 1 if alco == "Yes" else 0
        active = 1 if active == "Yes" else 0

        bmi = weight / ((height / 100) ** 2)
        age_group = 1 if age < 40 else 2 if age < 60 else 3

        if st.button("Predict Risk"):

            prob = model.predict_proba(np.array([[age, gender, height, weight,
                                                 ap_hi, ap_lo, cholesterol, gluc,
                                                 smoke, alco, active, bmi, age_group]]))[0][1]

            risk = "Low" if prob < 0.3 else "Moderate" if prob < 0.7 else "High"

            # Save prediction
            predictions.append({
                "user": st.session_state["user"],
                "name": name,
                "age": age,
                "bmi": round(bmi, 2),
                "risk": risk,
                "prob": float(prob),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_json(PRED_DB, predictions)

            st.success(f"{risk} Risk ({prob:.2f})")

    # ---------------- MY HISTORY ----------------
    elif menu == "My History":

        st.title("My Prediction History")

        df = pd.DataFrame(predictions)

        if df.empty:
            st.warning("No data yet")
        else:
            user_df = df[df["user"] == st.session_state["user"]]

            if user_df.empty:
                st.warning("No history found")
            else:
                st.dataframe(user_df)
                st.download_button("Download CSV", user_df.to_csv(index=False), "my_history.csv")

    # ---------------- ANALYTICS ----------------
    elif menu == "Analytics":

        st.title("Analytics")

        df = pd.DataFrame(predictions)

        if df.empty:
            st.warning("No data available")
        else:
            st.subheader("Risk Distribution")
            st.bar_chart(df["risk"].value_counts())

            st.subheader("BMI vs Probability")
            st.scatter_chart(df[["bmi", "prob"]])

    # ---------------- ADMIN ----------------
    elif menu == "Admin":

        st.title("Admin Panel")

        st.metric("Total Users", len(users))
        st.metric("Total Logins", history["total_logins"])

        st.subheader("Registered Users")

        for u, data in users.items():
            st.write(f"{data['name']} ({u})")