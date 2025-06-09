import streamlit as st
import joblib
import numpy as np

model = joblib.load('final_model.pkl')
scaler = joblib.load('scaler.pkl')


st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details below:")

# Inputs
age = st.number_input("Age", min_value=0.0, max_value=100.0, step=1.0)
fare = st.number_input("Fare", min_value=0.0, step=0.1)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['Male', 'Female'])
sibsp = st.number_input("SibSp", min_value=0, max_value=10, step=1)
parch = st.number_input("Parch", min_value=0, max_value=10, step=1)
embarked_q = st.checkbox("Embarked at Q")
embarked_s = st.checkbox("Embarked at S")

# Prepare input data (same feature order as used during model training)
input_data = np.array([[
    pclass,
    0 if sex == 'Male' else 1,
    age,
    sibsp,
    parch,
    fare,
    int(embarked_q),
    int(embarked_s)
]])

# Scale input
try:
    input_scaled = scaler.transform(input_data)

    # Predict and display result
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.success("🟢 Prediction: Survived")
    else:
        st.error("🔴 Prediction: Did Not Survive")
except Exception as e:
    st.error(f"⚠️ Error during prediction: {e}")

