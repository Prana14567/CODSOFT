import streamlit as st
import joblib
import numpy as np

model = joblib.load('final_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Titanic Survival Prediction")

age = st.number_input("Age")
fare = st.number_input("Fare")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['Male', 'Female'])
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, step=1)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, step=1)
embarked_q = st.checkbox("Embarked at Q?")
embarked_s = st.checkbox("Embarked at S?")

input_data = np.array([[pclass,
                        0 if sex == 'Male' else 1,
                        age,
                        sibsp,
                        parch,
                        fare,
                        int(embarked_q),
                        int(embarked_s)
                       ]])

# Scale the input
input_scaled = scaler.transform(input_data)

# Predict and display result
prediction = model.predict(input_scaled)
st.write("🟢 Survived" if prediction[0] == 1 else "🔴 Did not survive")
