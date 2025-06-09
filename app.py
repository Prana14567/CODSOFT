import streamlit as st
import joblib
import numpy as np

model = joblib.load('final_model.pkl')  
scaler = joblib.load('scaler.pkl')


st.title("🚢 Titanic Survival Prediction")

age = st.number_input("Age", min_value=0.0, step=1.0)
fare = st.number_input("Fare", min_value=0.0, step=1.0)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['Male', 'Female'])
embarked_q = st.checkbox("Embarked at Q?")
embarked_s = st.checkbox("Embarked at S?")


if st.button("Predict"):
    try:
        
        input_data = np.array([[pclass,
                                0 if sex == 'Male' else 1,
                                age,
                                fare,
                                int(embarked_q),
                                int(embarked_s)]])
        
        
        input_scaled = scaler.transform(input_data)

    
        prediction = model.predict(input_scaled)
        result = "🟢 Survived" if prediction[0] == 1 else "🔴 Did not survive"
        st.success(f"Prediction: {result}")
    
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
