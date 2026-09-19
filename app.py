import streamlit as st
import joblib
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

# 1. Load the saved model and scaler
model = tf.keras.models.load_model('my_bank_model.keras')
scaler = joblib.load('my_scaler.pkl')

# 2. Build the visual interface
st.title("Bank Churn Prediction Model")
st.write("Enter the customer's details to predict if they will leave the bank.")

# Create input boxes for the user
credit_score = st.number_input("Credit Score", value=600)
age = st.number_input("Age", value=40)
tenure = st.number_input("Tenure (Years)", value=3)
balance = st.number_input("Account Balance", value=60000.0)
salary = st.number_input("Estimated Salary", value=50000.0)

# 3. Predict when the button is clicked
if st.button("Predict Churn Risk"):
    # Insert the user inputs into your 12-item array
    # (Leaving Geography as France [2,0,0], Gender as Male [1], Products [2], Card [1], Active [1])
    customer_data = [[2, 0, 0, credit_score, 1, age, tenure, balance, 2, 1, 1, salary]]
    
    # Scale and run through the neural network
    scaled_data = scaler.transform(customer_data)
    prediction = model.predict(scaled_data)
    probability = prediction[0][0]
    
    # Display the result
    st.subheader(f"Churn Probability: {probability * 100:.2f}%")
    
    if probability > 0.5:
        st.error("High Risk: Customer is likely to churn.")
    else:
        st.success("Low Risk: Customer is likely to stay.")
