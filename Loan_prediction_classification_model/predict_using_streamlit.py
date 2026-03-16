import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from PIL import Image
import os
os.chdir('D:\\topmentor\\Cohort 127_ML_ Day 48\\MY SQL Project Python\\Data Set\\')
mapping = {'A':0, 'B':1, 'C':2, 'D':3}
reverse_mapping = {v:k for k,v in mapping.items()}

# set the page configuration to wide
st.set_page_config(layout="wide")
#pickle.load(open('kneighbors_model_loan_status.pkl','rb'))

# Load your pre-trained model
with open('kneighbors_model_loan_status.pkl', 'rb') as f:
    lm2 = pickle.load(f)

# Sidebar setup
image_sidebar=Image.open('loan.png') # replace with your image path
st.sidebar.image(image_sidebar, use_container_width=True)
st.sidebar.header('Loan Features')

def get_user_input():
    LOAN_AMOUNT = st.sidebar.number_input('Loan Amount (No)', min_value=0, max_value=1000000, step=5000, value=40000)
    DURATION = st.sidebar.number_input('Account Duration (Months)', min_value=0, max_value=100, step=3, value=12)
    PAYMENTS = st.sidebar.number_input('Payments (No)', min_value=0, max_value=1000000, step=5000, value=10000)
 

    
    # Explanation  for f'Body Size_{body_size}'
    # The value assigned to variable body_size conctinated with Body Size_
    # f is for formatted output string 

    # Example 
    # body_size='Compact'
    # x= f'Body Size_{body_size}'
    # print (x)

    # Output : Body Size_Compact
    
    user_data = {
        'LOAN_AMOUNT': LOAN_AMOUNT,
        'DURATION': DURATION,
        'PAYMENTS': PAYMENTS
    }
    return user_data

# Top banner
image_banner = Image.open('loan1.png')  # Replace with your image file
st.image(image_banner, use_container_width=True)

# Centred title
st.markdown("<h1 style='text-align: center;'>Loan Status Prediction App</h1>", unsafe_allow_html=True)

# Split layout into two columns
left_col, right_col = st.columns(2)

with left_col:
    st.header("Feature Details")
    
    # User inputs from sidebar
    user_data = get_user_input()
    st.write (user_data)
# Right column: Prediction Interface
with right_col:
    st.header("Predict Loan Status")
    
        # Transform the input into the required format
    def prepare_input(data, feature_list):
        input_data = {feature: data.get(feature, 0) for feature in feature_list}
        print('input_data:',input_data)
        return np.array([list(input_data.values())])

    # Feature list (same order as used during model training)
    features = [
        'LOAN_AMOUNT','DURATION','PAYMENTS'
    ]

    # Predict button
    if st.button("Predict"):
        input_array = prepare_input(user_data, features)
        prediction = lm2.predict(input_array)
        st.subheader("Predicted Loan Status")
        st.write(f"{reverse_mapping[prediction[0]]}")
        


