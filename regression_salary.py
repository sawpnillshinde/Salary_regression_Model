import streamlit as st
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder

model=tf.keras.models.load_model('Churn_regression.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('one_hot_geo.pkl','rb') as file:
    one_hot_geo=pickle.load(file)

with open('standard Scaler.pkl','rb') as file:
    Standard_Scaler=pickle.load(file)

#st.title("Salary Estimation Model")


geography=st.selectbox('Geography',one_hot_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited',[0,1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode 'Geography'
geo_encoded = one_hot_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale the input data using the StandardScaler
input_data_scaled = Standard_Scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_salary = prediction[0][0]

st.write(f'Estimated Salary: {prediction_salary:.2f}')
