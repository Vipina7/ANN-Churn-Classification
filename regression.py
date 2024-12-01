import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scaler
with open('onehot_encoder_geography.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler_reg.pkl','rb') as file:
    scaler_reg = pickle.load(file)

## Streamlit app
st.title('Salary Prediction')

# user input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0,1])
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number Of Products',1, 4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

# Example input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[age],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[exited]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out())

#combine one hot encoded df with the input data
input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis=1)

#scale the data
scale_columns = ['CreditScore','Age','Tenure','Balance','NumOfProducts']
input_data[scale_columns] = scaler_reg.transform(input_data[scale_columns])

# Predict Churn
prediction = model.predict(input_data)

st.write(f"The customer's estimated Salary:{prediction}")
