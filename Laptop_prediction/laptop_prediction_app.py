import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define the path to the directory containing the model and scaler files
model_directory = r'C:\Users\akash\OneDrive\Documents\streamlit\Laptop_prediction'

# Function to check and load files
def load_pickle_file(file_name):
    file_path = os.path.join(model_directory, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error(f"File not found: {file_path}")
        return None

# Load the trained model, scalers, and feature names
model = load_pickle_file('laptop.pkl')
scaler = load_pickle_file('scaler.pkl')
mmscaler = load_pickle_file('mmscaler.pkl')
feature_names = load_pickle_file('feature_names.pkl')

if model is None or scaler is None or mmscaler is None or feature_names is None:
    st.stop()

# Function to get user inputs
def user_input_features():
    form = st.form(key='laptop_form')
    Company = form.selectbox('Company', [
        'Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI', 'Microsoft', 'Toshiba',
        'Huawei', 'Xiaomi', 'Vero', 'Razer', 'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG'
    ])
    TypeName = form.selectbox('Type', ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', 'Convertible', 'Workstation'])
    Ram = form.slider('Ram (GB)', 2, 64, step=2)
    Weight = form.slider('Weight (kg)', 1.0, 5.0, step=0.1)
    ScreenResolution = form.selectbox('Screen Resolution', [
        '1920x1080', '1366x768', '3200x1800', '2560x1440', '2304x1440', '3840x2160'
    ])
    Touchscreen = form.selectbox('Touchscreen', ['Yes', 'No'])
    IPS_Panel = form.selectbox('IPS Panel', ['Yes', 'No'])
    CPU = form.selectbox('CPU', ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'AMD', 'Other'])
    GPU = form.selectbox('GPU', ['Intel', 'AMD', 'Nvidia'])
    storage = form.slider('Storage (GB)', 32, 2048, step=32)
    Inches = form.slider('Inches', 10, 18, step=1)
    submit_button = form.form_submit_button('Predict')

    user_data = {
        'Ram': Ram,
        'Weight': Weight,
        'Touchscreen': 1 if Touchscreen == 'Yes' else 0,
        'IPS Panel': 1 if IPS_Panel == 'Yes' else 0,
        'x resolution': int(ScreenResolution.split('x')[0]),
        'y resolution': int(ScreenResolution.split('x')[1]),
        'intel i3': 1 if CPU == 'Intel Core i3' else 0,
        'intel i5': 1 if CPU == 'Intel Core i5' else 0,
        'intel i7': 1 if CPU == 'Intel Core i7' else 0,
        'AMD': 1 if CPU == 'AMD' else 0,
        'other cpu': 1 if CPU == 'Other' else 0,
        'intel gpu': 1 if GPU == 'Intel' else 0,
        'AMD gpu': 1 if GPU == 'AMD' else 0,
        'Nvidia': 1 if GPU == 'Nvidia' else 0,
        'storage': storage,
        'Inches': Inches,
        'Ultrabook': 1 if TypeName == 'Ultrabook' else 0,
        'Notebook': 1 if TypeName == 'Notebook' else 0,
        'Netbook': 1 if TypeName == 'Netbook' else 0,
        'Gaming': 1 if TypeName == 'Gaming' else 0,
        'Convertible': 1 if TypeName == 'Convertible' else 0,
        'Workstation': 1 if TypeName == 'Workstation' else 0,
        **{company: 1 if company == Company else 0 for company in feature_names if company not in [
            'Ram', 'Weight', 'Touchscreen', 'IPS Panel', 'x resolution', 'y resolution', 'intel i3', 'intel i5',
            'intel i7', 'AMD', 'other cpu', 'intel gpu', 'AMD gpu', 'Nvidia', 'storage', 'Inches',
            'Ultrabook', 'Notebook', 'Netbook', 'Gaming', 'Convertible', 'Workstation'
        ]}
    }

    if submit_button:
        return pd.DataFrame(user_data, index=[0])
    return pd.DataFrame()

# Get user input
input_df = user_input_features()

st.write('#### User Input Features')
st.write(input_df)

# Ensure the input DataFrame has the same columns in the same order as feature_names
if not input_df.empty:
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Preprocess input data
if not input_df.empty:
    input_df[['x resolution', 'y resolution']] = scaler.fit_transform(input_df[['x resolution', 'y resolution']])
    input_df[['Ram', 'Weight', 'storage', 'Inches']] = mmscaler.fit_transform(input_df[['Ram', 'Weight', 'storage', 'Inches']])

# Display prediction
if not input_df.empty:
    prediction = model.predict(input_df)
    st.subheader('Predicted Price')
    st.write(f"Predicted Laptop Price :  ${round(prediction[0])}")
