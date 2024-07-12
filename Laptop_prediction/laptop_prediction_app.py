import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the trained model, scalers, and feature names
with open('laptop_prediction/laptop.pkl', 'rb') as file:
    model = pickle.load(file)

with open('laptop_prediction/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('laptop_prediction/mmscaler.pkl', 'rb') as file:
    mmscaler = pickle.load(file)

with open('laptop_prediction/feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)

# Function to get user inputs
def user_input_features():
    form = st.form(key='diabetes_form')
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
        'SSD': 1,
        'HDD': 0,
        'Flash Storage': 0,
        'Hybrid': 0,
        'Ultrabook': 1 if TypeName == 'Ultrabook' else 0,
        'Notebook': 1 if TypeName == 'Notebook' else 0,
        'Netbook': 1 if TypeName == 'Netbook' else 0,
        'Gaming': 1 if TypeName == 'Gaming' else 0,
        'Convertible': 1 if TypeName == 'Convertible' else 0,
        'Workstation': 1 if TypeName == 'Workstation' else 0,
        'Apple': 1 if Company == 'Apple' else 0,
        'HP': 1 if Company == 'HP' else 0,
        'Acer': 1 if Company == 'Acer' else 0,
        'Asus': 1 if Company == 'Asus' else 0,
        'Dell': 1 if Company == 'Dell' else 0,
        'Lenovo': 1 if Company == 'Lenovo' else 0,
        'Chuwi': 1 if Company == 'Chuwi' else 0,
        'MSI': 1 if Company == 'MSI' else 0,
        'Microsoft': 1 if Company == 'Microsoft' else 0,
        'Toshiba': 1 if Company == 'Toshiba' else 0,
        'Huawei': 1 if Company == 'Huawei' else 0,
        'Xiaomi': 1 if Company == 'Xiaomi' else 0,
        'Vero': 1 if Company == 'Vero' else 0,
        'Razer': 1 if Company == 'Razer' else 0,
        'Mediacom': 1 if Company == 'Mediacom' else 0,
        'Samsung': 1 if Company == 'Samsung' else 0,
        'Google': 1 if Company == 'Google' else 0,
        'Fujitsu': 1 if Company == 'Fujitsu' else 0,
        'LG': 1 if Company == 'LG' else 0
    }
    return pd.DataFrame(user_data, index=[0]),submit_button

# Function to apply scaling
def apply_scaling(df):
    df[['x resolution', 'y resolution']] = scaler.fit_transform(df[['x resolution', 'y resolution']])
    df[['Ram', 'Weight', 'storage', 'Inches']] = mmscaler.fit_transform(df[['Ram', 'Weight', 'storage', 'Inches']])
    return df

# Main function to run the app
def main():
    st.title('Laptop Price Prediction')

    st.write("Please provide the following details to predict the laptop price:")

    # Get user input
    input_df,submit= user_input_features()

    # Ensure input_df has the same columns as feature_names
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Show user input
    st.write("##### User Input Features")
    st.write(input_df)

    # Apply scaling to user input
    input_df = apply_scaling(input_df)

    # Predict using the model
    prediction = model.predict(input_df)
    if submit:

        # Show prediction
        st.write('#### Laptop Price')
        st.write(f"Predicted Laptop Price : ${prediction[0]:.2f}")

if __name__ == '__main__':
    main()
