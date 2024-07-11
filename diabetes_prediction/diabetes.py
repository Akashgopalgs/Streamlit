import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.title('Diabetes Prediction')

df = pd.read_csv('dataset/diabetes.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

def user_input_features():
    form = st.form(key='diabetes_form')
    pregnancies = form.number_input('Enter the number of Pregnancies', min_value=df['Pregnancies'].min(), max_value=df['Pregnancies'].max())
    glucose = form.number_input('Enter the Glucose Level', min_value=df['Glucose'].min(), max_value=df['Glucose'].max())
    bloodPressure = form.number_input('Enter your BP measurement', min_value=df['BloodPressure'].min(), max_value=df['BloodPressure'].max())
    skinThickness = form.number_input('Enter the thickness of your skin', min_value=df['SkinThickness'].min(), max_value=df['SkinThickness'].max())
    insulin = form.number_input('Enter your Insulin Level', min_value=df['Insulin'].min(), max_value=df['Insulin'].max())
    bmi = form.number_input('Enter your BMI', min_value=df['BMI'].min(), max_value=df['BMI'].max())
    diabetes_pedigree = form.number_input('Enter Diabetes Pedigree Function', min_value=df['DiabetesPedigreeFunction'].min(), max_value=df['DiabetesPedigreeFunction'].max())
    age = form.number_input('Enter your Age', min_value=df['Age'].min(), max_value=df['Age'].max())
    submit_button = form.form_submit_button('Predict')

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bloodPressure,
        'SkinThickness': skinThickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }

    features = pd.DataFrame(data, index=[0])
    return features, submit_button


df_input, submit = user_input_features()


st.subheader('User Input Parameters')
st.write(df_input)


reg = LogisticRegression(max_iter=1000)
reg.fit(X, y)

if submit:
    prediction = reg.predict(df_input)
    prediction_prob = reg.predict_proba(df_input)

    st.subheader('Prediction')
    st.write('Diabetes Prediction:', 'Positive' if prediction[0] == 1 else 'Negative')

    st.subheader('Prediction Probability')
    st.write(prediction_prob)
