import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


st.title("""
Iris Flower Prediction
""")

# function to recieve input from user
def user_input_features():
    form = st.form(key='iris_form')
    sepal_length = form.number_input('Sepal length (cm)', 4.3, 7.9, step=0.1)
    sepal_width = form.number_input('Sepal width (cm)', 2.0, 4.4, step=0.1)
    petal_length = form.number_input('Petal length (cm)', 1.0, 6.9, step=0.1)
    petal_width = form.number_input('Petal width (cm)', 0.1, 2.5, step=0.1)
    submit_button = form.form_submit_button(label='Predict')

    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features, submit_button


df, submit = user_input_features()

# Display user input parameters
st.subheader('User Input Parameters')
st.write(df)


data = pd.read_csv('dataset/iris.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train a RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)
prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)

if submit:
    prediction = clf.predict(df)
    prediction_proba = clf.predict_proba(df)

    st.subheader('Prediction')
    st.write('Predicted class:', prediction[0])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

