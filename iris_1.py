import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title = "Iris",
    page_icon = ":🌻:"
)

st.title('Iris')

df = pd.read_csv('dataset/iris.csv')

selected_x = st.selectbox('X axis',['sepal_length','sepal_width','petal_length','petal_width'])
selected_y = st.selectbox('Y axis',['sepal_length','sepal_width','petal_length','petal_width'])

fig, ax = plt.subplots()
sns.scatterplot(data=df, x=selected_x, y=selected_y, hue='species', palette='Set1', ax=ax)
ax.set_title(f'{selected_x} vs {selected_y}')
st.pyplot(fig)

st.write(df)

