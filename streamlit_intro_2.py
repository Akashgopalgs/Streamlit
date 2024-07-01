import streamlit as st
import numpy as np
import pandas as pd

st.title('North pole penguin')

penguin_df=pd.read_csv('dataset/penguins.csv.xls')
penguin_df
