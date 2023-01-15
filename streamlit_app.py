import time
import streamlit as st
import joblib
import pandas as pd
# from helper import *

model = joblib.load("final_model.pkl")
df1 = pd.read_csv("final_dataframe.csv")

st.title('*Enter Name* (artık gerçekten gir)')
