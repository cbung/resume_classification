import time
import streamlit as st
import joblib
import pandas as pd
from helper import *

model = joblib.load("final_model.pkl")
df1 = pd.read_csv("final_dataframe.csv")

st.markdown("<h1 style='text-align: center; color: ##00a6f9;'>Will Google Hire You?</h1>",
            unsafe_allow_html=True)
# st.markdown("<h5 style='text-align: center; color: #1B9E91;'>Google, Amazon gibi büyük şirketlerdeki veri ile ilgili meslekler yapan insanlardan aldığımız verilerle,"
#             " bu büyük şirketlerde çalışmak isteyen kişilerin kendilerini bu çalışanlarla kıyaslamalarına fırsat verip,"
#             " bu çalışanlar arasından hangi sınıfa dahil olduğunu görmesini sağlayoruz.</h5>",
#             unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #1B9E91;'>We use data from individuals working in data-related professions at large companies such as Google and Amazon to create a machine learning model. We then utilize this model to provide individuals who desire to work at these companies the opportunity to compare themselves to the existing employees and determine in which predetermined class they fall.</h5>",
            unsafe_allow_html=True)
