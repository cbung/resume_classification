import time
import streamlit as st
import joblib
import pandas as pd
from helper import *

model = joblib.load("final_model.pkl")
df1 = pd.read_csv("final_dataframe.csv")

st.markdown("<h1 style='text-align: center; color: ##00a6f9;'>İstanbul Ev Kiraları Tahminleme</h1>",
            unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #1B9E91;'>Google, Amazon gibi büyük şirketlerdeki veri ile ilgili meslekler yapan insanlardan aldığımız verilerle,"
            " bu büyük şirketlerde çalışmak isteyen kişilerin kendilerini kıyaslamalarına fırsat verip,"
            " bu çalışanlar arasından hangi sınıfa dahil olduğunu görmesini sağlayacak bir uygulama</h5>",
            unsafe_allow_html=True)
