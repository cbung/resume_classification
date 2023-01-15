import time
import streamlit as st
import joblib
import pandas as pd
from helper import *

model = joblib.load("final_model.pkl")
df = pd.read_csv("final_dataframe.csv")

st.markdown("<h1 style='text-align: center; color: ##00a6f9;'>Will Google Hire You?</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center; color: #1B9E91;'>We use data from individuals working in data-related professions at large companies such as Google and Amazon to create a machine learning model. We then utilize this model to provide individuals who desire to work at these companies the opportunity to compare themselves to the existing employees and determine in which predetermined class they fall.</h5>",
    unsafe_allow_html=True)

option_xp_lvl = st.sidebar.selectbox('Months Of Experience:', ('0-24 Months', '24-60 Months', "60-120 Months", "More Than 120 Months"))
if option_xp_lvl == '0-24 Months':
    option_xp_lvl = "Junior"
elif option_xp_lvl == '24-60 Months':
    option_xp_lvl = "Mid"
elif option_xp_lvl == '60-120 Months':
    option_xp_lvl = "Senior"
else:
    option_xp_lvl = "Master"

option_highest_degree = st.sidebar.selectbox('Highest Academic Degree:', ("Bachelor's Degree", "Master's Degree", "Doctorate Degree", "Other"))
if option_highest_degree == "Bachelor's Degree":
    option_highest_degree = "bachelor"
elif option_highest_degree == "Master's Degree":
    option_highest_degree = "master"
elif option_highest_degree == "Doctorate Degree":
    option_highest_degree = "doctor"
else:
    option_highest_degree = "other"

skill_list = [col[6:] for col in df.columns if col.__contains__("SKILL_")]

option_skills = st.sidebar.multiselect("Skills (Can Select Multiple Choices):", skill_list, [], max_selections=100)

selected_skill_list = [[]]
for ind_skill in enumerate(option_skills):
    selected_skill_list[0].append(f"SKILL_{ind_skill[1]}")


new_user = {
    "NEW_EXPERIENCE_LEVEL": option_xp_lvl,
    "NEW_HIGHEST_DEGREE": option_highest_degree,
    "skills": selected_skill_list
}

new_user_df = pd.DataFrame(new_user)

newframe = pd.DataFrame()
for skill in skill_list:
    newframe[f"SKILL_{skill}".upper()] = new_user_df["skills"].apply(lambda x: 0)

new_user_df = pd.concat([new_user_df, newframe], axis=1)

if st.sidebar.button("Save Choices"):
    for ind_skill in selected_skill_list[0]:
        new_user_df[f"{ind_skill}".upper()] = new_user_df["skills"].apply(lambda x: 1 if ind_skill in x else 0)

    new_user_df.drop(columns="skills", inplace=True)

    df = df.concat([df, new_user_df], axis=0)

    st.write("You: ", df)
