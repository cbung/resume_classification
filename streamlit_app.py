import time
import streamlit as st
import joblib
import pandas as pd
from helper import *

model = joblib.load("final_model.pkl")
df = pd.read_csv("final_dataframe.csv").drop(columns="Unnamed: 0")

st.markdown("<h1 style='text-align: center; color: #e6f2ff;'>Will Google Hire You?</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center; color: #80b3ff;'>We use data from individuals working in data-related professions at large companies such as Google and Amazon to create a machine learning model. We then utilize this model to provide individuals who desire to work at these companies the opportunity to compare themselves to the existing employees and determine in which predetermined class they fall.</h5>",
    unsafe_allow_html=True)

option_xp_lvl = st.sidebar.selectbox('Months Of Experience:', ('0-24 Months', '24-60 Months', "60-120 Months", "More Than 120 Months"))
if option_xp_lvl == '0-24 Months':
    option_xp_lvl = "Junior"
    option_junior = 1
    option_mid = 0
    option_senior = 0
    option_master = 0
elif option_xp_lvl == '24-60 Months':
    option_xp_lvl = "Mid-Level"
    option_junior = 0
    option_mid = 1
    option_senior = 0
    option_master = 0
elif option_xp_lvl == '60-120 Months':
    option_xp_lvl = "Senior"
    option_junior = 0
    option_mid = 0
    option_senior = 1
    option_master = 0
else:
    option_xp_lvl = "Principal"
    option_junior = 0
    option_mid = 0
    option_senior = 0
    option_master = 1

option_highest_degree = st.sidebar.selectbox('Highest Academic Degree:', ("Bachelor's Degree", "Master's Degree", "Doctorate Degree", "Other"))
if option_highest_degree == "Doctorate Degree":
    option_highest_degree = "Doctor"
    option_bachelors = 0
    option_doctors = 1
    option_masters = 0
elif option_highest_degree == "Master's Degree":
    option_highest_degree = "Master"
    option_bachelors = 0
    option_doctors = 0
    option_masters = 1
else:
    option_highest_degree = "Bachelor"
    option_bachelors = 1
    option_doctors = 0
    option_masters = 0

new_user_xp_edu = {
    "NEW_HIGHEST_DEGREE_doctor": option_doctors,
    "NEW_HIGHEST_DEGREE_master": option_masters,
    "NEW_EXPERIENCE_LEVEL_Mid": option_mid,
    "NEW_EXPERIENCE_LEVEL_Senior": option_senior,
    "NEW_EXPERIENCE_LEVEL_Master": option_master
}
new_user_xp_edu_df = pd.DataFrame(new_user_xp_edu, index=[0])

skill_list = [col[6:] for col in df.columns if col.__contains__("SKILL_")]
option_skills = st.sidebar.multiselect("Skills (Can Select Multiple Choices):", skill_list, [], max_selections=100)
selected_skill_list = [[]]
for ind_skill in enumerate(option_skills):
    selected_skill_list[0].append(f"SKILL_{ind_skill[1]}")


new_user_skill = {"skills": selected_skill_list}

new_user_skill_df = pd.DataFrame(new_user_skill)

newframe = pd.DataFrame()
for skill in skill_list:
    newframe[f"SKILL_{skill}".upper()] = new_user_skill_df["skills"].apply(lambda x: 0)
new_user_skill_df = pd.concat([new_user_skill_df, newframe], axis=1)


col1, col2, col3 = st.columns(3)
my_bar = st.progress(0)

from PIL import Image
image = Image.open('background_image.jpg')
st.image(image, use_column_width="always")
with col2:
    if st.button("\t", "Which class am I", "\t"):
        for ind_skill in selected_skill_list[0]:
            new_user_skill_df[f"{ind_skill}".upper()] = new_user_skill_df["skills"].apply(lambda x: 1 if ind_skill in x else 0)

        new_user_skill_df.drop(columns="skills", inplace=True)

        new_user_df = pd.concat([new_user_skill_df, new_user_xp_edu_df], axis=1)

        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1)
        new_user_pred = model.predict(new_user_df)

        st.success(f"{option_highest_degree}, {option_xp_lvl}, {new_user_pred[0]}".upper())


st.markdown("<h3 style='text-align: center; color: #1C9B41;'>Learn More About The Classes</h3>", unsafe_allow_html=True)




