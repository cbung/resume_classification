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

st.markdown("""---""")

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

col1, col2, col3 = st.columns([1, 5, 1])
button1, button2, button3 = st.columns([4, 3, 4])
my_bar = st.progress(0)

from PIL import Image
st.markdown("""---""")
with button2:
    pred_button = st.button("Which Class Are You")
    if pred_button:
        with col2:
            for ind_skill in selected_skill_list[0]:
                new_user_skill_df[f"{ind_skill}".upper()] = new_user_skill_df["skills"].apply(lambda x: 1 if ind_skill in x else 0)

            new_user_skill_df.drop(columns="skills", inplace=True)

            new_user_df = pd.concat([new_user_skill_df, new_user_xp_edu_df], axis=1)

            for percent_complete in range(100):
                time.sleep(0.0175)
                my_bar.progress(percent_complete + 1)
            st.balloons()

            new_user_pred = model.predict(new_user_df)
            new_user_pred = new_user_pred[0]

            # st.write("dataframe silinecek!")
            # st.write(new_user_df)

            # st.success(f"You are ... "
            #            f"{option_highest_degree.capitalize()}, {option_xp_lvl.capitalize()}, {new_user_pred}")
            image = Image.open(f'background_image/background_{new_user_pred}.png')
            st.image(image, use_column_width="always")

            if new_user_pred == 0:
                st.markdown(f"<h3 style='text-align: center; color: #1C9B41;'>Your Class is PlaceholderClass-1</h3>",
                            unsafe_allow_html=True)
                st.write("Açıklama 0")
            elif new_user_pred == 1:
                st.markdown(f"<h3 style='text-align: center; color: #1C9B41;'>Your Class is PlaceholderClass-2</h3>",
                            unsafe_allow_html=True)
                st.write("Açıklama 1")
            elif new_user_pred == 2:
                st.markdown(f"<h3 style='text-align: center; color: #1C9B41;'>Your Class is PlaceholderClass-3</h3>",
                            unsafe_allow_html=True)
                st.write("Açıklama 2")
            elif new_user_pred == 3:
                st.markdown(f"<h3 style='text-align: center; color: #1C9B41;'>Your Class is PlaceholderClass-4</h3>",
                            unsafe_allow_html=True)
                st.write("Açıklama 3")
            elif new_user_pred == 4:
                st.markdown(f"<h3 style='text-align: center; color: #1C9B41;'>Your Class is PlaceholderClass-5</h3>",
                            unsafe_allow_html=True)
                st.write("Açıklama 4")
            elif new_user_pred == 5:
                st.markdown(f"<h3 style='text-align: center; color: #1C9B41;'>Your Class is PlaceholderClass-6</h3>",
                            unsafe_allow_html=True)
                st.write("Açıklama 5")
            elif new_user_pred == 6:
                st.markdown(f"<h3 style='text-align: center; color: #1C9B41;'>Your Class is PlaceholderClass-7</h3>",
                            unsafe_allow_html=True)
                st.write("Açıklama 6")
            elif new_user_pred == 7:
                st.markdown(f"<h3 style='text-align: center; color: #1C9B41;'>Your Class is PlaceholderClass-8</h3>",
                            unsafe_allow_html=True)
                st.write("Açıklama 7")
            elif new_user_pred == 8:
                st.markdown(f"<h3 style='text-align: center; color: #1C9B41;'>Your Class is PlaceholderClass-9</h3>",
                            unsafe_allow_html=True)
                st.write("Açıklama 8")
            else:
                st.markdown(f"<h3 style='text-align: center; color: #1C9B41;'>Your Class is PlaceholderClass-10</h3>",
                            unsafe_allow_html=True)
                st.write("Açıklama 9")

while not pred_button:
    image_main = Image.open(f'background_image/background_main.jpg')
    st.image(image_main, use_column_width="always")
