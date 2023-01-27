
# Resume Classification (IT Monsters)

## Demo

[![IT-Monsters.png](https://i.postimg.cc/tggQQKfr/IT-Monsters.png)](https://postimg.cc/xXBFL49H)

https://cbung-resume-classification-streamlit-app-k41ep9.streamlit.app/

## Dataset: 
I used the data from individuals working in data-related professions at big tech companies such as Google and Amazon through web scraping.

#### Variables:
- Last Four Companies they worked in
    - organization_title_x: Title they work/worked as
    - organization_start_x: Organization start date
    - organization_end_x: Organization end date

- Last Three Facilities they educated in
    - education_degree_x: Degree they get
    - education_fos_x: What they studied
    - education_start_x: Education start date
    - education_end_x: Education end date

- Three Languages they know
    - language_x: Name of the languages 1 through 3
    - language_proficiency_x: Proficiency of the languages 1 through 3

- languages: All languages they know
- skills: All skills they have


## Operations:

- Web Scraping
- Creating the Dataset
- EDA
- Cleaning the Dataset
- Feature Extraction
- Creating a Target Variable (K-Means)
- Cluster Analysis (Naming and Describing the Clusters)
- Building a Machine Learning Model
- Prediction
- Deploying the App on Streamlit
