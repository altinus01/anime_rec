import streamlit as st
import data_handler
import model as model
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

preproc_url='data/ohe_data.csv'
filtered_data_url="data/filtered_anime_dataset.csv"
full_data_url="data/anime_dataset.csv"

#data_handler.full_data_updater(full_data_url, rating_filter=7.5)
#data_handler.preprocess_data(filtered_data_url)
preproc_data=model.data_loader(preproc_url)

def entry_remover(data:pd.DataFrame,entry:str)->pd.DataFrame:
    '''Remove a specific entry from the DataFrame.
    Args:
        data (pd.DataFrame): DataFrame to modify.
        entry (str): Entry to remove.
    Returns:
        pd.DataFrame: Modified DataFrame.
    '''
    # Remove a specific entry from the DataFrame
    # Return the modified DataFrame
    try:
        data=data[data['Name']!=entry]
    finally:
        return data




def make_clean_vector()->pd.DataFrame:
    themes_list_vec=['Action',
       'Adult Cast', 'Adventure', 'Anthropomorphic', 'Avant Garde',
       'Award Winning', 'Boys Love', 'CGDCT', 'Childcare', 'Combat Sports',
       'Comedy', 'Crossdressing', 'Delinquents', 'Detective', 'Drama', 'Ecchi',
       'Educational', 'Fantasy', 'Gag Humor', 'Girls Love', 'Gore', 'Gourmet',
       'Harem', 'High Stakes Game', 'Historical', 'Horror', 'Idols (Female)',
       'Idols (Male)', 'Isekai', 'Iyashikei', 'Kids', 'Love Polygon',
       'Love Status Quo', 'Magical Sex Shift', 'Mahou Shoujo', 'Martial Arts',
       'Mecha', 'Medical', 'Military', 'Music', 'Mystery', 'Mythology',
       'Organized Crime', 'Otaku Culture', 'Parody', 'Performing Arts', 'Pets',
       'Psychological', 'Racing', 'Reincarnation', 'Reverse Harem', 'Romance',
       'Samurai', 'School', 'Sci-Fi', 'Shounen', 'Showbiz', 'Slice of Life',
       'Space', 'Sports', 'Strategy Game', 'Super Power', 'Supernatural',
       'Survival', 'Suspense', 'Team Sports', 'Time Travel', 'Urban Fantasy',
       'Vampire', 'Video Game', 'Visual Arts', 'Workplace']
    zerolist=[float(0)]*len(themes_list_vec)
    example_vector=pd.DataFrame([zerolist], columns=themes_list_vec)
    return example_vector


st.set_page_config(page_title="Anime Recommender System", layout="wide", initial_sidebar_state="auto")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.title("Anime Recommender System")
st.write("Please select your preferences below:")
anime_list = ['Empty entree']
anime_list = anime_list + preproc_data['Name'].tolist()
col1, col2, col3, col4, col5, col6 = st.columns([0.15,0.15,0.15,0.15,0.15,0.25],vertical_alignment="top")
with col1:
    st.write("Select the animes you liked:")
    selected_anime_1 = st.selectbox("Select good anime 1", anime_list, placeholder="Select an anime")
    selected_anime_2 = st.selectbox("Select good anime 2", anime_list, placeholder="Select an anime")
    selected_anime_3 = st.selectbox("Select good anime 3", anime_list, placeholder="Select an anime")
    selected_anime_4 = st.selectbox("Select good anime 4", anime_list, placeholder="Select an anime")
    selected_anime_5 = st.selectbox("Select good anime 5", anime_list, placeholder="Select an anime")
with col2:
    st.write("Select the animes you didn't like:")
    selected_anime_6 = st.selectbox("Select bad anime 1", anime_list)
    selected_anime_7 = st.selectbox("Select bad anime 2", anime_list)
    selected_anime_8 = st.selectbox("Select bad anime 3", anime_list)
    selected_anime_9 = st.selectbox("Select bad anime 4", anime_list)
    selected_anime_10 = st.selectbox("Select bad anime 5", anime_list)

themes_list=['Empty entree','Action',
       'Adult Cast', 'Adventure', 'Anthropomorphic', 'Avant Garde',
       'Award Winning', 'Boys Love', 'CGDCT', 'Childcare', 'Combat Sports',
       'Comedy', 'Crossdressing', 'Delinquents', 'Detective', 'Drama', 'Ecchi',
       'Educational', 'Fantasy', 'Gag Humor', 'Girls Love', 'Gore', 'Gourmet',
       'Harem', 'High Stakes Game', 'Historical', 'Horror', 'Idols (Female)',
       'Idols (Male)', 'Isekai', 'Iyashikei', 'Kids', 'Love Polygon',
       'Love Status Quo', 'Magical Sex Shift', 'Mahou Shoujo', 'Martial Arts',
       'Mecha', 'Medical', 'Military', 'Music', 'Mystery', 'Mythology',
       'Organized Crime', 'Otaku Culture', 'Parody', 'Performing Arts', 'Pets',
       'Psychological', 'Racing', 'Reincarnation', 'Reverse Harem', 'Romance',
       'Samurai', 'School', 'Sci-Fi', 'Shounen', 'Showbiz', 'Slice of Life',
       'Space', 'Sports', 'Strategy Game', 'Super Power', 'Supernatural',
       'Survival', 'Suspense', 'Team Sports', 'Time Travel', 'Urban Fantasy',
       'Vampire', 'Video Game', 'Visual Arts', 'Workplace']
with col3:
    st.write("Select the themes you like:")
    selected_theme_1 = st.selectbox("Select good theme 1", themes_list)
    selected_theme_2 = st.selectbox("Select good theme 2", themes_list)
    selected_theme_3 = st.selectbox("Select good theme 3", themes_list)
    selected_theme_4 = st.selectbox("Select good theme 4", themes_list)
    selected_theme_5 = st.selectbox("Select good theme 5", themes_list)
with col4:
    st.write("Select the themes you don't like:")
    selected_theme_6 = st.selectbox("Select bad theme 1", themes_list)
    selected_theme_7 = st.selectbox("Select bad theme 2", themes_list)
    selected_theme_8 = st.selectbox("Select bad theme 3", themes_list)
    selected_theme_9 = st.selectbox("Select bad theme 4", themes_list)
    selected_theme_10 = st.selectbox("Select bad theme 5", themes_list)
with col5:
    years_list = ["Last 5 years", "Last 10 years", "Last 15 years", "Last 20 years", "Last 25 years", "Last 30 years"]
    time_box=st.selectbox("Time filter", options=years_list)
    if time_box=="Last 5 years":
        years_back=5
    elif time_box=="Last 10 years":
        years_back=10
    elif time_box=="Last 15 years":
        years_back=15
    elif time_box=="Last 20 years":
        years_back=20
    elif time_box=="Last 25 years":
        years_back=25
    else:
        years_back=30
    numb=st.selectbox("How many animes would you like?", options=[5, 10, 15, 20])
    if st.button("Calculate Reccomendations"):
        preproc_data=model.data_loader(preproc_url)
        # Filter out the selected animes

        example_vector=make_clean_vector()
        example_vector.loc[[0]]=example_vector.loc[[0]]+preproc_data.loc[[preproc_data['Name'] == selected_anime_1]].drop(columns=['Name', 'Rating', 'Release year']).values
        print(example_vector)
        print(preproc_data.loc[preproc_data['Name'] == selected_anime_1])
        prediction=model.predictor(user_prefs=example_vector, data=preproc_data, years_back=years_back, n=40)
        prediction=entry_remover(prediction,selected_anime_1)
        prediction=entry_remover(prediction,selected_anime_2)
        prediction=entry_remover(prediction,selected_anime_3)
        prediction=entry_remover(prediction,selected_anime_4)
        prediction=entry_remover(prediction,selected_anime_5)
        prediction=entry_remover(prediction,selected_anime_6)
        prediction=entry_remover(prediction,selected_anime_7)
        prediction=entry_remover(prediction,selected_anime_8)
        prediction=entry_remover(prediction,selected_anime_9)
        prediction=entry_remover(prediction,selected_anime_10)
        prediction=model.top_n(prediction, numb).reset_index(drop=True)
        print(prediction)
