import streamlit as st
import data_handler
import model as model
import pandas as pd
import warnings
import numpy as np
import os
from pathlib import Path
import base64

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Anime Recommender System", layout="wide", initial_sidebar_state="auto")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

BASE_DIR=Path(__file__).resolve().parent.parent
preproc_url=BASE_DIR/"data"/"ohe_data.csv"
filtered_data_url=BASE_DIR/"data"/"filtered_dataset.csv"
full_data_url=BASE_DIR/"data"/"anime_dataset.csv"

#data_handler.full_data_updater(full_data_url, rating_filter=7.5)
#data_handler.preprocess_data(filtered_data_url)
preproc_data=model.data_loader(preproc_url)


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call it in script main
set_background("assets/background.jpg")

st.markdown("""
<style>
/* Remove vertical scroll */
        html, body, [class*="css"]  {
            overflow: hidden !important;
        }

        /* Optional: force app container to fit screen height */
        .main {
            height: 100vh;
        }
/* Whole app background */
.stApp {{
        background-image: url("assets/background.jpg");
        background-attachment: fixed;
        background-size: cover;
    }}
}
/* Override primary red color globally */
    div[data-baseweb="tag"] {
        background-color: #1E90FF !important;   /* Blue tags */
        color: white !important;
    }
div[data-baseweb="select"] > div {
        border-color: #1E90FF !important;       /* Blue dropdown border */
    }
    div[data-baseweb="select"] option:checked,
    div[data-baseweb="select"] div[aria-selected="true"] {
        background-color: rgba(30,144,255,0.3) !important; /* Blue highlight */
    }

    /* Fix st.button (Calculate Recommendations) */
    div.stButton > button {
        background-color: #1E90FF !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background-color: #187bcd !important;  /* Darker blue on hover */
        color: #fff !important;
    }

/* Cancel text highlight on expanders */
    button[aria-expanded="true"],
    button[aria-expanded="false"] {
        color: inherit !important;   /* use the normal text color instead of Streamlit red */
    }

/* Cancel hover effect */
    button[aria-expanded="true"]:hover,
    button[aria-expanded="false"]:hover {
        color: inherit !important;
    }

/* Cancel the red arrow/caret */
    button[aria-expanded="true"] svg,
    button[aria-expanded="false"] svg {
        fill: inherit !important;   /* use the inherited text color */
    }




/* Page padding */
.block-container {
    padding-top: 2rem;
}

/* Make expanders glassy */
[data-testid="stExpander"] {
    background: rgba(0,0,0,0.55);
    border-radius: 10px;
    padding: 5px;
    border: 1px solid rgba(255,255,255,0.1);
}
/* Multiselect tags */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #1E90FF !important;  /* DodgerBlue */
        color: white !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
    }
/* Close (X) button inside tags */
    .stMultiSelect [data-baseweb="tag"] svg {
        fill: white !important;
    }

/* Expander header text */
[data-testid="stExpander"] [data-testid="stExpanderHeader"] p {
    font-weight: bold;
    color: #ffdd57;  /* soft yellow */
}

/* Dropdown highlight color */
    .stMultiSelect [data-baseweb="select"] div[role="option"][aria-selected="true"] {
        background-color: rgba(30, 144, 255, 0.3) !important;
    }
/* Hover state for options */
    .stMultiSelect [data-baseweb="select"] div[role="option"]:hover {
        background-color: rgba(30, 144, 255, 0.2) !important;
    }

/* Multiselect box styling */
[data-baseweb="select"] {
    background: rgba(20,20,20,0.65);
    border-radius: 6px;
}

.stSelectbox label, .stNumberInput label {
    color: #ffffff !important;
    font-weight: 600;
    text-shadow:
        0px 0px 8px rgba(0,0,0,1),   /* stronger glow */
        0px 0px 12px rgba(0,0,0,1); /* wider spread */
    }

/* Table / dataframe styling */
[data-testid="stTable"], [data-testid="stDataFrame"] {
    background: rgba(0,0,0,0.75) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


def remove_selected_entries(data:pd.DataFrame, entries:list)->pd.DataFrame:
    '''Remove multiple entries from the DataFrame at once.
    Args:
        data (pd.DataFrame): DataFrame to modify.
        entries (list): List of anime names to remove.
    Returns:
        pd.DataFrame: Modified DataFrame.
    '''
    # Filter out "Empty entree" and None values
    valid_entries = [entry for entry in entries if entry and entry != 'Empty entree']

    if valid_entries:
        # Remove all selected entries in one operation using isin()
        data = data[~data['Name'].isin(valid_entries)]

    return data

def build_preference_vector(liked_animes, disliked_animes, liked_themes, disliked_themes, preproc_data)->pd.DataFrame:
    """
    Build a normalized preference vector from user-selected liked/disliked animes and themes.

    Args:
        liked_animes (list): List of anime names the user likes
        disliked_animes (list): List of anime names the user dislikes
        liked_themes (list): List of themes the user likes
        disliked_themes (list): List of themes the user dislikes
        preproc_data (pd.DataFrame): Preprocessed anime dataset (OHE features)

    Returns:
        pd.DataFrame: A single-row DataFrame representing the normalized user preference vector
    """
    # feature columns = one-hot encoded genres/themes
    feature_cols = preproc_data.drop(columns=['Name', 'Rating', 'Release year']).columns
    vector = pd.DataFrame([[0]*len(feature_cols)], columns=feature_cols)

    # 1. Add liked anime profiles (positive weight)
    for anime in liked_animes:
        if anime and anime != "Empty entree":
            anime_row = preproc_data.loc[preproc_data['Name'] == anime]
            if not anime_row.empty:
                vector.loc[0] += anime_row.drop(columns=['Name','Rating','Release year']).values[0]

    # 2. Subtract disliked anime profiles (negative weight)
    for anime in disliked_animes:
        if anime and anime != "Empty entree":
            anime_row = preproc_data.loc[preproc_data['Name'] == anime]
            if not anime_row.empty:
                vector.loc[0] -= anime_row.drop(columns=['Name','Rating','Release year']).values[0]

    # 3. Boost liked themes
    for theme in liked_themes:
        if theme and theme != "Empty entree" and theme in vector.columns:
            vector.at[0, theme] += 1

    # 4. Downweight disliked themes
    for theme in disliked_themes:
        if theme and theme != "Empty entree" and theme in vector.columns:
            vector.at[0, theme] -= 1
    return vector
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



st.markdown(hide_st_style, unsafe_allow_html=True)
st.title("Anime Recommender System")
anime_list = []
anime_list = anime_list + preproc_data['Name'].tolist()
col1, col2, col3 = st.columns([0.35,0.15,0.50],vertical_alignment="top")
themes_list=['Action',
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
with col1:
    st.subheader("💙 Select your preferences")
    with st.expander("⭐ Select animes you liked", expanded=False):
        liked_animes = st.multiselect("Choose...", anime_list, default=[],key="likedA")
    with st.expander("👎 Select animes you didn't like", expanded=False):
        disliked_animes = st.multiselect("Choose...", anime_list, default=[],key="dislikedA")
    with st.expander("🎭 Select themes you like", expanded=False):
        liked_themes = st.multiselect("🎭 Choose...", themes_list, default=[],key="likedT")
    with st.expander("🚫 Select themes you don't like", expanded=False):
        disliked_themes = st.multiselect("Choose...", themes_list, default=[],key="dislikedT")

with col2:
    years_list = ["Last 5 years", "Last 10 years", "Last 15 years", "Last 20 years", "Last 25 years", "Last 30 years"]
    st.subheader("🔎 Filters")
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
    if st.button("Calculate Recommendations"):
        preproc_data = model.data_loader(preproc_url)

        # Build user preference vector
        example_vector = make_clean_vector()

        # Add all liked animes’ vectors if user actually picked them
        user_vector = build_preference_vector(liked_animes, disliked_animes, liked_themes, disliked_themes, preproc_data)

        # Run predictor
        prediction = model.predictor(user_prefs=user_vector, data=preproc_data, years_back=years_back, n=40)

        # Collect ALL selected anime (both liked and disliked)
        all_selected = liked_animes + disliked_animes

        # Remove selected titles in one clean step
        prediction = remove_selected_entries(prediction, all_selected)

        # Get top N recommendations
        prediction = model.top_n(prediction, numb).reset_index(drop=True)
        display_df = prediction[["Name", "Rating", "Release year"]].reset_index(drop=True)
        display_df.index += 1  # so ranking starts at 1
        display_df['Name'] = display_df.index.map(
        lambda i: "🥇 " + display_df.loc[i, 'Name'] if i == 1 else
                "🥈 " + display_df.loc[i, 'Name'] if i == 2 else
                "🥉 " + display_df.loc[i, 'Name'] if i == 3 else
                display_df.loc[i, 'Name']
)
        display_df["Rating"] = display_df["Rating"].map("{:.2f}".format)
        display_df["Info"] = display_df["Rating"].astype(str) + " ⭐ | " + display_df["Release year"].astype(str)
        display_df.index.name = "Rank"
        display_df = display_df[[ "Name", "Info"]]
        with col3:
            st.subheader("✨ Recommended Anime")
            st.table(display_df)
