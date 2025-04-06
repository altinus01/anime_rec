import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def data_loader(preproc_url:str)->pd.DataFrame:
    '''Load preprocessed data from a given URL.
    Args:
        preproc_url (str): URL to the preprocessed data.
    Returns:
        pd.DataFrame: Preprocessed data as a pandas DataFrame.
    '''
    # Load the preprocessed data
    # from the given URL
    # Check if the URL is valid and accessible
    # If not, print an error message
    # and return None
    try:
        data=pd.read_csv(preproc_url)
        print(u"\u2713 Preprocessed data imported!")
        return data
    except:
        print(u"\u2717 ERROR: Couldn't import preprocessed data! Check URL")


def time_filter(df:pd.DataFrame,years_back:int=10)->pd.DataFrame:
    '''Filter the DataFrame based on the release year.
    Args:
        df (pd.DataFrame): DataFrame to filter.
        years_back (int): Number of years to filter by.
    Returns:
        pd.DataFrame: Filtered DataFrame.
    '''
    # Filter the DataFrame based on the release year
    # Get the maximum release year from the DataFrame
    # Filter the DataFrame to include only rows with release year
    # greater than the maximum release year minus years_back
    # Return the filtered DataFrame
    top_time=df['Release year'].max()
    print(u"\u2713 "+f"Applied {years_back} year filter to animes!")
    return df[df['Release year']>top_time-years_back]



def cosine_similarity_entry(vector:pd.DataFrame,matrix:pd.DataFrame)->pd.DataFrame:
    '''Calculate cosine similarity between a vector and a matrix.
    Args:
        vector (pd.DataFrame): DataFrame containing the vector.
        matrix (pd.DataFrame): DataFrame containing the matrix.
    Returns:
        pd.DataFrame: DataFrame with cosine similarity values.
    '''
    # Calculate cosine similarity between a vector and a matrix
    elem=0
    matrix['Similarity']=0
    matrix['Similarity']=matrix['Similarity'].astype(float)
    vector["Similarity"]=0
    vector["Similarity"]=vector["Similarity"].astype(float)
    while elem < matrix.shape[0]:
        matrix.loc[elem, "Similarity"]=cosine_similarity(vector, matrix.loc[[elem]])
        elem+=1
    print(u"\u2713 Calculated most similar animes!")
    return matrix



def top_n(matrix:pd.DataFrame,n:int)->pd.DataFrame:
    '''Get the top n entries from a DataFrame.
    Args:
        matrix (pd.DataFrame): DataFrame to filter.
        n (int): Number of top entries to return.
    Returns:
        pd.DataFrame: DataFrame with top n entries.
    '''
    print(u"\u2713 "+f"Cut filtered animes to top {n}!")
    return matrix.iloc(axis=0)[0:n]




def predictor(user_prefs:pd.DataFrame, data:pd.DataFrame, years_back:int=15, n:int=15)->pd.DataFrame:
    '''Predict recommendations based on user preferences.
    Args:
        user_prefs (pd.DataFrame): User preferences DataFrame.
        data (pd.DataFrame): DataFrame containing the dataset.
        years_back (int): Number of years to filter by.
        n (int): Number of top recommendations to return.
    Returns:
        pd.DataFrame: DataFrame with top n recommendations.
    '''
    num_data=data.drop(columns=['Name', 'Rating', 'Release year'])
    sim_matrix=cosine_similarity_entry(user_prefs, num_data)
    sim_matrix.sort_values('Similarity',ascending=False,inplace=True)
    labeled_matrix=sim_matrix.join(data[['Name','Rating','Release year']],how='inner')
    labeled_matrix=time_filter(labeled_matrix, years_back)
    labeled_matrix=top_n(labeled_matrix, n)
    labeled_matrix=labeled_matrix.sort_values('Rating',ascending=False)
    labeled_matrix.drop(columns=['Action',
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
       'Vampire', 'Video Game', 'Visual Arts', 'Workplace'], inplace=True)
    labeled_matrix['Score']=labeled_matrix['Rating']*labeled_matrix['Similarity']
    labeled_matrix=labeled_matrix.sort_values('Score',ascending=False)
    final_table=labeled_matrix[['Name']].reset_index(drop=True)
    print(u"\u2713 Predictions made!")
    return final_table
