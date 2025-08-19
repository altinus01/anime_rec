import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def data_loader(preproc_url)->pd.DataFrame:
    '''Load preprocessed data from a given URL.
    Args:
        preproc_url (Path): URL to the preprocessed data.
    Returns:
        pd.DataFrame: Preprocessed data as a pandas DataFrame or None if error is found.
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
    except Exception as e:
        raise FileNotFoundError(f"❌ Couldn't import preprocessed data ({preproc_url}). Error: {e}")


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



def cosine_similarity_entry(vector: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Calculate cosine similarity between a vector and a matrix."""
    # Compute similarity between user vector and all anime rows
    sims = cosine_similarity(vector, matrix)[0]  # returns array of shape (n_samples,)
    matrix = matrix.copy()
    matrix['Similarity'] = sims
    print(u"\u2713 Calculated most similar animes (vectorized)!")
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



def predictor(user_prefs: pd.DataFrame, data: pd.DataFrame, years_back: int = 15, n: int = 15) -> pd.DataFrame:
    """Predict recommendations based on user preferences."""

    features = data.drop(columns=['Name', 'Rating', 'Release year'])
    sim_matrix = cosine_similarity_entry(user_prefs, features)

    # Attach metadata
    sim_matrix = sim_matrix.join(data[['Name', 'Rating', 'Release year']])

    # Apply filters
    sim_matrix = time_filter(sim_matrix, years_back)

    # Score = Similarity × Rating
    sim_matrix['Score'] = sim_matrix['Similarity'] * sim_matrix['Rating']

    # Get top n results
    sim_matrix = sim_matrix.sort_values('Score', ascending=False).head(n)
    final_table = sim_matrix[['Name','Rating','Similarity','Score','Release year']].reset_index(drop=True)

    print(u"\u2713 Predictions made!")
    return final_table
