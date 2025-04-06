import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

preproc_url='../data/ohe_data.csv'
def data_loader(preproc_url:str):
    try:
        data=pd.read_csv(preproc_url)
        print(u"\u2713 Preprocessed data imported!")
        return data
    except:
        print(u"\u2717 ERROR: Couldn't import preprocessed data! Check URL")

def predictor(user_prefs:pd.DataFrame, data:pd.DataFrame, years_back:int=15, n:int=15)->pd.DataFrame:
    def time_filter(df,years_back):
        top_time=df['Release year'].max()
        return df[df['Release year']>top_time-years_back]

    def cosine_similarity_entry(vector,matrix):
            elem=0
            matrix['Similarity']=0
            while elem < matrix.shape[0]:
                matrix['Similarity'][elem]=cosine_similarity(vector, matrix.loc[[elem]])
                elem+=1
            return matrix
    return data


preproc_data=data_loader(preproc_url)
prediction=predictor(user_prefs=preproc_data.loc[[0]], data=preproc_data, years_back=15, n=15)


    vectors=ohe_data.drop(columns=['Name', 'Image source', 'Rating', 'Release year'])
    vectors.shape[0]


    mat=cosine_similarity_entry(vectors.loc[[4]],vectors)
    mat.sort_values('Similarity',ascending=False,inplace=True)

    def top_n_similar(matrix,n):
        return matrix.iloc(axis=0)[0:n]

    top5=top_n_similar(mat,5)
    top5

    new=top5.join(ohe_data[['Name','Image source','Rating','Release year']]).sort_values('Rating',ascending=False)
    new

    new.drop(columns=['Action',
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
    new['Score']=new['Rating']*new['Similarity']

    new.sort_values('Score',ascending=False)

    final_list=new[['Name','Image source']].reset_index(drop=True)
    final_list
