import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import ast

def full_data_updater(data_url:str, rating_filter:float=7.5):
    try:
        data=pd.read_csv(data_url)
        print(u"\u2713 Full data imported!")
    except:
        print(u"\u2717 ERROR: Couldn't import full data! Check URL")
    data.drop(columns=['Image source','Synopsis','Rated by(number of users)', 'Rank', 'Popularity',
       'Number of episodes', 'Duration', 'Status', 'Aired', 'Producers',
       'Studio(s)','Demographic','Anime recomendations(by users and autorec)','Theme','English Name'],inplace=True, errors='ignore')
    data.dropna(inplace=True)
    boolean_mask=data['Rating']>rating_filter
    data=data[boolean_mask]
    data.drop_duplicates(keep='first',inplace=True)
    data.sort_values(by='Rating',ascending=False,inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['Release year']=data['Release time'].apply(lambda x: int(x.split()[1]))
    data.drop(columns=['Release time'],inplace=True)
    data.to_csv('data/filtered_anime_dataset.csv',index=False)
    print(u"\u2713 Data cleaned and filtered!")

def preprocess_data(data_url:str):
    try:
        data=pd.read_csv(data_url)
        print(u"\u2713 Filtered data imported!")
    except:
        print(u"\u2717 ERROR: Couldn't import filtered data! Check URL")
    data['Genres']=data['Genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x,str) else x)
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(mlb.fit_transform(data['Genres']),columns=mlb.classes_)
    filtered_data_encoded=pd.concat([data,genre_encoded],axis=1)
    filtered_data_encoded.drop('Genres',axis=1,inplace=True)
    filtered_data_encoded.to_csv('data/ohe_data.csv',index=False)
    print(u"\u2713 Data preprocessed!")
