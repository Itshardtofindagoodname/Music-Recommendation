import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

df = pd.read_csv('songdata.csv')
df['text'] = df['text'].fillna('')

vectorizer = TfidfVectorizer(stop_words='english') #Tf-IDF vectorizer -> token frequency and inverse document frequency
lyrics_vectors = vectorizer.fit_transform(df['text'])

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(lyrics_vectors)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def recommend_songs(song_name, n=5):
    song_idx = df[df['song'].str.contains(song_name, case=False, na=False)].index
    if len(song_idx) == 0:
        return {"error": "Song not found"}
    
    song_vector = lyrics_vectors[song_idx[0]]

    distances, indices = knn_model.kneighbors(song_vector, n_neighbors=n+1)
    indices = indices.flatten()[1:]
    
    recommendations = df.iloc[indices][['artist', 'song']].to_dict(orient='records')
    return recommendations

def recommend_artists(artist_name, n=5):
    artist_songs = df[df['artist'].str.contains(artist_name, case=False, na=False)]
    if artist_songs.empty:
        return {"error": "Artist not found"}

    artist_vector = np.mean(lyrics_vectors[artist_songs.index].toarray(), axis=0)
    
    distances, indices = knn_model.kneighbors([artist_vector], n_neighbors=n)
    indices = indices.flatten()
    
    recommendations = df.iloc[indices][['artist', 'song']].to_dict(orient='records')
    return recommendations

@app.get("/recommend/song/")
def get_song_recommendations(song_name: str, n: int = 5):
    return recommend_songs(song_name, n)

@app.get("/recommend/artist/")
def get_artist_recommendations(artist_name: str, n: int = 5):
    return recommend_artists(artist_name, n)