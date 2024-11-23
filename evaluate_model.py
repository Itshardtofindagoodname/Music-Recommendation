import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("songdata.csv")
df["text"] = df["text"].fillna("")

vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
lyrics_vectors = vectorizer.fit_transform(df["text"])

knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=11)
knn_model.fit(lyrics_vectors)

def recommend_songs(song_name, n=5):
    song_idx = df[df["song"].str.contains(song_name, case=False, na=False)].index
    if len(song_idx) == 0:
        return []

    song_vector = lyrics_vectors[song_idx[0]]

    distances, indices = knn_model.kneighbors(song_vector, n_neighbors=n+1)
    indices = indices.flatten()[1:]

    recommendations = df.iloc[indices][["song", "artist"]].to_dict(orient="records")
    return recommendations

test_data = [
    {"artist": "The Beatles", "song": "All I've Got To Do", "relevant": ["I'll Come Running"]},
    {"artist": "Guns N' Roses", "song": "Shadow Of Your Love", "relevant": ["Have You Seen Your Mother Baby", "Standing In The Shadow"]},
    {"artist": "ABBA", "song": "Ahe's My Kind Of Girl", "relevant": ["The Kind of Girl I Could Love", "What Kind of Girl"]},
    {"artist": "Michael Jackson", "song": "Man In The Mirror", "relevant": ["Human"]},
]

def evaluate_model(test_data, n=5):
    correct_recommendations = 0
    total_recommendations = len(test_data)

    for test_case in test_data:
        song_name = test_case["song"]
        relevant = test_case["relevant"]

        recommended_songs = recommend_songs(song_name, n)

        for rec in recommended_songs:
            if any(
                relevant_song.lower() in rec["song"].lower() or relevant_song.lower() in rec["artist"].lower()
                for relevant_song in relevant
            ):
                correct_recommendations += 1
                break

    precision = correct_recommendations / total_recommendations
    return precision

precision = evaluate_model(test_data, n=5)
print(f"Model Precision: {precision:.2f}")
