import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from main import app, recommend_songs, recommend_artists

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_data(monkeypatch):
    mock_df = pd.DataFrame({
        'artist': ['Artist1', 'Artist2', 'Artist3', 'Artist1', 'Artist2'],
        'song': ['Song1', 'Song2', 'Song3', 'Song4', 'Song5'],
        'text': ['lyrics one', 'lyrics two', 'lyrics three', 'lyrics four', 'lyrics five'],
        'link': ['link1', 'link2', 'link3', 'link4', 'link5']
    })
    vectorizer = TfidfVectorizer(stop_words='english')
    lyrics_vectors = vectorizer.fit_transform(mock_df['text'])
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(lyrics_vectors)
    import main
    monkeypatch.setattr(main, 'df', mock_df)
    monkeypatch.setattr(main, 'lyrics_vectors', lyrics_vectors)
    monkeypatch.setattr(main, 'vectorizer', vectorizer)
    monkeypatch.setattr(main, 'knn_model', knn_model)
    
    return mock_df

def test_recommend_songs_found():
    result = recommend_songs("Song1", n=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(item, dict) for item in result)
    assert all('artist' in item and 'song' in item for item in result)

def test_recommend_songs_not_found():
    result = recommend_songs("NonexistentSong")
    assert isinstance(result, dict)
    assert "error" in result
    assert result["error"] == "Song not found"

def test_recommend_artists_found():
    result = recommend_artists("Artist1", n=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(item, dict) for item in result)
    assert all('artist' in item and 'song' in item for item in result)

def test_recommend_artists_not_found():
    result = recommend_artists("NonexistentArtist")
    assert isinstance(result, dict)
    assert "error" in result
    assert result["error"] == "Artist not found"

def test_get_song_recommendations_endpoint():
    response = client.get("/recommend/song/?song_name=Song1&n=2")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

def test_get_song_recommendations_endpoint_not_found():
    response = client.get("/recommend/song/?song_name=NonexistentSong")
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"] == "Song not found"

def test_get_artist_recommendations_endpoint():
    response = client.get("/recommend/artist/?artist_name=Artist1&n=2")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

def test_get_artist_recommendations_endpoint_not_found():
    response = client.get("/recommend/artist/?artist_name=NonexistentArtist")
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"] == "Artist not found"

def test_recommend_songs_invalid_n():
    result = recommend_songs("Song1", n=0)
    assert isinstance(result, list)
    assert len(result) == 0

def test_recommend_artists_large_n():
    result = recommend_artists("Artist1", n=2)
    assert isinstance(result, list)
    assert len(result) <= 5

def test_recommend_songs_case_insensitive():
    result1 = recommend_songs("SONG1", n=2)
    result2 = recommend_songs("song1", n=2)
    assert result1 == result2

def test_vectorizer_output(setup_test_data):
    from main import lyrics_vectors
    assert lyrics_vectors.shape[0] == len(setup_test_data)
    assert not np.isnan(lyrics_vectors.toarray()).any()

def test_knn_model_validity(setup_test_data):
    from main import knn_model, lyrics_vectors
    distances, indices = knn_model.kneighbors(lyrics_vectors[0:1], n_neighbors=2)
    assert len(indices[0]) == 2
    assert len(distances[0]) == 2

def test_response_time():
    import time
    start_time = time.time()
    recommend_songs("Song1", n=2)
    end_time = time.time()
    assert end_time - start_time < 0.1