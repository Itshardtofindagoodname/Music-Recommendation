import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("songdata.csv")
df["text"] = df["text"].fillna("")

vectorizer = TfidfVectorizer(stop_words="english", max_features=15000, ngram_range=(1, 3), min_df=2, max_df=0.95)
lyrics_vectors = vectorizer.fit_transform(df["text"])

knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=11, p=2)
knn_model.fit(lyrics_vectors)

def recommend_songs(song_name, n=5):
    """
    Recommend similar songs based on lyrics
    """
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
    {"artist": "Michael Jackson", "song": "Man In The Mirror", "relevant": ["Mirror, Mirror"]},
    {"artist": "Ray Charles", "song": "Ellie my Love", "relevant": ["And You My Love"]},
]

def evaluate_model_basic(test_data, n=5):
    """
    Basic evaluation using precision
    """
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

def evaluate_model_enhanced(test_data, n=5):
    """
    Comprehensive evaluation with multiple metrics
    """
    metrics = {
        'precision': 0,
        'recall': 0,
        'mean_reciprocal_rank': 0,
        'hit_rate': 0,
        'diversity': 0,
        'average_similarity': 0
    }
    
    total_cases = len(test_data)
    total_relevant_found = 0
    total_possible_relevant = 0
    reciprocal_ranks = []
    
    for test_case in test_data:
        song_name = test_case["song"]
        relevant_songs = test_case["relevant"]

        recommended_songs = recommend_songs(song_name, n)
        if not recommended_songs:
            continue

        rec_indices = [
            df[(df['song'] == rec['song']) & (df['artist'] == rec['artist'])].index[0]
            for rec in recommended_songs
            if len(df[(df['song'] == rec['song']) & (df['artist'] == rec['artist'])]) > 0
        ]
        
        if len(rec_indices) > 1:
            rec_vectors = lyrics_vectors[rec_indices]
            similarities = cosine_similarity(rec_vectors)
            avg_sim = (similarities.sum() - len(rec_indices)) / (len(rec_indices) * (len(rec_indices) - 1))
            metrics['diversity'] += 1 - avg_sim
            
        found_relevant = False
        for i, rec in enumerate(recommended_songs):
            is_relevant = any(
                relevant_song.lower() in rec["song"].lower() or 
                relevant_song.lower() in rec["artist"].lower()
                for relevant_song in relevant_songs
            )
            if is_relevant:
                found_relevant = True
                total_relevant_found += 1
                reciprocal_ranks.append(1 / (i + 1))
                break
        
        if not found_relevant:
            reciprocal_ranks.append(0)
            
        total_possible_relevant += len(relevant_songs)

    metrics['precision'] = total_relevant_found / total_cases
    metrics['recall'] = total_relevant_found / total_possible_relevant if total_possible_relevant > 0 else 0
    metrics['mean_reciprocal_rank'] = np.mean(reciprocal_ranks)
    metrics['hit_rate'] = sum(1 for rr in reciprocal_ranks if rr > 0) / total_cases
    metrics['diversity'] = metrics['diversity'] / total_cases
    
    return metrics

def print_evaluation_report(metrics, basic_precision=None):
    """
    Print a formatted evaluation report
    """
    print("\n=== Song Recommendation System Evaluation ===")
    if basic_precision is not None:
        print(f"Basic Precision: {basic_precision:.3f}")
    print(f"Enhanced Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.3f}")
    print(f"Hit Rate: {metrics['hit_rate']:.3f}")
    print(f"Diversity: {metrics['diversity']:.3f}")

    if metrics['precision'] + metrics['recall'] > 0:
        f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        print(f"F1 Score: {f1:.3f}")

def demonstrate_recommendations(song_name, n=5):
    """
    Demonstrate recommendations for a specific song
    """
    print(f"\nGetting recommendations for: {song_name}")
    recommendations = recommend_songs(song_name, n)
    if recommendations:
        print("\nRecommended songs:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['song']} by {rec['artist']}")
    else:
        print("No recommendations found.")

basic_precision = evaluate_model_basic(test_data)
enhanced_metrics = evaluate_model_enhanced(test_data)

print(f"\nOriginal Model Precision: {basic_precision:.2f}")
print_evaluation_report(enhanced_metrics, basic_precision)

print("\n=== Example Recommendations ===")
demonstrate_recommendations("Ellie my Love")