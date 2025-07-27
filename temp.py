from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import Document
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load model
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

# Prepare document list from movie metadata
def prepare_movie_documents(movies_df):
    documents = []
    for i, row in movies_df.iterrows():
        meta_text = f"{row['title']} {', '.join(row['Genres'])} {row['overview']}"  # assuming 'overview' exists
        documents.append(Document(page_content=meta_text, metadata={"index": i}))
    return documents

movie_documents = prepare_movie_documents(movies)
db_movies = Chroma.from_documents(documents=movie_documents, embedding=embedding_model)

def recommend_movies_by_mood_semantic(
    mood, user_history_titles=None, query=None, top_n=10, alpha=0.4, beta=0.3, gamma=0.3
):
    genre_weights = mood_genre_mapping.get(mood, {})
    scores = []

    # Normalize genre_weights
    if genre_weights:
        total = sum(genre_weights.values())
        genre_weights = {g: w / total for g, w in genre_weights.items()}

    # User history similarity (TF-IDF)
    user_sim = np.zeros(len(movies))
    user_history_titles_lower = [t.lower() for t in user_history_titles] if user_history_titles else []

    if user_history_titles:
        user_history_indices = movies[movies['title'].str.lower().isin(user_history_titles_lower)].index.tolist()
        if user_history_indices:
            user_profile_vector = np.mean(tfidf_matrix[user_history_indices], axis=0).A1
            user_sim = cosine_similarity([user_profile_vector], tfidf_matrix).flatten()

    # Semantic vector similarity using embedding
    semantic_scores = np.zeros(len(movies))
    if query:
        results = db_movies.similarity_search(query, k=len(movies))
        for rank, result in enumerate(results):
            idx = result.metadata["index"]
            semantic_scores[idx] = 1.0 - rank / len(results)  # higher score for top results

    # Precompute mood scores
    mood_scores_raw = []
    for _, row in movies.iterrows():
        if row['title'].lower() in user_history_titles_lower:
            mood_scores_raw.append(0)
        else:
            genres = row['Genres']
            mood_score = sum([genre_weights.get(g, 0) for g in genres])
            mood_scores_raw.append(mood_score)
    max_mood_score = max(mood_scores_raw) or 1

    for idx, row in movies.iterrows():
        if row['title'].lower() in user_history_titles_lower:
            continue

        genres = row['Genres']
        raw_mood_score = sum([genre_weights.get(g, 0) for g in genres])
        mood_score = raw_mood_score / max_mood_score

        sim_score = user_sim[idx]
        rating_score = row['weighted_rating_norm']
        tmdb_score = row['tmdb_score']
        sem_score = semantic_scores[idx]

        # Include semantic similarity in sim_score (optional: tune this ratio)
        combined_sim_score = 0.7 * sim_score + 0.3 * sem_score

        final = alpha * mood_score + beta * combined_sim_score + gamma * rating_score

        scores.append((
            row['title'], final, genres,
            row['poster_path'], row['release_date'],
            row['Movie_id'], tmdb_score
        ))

    scores.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(
        scores[:top_n],
        columns=['title', 'score', 'Genres', 'poster_path', 'release_date', 'Movie_id', 'tmdb_score']
    )
