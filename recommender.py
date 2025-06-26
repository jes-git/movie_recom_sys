import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data
movies = pd.read_csv("movies.csv")

# Preprocess genres
movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Reset index of movies DataFrame
movies = movies.reset_index()

# Function to get recommendations
def recommend(title, num_recommendations=10):
    # Convert title to lowercase for case-insensitive match
    title = title.lower()
    # Find the index of the movie that matches the title
    indices = movies[movies['title'].str.lower() == title].index

    if len(indices) == 0:
        return ["Movie not found in database."]

    idx = indices[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]

    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

