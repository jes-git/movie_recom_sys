import streamlit as st
import pickle
import pandas as pd

# Load movie list and similarity matrix
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    
    recommended_movies = [movies.iloc[i[0]].title for i in movie_list]
    return recommended_movies

# Streamlit App UI
st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
    "Select a movie from the list below to get recommendations:",
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    st.subheader("Top 10 Recommendations:")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")

