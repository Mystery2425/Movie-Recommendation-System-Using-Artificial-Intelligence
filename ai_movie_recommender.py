# Install required libraries
!pip install pandas numpy scikit-learn
import pandas as pd
import numpy as np
from imdb import IMDb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Initialize IMDb object
ia = IMDb()

# List of IMDb movie IDs you want to use (you can expand this list)
movie_ids = ['0111161', '0068646', '0071562', '0468569', '0050083']

# Function to get movie data from IMDb
def get_movie_data(movie_ids):
    movies_data = []
    
    for movie_id in movie_ids:
        movie = ia.get_movie(movie_id)
        title = movie['title']
        genres = ' '.join(movie.get('genres', []))
        rating = movie.get('rating', 0)
        
        movies_data.append({
            'id': movie_id,
            'title': title,
            'genres': genres,
            'rating': rating
        })
    
    return pd.DataFrame(movies_data)

# Fetch movie data
movies_df = get_movie_data(movie_ids)
print(movies_df)

# Create a CountVectorizer based on genres for similarity computation
count_vectorizer = CountVectorizer(stop_words='english')
genres_matrix = count_vectorizer.fit_transform(movies_df['genres'])

# Compute cosine similarity based on genres
cosine_sim = cosine_similarity(genres_matrix, genres_matrix)

# Function to get movie recommendations
def recommend_movies(movie_title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = movies_df.index[movies_df['title'] == movie_title].tolist()[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return movies_df.iloc[movie_indices][['title', 'rating']]

# Example usage
print("\nRecommended movies for 'The Shawshank Redemption':")
recommendations = recommend_movies('The Shawshank Redemption')
print(recommendations)

النتائج
   id                     title                       genres  rating
0  0111161  The Shawshank Redemption                        Drama     9.3
1  0068646             The Godfather                  Crime Drama     9.2
2  0071562     The Godfather Part II                  Crime Drama     9.0
3  0468569           The Dark Knight  Action Crime Drama Thriller     9.0
4  0050083              12 Angry Men                  Crime Drama     9.0



Recommended movies for 'The Shawshank Redemption':
                   title  rating
1          The Godfather     9.2
2  The Godfather Part II     9.0
4           12 Angry Men     9.0
3        The Dark Knight     9.0


 
