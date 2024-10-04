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

تطوير 
!pip install pandas numpy scikit-learn imdbpy
import pandas as pd
import numpy as np
from imdb import IMDb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
# Initialize IMDb object
ia = IMDb()
# List of IMDb movie IDs
movie_ids = ['0111161', '0068646', '0071562', '0468569', '0050083']  # Add more movie IDs as needed
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
print("Movies DataFrame:")
print(movies_df)
# Create a CountVectorizer based on genres for similarity computation
count_vectorizer = CountVectorizer(stop_words='english')
genres_matrix = count_vectorizer.fit_transform(movies_df['genres'])
# Compute cosine similarity based on genres
cosine_sim = cosine_similarity(genres_matrix, genres_matrix)
# Simulated user data (in a real application, you would capture this dynamically)
user_data = {
    'user_id': 1,
    'gender': 'female',
    'age_group': '18-24',  # Options could be 'under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'
    'watch_data': {
        'The Shawshank Redemption': 100,  # Watch duration in minutes
        'The Godfather': 30,
        'The Dark Knight': 150,
        '12 Angry Men': 45,
        'Schindler\'s List': 75,
        'Pulp Fiction': 120
    }
}
# Simulated user data (in a real application, you would capture this dynamically)
user_data = {
    'user_id': 1,
    'gender': 'female',
    'age_group': '18-24',  # Options could be 'under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'
    'watch_data': {
        'The Shawshank Redemption': 100,  # Watch duration in minutes
        'The Godfather': 30,
        'The Dark Knight': 150,
        '12 Angry Men': 45,
        'Schindler\'s List': 75,
        'Pulp Fiction': 120
    }
}
# Function to recommend movies based on watch duration and demographics
def recommend_movies(user_data, movies_df, cosine_sim):
    recommendations = []
    
    # Total watch time and rating count
    total_watch_time = sum(user_data['watch_data'].values())
    
    for movie_title, watch_time in user_data['watch_data'].items():
        # Check if the movie exists in the dataframe
        if movie_title in movies_df['title'].values:
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

            for i in movie_indices:
                recommended_title = movies_df.iloc[i]['title']
                recommended_rating = movies_df.iloc[i]['rating']
                recommendations.append({
                    'title': recommended_title,
                    'rating': recommended_rating,
                    'similarity_score': sim_scores[movie_indices.index(i)][1],
                    'watch_time_ratio': watch_time / 120  # Assuming the average movie length is 120 minutes
                })

    # Filter recommendations based on watch time ratio
    filtered_recommendations = [rec for rec in recommendations if rec['watch_time_ratio'] >= 0.5]  # Keep recommendations with >= 50% watch time
    sorted_recommendations = sorted(filtered_recommendations, key=lambda x: x['similarity_score'], reverse=True)
    
    return pd.DataFrame(sorted_recommendations)
    # Get recommendations for the user
recommended_movies_df = recommend_movies(user_data, movies_df, cosine_sim)
print("\nRecommended Movies:")
print(recommended_movies_df[['title', 'rating', 'similarity_score']])

النتائج
Recommended Movies:
                      title  rating  similarity_score
0             The Godfather     9.2          0.707107
1     The Godfather Part II     9.0          0.707107
2              12 Angry Men     9.0          0.707107
3             The Godfather     9.2          0.707107
4     The Godfather Part II     9.0          0.707107
5              12 Angry Men     9.0          0.707107
6           The Dark Knight     9.0          0.500000
7  The Shawshank Redemption     9.3          0.500000

تطوير
!pip install pandas numpy scikit-learn imdbpy textblob tensorflow
# Dummy comments for movies (for sentiment analysis)
comments = {
    'The Shawshank Redemption': "Incredible story and amazing performances!",
    'The Godfather': "A masterpiece of cinema, truly timeless.",
    'The Dark Knight': "Intense and thrilling, Heath Ledger was phenomenal.",
    '12 Angry Men': "Thought-provoking and brilliantly written.",
    'Schindler\'s List': "A haunting portrayal of history.",
    'Pulp Fiction': "Unique storytelling and iconic characters."
}

# Add sentiment analysis to movies_df
def safe_analyze_sentiment(movie_title):
    if movie_title in comments:
        return analyze_sentiment(comments[movie_title])
    else:
        return 0  # or any default sentiment value you prefer

movies_df['sentiment'] = [safe_analyze_sentiment(movie) for movie in movies_df['title']]
# Function to recommend movies based on watch duration and demographics
def recommend_movies(user_data, movies_df, cosine_sim):
    recommendations = []
    
    # Total watch time and rating count
    total_watch_time = sum(user_data['watch_data'].values())
    
    for movie_title, watch_time in user_data['watch_data'].items():
        # Check if the movie is in movies_df
        if movie_title in movies_df['title'].values:
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
            
            for i in movie_indices:
                recommended_title = movies_df.iloc[i]['title']
                recommended_rating = movies_df.iloc[i]['rating']
                recommended_sentiment = movies_df.iloc[i]['sentiment']
                recommendations.append({
                    'title': recommended_title,
                    'rating': recommended_rating,
                    'similarity_score': sim_scores[movie_indices.index(i)][1],
                    'watch_time_ratio': watch_time / 120,  # Assuming the average movie length is 120 minutes
                    'sentiment': recommended_sentiment
                })

    # Filter recommendations based on watch time ratio
    filtered_recommendations = [rec for rec in recommendations if rec['watch_time_ratio'] >= 0.5]  # Keep recommendations with >= 50% watch time
    sorted_recommendations = sorted(filtered_recommendations, key=lambda x: x['similarity_score'], reverse=True)
    
    return pd.DataFrame(sorted_recommendations)

# Get recommendations for the user
recommended_movies_df = recommend_movies(user_data, movies_df, cosine_sim)
print("\nRecommended Movies:")
print(recommended_movies_df[['title', 'rating', 'similarity_score', 'sentiment']])

النتائج 
Recommended Movies:
                      title  rating  similarity_score  sentiment
0             The Godfather     9.2          0.707107   0.000000
1     The Godfather Part II     9.0          0.707107   0.000000
2              12 Angry Men     9.0          0.707107   0.650000
3             The Godfather     9.2          0.707107   0.000000
4     The Godfather Part II     9.0          0.707107   0.000000
5              12 Angry Men     9.0          0.707107   0.650000
6           The Dark Knight     9.0          0.500000   0.316667
7  The Shawshank Redemption     9.3          0.500000   0.825000


print("User watch data titles:")
print(user_data['watch_data'].keys())

print("Movies in movies_df:")
print(movies_df['title'].tolist())

النتائج
User watch data titles:
dict_keys(['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', '12 Angry Men', "Schindler's List", 'Pulp Fiction'])
Movies in movies_df:
['The Shawshank Redemption', 'The Godfather', 'The Godfather Part II', 'The Dark Knight', '12 Angry Men']


    
 
