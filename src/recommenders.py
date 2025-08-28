"""
Movie Recommendation System Implementations
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class BaseRecommender:
    """Base class for all recommenders"""
    
    def __init__(self, name="Base Recommender"):
        self.name = name
        self.is_fitted = False
        
    def fit(self, ratings_df, movies_df=None, users_df=None):
        """Fit the recommender model"""
        raise NotImplementedError
        
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations for a user"""
        raise NotImplementedError
        
    def predict_rating(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        raise NotImplementedError

class PopularityRecommender(BaseRecommender):
    """Popularity-based recommender - recommends most popular movies"""
    
    def __init__(self):
        super().__init__("Popularity-based Recommender")
        
    def fit(self, ratings_df, movies_df=None, users_df=None):
        """Fit by calculating movie popularity scores"""
        # Calculate popularity score (weighted rating)
        movie_stats = ratings_df.groupby('movie_id').agg({
            'rating': ['count', 'mean']
        }).round(2)
        movie_stats.columns = ['num_ratings', 'avg_rating']
        movie_stats = movie_stats.reset_index()
        
        # Calculate weighted rating (IMDB formula)
        # WR = (v/(v+m)) * R + (m/(v+m)) * C
        # v = number of votes, m = minimum votes, R = average rating, C = mean vote across all movies
        
        m = movie_stats['num_ratings'].quantile(0.7)  # Minimum votes required
        C = ratings_df['rating'].mean()  # Mean rating across all movies
        
        def weighted_rating(row):
            v = row['num_ratings']
            R = row['avg_rating']
            return (v/(v+m)) * R + (m/(v+m)) * C
            
        movie_stats['popularity_score'] = movie_stats.apply(weighted_rating, axis=1)
        
        # Sort by popularity score
        self.popular_movies = movie_stats.sort_values('popularity_score', ascending=False)
        
        if movies_df is not None:
            self.popular_movies = self.popular_movies.merge(
                movies_df[['movie_id', 'title']], 
                on='movie_id'
            )
            
        self.is_fitted = True
        return self
        
    def recommend(self, user_id, n_recommendations=10):
        """Recommend top popular movies"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
            
        recommendations = self.popular_movies.head(n_recommendations)[['movie_id', 'title', 'popularity_score']]
        return recommendations.to_dict('records')
        
    def predict_rating(self, user_id, movie_id):
        """Predict rating as the movie's average rating"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        movie_data = self.popular_movies[self.popular_movies['movie_id'] == movie_id]
        if len(movie_data) > 0:
            return movie_data['avg_rating'].iloc[0]
        else:
            return self.popular_movies['avg_rating'].mean()

class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using movie genres"""
    
    def __init__(self):
        super().__init__("Content-based Recommender")
        
    def fit(self, ratings_df, movies_df, users_df=None):
        """Fit by creating movie content similarity matrix"""
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Create genre features
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                     'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        # Create content matrix (genres)
        self.content_matrix = movies_df[['movie_id'] + genre_cols].set_index('movie_id')
        
        # Calculate cosine similarity between movies
        self.movie_similarity = cosine_similarity(self.content_matrix)
        self.movie_similarity_df = pd.DataFrame(
            self.movie_similarity,
            index=self.content_matrix.index,
            columns=self.content_matrix.index
        )
        
        # Create user profiles based on rated movies
        self.user_profiles = self._create_user_profiles()
        
        self.is_fitted = True
        return self
        
    def _create_user_profiles(self):
        """Create user profiles based on genres of movies they rated highly"""
        user_profiles = {}
        
        for user_id in self.ratings_df['user_id'].unique():
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
            
            # Get movies rated 4 or 5 (liked movies)
            liked_movies = user_ratings[user_ratings['rating'] >= 4]['movie_id'].tolist()
            
            if liked_movies:
                # Get genre preferences
                liked_movie_genres = self.content_matrix.loc[
                    self.content_matrix.index.isin(liked_movies)
                ]
                
                # Create weighted profile (higher ratings = higher weight)
                profile = np.zeros(len(self.content_matrix.columns))
                
                for _, movie_rating in user_ratings.iterrows():
                    if movie_rating['movie_id'] in self.content_matrix.index:
                        movie_genres = self.content_matrix.loc[movie_rating['movie_id']].values
                        weight = movie_rating['rating'] / 5.0  # Normalize rating
                        profile += movie_genres * weight
                        
                # Normalize profile
                if np.sum(profile) > 0:
                    profile = profile / np.sum(profile)
                    
                user_profiles[user_id] = profile
            else:
                # Default profile for users with no high ratings
                user_profiles[user_id] = np.ones(len(self.content_matrix.columns)) / len(self.content_matrix.columns)
                
        return user_profiles
        
    def recommend(self, user_id, n_recommendations=10):
        """Recommend movies similar to user's preferences"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
            
        if user_id not in self.user_profiles:
            # New user - recommend popular movies
            popular_movies = self.ratings_df.groupby('movie_id')['rating'].mean().sort_values(ascending=False)
            recommendations = popular_movies.head(n_recommendations).index.tolist()
        else:
            # Get user profile
            user_profile = self.user_profiles[user_id]
            
            # Calculate similarity between user profile and all movies
            movie_scores = {}
            for movie_id in self.content_matrix.index:
                movie_genres = self.content_matrix.loc[movie_id].values
                similarity = np.dot(user_profile, movie_genres)
                movie_scores[movie_id] = similarity
                
            # Remove movies already rated by user
            user_rated_movies = self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id'].tolist()
            for movie_id in user_rated_movies:
                movie_scores.pop(movie_id, None)
                
            # Sort by similarity score
            recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = [movie_id for movie_id, score in recommendations[:n_recommendations]]
            
        # Add movie titles
        result = []
        for movie_id in recommendations:
            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
            if len(movie_info) > 0:
                result.append({
                    'movie_id': movie_id,
                    'title': movie_info['title'].iloc[0],
                    'similarity_score': movie_scores.get(movie_id, 0)
                })
                
        return result
        
    def predict_rating(self, user_id, movie_id):
        """Predict rating based on content similarity"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Get user's average rating
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        if len(user_ratings) == 0:
            return self.ratings_df['rating'].mean()
            
        user_avg = user_ratings['rating'].mean()
        
        # Find similar movies that user has rated
        if movie_id in self.movie_similarity_df.index:
            similar_movies = self.movie_similarity_df[movie_id].sort_values(ascending=False)
            
            # Get ratings for similar movies
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_movie_id, similarity in similar_movies.items():
                if similar_movie_id != movie_id:
                    user_movie_rating = user_ratings[user_ratings['movie_id'] == similar_movie_id]
                    if len(user_movie_rating) > 0:
                        rating = user_movie_rating['rating'].iloc[0]
                        weighted_sum += similarity * rating
                        similarity_sum += abs(similarity)
                        
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                return max(1, min(5, predicted_rating))  # Clamp to 1-5 range
                
        return user_avg

class CollaborativeFilteringRecommender(BaseRecommender):
    """Collaborative Filtering using user-user and item-item similarity"""
    
    def __init__(self, method='user_based'):
        super().__init__(f"Collaborative Filtering ({method})")
        self.method = method  # 'user_based' or 'item_based'
        
    def fit(self, ratings_df, movies_df=None, users_df=None):
        """Fit by creating user-item matrix and similarity matrices"""
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Create user-item rating matrix
        self.rating_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        if self.method == 'user_based':
            # Calculate user-user similarity
            self.similarity_matrix = cosine_similarity(self.rating_matrix)
            self.similarity_df = pd.DataFrame(
                self.similarity_matrix,
                index=self.rating_matrix.index,
                columns=self.rating_matrix.index
            )
        else:  # item_based
            # Calculate item-item similarity
            self.similarity_matrix = cosine_similarity(self.rating_matrix.T)
            self.similarity_df = pd.DataFrame(
                self.similarity_matrix,
                index=self.rating_matrix.columns,
                columns=self.rating_matrix.columns
            )
            
        self.is_fitted = True
        return self
        
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations using collaborative filtering"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
            
        if user_id not in self.rating_matrix.index:
            # New user - recommend popular movies
            popular_movies = self.ratings_df.groupby('movie_id')['rating'].mean().sort_values(ascending=False)
            recommendations = popular_movies.head(n_recommendations).index.tolist()
        else:
            if self.method == 'user_based':
                recommendations = self._user_based_recommend(user_id, n_recommendations)
            else:
                recommendations = self._item_based_recommend(user_id, n_recommendations)
                
        # Add movie titles
        result = []
        for movie_id in recommendations:
            if self.movies_df is not None:
                movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
                if len(movie_info) > 0:
                    result.append({
                        'movie_id': movie_id,
                        'title': movie_info['title'].iloc[0]
                    })
            else:
                result.append({'movie_id': movie_id})
                
        return result
        
    def _user_based_recommend(self, user_id, n_recommendations):
        """User-based collaborative filtering"""
        # Get similar users
        similar_users = self.similarity_df[user_id].sort_values(ascending=False)[1:11]  # Top 10 similar users
        
        # Get movies rated by similar users but not by target user
        user_movies = set(self.rating_matrix.loc[user_id][self.rating_matrix.loc[user_id] > 0].index)
        
        movie_scores = {}
        for similar_user_id, similarity in similar_users.items():
            if similarity > 0:  # Only consider positively correlated users
                similar_user_movies = self.rating_matrix.loc[similar_user_id]
                for movie_id, rating in similar_user_movies.items():
                    if rating > 0 and movie_id not in user_movies:
                        if movie_id not in movie_scores:
                            movie_scores[movie_id] = 0
                        movie_scores[movie_id] += similarity * rating
                        
        # Sort by score
        recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in recommendations[:n_recommendations]]
        
    def _item_based_recommend(self, user_id, n_recommendations):
        """Item-based collaborative filtering"""
        user_ratings = self.rating_matrix.loc[user_id]
        user_movies = user_ratings[user_ratings > 0]
        
        movie_scores = {}
        for movie_id, rating in user_movies.items():
            # Get similar movies
            similar_movies = self.similarity_df[movie_id].sort_values(ascending=False)[1:11]
            
            for similar_movie_id, similarity in similar_movies.items():
                if similarity > 0 and user_ratings[similar_movie_id] == 0:  # Not rated by user
                    if similar_movie_id not in movie_scores:
                        movie_scores[similar_movie_id] = 0
                    movie_scores[similar_movie_id] += similarity * rating
                    
        # Sort by score
        recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in recommendations[:n_recommendations]]
        
    def predict_rating(self, user_id, movie_id):
        """Predict rating using collaborative filtering"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if user_id not in self.rating_matrix.index:
            return self.ratings_df['rating'].mean()
            
        if self.method == 'user_based':
            return self._user_based_predict(user_id, movie_id)
        else:
            return self._item_based_predict(user_id, movie_id)
            
    def _user_based_predict(self, user_id, movie_id):
        """User-based rating prediction"""
        # Get similar users who rated this movie
        similar_users = self.similarity_df[user_id].sort_values(ascending=False)[1:]
        
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user_id, similarity in similar_users.items():
            if similarity > 0:
                similar_user_rating = self.rating_matrix.loc[similar_user_id, movie_id]
                if similar_user_rating > 0:
                    weighted_sum += similarity * similar_user_rating
                    similarity_sum += abs(similarity)
                    
        if similarity_sum > 0:
            predicted_rating = weighted_sum / similarity_sum
            return max(1, min(5, predicted_rating))
        else:
            return self.ratings_df['rating'].mean()
            
    def _item_based_predict(self, user_id, movie_id):
        """Item-based rating prediction"""
        user_ratings = self.rating_matrix.loc[user_id]
        
        # Get similar movies that user has rated
        similar_movies = self.similarity_df[movie_id].sort_values(ascending=False)[1:]
        
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_movie_id, similarity in similar_movies.items():
            if similarity > 0:
                user_movie_rating = user_ratings[similar_movie_id]
                if user_movie_rating > 0:
                    weighted_sum += similarity * user_movie_rating
                    similarity_sum += abs(similarity)
                    
        if similarity_sum > 0:
            predicted_rating = weighted_sum / similarity_sum
            return max(1, min(5, predicted_rating))
        else:
            return user_ratings[user_ratings > 0].mean()

class MatrixFactorizationRecommender(BaseRecommender):
    """Matrix Factorization using SVD"""
    
    def __init__(self, n_components=50, random_state=42):
        super().__init__("Matrix Factorization (SVD)")
        self.n_components = n_components
        self.random_state = random_state
        
    def fit(self, ratings_df, movies_df=None, users_df=None):
        """Fit SVD model"""
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Create user-item rating matrix
        self.rating_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        # Apply SVD
        self.svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state
        )
        
        # Fit SVD on the rating matrix
        self.user_factors = self.svd.fit_transform(self.rating_matrix)
        self.movie_factors = self.svd.components_.T
        
        # Calculate global mean
        self.global_mean = ratings_df['rating'].mean()
        
        # Calculate user and movie biases
        self.user_bias = {}
        self.movie_bias = {}
        
        for user_id in self.rating_matrix.index:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]['rating']
            self.user_bias[user_id] = user_ratings.mean() - self.global_mean
            
        for movie_id in self.rating_matrix.columns:
            movie_ratings = ratings_df[ratings_df['movie_id'] == movie_id]['rating']
            self.movie_bias[movie_id] = movie_ratings.mean() - self.global_mean
            
        self.is_fitted = True
        return self
        
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations using matrix factorization"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
            
        if user_id not in self.rating_matrix.index:
            # New user - recommend popular movies
            popular_movies = self.ratings_df.groupby('movie_id')['rating'].mean().sort_values(ascending=False)
            recommendations = popular_movies.head(n_recommendations).index.tolist()
        else:
            # Get user index
            user_idx = list(self.rating_matrix.index).index(user_id)
            
            # Predict ratings for all movies
            movie_scores = {}
            for movie_idx, movie_id in enumerate(self.rating_matrix.columns):
                # Skip movies already rated
                if self.rating_matrix.iloc[user_idx, movie_idx] == 0:
                    predicted_rating = self.predict_rating(user_id, movie_id)
                    movie_scores[movie_id] = predicted_rating
                    
            # Sort by predicted rating
            recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = [movie_id for movie_id, score in recommendations[:n_recommendations]]
            
        # Add movie titles
        result = []
        for movie_id in recommendations:
            if self.movies_df is not None:
                movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
                if len(movie_info) > 0:
                    result.append({
                        'movie_id': movie_id,
                        'title': movie_info['title'].iloc[0]
                    })
            else:
                result.append({'movie_id': movie_id})
                
        return result
        
    def predict_rating(self, user_id, movie_id):
        """Predict rating using matrix factorization"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if user_id not in self.rating_matrix.index or movie_id not in self.rating_matrix.columns:
            return self.global_mean
            
        # Get indices
        user_idx = list(self.rating_matrix.index).index(user_id)
        movie_idx = list(self.rating_matrix.columns).index(movie_id)
        
        # Calculate prediction: global_mean + user_bias + movie_bias + dot_product
        prediction = self.global_mean
        prediction += self.user_bias.get(user_id, 0)
        prediction += self.movie_bias.get(movie_id, 0)
        prediction += np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])
        
        # Clamp to valid rating range
        return max(1, min(5, prediction))
