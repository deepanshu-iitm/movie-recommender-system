"""
Streamlit Web App for Movie Recommendation System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import pickle

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))
from data_loader import MovieLensDataLoader

# Page config
st.set_page_config(
    page_title="Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        background-color: var(--background-color, #f8f9fa);
        color: var(--text-color, #000000);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
        border: 1px solid var(--border-color, #dee2e6);
    }
    .metric-card {
        background-color: var(--secondary-background-color, #e3f2fd);
        color: var(--text-color, #000000);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid var(--border-color, #dee2e6);
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .movie-card {
            background-color: #2d3748 !important;
            color: #ffffff !important;
            border: 1px solid #4a5568 !important;
        }
        .metric-card {
            background-color: #2d3748 !important;
            color: #ffffff !important;
            border: 1px solid #4a5568 !important;
        }
    }
    /* Streamlit dark theme override */
    [data-testid="stAppViewContainer"][data-theme="dark"] .movie-card {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
    [data-testid="stAppViewContainer"][data-theme="dark"] .metric-card {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        loader = MovieLensDataLoader()
        data = loader.load_all_data()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_evaluation_results():
    """Load evaluation results"""
    try:
        models_dir = Path("models")
        
        # Load comparison results
        comparison_path = models_dir / "model_comparison.csv"
        if comparison_path.exists():
            comparison_df = pd.read_csv(comparison_path)
            return comparison_df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading evaluation results: {e}")
        return None

@st.cache_data
def get_movie_recommendations(user_id, model_name, n_recommendations=10):
    """Get movie recommendations for a user using specified model"""
    try:
        models_dir = Path("models")
        model_path = models_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            st.error(f"Model {model_name} not found at {model_path}")
            return []
        
        # Load the trained model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Get recommendations
        recommendations = model.recommend(user_id, n_recommendations=n_recommendations)
        
        # Load movie data to get titles
        data = load_data()
        if data and 'movies' in data:
            movies_df = data['movies']
            
            # Add movie titles to recommendations
            for rec in recommendations:
                movie_info = movies_df[movies_df['movie_id'] == rec['movie_id']]
                if not movie_info.empty:
                    rec['title'] = movie_info.iloc[0]['title']
                else:
                    rec['title'] = f"Movie ID: {rec['movie_id']}"
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Load data
    data = load_data()
    if data is None:
        st.error("❌ Please ensure the dataset is downloaded and run the training script first!")
        st.info("Run: `python src/train_models.py` to train the models")
        return
    
    ratings = data['ratings']
    movies = data['movies']
    users = data['users']
    
    # Sidebar options
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Get Recommendations", "Model Performance", "Data Analysis", "Movie Search"]
    )
    
    if page == "Home":
        show_home_page(ratings, movies, users)
    elif page == "Get Recommendations":
        show_recommendations_page(ratings, movies, users)
    elif page == "Model Performance":
        show_performance_page()
    elif page == "Data Analysis":
        show_analysis_page(ratings, movies, users)
    elif page == "Movie Search":
        show_search_page(movies, ratings)

def show_home_page(ratings, movies, users):
    """Home page with overview"""
    st.markdown("## Welcome to the Movie Recommendation System!")
    
    st.markdown("""
    This system uses multiple machine learning approaches to recommend movies:
    - **Popularity-based**: Recommends trending movies
    - **Content-based**: Recommends similar movies based on genres
    - **Collaborative Filtering**: Recommends based on similar users/items
    - **Matrix Factorization**: Uses latent factors to make predictions
    """)
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(ratings):,}</h3>
            <p>Total Ratings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(movies):,}</h3>
            <p>Movies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(users):,}</h3>
            <p>Users</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = ratings['rating'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_rating:.1f}</h3>
            <p>Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Popular movies
    st.markdown("## Most Popular Movies")
    popular_movies = ratings.groupby('movie_id').agg({
        'rating': ['count', 'mean']
    }).round(2)
    popular_movies.columns = ['num_ratings', 'avg_rating']
    popular_movies = popular_movies.reset_index()
    popular_movies = popular_movies.merge(movies[['movie_id', 'title']], on='movie_id')
    popular_movies = popular_movies.sort_values('num_ratings', ascending=False).head(10)
    
    for _, movie in popular_movies.iterrows():
        st.markdown(f"""
        <div class="movie-card">
            <strong>{movie['title']}</strong><br>
            {movie['avg_rating']:.1f}/5 | {movie['num_ratings']} ratings
        </div>
        """, unsafe_allow_html=True)

def show_recommendations_page(ratings, movies, users):
    """Recommendations page"""
    st.markdown("## 🎯 Get Personalized Recommendations")
    
    # Model selection
    available_models = [
        "popularity-based_recommender",
        "content-based_recommender", 
        "collaborative_filtering_user_based",
        "collaborative_filtering_item_based",
        "matrix_factorization_svd"
    ]
    
    model_display_names = {
        "popularity-based_recommender": "Popularity-based",
        "content-based_recommender": "Content-based",
        "collaborative_filtering_user_based": "User-based CF",
        "collaborative_filtering_item_based": "Item-based CF",
        "matrix_factorization_svd": "Matrix Factorization"
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Choose recommendation algorithm:",
            available_models,
            format_func=lambda x: model_display_names.get(x, x)
        )
    
    with col2:
        n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
    
    # User selection
    user_ids = sorted(ratings['user_id'].unique())
    selected_user = st.selectbox("Select User ID:", user_ids)
    
    if st.button("Get Recommendations", type="primary"):
        try:
            with st.spinner("Generating recommendations..."):
                recommendations = get_movie_recommendations(
                    selected_user, selected_model, n_recommendations
                )
            
            if recommendations:
                st.success(f"Generated {len(recommendations)} recommendations!")
                
                # Show user's rating history
                user_history = ratings[ratings['user_id'] == selected_user].merge(
                    movies[['movie_id', 'title']], on='movie_id'
                ).sort_values('rating', ascending=False).head(5)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### User's Top-Rated Movies")
                    for _, movie in user_history.iterrows():
                        st.markdown(f"""
                        <div class="movie-card">
                            <strong>{movie['title']}</strong><br>
                            Rating: {movie['rating']}/5
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Recommendations")
                    for i, rec in enumerate(recommendations, 1):
                        title = rec.get('title', f"Movie ID: {rec['movie_id']}")
                        score = rec.get('similarity_score', rec.get('popularity_score', ''))
                        score_text = f" | Score: {score:.3f}" if score else ""
                        
                        st.markdown(f"""
                        <div class="movie-card">
                            <strong>{i}. {title}</strong><br>
                            🎬 Movie ID: {rec['movie_id']}{score_text}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("❌ No recommendations generated. Please check if models are trained.")
                
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {e}")
            st.info("Make sure to run `python src/train_models.py` first to train the models.")

def show_performance_page():
    """Model performance comparison page"""
    st.markdown("## 📊 Model Performance Comparison")
    
    comparison_df = load_evaluation_results()
    
    if comparison_df is not None:
        st.markdown("### 🏆 Overall Performance")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE comparison
            fig_rmse = px.bar(
                comparison_df, 
                x='Model', 
                y='RMSE',
                title='RMSE Comparison (Lower is Better)',
                color='RMSE',
                color_continuous_scale='Reds_r'
            )
            fig_rmse.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            # Precision@10 comparison
            fig_precision = px.bar(
                comparison_df,
                x='Model',
                y='Precision@10', 
                title='Precision@10 Comparison (Higher is Better)',
                color='Precision@10',
                color_continuous_scale='Blues'
            )
            fig_precision.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_precision, use_container_width=True)
        
        # Radar chart for comprehensive comparison
        st.markdown("### 🕸️ Comprehensive Performance Radar")
        
        # Normalize metrics for radar chart
        metrics = ['RMSE', 'MAE', 'Precision@10', 'Recall@10', 'NDCG@10', 'Diversity']
        
        fig_radar = go.Figure()
        
        for _, row in comparison_df.iterrows():
            # Invert RMSE and MAE (lower is better)
            values = [
                1 - (row['RMSE'] / comparison_df['RMSE'].max()),  # Inverted RMSE
                1 - (row['MAE'] / comparison_df['MAE'].max()),    # Inverted MAE
                row['Precision@10'],
                row['Recall@10'], 
                row['NDCG@10'],
                row['Diversity']
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=row['Model']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart (Normalized)"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
    else:
        st.warning("⚠️ No evaluation results found. Please run the training script first.")
        st.code("python src/train_models.py")

def show_analysis_page(ratings, movies, users):
    """Data analysis page"""
    st.markdown("## 📈 Dataset Analysis")
    
    # Rating distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ratings = px.histogram(
            ratings, 
            x='rating',
            title='Rating Distribution',
            nbins=5
        )
        st.plotly_chart(fig_ratings, use_container_width=True)
    
    with col2:
        # User activity
        user_activity = ratings.groupby('user_id').size().reset_index(name='num_ratings')
        fig_activity = px.histogram(
            user_activity,
            x='num_ratings',
            title='User Activity Distribution',
            nbins=30
        )
        st.plotly_chart(fig_activity, use_container_width=True)
    
    # Genre analysis
    if 'genres' in movies.columns:
        st.markdown("### 🎭 Genre Analysis")
        
        # Get genre columns
        genre_cols = [col for col in movies.columns if col in [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]]
        
        if genre_cols:
            genre_counts = movies[genre_cols].sum().sort_values(ascending=True)
            
            fig_genres = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title='Movies by Genre'
            )
            st.plotly_chart(fig_genres, use_container_width=True)
    
    # Temporal analysis
    if 'timestamp' in ratings.columns:
        st.markdown("### ⏰ Temporal Analysis")
        
        ratings['date'] = ratings['timestamp'].dt.date
        daily_ratings = ratings.groupby('date').size().reset_index(name='num_ratings')
        
        fig_temporal = px.line(
            daily_ratings,
            x='date',
            y='num_ratings',
            title='Ratings Over Time'
        )
        st.plotly_chart(fig_temporal, use_container_width=True)

def show_search_page(movies, ratings):
    """Movie search page"""
    st.markdown("## 🔍 Movie Search & Information")
    
    # Search functionality
    search_term = st.text_input("🔍 Search for movies:", placeholder="Enter movie title...")
    
    if search_term:
        # Filter movies
        filtered_movies = movies[movies['title'].str.contains(search_term, case=False, na=False)]
        
        if len(filtered_movies) > 0:
            st.success(f"Found {len(filtered_movies)} movies matching '{search_term}'")
            
            for _, movie in filtered_movies.head(10).iterrows():
                # Get movie statistics
                movie_ratings = ratings[ratings['movie_id'] == movie['movie_id']]
                
                if len(movie_ratings) > 0:
                    avg_rating = movie_ratings['rating'].mean()
                    num_ratings = len(movie_ratings)
                    rating_info = f"⭐ {avg_rating:.1f}/5 ({num_ratings} ratings)"
                else:
                    rating_info = "No ratings yet"
                
                # Display genres if available
                if 'genres' in movie and isinstance(movie['genres'], list):
                    genres_str = " | ".join(movie['genres'][:3])  # Show first 3 genres
                else:
                    genres_str = "Unknown genres"
                
                year_info = f" ({int(movie['year'])})" if pd.notna(movie.get('year')) else ""
                
                st.markdown(f"""
                <div class="movie-card">
                    <strong>{movie['title']}{year_info}</strong><br>
                    🎭 {genres_str}<br>
                    {rating_info}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"No movies found matching '{search_term}'")
    
    # Random movie suggestion
    if st.button("🎲 Suggest Random Movie"):
        random_movie = movies.sample(1).iloc[0]
        movie_ratings = ratings[ratings['movie_id'] == random_movie['movie_id']]
        
        if len(movie_ratings) > 0:
            avg_rating = movie_ratings['rating'].mean()
            num_ratings = len(movie_ratings)
            rating_info = f"⭐ {avg_rating:.1f}/5 ({num_ratings} ratings)"
        else:
            rating_info = "No ratings yet"
        
        year_info = f" ({int(random_movie['year'])})" if pd.notna(random_movie.get('year')) else ""
        
        st.markdown(f"""
        <div class="movie-card">
            <h3>🎬 Random Movie Suggestion</h3>
            <strong>{random_movie['title']}{year_info}</strong><br>
            {rating_info}
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
