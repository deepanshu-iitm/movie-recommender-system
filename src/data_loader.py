"""
Data loader for MovieLens 100K dataset
"""
import os
import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path

class MovieLensDataLoader:
    """Load and preprocess MovieLens 100K dataset"""
    
    def __init__(self, data_dir="data"):
        # Handle relative paths by making them relative to the project root
        if not Path(data_dir).is_absolute():
            # Find project root by looking for key files
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:
                if (current_dir / "requirements.txt").exists() or (current_dir / "README.md").exists():
                    project_root = current_dir
                    break
                current_dir = current_dir.parent
            else:
                # Fallback to current working directory
                project_root = Path.cwd()
            
            self.data_dir = project_root / data_dir
        else:
            self.data_dir = Path(data_dir)
            
        self.data_dir.mkdir(exist_ok=True)
        # Using Kaggle dataset - user should download manually or use Kaggle API
        self.kaggle_dataset = "prajitdatta/movielens-100k-dataset"
        
    def download_dataset(self):
        """Download MovieLens 100K dataset from Kaggle"""
        dataset_path = self.data_dir / "movielens-100k-dataset"
        zip_path = self.data_dir / "movielens-100k-dataset.zip"
        
        # Check if dataset is already extracted
        if dataset_path.exists() and len(list(dataset_path.glob("*.csv"))) > 0:
            print("Dataset already exists!")
            return
            
        # Check if zip file exists and extract it
        if zip_path.exists():
            print("Found downloaded zip file, extracting...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                print("Dataset extracted successfully!")
                return
            except Exception as e:
                print(f"Failed to extract zip file: {e}")
        
        print("Downloading MovieLens 100K dataset from Kaggle...")
        
        try:
            # Try using Kaggle API
            import kaggle
            kaggle.api.dataset_download_files(
                self.kaggle_dataset, 
                path=self.data_dir, 
                unzip=True
            )
            print("Dataset downloaded successfully using Kaggle API!")
            
            # If unzip didn't work, try manual extraction
            if zip_path.exists() and not dataset_path.exists():
                print("Extracting downloaded zip file...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                print("Dataset extracted successfully!")
                
        except Exception as e:
            print(f"Kaggle API failed: {e}")
            print("\nPlease download the dataset manually from:")
            print(f"https://www.kaggle.com/datasets/{self.kaggle_dataset}")
            print(f"Extract the files to: {dataset_path}")
            print("\nAlternatively, install and configure Kaggle API:")
            print("1. pip install kaggle")
            print("2. Get API token from Kaggle Account settings")
            print("3. Place kaggle.json in ~/.kaggle/ directory")
            raise
        
    def load_ratings(self):
        """Load ratings data"""
        # Try different possible file locations
        possible_paths = [
            self.data_dir / "ml-100k" / "u.data",
            self.data_dir / "movielens-100k-dataset" / "u.data",
            self.data_dir / "u.data",
            self.data_dir / "ratings.csv"
        ]
        
        ratings_path = None
        print(f"Looking for ratings file in these locations:")
        for path in possible_paths:
            print(f"  Checking: {path.absolute()} - {'EXISTS' if path.exists() else 'NOT FOUND'}")
            if path.exists():
                ratings_path = path
                break
                
        if ratings_path is None:
            print(f"Current working directory: {Path.cwd()}")
            print(f"Data directory: {self.data_dir.absolute()}")
            raise FileNotFoundError("Ratings file not found. Please ensure dataset is downloaded.")
        
        # Load based on file extension
        if ratings_path.suffix == '.csv':
            ratings = pd.read_csv(ratings_path)
            # Standardize column names
            if 'userId' in ratings.columns:
                ratings = ratings.rename(columns={
                    'userId': 'user_id',
                    'movieId': 'movie_id'
                })
        else:
            # Original format
            rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
            ratings = pd.read_csv(
                ratings_path, 
                sep='\t', 
                names=rating_cols,
                encoding='latin-1'
            )
        
        # Convert timestamp to datetime if it exists and is numeric
        if 'timestamp' in ratings.columns:
            if ratings['timestamp'].dtype in ['int64', 'float64']:
                ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        
        return ratings
    
    def load_movies(self):
        """Load movies data"""
        # Try different possible file locations
        possible_paths = [
            self.data_dir / "ml-100k" / "u.item",
            self.data_dir / "movielens-100k-dataset" / "u.item",
            self.data_dir / "u.item",
            self.data_dir / "movies.csv"
        ]
        
        movies_path = None
        for path in possible_paths:
            if path.exists():
                movies_path = path
                break
                
        if movies_path is None:
            raise FileNotFoundError("Movies file not found. Please ensure dataset is downloaded.")
        
        # Load based on file format
        if movies_path.suffix == '.csv':
            movies = pd.read_csv(movies_path)
            # Standardize column names
            if 'movieId' in movies.columns:
                movies = movies.rename(columns={'movieId': 'movie_id'})
            
            # Parse genres if they're in string format
            if 'genres' in movies.columns and isinstance(movies['genres'].iloc[0], str):
                # Convert genre string to binary columns
                all_genres = set()
                for genres_str in movies['genres'].dropna():
                    all_genres.update(genres_str.split('|'))
                
                # Create binary genre columns
                for genre in sorted(all_genres):
                    if genre != '(no genres listed)':
                        movies[genre] = movies['genres'].str.contains(genre, na=False).astype(int)
        else:
            # Original format
            movie_cols = [
                'movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
                'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ]
            
            movies = pd.read_csv(
                movies_path,
                sep='|',
                names=movie_cols,
                encoding='latin-1'
            )
        
        # Extract year from title if not already present
        if 'year' not in movies.columns:
            movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
            movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
        
        # Create genre list for each movie if not already present
        if 'genres' not in movies.columns:
            genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                         'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                         'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            
            # Filter to only existing genre columns
            existing_genre_cols = [col for col in genre_cols if col in movies.columns]
            
            if existing_genre_cols:
                movies['genres'] = movies[existing_genre_cols].apply(
                    lambda x: [genre for genre, val in zip(existing_genre_cols, x) if val == 1], 
                    axis=1
                )
        
        return movies
    
    def load_users(self):
        """Load users data"""
        # Try different possible file locations
        possible_paths = [
            self.data_dir / "ml-100k" / "u.user",
            self.data_dir / "movielens-100k-dataset" / "u.user",
            self.data_dir / "u.user",
            self.data_dir / "users.csv"
        ]
        
        users_path = None
        for path in possible_paths:
            if path.exists():
                users_path = path
                break
        
        if users_path is None:
            print("Users file not found. Creating minimal user data from ratings.")
            # Create minimal user data if file doesn't exist
            ratings = self.load_ratings()
            users = pd.DataFrame({
                'user_id': ratings['user_id'].unique(),
                'age': 25,  # Default age
                'gender': 'M',  # Default gender
                'occupation': 'other',  # Default occupation
                'zip_code': '00000'  # Default zip
            })
            return users
        
        # Load based on file format
        if users_path.suffix == '.csv':
            users = pd.read_csv(users_path)
            # Standardize column names
            if 'userId' in users.columns:
                users = users.rename(columns={'userId': 'user_id'})
        else:
            # Original format
            user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
            users = pd.read_csv(
                users_path,
                sep='|',
                names=user_cols,
                encoding='latin-1'
            )
        
        return users
    
    def load_all_data(self):
        """Load all datasets and return as dictionary"""
        if not (self.data_dir / "ml-100k").exists():
            self.download_dataset()
        else:
            print("Found local ml-100k dataset. Skipping download.")
        
        data = {
            'ratings': self.load_ratings(),
            'movies': self.load_movies(),
            'users': self.load_users()
        }
        
        print(f"Loaded data:")
        print(f"- Ratings: {len(data['ratings'])} records")
        print(f"- Movies: {len(data['movies'])} movies")
        print(f"- Users: {len(data['users'])} users")
        
        return data
    
    def create_rating_matrix(self, ratings_df):
        """Create user-item rating matrix"""
        rating_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id', 
            values='rating',
            fill_value=0
        )
        
        return rating_matrix

if __name__ == "__main__":
    # Test the data loader
    loader = MovieLensDataLoader()
    data = loader.load_all_data()
    
    # Display basic statistics
    print("\n=== Dataset Statistics ===")
    print(f"Rating range: {data['ratings']['rating'].min()} - {data['ratings']['rating'].max()}")
    print(f"Average rating: {data['ratings']['rating'].mean():.2f}")
    print(f"Most active user: {data['ratings']['user_id'].value_counts().index[0]} with {data['ratings']['user_id'].value_counts().iloc[0]} ratings")
    print(f"Most rated movie: {data['ratings']['movie_id'].value_counts().index[0]} with {data['ratings']['movie_id'].value_counts().iloc[0]} ratings")
