"""
Main script to train and evaluate all recommendation models
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_loader import MovieLensDataLoader
from recommenders import (
    PopularityRecommender,
    ContentBasedRecommender, 
    CollaborativeFilteringRecommender,
    MatrixFactorizationRecommender
)
from evaluation import RecommenderEvaluator
import pickle
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main training and evaluation pipeline"""
    print("Movie Recommendation System Training Pipeline")
    print("=" * 60)
    
    # 1. Load Data
    print("\nLoading MovieLens 100K Dataset...")
    loader = MovieLensDataLoader()
    
    try:
        data = loader.load_all_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download the dataset first:")
        return
    
    ratings = data['ratings']
    movies = data['movies']
    users = data['users']
    
    print(f"Data loaded successfully!")
    print(f"   - Ratings: {len(ratings):,}")
    print(f"   - Movies: {len(movies):,}")
    print(f"   - Users: {len(users):,}")
    
    # 2. Split Data
    print("\nSplitting data into train/test sets...")
    evaluator = RecommenderEvaluator()
    train_ratings, test_ratings = evaluator.train_test_split_ratings(ratings)
    
    print(f"   - Training ratings: {len(train_ratings):,}")
    print(f"   - Test ratings: {len(test_ratings):,}")
    
    # 3. Initialize Recommenders
    print("\nInitializing recommendation models...")
    
    recommenders = [
        PopularityRecommender(),
        ContentBasedRecommender(),
        CollaborativeFilteringRecommender(method='user_based'),
        CollaborativeFilteringRecommender(method='item_based'),
        MatrixFactorizationRecommender(n_components=50)
    ]
    
    # 4. Train Models
    print("\nTraining models...")
    trained_recommenders = []
    
    for recommender in recommenders:
        try:
            print(f"   Training {recommender.name}...")
            recommender.fit(train_ratings, movies, users)
            trained_recommenders.append(recommender)
            print(f"   {recommender.name} trained successfully!")
        except Exception as e:
            print(f"   Error training {recommender.name}: {e}")
            continue
    
    # 5. Evaluate Models
    print("\nEvaluating models...")
    results, comparison_df = evaluator.compare_recommenders(
        trained_recommenders, train_ratings, test_ratings, movies
    )
    
    # 6. Save Results
    print("\nSaving results...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save trained models
    for recommender in trained_recommenders:
        model_path = models_dir / f"{recommender.name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(recommender, f)
            print(f"   Saved {recommender.name} to {model_path}")
        except Exception as e:
            print(f"   Error saving {recommender.name}: {e}")
    
    # Save evaluation results
    results_path = models_dir / "evaluation_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save comparison DataFrame
    comparison_path = models_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"   Evaluation results saved to {results_path}")
    print(f"   Model comparison saved to {comparison_path}")
    
    # 7. Display Best Model
    print("\nBest Performing Models:")
    print("-" * 40)
    
    best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin()]
    best_precision = comparison_df.loc[comparison_df['Precision@10'].idxmax()]
    best_ndcg = comparison_df.loc[comparison_df['NDCG@10'].idxmax()]
    
    print(f"Best RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
    print(f"Best Precision@10: {best_precision['Model']} ({best_precision['Precision@10']:.4f})")
    print(f"Best NDCG@10: {best_ndcg['Model']} ({best_ndcg['NDCG@10']:.4f})")
    
    # 8. Sample Recommendations
    print("\nSample Recommendations:")
    print("-" * 40)
    
    # Get a sample user
    sample_user = train_ratings['user_id'].iloc[0]
    print(f"\nRecommendations for User {sample_user}:")
    
    # Show user's rating history
    user_history = train_ratings[train_ratings['user_id'] == sample_user].merge(
        movies[['movie_id', 'title']], on='movie_id'
    ).sort_values('rating', ascending=False).head(5)
    
    print(f"\nUser's top-rated movies:")
    for _, row in user_history.iterrows():
        print(f"   {row['rating']}/5 - {row['title']}")
    
    # Show recommendations from best model
    best_model_name = best_ndcg['Model']
    best_model = next(r for r in trained_recommenders if r.name == best_model_name)
    
    recommendations = best_model.recommend(sample_user, n_recommendations=5)
    print(f"\nTop 5 recommendations from {best_model_name}:")
    for i, rec in enumerate(recommendations, 1):
        title = rec.get('title', f"Movie ID: {rec['movie_id']}")
        print(f"   {i}. {title}")
    
    print(f"\nTraining pipeline completed successfully!")
    print(f"Models and results saved in: {models_dir.absolute()}")
    print(f"\nNext steps:")
    print(f"   1. Run 'streamlit run app.py' to launch the web interface")
    print(f"   2. Explore the Jupyter notebook for detailed analysis")
    print(f"   3. Experiment with different model parameters")

def load_trained_model(model_name):
    """Load a trained model from disk"""
    models_dir = Path("models")
    model_path = models_dir / f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Model {model_name} not found at {model_path}")

def get_movie_recommendations(user_id, model_name="matrix_factorization_svd", n_recommendations=10):
    """Get recommendations for a specific user using a trained model"""
    try:
        model = load_trained_model(model_name)
        recommendations = model.recommend(user_id, n_recommendations)
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return []

if __name__ == "__main__":
    main()
