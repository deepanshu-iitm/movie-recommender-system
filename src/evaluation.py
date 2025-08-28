"""
Evaluation metrics for recommendation systems
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class RecommenderEvaluator:
    """Evaluate recommendation systems using various metrics"""
    
    def __init__(self):
        self.metrics = {}
        
    def train_test_split_ratings(self, ratings_df, test_size=0.2, random_state=42):
        """Split ratings into train and test sets"""
        return train_test_split(
            ratings_df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=ratings_df['user_id']  # Ensure each user has ratings in both sets
        )
    
    def calculate_rmse(self, y_true, y_pred):
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mae(self, y_true, y_pred):
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    def evaluate_rating_prediction(self, recommender, test_ratings, sample_size=1000):
        """Evaluate rating prediction accuracy"""
        predictions = []
        actuals = []
        
        # Sample a subset for faster evaluation
        if len(test_ratings) > sample_size:
            test_sample = test_ratings.sample(n=sample_size, random_state=42)
            print(f"Evaluating {recommender.name} on {sample_size} sampled test ratings...")
        else:
            test_sample = test_ratings
            print(f"Evaluating {recommender.name} on {len(test_ratings)} test ratings...")
        
        total_rows = len(test_sample)
        for idx, (_, row) in enumerate(test_sample.iterrows()):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{total_rows} ({idx/total_rows*100:.1f}%)")
            
            try:
                pred_rating = recommender.predict_rating(row['user_id'], row['movie_id'])
                predictions.append(pred_rating)
                actuals.append(row['rating'])
            except Exception as e:
                # Skip problematic predictions
                continue
                
        if len(predictions) == 0:
            return {'rmse': float('inf'), 'mae': float('inf'), 'coverage': 0.0}
            
        rmse = self.calculate_rmse(actuals, predictions)
        mae = self.calculate_mae(actuals, predictions)
        coverage = len(predictions) / len(test_ratings)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'num_predictions': len(predictions)
        }
    
    def precision_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
            
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        
        return relevant_recommended / min(k, len(recommended_k))
    
    def recall_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate Recall@K"""
        if len(relevant_items) == 0:
            return 0.0
            
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        
        return relevant_recommended / len(relevant_items)
    
    def f1_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate F1@K"""
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate Normalized Discounted Cumulative Gain@K"""
        def dcg_at_k(r, k):
            """Calculate DCG@K"""
            r = np.asfarray(r)[:k]
            if r.size:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            return 0.0
        
        # Create relevance scores (1 for relevant, 0 for not relevant)
        relevance_scores = [1 if item in relevant_items else 0 for item in recommended_items[:k]]
        
        # Calculate DCG
        dcg = dcg_at_k(relevance_scores, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1] * min(len(relevant_items), k)
        idcg = dcg_at_k(ideal_relevance, k)
        
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    def evaluate_recommendations(self, recommender, test_ratings, k_values=[5, 10, 20], max_users=50):
        """Evaluate recommendation quality using ranking metrics"""
        print(f"Evaluating {recommender.name} recommendations...")
        
        # Group test ratings by user
        user_test_items = test_ratings.groupby('user_id')['movie_id'].apply(list).to_dict()
        
        # Define relevant items as those rated >= 4
        user_relevant_items = test_ratings[test_ratings['rating'] >= 4].groupby('user_id')['movie_id'].apply(list).to_dict()
        
        # Sample users for faster evaluation
        if len(user_test_items) > max_users:
            sampled_users = list(user_test_items.keys())[:max_users]
            print(f"  Sampling {max_users} users for evaluation...")
        else:
            sampled_users = list(user_test_items.keys())
        
        results = {}
        
        for k in k_values:
            precisions = []
            recalls = []
            f1s = []
            ndcgs = []
            
            for idx, user_id in enumerate(sampled_users):
                if idx % 10 == 0:
                    print(f"    Progress: {idx}/{len(sampled_users)} users")
                
                try:
                    # Get recommendations
                    recommendations = recommender.recommend(user_id, n_recommendations=k)
                    recommended_items = [rec['movie_id'] for rec in recommendations]
                    
                    # Get relevant items for this user
                    relevant_items = user_relevant_items.get(user_id, [])
                    
                    # Calculate metrics
                    precision = self.precision_at_k(recommended_items, relevant_items, k)
                    recall = self.recall_at_k(recommended_items, relevant_items, k)
                    f1 = self.f1_at_k(recommended_items, relevant_items, k)
                    ndcg = self.ndcg_at_k(recommended_items, relevant_items, k)
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)
                    ndcgs.append(ndcg)
                    
                except Exception as e:
                    # Skip problematic users
                    continue
            
            results[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
            results[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
            results[f'f1@{k}'] = np.mean(f1s) if f1s else 0.0
            results[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
        
        return results
    
    def evaluate_diversity(self, recommender, test_users, movies_df, n_recommendations=10):
        """Evaluate recommendation diversity"""
        print(f"Evaluating {recommender.name} diversity...")
        
        all_recommendations = []
        
        for user_id in test_users[:50]:  # Sample 50 users for efficiency
            try:
                recommendations = recommender.recommend(user_id, n_recommendations)
                recommended_items = [rec['movie_id'] for rec in recommendations]
                all_recommendations.extend(recommended_items)
            except:
                continue
        
        if not all_recommendations:
            return {'diversity': 0.0, 'coverage': 0.0}
        
        # Calculate diversity metrics
        unique_items = len(set(all_recommendations))
        total_items = len(all_recommendations)
        total_movies = len(movies_df)
        
        diversity = unique_items / total_items if total_items > 0 else 0.0
        coverage = unique_items / total_movies if total_movies > 0 else 0.0
        
        return {
            'diversity': diversity,
            'coverage': coverage,
            'unique_recommendations': unique_items,
            'total_recommendations': total_items
        }
    
    def comprehensive_evaluation(self, recommender, train_ratings, test_ratings, movies_df):
        """Run comprehensive evaluation of a recommender"""
        print(f"\n{'='*50}")
        print(f"Comprehensive Evaluation: {recommender.name}")
        print(f"{'='*50}")
        
        results = {}
        
        # 1. Rating prediction accuracy
        print("\n1. Rating Prediction Accuracy:")
        rating_metrics = self.evaluate_rating_prediction(recommender, test_ratings)
        results['rating_prediction'] = rating_metrics
        
        print(f"   RMSE: {rating_metrics['rmse']:.4f}")
        print(f"   MAE: {rating_metrics['mae']:.4f}")
        print(f"   Coverage: {rating_metrics['coverage']:.4f}")
        
        # 2. Recommendation quality
        print("\n2. Recommendation Quality:")
        rec_metrics = self.evaluate_recommendations(recommender, test_ratings)
        results['recommendation_quality'] = rec_metrics
        
        for metric, value in rec_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        # 3. Diversity
        print("\n3. Diversity:")
        test_users = test_ratings['user_id'].unique()
        diversity_metrics = self.evaluate_diversity(recommender, test_users, movies_df)
        results['diversity'] = diversity_metrics
        
        for metric, value in diversity_metrics.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")
        
        return results
    
    def compare_recommenders(self, recommenders, train_ratings, test_ratings, movies_df):
        """Compare multiple recommenders"""
        print(f"\n{'='*60}")
        print("RECOMMENDER SYSTEM COMPARISON")
        print(f"{'='*60}")
        
        all_results = {}
        
        for recommender in recommenders:
            try:
                results = self.comprehensive_evaluation(
                    recommender, train_ratings, test_ratings, movies_df
                )
                all_results[recommender.name] = results
            except Exception as e:
                print(f"Error evaluating {recommender.name}: {e}")
                continue
        
        # Create comparison summary
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        
        comparison_df = pd.DataFrame()
        
        for name, results in all_results.items():
            row_data = {
                'Model': name,
                'RMSE': results['rating_prediction']['rmse'],
                'MAE': results['rating_prediction']['mae'],
                'Precision@10': results['recommendation_quality']['precision@10'],
                'Recall@10': results['recommendation_quality']['recall@10'],
                'NDCG@10': results['recommendation_quality']['ndcg@10'],
                'Diversity': results['diversity']['diversity'],
                'Coverage': results['diversity']['coverage']
            }
            comparison_df = pd.concat([comparison_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Sort by NDCG@10 (higher is better)
        comparison_df = comparison_df.sort_values('NDCG@10', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return all_results, comparison_df
