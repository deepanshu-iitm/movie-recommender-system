# Movie Recommendation System

A comprehensive movie recommendation system built using the MovieLens 100K dataset. This project implements multiple recommendation approaches including popularity-based, content-based filtering, collaborative filtering, and matrix factorization.

## Features

- **Popularity-based Recommender**: Recommends top-rated movies
- **Content-based Filtering**: Recommends movies similar to user preferences based on genres
- **Collaborative Filtering**: User-user and item-item similarity recommendations
- **Matrix Factorization**: SVD-based latent factor model
- **Comprehensive Evaluation**: RMSE, MAE, Precision@K, Recall@K metrics
- **Interactive Web App**: Streamlit interface for movie recommendations

## Dataset

Uses the MovieLens 100K dataset containing:
- 100,000 ratings (1-5 scale)
- 943 users
- 1,682 movies
- Movie metadata (genres, titles, years)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the data exploration notebook:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

2. Train and evaluate models:
```bash
python src/train_models.py
```

3. Launch the Streamlit app:
```bash
streamlit run app.py
```




