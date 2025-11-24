"""Unit tests for travel recommendation system."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.loader import TravelDataLoader, create_train_test_split, create_negative_samples
from src.models.recommenders import (
    PopularityRecommender,
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender,
    MatrixFactorizationRecommender
)
from src.evaluation.metrics import RecommendationMetrics


class TestTravelDataLoader:
    """Test cases for TravelDataLoader."""
    
    def test_init(self):
        """Test initialization."""
        loader = TravelDataLoader("test_data")
        assert loader.data_dir == "test_data"
        assert loader.interactions_df is None
        assert loader.items_df is None
        assert loader.users_df is None
    
    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        """Test data loading."""
        # Mock data
        mock_interactions = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': [1, 2, 3],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'rating': [5, 4, 3]
        })
        
        mock_items = pd.DataFrame({
            'item_id': [1, 2, 3],
            'title': ['Paris', 'Tokyo', 'Sydney'],
            'description': ['City of Light', 'Modern Metropolis', 'Coastal City']
        })
        
        mock_users = pd.DataFrame({
            'user_id': [1, 2, 3],
            'age_group': ['25-35', '35-45', '25-35']
        })
        
        mock_read_csv.side_effect = [mock_interactions, mock_items, mock_users]
        
        loader = TravelDataLoader("test_data")
        interactions_df, items_df, users_df = loader.load_data()
        
        assert len(interactions_df) == 3
        assert len(items_df) == 3
        assert len(users_df) == 3
        assert mock_read_csv.call_count == 3


class TestRecommendationModels:
    """Test cases for recommendation models."""
    
    def setup_method(self):
        """Set up test data."""
        self.interactions_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [1, 2, 1, 3, 2, 3],
            'rating': [5, 4, 3, 5, 4, 3],
            'timestamp': pd.date_range('2023-01-01', periods=6)
        })
        
        self.items_df = pd.DataFrame({
            'item_id': [1, 2, 3],
            'title': ['Paris', 'Tokyo', 'Sydney'],
            'description': ['City of Light', 'Modern Metropolis', 'Coastal City'],
            'activities': ['Museums', 'Technology', 'Beaches'],
            'culture': ['Romantic', 'Traditional', 'Relaxed']
        })
        
        self.users_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'age_group': ['25-35', '35-45', '25-35']
        })
    
    def test_popularity_recommender(self):
        """Test PopularityRecommender."""
        model = PopularityRecommender(random_state=42)
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        recommendations = model.recommend(1, top_k=2)
        assert len(recommendations) == 2
        assert isinstance(recommendations, list)
    
    def test_content_based_recommender(self):
        """Test ContentBasedRecommender."""
        model = ContentBasedRecommender(random_state=42)
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        recommendations = model.recommend(1, top_k=2)
        assert len(recommendations) == 2
        assert isinstance(recommendations, list)
    
    def test_collaborative_filtering_recommender(self):
        """Test CollaborativeFilteringRecommender."""
        model = CollaborativeFilteringRecommender(random_state=42)
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        recommendations = model.recommend(1, top_k=2)
        assert len(recommendations) == 2
        assert isinstance(recommendations, list)
    
    def test_hybrid_recommender(self):
        """Test HybridRecommender."""
        model = HybridRecommender(random_state=42)
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        recommendations = model.recommend(1, top_k=2)
        assert len(recommendations) == 2
        assert isinstance(recommendations, list)
    
    def test_matrix_factorization_recommender(self):
        """Test MatrixFactorizationRecommender."""
        model = MatrixFactorizationRecommender(random_state=42)
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        recommendations = model.recommend(1, top_k=2)
        assert len(recommendations) == 2
        assert isinstance(recommendations, list)


class TestRecommendationMetrics:
    """Test cases for RecommendationMetrics."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        y_true = [1, 2, 3]
        y_pred = [1, 4, 5]
        
        precision = RecommendationMetrics.precision_at_k(y_true, y_pred, k=3)
        assert precision == 1/3  # Only item 1 is correct
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        y_true = [1, 2, 3]
        y_pred = [1, 4, 5]
        
        recall = RecommendationMetrics.recall_at_k(y_true, y_pred, k=3)
        assert recall == 1/3  # Only item 1 is found out of 3 true items
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        y_true = [1, 2, 3]
        y_pred = [1, 4, 5]
        
        ndcg = RecommendationMetrics.ndcg_at_k(y_true, y_pred, k=3)
        assert 0 <= ndcg <= 1
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        y_true = [1, 2, 3]
        y_pred = [1, 4, 5]
        
        hit_rate = RecommendationMetrics.hit_rate_at_k(y_true, y_pred, k=3)
        assert hit_rate == 1.0  # At least one hit
    
    def test_coverage(self):
        """Test coverage calculation."""
        predictions = {
            1: [1, 2, 3],
            2: [2, 3, 4],
            3: [3, 4, 5]
        }
        all_items = [1, 2, 3, 4, 5, 6]
        
        coverage = RecommendationMetrics.coverage(predictions, all_items)
        assert coverage == 5/6  # Items 1-5 are recommended out of 6 total
    
    def test_diversity(self):
        """Test diversity calculation."""
        predictions = {
            1: [1, 2, 3],
            2: [1, 1, 2],  # Less diverse
            3: [4, 5, 6]
        }
        
        diversity = RecommendationMetrics.diversity(predictions)
        assert 0 <= diversity <= 1


class TestDataProcessing:
    """Test cases for data processing functions."""
    
    def test_create_train_test_split(self):
        """Test train/test split creation."""
        interactions_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [1, 2, 1, 3, 2, 3],
            'rating': [5, 4, 3, 5, 4, 3],
            'timestamp': pd.date_range('2023-01-01', periods=6)
        })
        
        train_df, test_df = create_train_test_split(interactions_df, test_size=0.5)
        
        assert len(train_df) + len(test_df) == len(interactions_df)
        assert len(train_df) == 3  # 50% of 6
        assert len(test_df) == 3
    
    def test_create_negative_samples(self):
        """Test negative sample creation."""
        interactions_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'item_id': [1, 2, 1, 3],
            'rating': [5, 4, 3, 5],
            'timestamp': pd.date_range('2023-01-01', periods=4)
        })
        
        items_df = pd.DataFrame({
            'item_id': [1, 2, 3, 4, 5]
        })
        
        combined_df = create_negative_samples(interactions_df, items_df, num_negatives=2)
        
        # Should have original interactions plus negative samples
        assert len(combined_df) > len(interactions_df)
        
        # Check that negative samples have rating 0
        negative_samples = combined_df[combined_df['interaction_type'] == 'negative']
        assert all(negative_samples['rating'] == 0)


if __name__ == "__main__":
    pytest.main([__file__])
