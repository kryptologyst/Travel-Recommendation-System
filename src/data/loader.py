"""Data loading and preprocessing utilities for travel recommendation system."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TravelDataLoader:
    """Loads and preprocesses travel recommendation data."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the raw data files
        """
        self.data_dir = Path(data_dir)
        self.interactions_df: Optional[pd.DataFrame] = None
        self.items_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all data files.
        
        Returns:
            Tuple of (interactions_df, items_df, users_df)
        """
        logger.info("Loading travel recommendation data...")
        
        # Load interactions
        interactions_path = self.data_dir / "interactions.csv"
        self.interactions_df = pd.read_csv(interactions_path)
        self.interactions_df['timestamp'] = pd.to_datetime(self.interactions_df['timestamp'])
        
        # Load items (destinations)
        items_path = self.data_dir / "items.csv"
        self.items_df = pd.read_csv(items_path)
        
        # Load users
        users_path = self.data_dir / "users.csv"
        self.users_df = pd.read_csv(users_path)
        
        logger.info(f"Loaded {len(self.interactions_df)} interactions, "
                   f"{len(self.items_df)} destinations, "
                   f"{len(self.users_df)} users")
        
        return self.interactions_df, self.items_df, self.users_df
    
    def get_user_item_matrix(self) -> pd.DataFrame:
        """Create user-item interaction matrix.
        
        Returns:
            User-item matrix with ratings as values
        """
        if self.interactions_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        matrix = self.interactions_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        return matrix
    
    def get_item_features(self) -> pd.DataFrame:
        """Extract item features for content-based filtering.
        
        Returns:
            DataFrame with item features
        """
        if self.items_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Create feature columns from categorical variables
        features_df = self.items_df.copy()
        
        # One-hot encode categorical features
        categorical_features = ['climate', 'travel_style', 'budget_level']
        for feature in categorical_features:
            dummies = pd.get_dummies(features_df[feature], prefix=feature)
            features_df = pd.concat([features_df, dummies], axis=1)
        
        # Normalize numerical features
        numerical_features = ['rating_avg', 'popularity_score']
        for feature in numerical_features:
            features_df[f'{feature}_normalized'] = (
                features_df[feature] - features_df[feature].min()
            ) / (features_df[feature].max() - features_df[feature].min())
        
        return features_df
    
    def get_user_features(self) -> pd.DataFrame:
        """Extract user features for hybrid recommendations.
        
        Returns:
            DataFrame with user features
        """
        if self.users_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        features_df = self.users_df.copy()
        
        # One-hot encode categorical features
        categorical_features = ['age_group', 'travel_style_preference', 
                               'budget_preference', 'climate_preference']
        for feature in categorical_features:
            dummies = pd.get_dummies(features_df[feature], prefix=feature)
            features_df = pd.concat([features_df, dummies], axis=1)
        
        return features_df
    
    def create_text_features(self) -> Dict[int, str]:
        """Create text descriptions for destinations.
        
        Returns:
            Dictionary mapping item_id to text description
        """
        if self.items_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        text_features = {}
        for _, row in self.items_df.iterrows():
            text = f"{row['title']} in {row['country']}. "
            text += f"Climate: {row['climate']}. "
            text += f"Activities: {row['activities']}. "
            text += f"Culture: {row['culture']}. "
            text += f"Travel style: {row['travel_style']}. "
            text += f"Budget level: {row['budget_level']}. "
            text += row['description']
            
            text_features[row['item_id']] = text
            
        return text_features


def create_train_test_split(interactions_df: pd.DataFrame, 
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create time-aware train/test split.
    
    Args:
        interactions_df: DataFrame with interactions
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    np.random.seed(random_state)
    
    # Sort by timestamp
    interactions_df = interactions_df.sort_values('timestamp')
    
    # Split chronologically
    split_idx = int(len(interactions_df) * (1 - test_size))
    train_df = interactions_df.iloc[:split_idx].copy()
    test_df = interactions_df.iloc[split_idx:].copy()
    
    logger.info(f"Train set: {len(train_df)} interactions")
    logger.info(f"Test set: {len(test_df)} interactions")
    
    return train_df, test_df


def create_negative_samples(interactions_df: pd.DataFrame, 
                          items_df: pd.DataFrame,
                          num_negatives: int = 4,
                          random_state: int = 42) -> pd.DataFrame:
    """Create negative samples for implicit feedback.
    
    Args:
        interactions_df: DataFrame with positive interactions
        items_df: DataFrame with all items
        num_negatives: Number of negative samples per positive
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with negative samples added
    """
    np.random.seed(random_state)
    
    negative_samples = []
    all_items = set(items_df['item_id'].unique())
    
    for user_id in interactions_df['user_id'].unique():
        user_items = set(interactions_df[interactions_df['user_id'] == user_id]['item_id'])
        available_items = list(all_items - user_items)
        
        if len(available_items) >= num_negatives:
            sampled_items = np.random.choice(
                available_items, 
                size=num_negatives, 
                replace=False
            )
            
            for item_id in sampled_items:
                negative_samples.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': 0,
                    'interaction_type': 'negative',
                    'timestamp': interactions_df['timestamp'].max()
                })
    
    negative_df = pd.DataFrame(negative_samples)
    combined_df = pd.concat([interactions_df, negative_df], ignore_index=True)
    
    logger.info(f"Added {len(negative_df)} negative samples")
    
    return combined_df
