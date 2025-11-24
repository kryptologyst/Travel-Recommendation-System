"""Utility functions for travel recommendation system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across libraries.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_model(model: Any, filepath: str) -> None:
    """Save a trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Path to save the model
    """
    try:
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def load_model(filepath: str) -> Any:
    """Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    try:
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def create_user_profile(user_id: int, interactions_df: pd.DataFrame, 
                       items_df: pd.DataFrame) -> Dict[str, Any]:
    """Create a detailed user profile from interaction history.
    
    Args:
        user_id: User identifier
        interactions_df: User interactions data
        items_df: Items data
        
    Returns:
        Dictionary containing user profile information
    """
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    
    if len(user_interactions) == 0:
        return {"user_id": user_id, "interaction_count": 0}
    
    # Get interacted items
    interacted_items = user_interactions['item_id'].tolist()
    item_details = items_df[items_df['item_id'].isin(interacted_items)]
    
    # Calculate preferences
    preferences = {
        "user_id": user_id,
        "interaction_count": len(user_interactions),
        "avg_rating": user_interactions['rating'].mean(),
        "preferred_climates": item_details['climate'].value_counts().to_dict(),
        "preferred_activities": item_details['activities'].value_counts().to_dict(),
        "preferred_cultures": item_details['culture'].value_counts().to_dict(),
        "preferred_budget_levels": item_details['budget_level'].value_counts().to_dict(),
        "preferred_travel_styles": item_details['travel_style'].value_counts().to_dict(),
        "interacted_countries": item_details['country'].unique().tolist(),
        "total_countries": len(item_details['country'].unique())
    }
    
    return preferences


def calculate_item_popularity(interactions_df: pd.DataFrame) -> Dict[int, float]:
    """Calculate popularity scores for all items.
    
    Args:
        interactions_df: User interactions data
        
    Returns:
        Dictionary mapping item_id to popularity score
    """
    item_stats = interactions_df.groupby('item_id').agg({
        'rating': ['count', 'mean'],
        'user_id': 'nunique'
    }).round(4)
    
    item_stats.columns = ['interaction_count', 'avg_rating', 'unique_users']
    
    # Normalize popularity score (0-1)
    max_interactions = item_stats['interaction_count'].max()
    max_users = item_stats['unique_users'].max()
    
    item_stats['popularity_score'] = (
        0.6 * (item_stats['interaction_count'] / max_interactions) +
        0.4 * (item_stats['unique_users'] / max_users)
    )
    
    return item_stats['popularity_score'].to_dict()


def get_recommendation_explanation(user_id: int, item_id: int, 
                                 model_name: str, 
                                 user_profile: Optional[Dict] = None,
                                 item_features: Optional[Dict] = None) -> str:
    """Generate explanation for a recommendation.
    
    Args:
        user_id: User identifier
        item_id: Recommended item identifier
        model_name: Name of the recommendation model
        user_profile: Optional user profile information
        item_features: Optional item features
        
    Returns:
        Explanation string
    """
    explanations = {
        "Popularity": f"This destination is popular among travelers.",
        "Content-Based": f"This destination matches your preferences for {item_features.get('climate', 'travel experiences')}.",
        "Collaborative Filtering": f"Users with similar preferences to you have enjoyed this destination.",
        "Hybrid": f"This destination combines popularity, content similarity, and collaborative signals.",
        "Matrix Factorization": f"Our algorithm found this destination matches your latent preferences."
    }
    
    base_explanation = explanations.get(model_name, "This destination was recommended based on your preferences.")
    
    if user_profile and item_features:
        # Add specific details if available
        if 'preferred_climates' in user_profile:
            preferred_climate = max(user_profile['preferred_climates'], key=user_profile['preferred_climates'].get)
            if item_features.get('climate') == preferred_climate:
                base_explanation += f" It matches your preferred climate: {preferred_climate}."
    
    return base_explanation


def validate_data_quality(interactions_df: pd.DataFrame, 
                         items_df: pd.DataFrame, 
                         users_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and return statistics.
    
    Args:
        interactions_df: User interactions data
        items_df: Items data
        users_df: Users data
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_metrics = {
        "interactions": {
            "total_count": len(interactions_df),
            "unique_users": interactions_df['user_id'].nunique(),
            "unique_items": interactions_df['item_id'].nunique(),
            "rating_range": (interactions_df['rating'].min(), interactions_df['rating'].max()),
            "avg_rating": interactions_df['rating'].mean(),
            "missing_values": interactions_df.isnull().sum().to_dict()
        },
        "items": {
            "total_count": len(items_df),
            "missing_values": items_df.isnull().sum().to_dict(),
            "unique_countries": items_df['country'].nunique(),
            "climate_types": items_df['climate'].nunique(),
            "budget_levels": items_df['budget_level'].nunique()
        },
        "users": {
            "total_count": len(users_df),
            "missing_values": users_df.isnull().sum().to_dict(),
            "age_groups": users_df['age_group'].nunique(),
            "travel_styles": users_df['travel_style_preference'].nunique()
        }
    }
    
    return quality_metrics


def create_interaction_summary(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary of user-item interactions.
    
    Args:
        interactions_df: User interactions data
        
    Returns:
        DataFrame with interaction summary
    """
    summary = interactions_df.groupby(['user_id', 'item_id']).agg({
        'rating': ['count', 'mean', 'max'],
        'timestamp': ['min', 'max']
    }).round(4)
    
    summary.columns = ['interaction_count', 'avg_rating', 'max_rating', 'first_interaction', 'last_interaction']
    summary = summary.reset_index()
    
    return summary


def export_results(results_df: pd.DataFrame, 
                  output_dir: str = "data/processed",
                  format: str = "csv") -> None:
    """Export evaluation results in various formats.
    
    Args:
        results_df: Results DataFrame
        output_dir: Output directory
        format: Export format (csv, json, excel)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "csv":
        filepath = output_path / "evaluation_results.csv"
        results_df.to_csv(filepath, index=False)
    elif format.lower() == "json":
        filepath = output_path / "evaluation_results.json"
        results_df.to_json(filepath, orient='records', indent=2)
    elif format.lower() == "excel":
        filepath = output_path / "evaluation_results.xlsx"
        results_df.to_excel(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results exported to {filepath}")


def create_model_comparison_chart(results_df: pd.DataFrame, 
                                metric: str = "ndcg@10") -> None:
    """Create a comparison chart for model performance.
    
    Args:
        results_df: Results DataFrame
        metric: Metric to visualize
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='model', y=metric)
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = Path("data/processed")
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / f"model_comparison_{metric}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison chart saved for {metric}")
        
    except ImportError:
        logger.warning("Matplotlib/Seaborn not available for chart creation")
    except Exception as e:
        logger.error(f"Failed to create comparison chart: {e}")
