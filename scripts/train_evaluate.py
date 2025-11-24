"""Main training and evaluation script for travel recommendation system."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import yaml
from typing import Dict, Any

from src.data.loader import TravelDataLoader, create_train_test_split
from src.models.recommenders import (
    PopularityRecommender,
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender,
    MatrixFactorizationRecommender
)
from src.evaluation.metrics import create_evaluation_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
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


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'data': {
                'data_dir': 'data/raw',
                'test_size': 0.2,
                'random_state': 42
            },
            'models': {
                'popularity': {'random_state': 42},
                'content_based': {'random_state': 42},
                'collaborative_filtering': {'n_neighbors': 5, 'random_state': 42},
                'hybrid': {
                    'content_weight': 0.4,
                    'collab_weight': 0.4,
                    'popularity_weight': 0.2,
                    'random_state': 42
                },
                'matrix_factorization': {
                    'n_factors': 50,
                    'n_iterations': 20,
                    'regularization': 0.01,
                    'random_state': 42
                }
            },
            'evaluation': {
                'k_values': [5, 10, 20]
            }
        }
    
    return config


def train_models(config: Dict[str, Any], 
                train_data: pd.DataFrame,
                items_df: pd.DataFrame,
                users_df: pd.DataFrame) -> Dict[str, Any]:
    """Train all recommendation models.
    
    Args:
        config: Configuration dictionary
        train_data: Training data
        items_df: Items data
        users_df: Users data
        
    Returns:
        Dictionary of trained models
    """
    logger.info("Training recommendation models...")
    
    models = {}
    model_configs = config['models']
    
    # Train Popularity Recommender
    logger.info("Training Popularity Recommender...")
    models['Popularity'] = PopularityRecommender(**model_configs['popularity'])
    models['Popularity'].fit(train_data, items_df, users_df)
    
    # Train Content-Based Recommender
    logger.info("Training Content-Based Recommender...")
    models['Content-Based'] = ContentBasedRecommender(**model_configs['content_based'])
    models['Content-Based'].fit(train_data, items_df, users_df)
    
    # Train Collaborative Filtering Recommender
    logger.info("Training Collaborative Filtering Recommender...")
    models['Collaborative Filtering'] = CollaborativeFilteringRecommender(**model_configs['collaborative_filtering'])
    models['Collaborative Filtering'].fit(train_data, items_df, users_df)
    
    # Train Hybrid Recommender
    logger.info("Training Hybrid Recommender...")
    models['Hybrid'] = HybridRecommender(**model_configs['hybrid'])
    models['Hybrid'].fit(train_data, items_df, users_df)
    
    # Train Matrix Factorization Recommender
    logger.info("Training Matrix Factorization Recommender...")
    models['Matrix Factorization'] = MatrixFactorizationRecommender(**model_configs['matrix_factorization'])
    models['Matrix Factorization'].fit(train_data, items_df, users_df)
    
    logger.info("All models trained successfully!")
    return models


def evaluate_models(models: Dict[str, Any], 
                   test_data: pd.DataFrame,
                   config: Dict[str, Any]) -> pd.DataFrame:
    """Evaluate all trained models.
    
    Args:
        models: Dictionary of trained models
        test_data: Test data
        config: Configuration dictionary
        
    Returns:
        Evaluation results DataFrame
    """
    logger.info("Evaluating models...")
    
    k_values = config['evaluation']['k_values']
    results_df = create_evaluation_report(models, test_data, k_values)
    
    logger.info("Evaluation completed!")
    return results_df


def save_results(results_df: pd.DataFrame, 
                output_dir: str = "data/processed") -> None:
    """Save evaluation results.
    
    Args:
        results_df: Results DataFrame
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "evaluation_results.csv"
    results_df.to_csv(results_file, index=False)
    
    logger.info(f"Results saved to {results_file}")


def main():
    """Main training and evaluation pipeline."""
    logger.info("Starting Travel Recommendation System Training...")
    
    # Load configuration
    config = load_config()
    
    # Set random seeds
    set_random_seeds(config['data']['random_state'])
    
    # Load data
    logger.info("Loading data...")
    data_loader = TravelDataLoader(config['data']['data_dir'])
    interactions_df, items_df, users_df = data_loader.load_data()
    
    # Create train/test split
    logger.info("Creating train/test split...")
    train_data, test_data = create_train_test_split(
        interactions_df, 
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # Train models
    models = train_models(config, train_data, items_df, users_df)
    
    # Evaluate models
    results_df = evaluate_models(models, test_data, config)
    
    # Print results
    logger.info("Evaluation Results:")
    print("\n" + "="*80)
    print("TRAVEL RECOMMENDATION SYSTEM - MODEL COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False, float_format='%.4f'))
    print("="*80)
    
    # Save results
    save_results(results_df)
    
    # Generate sample recommendations
    logger.info("Generating sample recommendations...")
    sample_user_id = 1
    print(f"\nSample Recommendations for User {sample_user_id}:")
    print("-" * 50)
    
    for model_name, model in models.items():
        try:
            recommendations = model.recommend(sample_user_id, top_k=5)
            print(f"{model_name}: {recommendations}")
        except Exception as e:
            logger.warning(f"Failed to generate recommendations for {model_name}: {e}")
    
    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
