"""Evaluation metrics for travel recommendation system."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """Class for computing recommendation evaluation metrics."""
    
    @staticmethod
    def precision_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
        """Compute Precision@K.
        
        Args:
            y_true: List of true item IDs
            y_pred: List of predicted item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
            
        y_pred_k = y_pred[:k]
        if len(y_pred_k) == 0:
            return 0.0
            
        hits = len(set(y_true) & set(y_pred_k))
        return hits / len(y_pred_k)
    
    @staticmethod
    def recall_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
        """Compute Recall@K.
        
        Args:
            y_true: List of true item IDs
            y_pred: List of predicted item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(y_true) == 0:
            return 0.0
            
        y_pred_k = y_pred[:k]
        hits = len(set(y_true) & set(y_pred_k))
        return hits / len(y_true)
    
    @staticmethod
    def ndcg_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
        """Compute NDCG@K.
        
        Args:
            y_true: List of true item IDs
            y_pred: List of predicted item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        if k == 0 or len(y_true) == 0:
            return 0.0
            
        y_pred_k = y_pred[:k]
        
        # Compute DCG
        dcg = 0.0
        for i, item in enumerate(y_pred_k):
            if item in y_true:
                dcg += 1.0 / np.log2(i + 2)
        
        # Compute IDCG
        idcg = 0.0
        for i in range(min(len(y_true), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def map_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
        """Compute MAP@K.
        
        Args:
            y_true: List of true item IDs
            y_pred: List of predicted item IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score
        """
        if k == 0 or len(y_true) == 0:
            return 0.0
            
        y_pred_k = y_pred[:k]
        hits = 0
        precision_sum = 0.0
        
        for i, item in enumerate(y_pred_k):
            if item in y_true:
                hits += 1
                precision_sum += hits / (i + 1)
        
        return precision_sum / len(y_true) if len(y_true) > 0 else 0.0
    
    @staticmethod
    def hit_rate_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
        """Compute Hit Rate@K.
        
        Args:
            y_true: List of true item IDs
            y_pred: List of predicted item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K score
        """
        if k == 0:
            return 0.0
            
        y_pred_k = y_pred[:k]
        hits = len(set(y_true) & set(y_pred_k))
        return 1.0 if hits > 0 else 0.0
    
    @staticmethod
    def coverage(predictions: Dict[int, List[int]], 
                all_items: List[int]) -> float:
        """Compute catalog coverage.
        
        Args:
            predictions: Dictionary mapping user_id to recommended items
            all_items: List of all available items
            
        Returns:
            Coverage score
        """
        if not predictions:
            return 0.0
            
        recommended_items = set()
        for user_recs in predictions.values():
            recommended_items.update(user_recs)
        
        return len(recommended_items) / len(all_items)
    
    @staticmethod
    def diversity(predictions: Dict[int, List[int]], 
                 item_features: Optional[pd.DataFrame] = None) -> float:
        """Compute recommendation diversity.
        
        Args:
            predictions: Dictionary mapping user_id to recommended items
            item_features: Optional DataFrame with item features
            
        Returns:
            Diversity score
        """
        if not predictions:
            return 0.0
            
        total_diversity = 0.0
        count = 0
        
        for user_recs in predictions.values():
            if len(user_recs) < 2:
                continue
                
            # Simple diversity: count unique items
            unique_items = len(set(user_recs))
            diversity_score = unique_items / len(user_recs)
            total_diversity += diversity_score
            count += 1
        
        return total_diversity / count if count > 0 else 0.0
    
    @staticmethod
    def popularity_bias(predictions: Dict[int, List[int]], 
                       item_popularity: Dict[int, float]) -> float:
        """Compute popularity bias in recommendations.
        
        Args:
            predictions: Dictionary mapping user_id to recommended items
            item_popularity: Dictionary mapping item_id to popularity score
            
        Returns:
            Average popularity of recommended items
        """
        if not predictions:
            return 0.0
            
        total_popularity = 0.0
        total_items = 0
        
        for user_recs in predictions.values():
            for item in user_recs:
                if item in item_popularity:
                    total_popularity += item_popularity[item]
                    total_items += 1
        
        return total_popularity / total_items if total_items > 0 else 0.0


def evaluate_model(model, test_data: pd.DataFrame, 
                 k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """Evaluate a recommendation model.
    
    Args:
        model: Trained recommendation model
        test_data: Test dataset
        k_values: List of K values for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    # Group test data by user
    user_test_items = test_data.groupby('user_id')['item_id'].apply(list).to_dict()
    
    # Get predictions for each user
    predictions = {}
    for user_id in user_test_items.keys():
        try:
            user_preds = model.recommend(user_id, top_k=max(k_values))
            predictions[user_id] = user_preds
        except Exception as e:
            logger.warning(f"Failed to get predictions for user {user_id}: {e}")
            predictions[user_id] = []
    
    # Compute metrics for each K
    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []
        maps = []
        hit_rates = []
        
        for user_id, true_items in user_test_items.items():
            if user_id in predictions:
                pred_items = predictions[user_id]
                
                precisions.append(RecommendationMetrics.precision_at_k(true_items, pred_items, k))
                recalls.append(RecommendationMetrics.recall_at_k(true_items, pred_items, k))
                ndcgs.append(RecommendationMetrics.ndcg_at_k(true_items, pred_items, k))
                maps.append(RecommendationMetrics.map_at_k(true_items, pred_items, k))
                hit_rates.append(RecommendationMetrics.hit_rate_at_k(true_items, pred_items, k))
        
        metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
        metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
        metrics[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
        metrics[f'map@{k}'] = np.mean(maps) if maps else 0.0
        metrics[f'hit_rate@{k}'] = np.mean(hit_rates) if hit_rates else 0.0
    
    return metrics


def create_evaluation_report(models: Dict[str, object], 
                           test_data: pd.DataFrame,
                           k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Create a comprehensive evaluation report comparing multiple models.
    
    Args:
        models: Dictionary mapping model names to model objects
        test_data: Test dataset
        k_values: List of K values for evaluation
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, test_data, k_values)
        metrics['model'] = model_name
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Sort by NDCG@10 (or another primary metric)
    if 'ndcg@10' in results_df.columns:
        results_df = results_df.sort_values('ndcg@10', ascending=False)
    
    return results_df
