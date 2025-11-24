"""Travel recommendation system package."""

__version__ = "1.0.0"
__author__ = "AI Projects"
__email__ = "ai@example.com"

from .data.loader import TravelDataLoader
from .models.recommenders import (
    PopularityRecommender,
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender,
    MatrixFactorizationRecommender
)
from .evaluation.metrics import RecommendationMetrics

__all__ = [
    "TravelDataLoader",
    "PopularityRecommender",
    "ContentBasedRecommender", 
    "CollaborativeFilteringRecommender",
    "HybridRecommender",
    "MatrixFactorizationRecommender",
    "RecommendationMetrics"
]
