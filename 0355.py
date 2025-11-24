"""
Project 355: Travel Recommendation System - Modernized Example

This is a simple demonstration of the modernized travel recommendation system.
For the full implementation, see the src/ directory and run:

    python scripts/train_evaluate.py  # Train and evaluate all models
    streamlit run demo.py             # Launch interactive demo

This example shows the basic content-based filtering approach that was
modernized into a comprehensive recommendation system.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def simple_travel_recommendation(destinations: List[str], 
                               destination_features: List[str],
                               user_profile: str, 
                               top_n: int = 3) -> List[str]:
    """
    Simple content-based travel recommendation using TF-IDF and cosine similarity.
    
    Args:
        destinations: List of destination names
        destination_features: List of destination feature descriptions
        user_profile: User preference description
        top_n: Number of recommendations to return
        
    Returns:
        List of recommended destination names
    """
    logger.info("Computing recommendations using content-based filtering...")
    
    # Use TF-IDF to convert destination features and user profile into numerical features
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(destination_features + [user_profile])
    
    # Compute cosine similarity between user profile and destination features
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get indices of most similar destinations
    similar_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    recommended_destinations = [destinations[i] for i in similar_indices]
    
    logger.info(f"Generated {len(recommended_destinations)} recommendations")
    return recommended_destinations


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("TRAVEL RECOMMENDATION SYSTEM - SIMPLE DEMO")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # 1. Define travel destinations and their features
    destinations = ['Paris', 'New York', 'Tokyo', 'Sydney', 'Barcelona']
    destination_features = [
        "Romantic city with iconic landmarks, museums, and fine dining experiences.",
        "Vibrant city with famous landmarks, diverse culture, and bustling nightlife.",
        "A city with cutting-edge technology, rich culture, and unique culinary experiences.",
        "A city with beautiful beaches, iconic landmarks, and a laid-back lifestyle.",
        "A city known for its architecture, history, beaches, and vibrant nightlife."
    ]
    
    # 2. Define user preferences
    user_profile = "I enjoy cultural experiences, exploring museums, and experiencing unique cuisines."
    
    print(f"User Profile: {user_profile}")
    print(f"Available Destinations: {', '.join(destinations)}")
    print()
    
    # 3. Generate recommendations
    recommended_destinations = simple_travel_recommendation(
        destinations, destination_features, user_profile, top_n=3
    )
    
    # 4. Display results
    print("RECOMMENDATIONS:")
    print("-" * 30)
    for i, destination in enumerate(recommended_destinations, 1):
        print(f"{i}. {destination}")
    
    print()
    print("=" * 60)
    print("For the full system with multiple models, evaluation metrics,")
    print("and interactive demo, see the src/ directory and README.md")
    print("=" * 60)


if __name__ == "__main__":
    main()