"""Streamlit demo application for travel recommendation system."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
import yaml
import pickle

from src.data.loader import TravelDataLoader
from src.models.recommenders import (
    PopularityRecommender,
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender,
    MatrixFactorizationRecommender
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Travel Recommendation System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache data."""
    try:
        data_loader = TravelDataLoader("data/raw")
        interactions_df, items_df, users_df = data_loader.load_data()
        return interactions_df, items_df, users_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


@st.cache_resource
def load_models():
    """Load and cache trained models."""
    try:
        # Load configuration
        with open("configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Load data
        data_loader = TravelDataLoader("data/raw")
        interactions_df, items_df, users_df = data_loader.load_data()
        
        # Train models (in production, these would be pre-trained)
        models = {}
        
        # Train Popularity Recommender
        models['Popularity'] = PopularityRecommender(random_state=42)
        models['Popularity'].fit(interactions_df, items_df, users_df)
        
        # Train Content-Based Recommender
        models['Content-Based'] = ContentBasedRecommender(random_state=42)
        models['Content-Based'].fit(interactions_df, items_df, users_df)
        
        # Train Collaborative Filtering Recommender
        models['Collaborative Filtering'] = CollaborativeFilteringRecommender(random_state=42)
        models['Collaborative Filtering'].fit(interactions_df, items_df, users_df)
        
        # Train Hybrid Recommender
        models['Hybrid'] = HybridRecommender(random_state=42)
        models['Hybrid'].fit(interactions_df, items_df, users_df)
        
        # Train Matrix Factorization Recommender
        models['Matrix Factorization'] = MatrixFactorizationRecommender(random_state=42)
        models['Matrix Factorization'].fit(interactions_df, items_df, users_df)
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">✈️ Travel Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    interactions_df, items_df, users_df = load_data()
    if interactions_df is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Load models
    models = load_models()
    if not models:
        st.error("Failed to load models. Please train the models first.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Recommendations", "Data Overview", "Model Comparison", "Item Similarity"]
    )
    
    if page == "Recommendations":
        show_recommendations_page(models, items_df, users_df)
    elif page == "Data Overview":
        show_data_overview_page(interactions_df, items_df, users_df)
    elif page == "Model Comparison":
        show_model_comparison_page(models, interactions_df)
    elif page == "Item Similarity":
        show_item_similarity_page(models, items_df)


def show_recommendations_page(models, items_df, users_df):
    """Show the recommendations page."""
    st.header("🎯 Personalized Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("User Selection")
        
        # User selection
        user_id = st.selectbox(
            "Select a user:",
            options=sorted(users_df['user_id'].unique()),
            format_func=lambda x: f"User {x}"
        )
        
        # Model selection
        model_name = st.selectbox(
            "Select recommendation model:",
            options=list(models.keys())
        )
        
        # Number of recommendations
        top_k = st.slider("Number of recommendations:", 1, 20, 10)
        
        # User profile display
        if user_id in users_df['user_id'].values:
            user_profile = users_df[users_df['user_id'] == user_id].iloc[0]
            st.subheader("User Profile")
            st.write(f"**Age Group:** {user_profile['age_group']}")
            st.write(f"**Travel Style:** {user_profile['travel_style_preference']}")
            st.write(f"**Budget:** {user_profile['budget_preference']}")
            st.write(f"**Climate:** {user_profile['climate_preference']}")
            st.write(f"**Activities:** {user_profile['activity_preference']}")
            st.write(f"**Culture:** {user_profile['culture_preference']}")
    
    with col2:
        st.subheader("Recommendations")
        
        if st.button("Get Recommendations"):
            try:
                model = models[model_name]
                recommendations = model.recommend(user_id, top_k)
                
                st.success(f"Top {len(recommendations)} recommendations using {model_name}:")
                
                for i, item_id in enumerate(recommendations, 1):
                    item_info = items_df[items_df['item_id'] == item_id].iloc[0]
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{i}. {item_info['title']}</h4>
                            <p><strong>Country:</strong> {item_info['country']}</p>
                            <p><strong>Climate:</strong> {item_info['climate']}</p>
                            <p><strong>Activities:</strong> {item_info['activities']}</p>
                            <p><strong>Culture:</strong> {item_info['culture']}</p>
                            <p><strong>Budget Level:</strong> {item_info['budget_level']}</p>
                            <p><strong>Description:</strong> {item_info['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")


def show_data_overview_page(interactions_df, items_df, users_df):
    """Show the data overview page."""
    st.header("📊 Data Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(users_df))
    
    with col2:
        st.metric("Total Destinations", len(items_df))
    
    with col3:
        st.metric("Total Interactions", len(interactions_df))
    
    with col4:
        avg_rating = interactions_df['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        rating_counts = interactions_df['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index, 
            y=rating_counts.values,
            title="Distribution of Ratings",
            labels={'x': 'Rating', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Destinations by Popularity")
        destination_counts = interactions_df['item_id'].value_counts().head(10)
        destination_names = items_df.set_index('item_id')['title'].to_dict()
        destination_names = [destination_names.get(x, f"Item {x}") for x in destination_counts.index]
        
        fig = px.bar(
            x=destination_counts.values,
            y=destination_names,
            orientation='h',
            title="Most Popular Destinations",
            labels={'x': 'Number of Interactions', 'y': 'Destination'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data tables
    st.subheader("Sample Data")
    
    tab1, tab2, tab3 = st.tabs(["Interactions", "Destinations", "Users"])
    
    with tab1:
        st.dataframe(interactions_df.head(10))
    
    with tab2:
        st.dataframe(items_df.head(10))
    
    with tab3:
        st.dataframe(users_df.head(10))


def show_model_comparison_page(models, interactions_df):
    """Show the model comparison page."""
    st.header("⚖️ Model Comparison")
    
    st.subheader("Model Performance Metrics")
    
    # This would typically load pre-computed evaluation results
    # For demo purposes, we'll show a placeholder
    st.info("Model evaluation results would be displayed here. Run the training script to generate evaluation metrics.")
    
    # Sample comparison chart
    st.subheader("Sample Performance Comparison")
    
    # Mock data for demonstration
    metrics_data = {
        'Model': ['Popularity', 'Content-Based', 'Collaborative Filtering', 'Hybrid', 'Matrix Factorization'],
        'Precision@10': [0.15, 0.22, 0.28, 0.31, 0.29],
        'Recall@10': [0.12, 0.18, 0.24, 0.27, 0.25],
        'NDCG@10': [0.20, 0.25, 0.32, 0.35, 0.33]
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Create comparison chart
    fig = go.Figure()
    
    for metric in ['Precision@10', 'Recall@10', 'NDCG@10']:
        fig.add_trace(go.Bar(
            name=metric,
            x=df['Model'],
            y=df[metric]
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model details
    st.subheader("Model Descriptions")
    
    model_descriptions = {
        'Popularity': "Recommends the most popular destinations based on interaction frequency.",
        'Content-Based': "Uses destination features and user preferences to find similar items.",
        'Collaborative Filtering': "Finds users with similar preferences and recommends items they liked.",
        'Hybrid': "Combines multiple approaches for improved recommendation quality.",
        'Matrix Factorization': "Learns latent factors to model user-item interactions."
    }
    
    for model_name, description in model_descriptions.items():
        st.write(f"**{model_name}:** {description}")


def show_item_similarity_page(models, items_df):
    """Show the item similarity page."""
    st.header("🔍 Item Similarity Explorer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Destination")
        
        # Destination selection
        destination = st.selectbox(
            "Choose a destination:",
            options=items_df['title'].tolist()
        )
        
        # Number of similar items
        num_similar = st.slider("Number of similar destinations:", 1, 10, 5)
        
        # Model for similarity
        similarity_model = st.selectbox(
            "Similarity model:",
            options=['Content-Based', 'Collaborative Filtering']
        )
    
    with col2:
        st.subheader("Similar Destinations")
        
        if st.button("Find Similar Destinations"):
            try:
                # Get destination ID
                dest_id = items_df[items_df['title'] == destination]['item_id'].iloc[0]
                
                # Get similar destinations (simplified - in practice, you'd implement similarity)
                similar_items = items_df[items_df['item_id'] != dest_id].head(num_similar)
                
                st.success(f"Destinations similar to {destination}:")
                
                for _, item in similar_items.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{item['title']}</h4>
                            <p><strong>Country:</strong> {item['country']}</p>
                            <p><strong>Climate:</strong> {item['climate']}</p>
                            <p><strong>Activities:</strong> {item['activities']}</p>
                            <p><strong>Culture:</strong> {item['culture']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error finding similar destinations: {e}")


if __name__ == "__main__":
    main()
