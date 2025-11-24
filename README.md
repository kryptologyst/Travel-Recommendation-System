# Travel Recommendation System

A comprehensive travel recommendation system that uses multiple recommendation approaches to suggest personalized travel destinations to users based on their preferences, interaction history, and destination features.

## Features

- **Multiple Recommendation Models**: Popularity-based, Content-based, Collaborative Filtering, Hybrid, and Matrix Factorization
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate, Coverage, and Diversity metrics
- **Interactive Demo**: Streamlit web application for exploring recommendations
- **Realistic Data**: Generated travel destination data with user interactions
- **Production-Ready Structure**: Clean code with type hints, docstrings, and proper organization

## Project Structure

```
travel_recommendation_system/
├── src/
│   ├── data/
│   │   └── loader.py              # Data loading and preprocessing
│   ├── models/
│   │   └── recommenders.py       # Recommendation models
│   ├── evaluation/
│   │   └── metrics.py            # Evaluation metrics
│   └── utils/
├── data/
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data and results
├── configs/
│   └── config.yaml              # Configuration file
├── scripts/
│   └── train_evaluate.py        # Training and evaluation script
├── tests/
│   └── test_recommendation_system.py
├── notebooks/                   # Jupyter notebooks for analysis
├── assets/                      # Images and other assets
├── requirements.txt
├── .gitignore
├── demo.py                      # Streamlit demo application
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Travel-Recommendation-System.git
cd Travel-Recommendation-System

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

The project includes realistic travel data in the `data/raw/` directory:

- `interactions.csv`: User-item interactions with ratings and timestamps
- `items.csv`: Destination features (climate, activities, culture, etc.)
- `users.csv`: User profiles and preferences

### 3. Training and Evaluation

```bash
# Train all models and evaluate performance
python scripts/train_evaluate.py
```

This will:
- Load and preprocess the data
- Train all recommendation models
- Evaluate models using multiple metrics
- Save results to `data/processed/evaluation_results.csv`

### 4. Interactive Demo

```bash
# Launch the Streamlit demo
streamlit run demo.py
```

The demo provides:
- Personalized recommendations for selected users
- Data overview and visualizations
- Model comparison charts
- Item similarity exploration

## Models

### 1. Popularity Recommender
Recommends the most popular destinations based on interaction frequency.

### 2. Content-Based Recommender
Uses TF-IDF vectorization of destination features to find similar items based on user preferences.

### 3. Collaborative Filtering Recommender
Finds users with similar preferences using cosine similarity and recommends items they liked.

### 4. Hybrid Recommender
Combines content-based, collaborative filtering, and popularity approaches with weighted scores.

### 5. Matrix Factorization Recommender
Uses Alternating Least Squares (ALS) to learn latent factors for user-item interactions.

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **Coverage**: Fraction of catalog items that are recommended
- **Diversity**: Measure of recommendation variety
- **Popularity Bias**: Average popularity of recommended items

## Configuration

Edit `configs/config.yaml` to customize:

- Data paths and parameters
- Model hyperparameters
- Evaluation settings
- Demo configuration

## Data Schema

### Interactions (`interactions.csv`)
- `user_id`: User identifier
- `item_id`: Destination identifier
- `timestamp`: Interaction timestamp
- `rating`: User rating (1-5)
- `interaction_type`: Type of interaction (view, bookmark, etc.)

### Items (`items.csv`)
- `item_id`: Destination identifier
- `title`: Destination name
- `description`: Detailed description
- `country`: Country location
- `climate`: Climate type
- `activities`: Available activities
- `culture`: Cultural characteristics
- `budget_level`: Budget requirement (Low/Medium/High)
- `travel_style`: Travel style category
- `rating_avg`: Average rating
- `popularity_score`: Popularity score

### Users (`users.csv`)
- `user_id`: User identifier
- `age_group`: Age group category
- `travel_style_preference`: Preferred travel style
- `budget_preference`: Budget preference
- `climate_preference`: Preferred climate
- `activity_preference`: Preferred activities
- `culture_preference`: Preferred culture type
- `travel_experience`: Travel experience level

## Usage Examples

### Basic Recommendation

```python
from src.data.loader import TravelDataLoader
from src.models.recommenders import HybridRecommender

# Load data
loader = TravelDataLoader("data/raw")
interactions_df, items_df, users_df = loader.load_data()

# Train model
model = HybridRecommender()
model.fit(interactions_df, items_df, users_df)

# Get recommendations
recommendations = model.recommend(user_id=1, top_k=5)
print(f"Recommendations: {recommendations}")
```

### Model Evaluation

```python
from src.evaluation.metrics import evaluate_model

# Evaluate model
metrics = evaluate_model(model, test_data, k_values=[5, 10, 20])
print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

### Code Quality

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pytest** for testing

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## API Reference

### DataLoader

```python
class TravelDataLoader:
    def __init__(self, data_dir: str = "data/raw")
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    def get_user_item_matrix(self) -> pd.DataFrame
    def get_item_features(self) -> pd.DataFrame
    def get_user_features(self) -> pd.DataFrame
    def create_text_features(self) -> Dict[int, str]
```

### Recommendation Models

All models inherit from `BaseRecommender` and implement:

```python
def fit(self, interactions_df: pd.DataFrame, 
        items_df: pd.DataFrame, 
        users_df: Optional[pd.DataFrame] = None) -> None

def recommend(self, user_id: int, top_k: int = 10) -> List[int]
```

### Evaluation Metrics

```python
class RecommendationMetrics:
    @staticmethod
    def precision_at_k(y_true: List[int], y_pred: List[int], k: int) -> float
    @staticmethod
    def recall_at_k(y_true: List[int], y_pred: List[int], k: int) -> float
    @staticmethod
    def ndcg_at_k(y_true: List[int], y_pred: List[int], k: int) -> float
    @staticmethod
    def map_at_k(y_true: List[int], y_pred: List[int], k: int) -> float
    @staticmethod
    def hit_rate_at_k(y_true: List[int], y_pred: List[int], k: int) -> float
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with scikit-learn, pandas, numpy, and streamlit
- Inspired by modern recommendation system research
- Uses realistic travel data for demonstration purposes
# Travel-Recommendation-System
