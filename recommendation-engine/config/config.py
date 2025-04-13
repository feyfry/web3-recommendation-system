"""
Configuration for Web3 recommendation system.
"""

import os

# API Configuration
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY = "CG-Tofppzfh4tBV1JWpW5bSpJjK"  # Demo API key

# Data Collection Configuration
TOP_COINS_LIMIT = 250  # Number of top coins to collect
CATEGORIES = [
    "layer-1",
    "smart-contract-platform", 
    "decentralized-finance-defi",
    "non-fungible-tokens-nft",
    "gaming",
    "meme-token",
    "stablecoins",
    "metaverse",
    "real-world-assets-rwa"
]

# Path Configuration
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
MODELS_PATH = os.path.join(PROCESSED_DATA_PATH, "models")
LOGS_PATH = "logs"

# Ensure all directories exist
for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)

# Recommendation Engine Configuration
SIMILARITY_THRESHOLD = 0.001  # Reduced from 0.1
NUM_RECOMMENDATIONS = 10
USER_BASED_WEIGHT = 0.1  # Reduced from 0.5
ITEM_BASED_WEIGHT = 0.1  # Reduced from 0.5
POPULARITY_WEIGHT = 0.3
TREND_WEIGHT = 0.3
FEATURE_WEIGHT = 0.8    # Increased from 0.6

# Model Feature Weights
MARKET_CAP_WEIGHT = 0.25
VOLUME_WEIGHT = 0.15
PRICE_CHANGE_WEIGHT = 0.20
SOCIAL_SCORE_WEIGHT = 0.15
DEVELOPER_SCORE_WEIGHT = 0.10
CATEGORY_SIMILARITY_WEIGHT = 0.25  # Increased from 0.15

# Database Configuration
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "web3_recommender"
DB_USER = "postgres"
DB_PASSWORD = ""

# Feature Engineering Settings
CATEGORICAL_FEATURES = ["chain", "category"]
NUMERICAL_FEATURES = [
    "market_cap", 
    "volume_24h", 
    "price_change_24h", 
    "price_change_7d",
    "reddit_subscribers", 
    "twitter_followers",
    "github_stars"
]

# Neural Collaborative Filtering (NCF) Configuration
NCF_EMBEDDING_SIZE = 64     # Increased from 32
NCF_LAYERS = [128, 64, 32, 16]
NCF_LEARNING_RATE = 0.0005
NCF_BATCH_SIZE = 128
NCF_NUM_EPOCHS = 10          # Full training epochs
NCF_QUICK_EPOCHS = 5         # Quick training for runtime usage 
NCF_EVAL_EPOCHS = 15         # Detailed evaluation
NCF_VALIDATION_RATIO = 0.2
NCF_NEGATIVE_RATIO = 4  # Number of negative samples per positive sample

# Cold Start Configuration
COLD_START_USERS = 20  # Modified
COLD_START_INTERACTIONS = 5  # Unchanged

# Evaluation Configuration
EVALUATION_SPLIT = 0.2  # Reduced from 0.3
EVALUATION_RANDOM_SEED = 42
EVALUATION_K_VALUES = [5, 10, 20]  # For precision@k, recall@k, etc.

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = False
API_SECRET_KEY = "your-secret-api-key"  # Use environment variables in production

# Caching Configuration
CACHE_ENABLED = True
CACHE_TTL = 3600  # Time-to-live in seconds (1 hour)
RECOMMENDATION_CACHE_TTL = 43200  # 12 hours

# Update Configuration
UPDATE_INTERVAL = 24  # Hours between data updates
TRENDING_UPDATE_INTERVAL = 6  # Hours between trending data updates

# Performance Configuration
MAX_THREADS = 4
REQUEST_TIMEOUT = 30  # Seconds
RETRY_COUNT = 3
RATE_LIMIT_DELAY = 2  # Seconds between API requests