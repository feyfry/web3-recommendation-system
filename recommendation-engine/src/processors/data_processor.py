"""
Enhanced module for data processing and feature engineering
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
import sys
import traceback
import csv
from typing import Dict, List, Tuple, Optional, Union, Any
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from functools import partial

# Add root path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import (
    RAW_DATA_PATH, 
    PROCESSED_DATA_PATH, 
    CATEGORICAL_FEATURES, 
    NUMERICAL_FEATURES
)

# Setup logging
from central_logging import get_logger
logger = get_logger(__name__)

from src.utils.data_utils import DateTimeEncoder, safe_convert


class DataProcessor:
    """
    Enhanced class for processing and feature engineering on cryptocurrency data
    """
    
    def __init__(self, debug_mode=False):
        """
        Initialize processor with enhanced options
        
        Args:
            debug_mode (bool): Enable detailed debugging
        """
        # Create directory for processed data if it doesn't exist
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        
        # Initialize scalers - use RobustScaler for better handling of outliers
        self.num_scaler = RobustScaler()
        self.market_cap_scaler = MinMaxScaler()  # Separate scaler for market cap
        self.volume_scaler = MinMaxScaler()      # Separate scaler for volume
        self.social_scaler = MinMaxScaler()      # Separate scaler for social metrics
        
        self.categorical_encoders = {}
        self.debug_mode = debug_mode or os.environ.get('DEBUG_MODE', 'false').lower() == 'true'
        
        # Define blockchain platforms with aliases for better matching
        self.blockchain_platforms = {
            'ethereum': ['eth', 'erc20', 'erc-20', 'erc721', 'erc-721'],
            'binance-smart-chain': ['bnb', 'bsc', 'bep20', 'bep-20'],
            'solana': ['sol'],
            'polygon-pos': ['polygon', 'matic'],
            'avalanche': ['avax'],
            'tron': ['trx'],
            'fantom': ['ftm'],
            'arbitrum': ['arb', 'arbitrum-one'],
            'optimism': ['op'],
            'cosmos': ['atom', 'cosmos-hub'],
            'polkadot': ['dot'],
            'cardano': ['ada']
        }
        
        # Define category mappings for normalization
        self.category_mappings = {
            'defi': ['defi', 'decentralized-finance', 'decentralized-finance-defi', 'yield-farming', 'lending', 'dex'],
            'nft': ['nft', 'non-fungible-tokens', 'non-fungible-tokens-nft', 'collectibles'],
            'layer-1': ['layer-1', 'layer1', 'l1', 'blockchain-service', 'smart-contract-platform'],
            'layer-2': ['layer-2', 'layer2', 'l2', 'scaling'],
            'gaming': ['gaming', 'play-to-earn', 'p2e', 'game', 'metaverse', 'gaming-guild'],
            'privacy': ['privacy', 'anonymity', 'zero-knowledge'],
            'stablecoin': ['stablecoin', 'stablecoin-algorithmically-stabilized', 'stablecoin-asset-backed'],
            'meme': ['meme', 'meme-token', 'dog', 'inu', 'cat', 'food'],
            'dao': ['dao', 'governance'],
            'exchange': ['exchange', 'exchange-token', 'exchange-based', 'centralized-exchange'],
            'infrastructure': ['infrastructure', 'oracle', 'bridge', 'interoperability'],
            'real-world-assets': ['real-world-assets', 'rwa', 'tokenized-asset', 'tokenization']
        }
        
        # Additional parameters for more realistic synthetic data
        self.user_persona_weights = {
            'defi_enthusiast': {
                'categories': {'defi': 0.6, 'layer-1': 0.2, 'stablecoin': 0.2},
                'chains': {'ethereum': 0.5, 'avalanche': 0.2, 'solana': 0.2, 'polygon-pos': 0.1}
            },
            'nft_collector': {
                'categories': {'nft': 0.7, 'gaming': 0.2, 'metaverse': 0.1},
                'chains': {'ethereum': 0.4, 'solana': 0.3, 'polygon-pos': 0.3}
            },
            'trader': {
                'categories': {'exchange': 0.4, 'stablecoin': 0.3, 'defi': 0.3},
                'chains': {'binance-smart-chain': 0.4, 'ethereum': 0.3, 'solana': 0.3}
            },
            'gamer': {
                'categories': {'gaming': 0.6, 'nft': 0.2, 'metaverse': 0.2},
                'chains': {'solana': 0.4, 'polygon-pos': 0.3, 'binance-smart-chain': 0.2, 'ethereum': 0.1}
            },
            'meme_investor': {
                'categories': {'meme': 0.7, 'dog': 0.2, 'cat': 0.1},
                'chains': {'ethereum': 0.4, 'solana': 0.3, 'binance-smart-chain': 0.3}
            }
        }
    
    def load_latest_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load the latest raw data with enhanced error handling
        
        Returns:
            tuple: (projects_df, categories_df, trending_df)
        """
        logger.info("Loading latest raw data")
        
        try:
            # Check directory contents for debugging
            logger.info(f"RAW_DATA_PATH: {os.path.abspath(RAW_DATA_PATH)}")
            logger.info(f"Current working directory: {os.getcwd()}")
            
            raw_files = os.listdir(RAW_DATA_PATH)
            logger.info(f"Found {len(raw_files)} files in RAW_DATA_PATH")
            
            # Find web3_projects CSV files
            projects_files = [f for f in raw_files if f.startswith("web3_projects_") and f.endswith(".csv")]
            logger.info(f"Found {len(projects_files)} project CSV files")
            
            projects_df = None
            
            # Try loading from CSV files
            if projects_files:
                latest_projects_file = max(projects_files)
                projects_path = os.path.join(RAW_DATA_PATH, latest_projects_file)
                try:
                    projects_df = pd.read_csv(projects_path, low_memory=False)
                    logger.info(f"Loaded {len(projects_df)} projects from {latest_projects_file}")
                    
                    # Check minimum required columns
                    required_cols = ['id', 'name', 'symbol']
                    missing_cols = [col for col in required_cols if col not in projects_df.columns]
                    if missing_cols:
                        logger.warning(f"Missing essential columns in projects data: {missing_cols}")
                except Exception as e:
                    logger.error(f"Error loading projects CSV: {e}")
                    projects_df = None
            
            # If CSV loading failed, try JSON files
            if projects_df is None:
                logger.info("Attempting to load data from JSON files")
                json_files = [f for f in raw_files if (f.startswith("coins_markets_") or f.startswith("web3_projects_")) and f.endswith(".json")]
                
                if json_files:
                    all_market_data = []
                    for json_file in json_files:
                        try:
                            with open(os.path.join(RAW_DATA_PATH, json_file), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    all_market_data.append(pd.DataFrame(data))
                                    logger.info(f"Loaded {len(data)} records from {json_file}")
                                else:
                                    logger.warning(f"JSON file {json_file} does not contain a list")
                        except Exception as e:
                            logger.error(f"Error loading JSON file {json_file}: {e}")
                    
                    if all_market_data:
                        # Combine all data frames
                        projects_df = pd.concat(all_market_data, ignore_index=True)
                        projects_df = projects_df.drop_duplicates(subset='id')
                        logger.info(f"Created combined DataFrame with {len(projects_df)} projects")
                        
                        # Save as CSV for future use
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = os.path.join(RAW_DATA_PATH, f"web3_projects_{timestamp}.csv")
                        projects_df.to_csv(backup_path, index=False)
                        logger.info(f"Saved combined data to {backup_path}")
            
            # Final fallback - look in processed directory
            if projects_df is None:
                logger.warning("No projects data found in RAW_DATA_PATH, checking PROCESSED_DATA_PATH")
                
                processed_files = [f for f in os.listdir(PROCESSED_DATA_PATH) 
                                if f.startswith("processed_projects_") and f.endswith(".csv")]
                
                if processed_files:
                    latest_file = max(processed_files)
                    projects_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, latest_file), low_memory=False)
                    logger.info(f"Loaded {len(projects_df)} projects from processed data {latest_file}")
                else:
                    # Try standard filename
                    standard_path = os.path.join(PROCESSED_DATA_PATH, "processed_projects.csv")
                    if os.path.exists(standard_path):
                        projects_df = pd.read_csv(standard_path, low_memory=False)
                        logger.info(f"Loaded {len(projects_df)} projects from standard processed file")
            
            # If still no data, we can't proceed
            if projects_df is None:
                logger.error("Could not load projects data from any source")
                return None, None, None
            
            # Now try to load categories and trending data
            categories_df = None
            trending_df = None
            
            # Look for categories files
            categories_files = [f for f in raw_files if f.startswith("categories_") and f.endswith(".csv")]
            if categories_files:
                latest_categories_file = max(categories_files)
                try:
                    categories_df = pd.read_csv(os.path.join(RAW_DATA_PATH, latest_categories_file))
                    logger.info(f"Loaded categories data from {latest_categories_file}")
                except Exception as e:
                    logger.error(f"Error loading categories CSV: {e}")
                    # Try loading categories from JSON
                    categories_json = [f for f in raw_files if f.startswith("coins_categories") and f.endswith(".json")]
                    if categories_json:
                        try:
                            with open(os.path.join(RAW_DATA_PATH, max(categories_json)), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                categories_df = pd.DataFrame(data)
                                logger.info("Loaded categories data from JSON")
                        except Exception as e:
                            logger.error(f"Error loading categories JSON: {e}")
            
            # Look for trending files
            trending_files = [f for f in raw_files if f.startswith("trending_") and f.endswith(".csv")]
            if trending_files:
                latest_trending_file = max(trending_files)
                try:
                    trending_df = pd.read_csv(os.path.join(RAW_DATA_PATH, latest_trending_file))
                    logger.info(f"Loaded trending data from {latest_trending_file}")
                except Exception as e:
                    logger.error(f"Error loading trending CSV: {e}")
                    # Try loading trending from JSON
                    trending_json = [f for f in raw_files if "trending" in f and f.endswith(".json")]
                    if trending_json:
                        try:
                            with open(os.path.join(RAW_DATA_PATH, max(trending_json)), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if 'coins' in data:
                                    trending_coins = [item['item'] for item in data['coins']]
                                    trending_df = pd.DataFrame(trending_coins)
                                    logger.info("Loaded trending data from JSON")
                        except Exception as e:
                            logger.error(f"Error loading trending JSON: {e}")
            
            return projects_df, categories_df, trending_df
            
        except Exception as e:
            logger.error(f"Error in load_latest_data: {e}")
            logger.error(traceback.format_exc())
            return None, None, None

    def _load_coin_details(self) -> Dict[str, Dict[str, Any]]:
        """
        Enhanced method to load all coin details from JSON files
        with improved JSON parsing and error recovery
        
        Returns:
            dict: Dictionary with coin ID as key and details as value
        """
        logger.info("Loading coin details from JSON files")
        
        try:
            # Get all detail files
            detail_files = [f for f in os.listdir(RAW_DATA_PATH) if f.startswith('coin_details_') and f.endswith('.json')]
            logger.info(f"Found {len(detail_files)} coin detail files")
            
            if not detail_files:
                logger.warning("No coin detail files found")
                return {}
            
            # Mapping to store detail data
            coin_details = {}
            file_count = 0
            error_count = 0
            
            # Process each detail file
            for file in detail_files:
                try:
                    file_path = os.path.join(RAW_DATA_PATH, file)
                    
                    # Extract coin ID from filename as fallback
                    filename_coin_id = file[len('coin_details_'):-len('.json')]
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            detail_data = json.loads(f.read())
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {file}, trying alternative parsing")
                            # Try reading with different encoding or line by line
                            with open(file_path, 'r', encoding='utf-8-sig') as f2:
                                content = f2.read()
                                content = content.replace('\r', '').replace('\n', '')
                                try:
                                    detail_data = json.loads(content)
                                except:
                                    logger.error(f"Failed to parse JSON in {file}")
                                    continue
                    
                    # Get coin ID from data or filename
                    coin_id = detail_data.get('id', filename_coin_id)
                    
                    # Extract platforms with robust parsing
                    platforms = {}
                    if 'platforms' in detail_data:
                        platforms_data = detail_data['platforms']
                        if isinstance(platforms_data, str):
                            try:
                                platforms = json.loads(platforms_data)
                            except json.JSONDecodeError:
                                try:
                                    # Clean and try again
                                    cleaned = platforms_data.strip()
                                    if cleaned.startswith("'") and cleaned.endswith("'"):
                                        cleaned = cleaned[1:-1]
                                    if cleaned.startswith('"') and cleaned.endswith('"'):
                                        cleaned = cleaned[1:-1]
                                    platforms = json.loads(cleaned)
                                except:
                                    try:
                                        # Last resort - try eval (safe since we know it's from our API)
                                        platforms = eval(platforms_data)
                                    except:
                                        platforms = {}
                        elif isinstance(platforms_data, dict):
                            platforms = platforms_data
                    
                    # Extract categories with robust parsing
                    categories = []
                    if 'categories' in detail_data:
                        categories_data = detail_data['categories']
                        if isinstance(categories_data, str):
                            try:
                                categories = json.loads(categories_data)
                            except json.JSONDecodeError:
                                try:
                                    # Clean and try again
                                    cleaned = categories_data.strip()
                                    if cleaned.startswith("'") and cleaned.endswith("'"):
                                        cleaned = cleaned[1:-1]
                                    if cleaned.startswith('"') and cleaned.endswith('"'):
                                        cleaned = cleaned[1:-1]
                                    categories = json.loads(cleaned)
                                except:
                                    try:
                                        # Last resort - try eval
                                        categories = eval(categories_data)
                                    except:
                                        categories = []
                        elif isinstance(categories_data, list):
                            categories = categories_data
                    
                    # Get social and developer data safely
                    community_data = detail_data.get('community_data', {}) or {}
                    developer_data = detail_data.get('developer_data', {}) or {}
                    
                    # Extract sentiment data if available
                    sentiment_data = detail_data.get('sentiment_votes_up_percentage', 50)
                    
                    # Store the details
                    coin_details[coin_id] = {
                        'reddit_subscribers': int(safe_convert(community_data.get('reddit_subscribers'), int, 0)),
                        'twitter_followers': int(safe_convert(community_data.get('twitter_followers'), int, 0)),
                        'github_stars': int(safe_convert(developer_data.get('stars'), int, 0)),
                        'github_subscribers': int(safe_convert(developer_data.get('subscribers'), int, 0)),
                        'github_forks': int(safe_convert(developer_data.get('forks'), int, 0)),
                        'categories': categories,
                        'platforms': platforms,
                        'sentiment_positive': float(safe_convert(sentiment_data, float, 50.0)),
                        # Additional useful data
                        'description': detail_data.get('description', {}).get('en', '') if isinstance(detail_data.get('description'), dict) else '',
                        'genesis_date': detail_data.get('genesis_date', None),
                        'market_cap_rank': int(safe_convert(detail_data.get('market_cap_rank'), int, 9999))
                    }
                    
                    # Add developer score if available
                    if 'developer_score' in detail_data:
                        coin_details[coin_id]['developer_score'] = float(safe_convert(detail_data.get('developer_score'), float, 0.0))
                    
                    # Add social score if available
                    if 'community_score' in detail_data:
                        coin_details[coin_id]['social_score'] = float(safe_convert(detail_data.get('community_score'), float, 0.0))
                    
                    # Increment counter and log progress
                    file_count += 1
                    if file_count % 100 == 0:
                        logger.info(f"Processed {file_count} detail files")
                    
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Error processing detail file {file}: {e}")
                    if error_count < 5:  # Only log first few errors in detail
                        logger.debug(traceback.format_exc())
                    continue
            
            # Log summary
            logger.info(f"Successfully processed {file_count} detail files with {error_count} errors")
            logger.info(f"Collected details for {len(coin_details)} coins")
            
            return coin_details
            
        except Exception as e:
            logger.error(f"Error loading coin details: {e}")
            logger.error(traceback.format_exc())
            return {}

    def clean_projects_data(self, projects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and enhance project data with improved feature generation
        
        Args:
            projects_df: DataFrame of projects
                    
        Returns:
            pd.DataFrame: Cleaned and enhanced DataFrame
        """
        if projects_df is None or projects_df.empty:
            logger.error("Projects DataFrame is empty or None")
            
            # Fill NA values for categorical columns
            if 'category' in df.columns:
                df['category'] = df['category'].fillna('unknown')

            if 'primary_category' in df.columns:
                df['primary_category'] = df['primary_category'].fillna('unknown')

            if 'chain' in df.columns:
                df['chain'] = df['chain'].fillna('unknown')

            return pd.DataFrame()
                    
        logger.info("Cleaning projects data")
                
        # Make a copy to avoid SettingWithCopyWarning
        df = projects_df.copy()
        
        # Ensure essential columns exist
        essential_columns = ['id', 'name', 'symbol']
        for col in essential_columns:
            if col not in df.columns:
                logger.error(f"Essential column '{col}' not found in data")
                return pd.DataFrame()
        
        # Add or fill required columns with defaults
        required_columns = ['market_cap', 'total_volume', 'current_price']
        for col in required_columns:
            if col not in df.columns:
                logger.info(f"Column '{col}' not found, creating with default value 0")
                df[col] = 0
            else:
                # Handle missing values
                df[col] = df[col].fillna(0)
        
        # Add social and developer columns with defaults
        social_columns = ['reddit_subscribers', 'twitter_followers', 'github_stars', 'github_forks', 'github_subscribers']
        for col in social_columns:
            if col not in df.columns:
                logger.info(f"Column '{col}' not found, creating with default value 0")
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)
        
        # Ensure JSON columns exist with correct defaults
        if 'platforms' not in df.columns:
            logger.info("Column 'platforms' not found, creating with default empty dict")
            df['platforms'] = [{} for _ in range(len(df))]
        
        if 'categories' not in df.columns:
            logger.info("Column 'categories' not found, creating with default empty list")
            df['categories'] = [[] for _ in range(len(df))]
        
        # Create chain and primary_category with default values
        if 'chain' not in df.columns:
            df['chain'] = 'unknown'
        
        if 'primary_category' not in df.columns:
            df['primary_category'] = 'unknown'
        
        # Enrich with coin details
        coin_details = self._load_coin_details()
        self._enrich_with_coin_details(df, coin_details)
        
        # Handle missing values and normalize/convert data types
        df = self._handle_missing_values(df)
        
        # Calculate enhanced metrics
        df = self._calculate_enhanced_metrics(df)
        
        # Extract and normalize chain/category data
        # df = self._extract_chain_and_category(df)
        
        # Ensure JSON columns are properly formatted
        df = self.convert_json_columns(df, ['platforms', 'categories'])
        
        # Add text features if description available
        if 'description' in df.columns:
            df = self._add_text_features(df)
        
        logger.info("Projects data cleaned successfully")
        
        # Final verification
        df = self._verify_data_integrity(df)
        
        return df
    
    def _enrich_with_coin_details(self, df: pd.DataFrame, coin_details: Dict[str, Dict[str, Any]]) -> None:
        """
        Enhance DataFrame with detailed coin information
        
        Args:
            df: DataFrame to enrich
            coin_details: Dictionary of coin details
        """
        logger.info("Enriching data with coin details")
        
        if not coin_details:
            logger.warning("No coin details available for enrichment")
            return
        
        # Sample for verification
        sample_coin = next(iter(coin_details.keys())) if coin_details else None
        if sample_coin:
            logger.info(f"Sample detail for enrichment: {sample_coin}")
            logger.info(f"  - Categories type: {type(coin_details[sample_coin]['categories'])}")
            logger.info(f"  - Platforms type: {type(coin_details[sample_coin]['platforms'])}")
        
        # Track enrichment statistics
        enriched_count = 0
        missing_count = 0
        category_extracted = 0
        chain_extracted = 0
        
        for idx, row in df.iterrows():
            try:
                coin_id = row['id']
                if coin_id in coin_details:
                    details = coin_details[coin_id]
                    
                    # Update social metrics
                    social_metrics = ['reddit_subscribers', 'twitter_followers', 'github_stars', 'github_subscribers', 'github_forks']
                    for metric in social_metrics:
                        if metric in details:
                            df.at[idx, metric] = details[metric]
                    
                    # Update developer and social scores if available
                    if 'developer_score' in details:
                        df.at[idx, 'developer_score'] = details['developer_score']
                    
                    if 'social_score' in details:
                        df.at[idx, 'social_score'] = details['social_score']
                    
                    # Update sentiment if available
                    if 'sentiment_positive' in details:
                        df.at[idx, 'sentiment_positive'] = details['sentiment_positive']
                    
                    # Update genesis date if available
                    if 'genesis_date' in details and details['genesis_date']:
                        df.at[idx, 'genesis_date'] = details['genesis_date']
                    
                    # Update market cap rank if available
                    if 'market_cap_rank' in details:
                        df.at[idx, 'market_cap_rank'] = details['market_cap_rank']
                    
                    # Update description if available
                    if 'description' in details and details['description']:
                        df.at[idx, 'description'] = details['description']
                    
                    # Update categories and platforms
                    if details['categories']:
                        df.at[idx, 'categories'] = list(details['categories'])
                    
                    if details['platforms']:
                        df.at[idx, 'platforms'] = dict(details['platforms'])
                    
                    # Extract chain with improved algorithm
                    if details['platforms'] and isinstance(details['platforms'], dict):
                        extracted_chain = self._extract_primary_chain(details['platforms'])
                        if extracted_chain:
                            df.at[idx, 'chain'] = extracted_chain
                            chain_extracted += 1
                    
                    # Extract primary category with improved algorithm
                    if details['categories'] and isinstance(details['categories'], list):
                        extracted_category = self._extract_primary_category(details['categories'])
                        if extracted_category:
                            df.at[idx, 'primary_category'] = extracted_category
                            category_extracted += 1
                    
                    enriched_count += 1
                else:
                    missing_count += 1
            except Exception as e:
                logger.warning(f"Error updating row {idx}: {e}")
        
        logger.info(f"Enriched {enriched_count} rows, {missing_count} coins not found in details")
        logger.info(f"Extracted chain for {chain_extracted} coins, primary category for {category_extracted} coins")
    
    def _extract_primary_chain(self, platforms: Dict[str, Any]) -> str:
        """
        Extract primary blockchain with improved matching
        
        Args:
            platforms: Dictionary of platforms
            
        Returns:
            str: Primary chain name
        """
        if not platforms:
            return 'unknown'
        
        # Priority chains
        priority_chains = ['ethereum', 'binance-smart-chain', 'solana', 'polygon-pos', 'avalanche']
        
        # First try exact match with priority chains
        for chain in priority_chains:
            if chain in platforms:
                return chain
        
        # Try normalized matching with aliases
        normalized_platforms = {k.lower(): k for k in platforms.keys()}
        
        for chain_name, aliases in self.blockchain_platforms.items():
            # Check direct match
            if chain_name in normalized_platforms:
                return chain_name
            
            # Check aliases
            for alias in aliases:
                if alias in normalized_platforms:
                    return chain_name
        
        # Fallback to first platform
        if platforms:
            return next(iter(platforms.keys()))
        
        return 'unknown'
    
    def _extract_primary_category(self, categories: List[str]) -> str:
        """
        Extract primary category with improved matching
        
        Args:
            categories: List of categories
            
        Returns:
            str: Primary category name
        """
        if not categories:
            return 'unknown'
        
        # Normalized categories
        normalized_categories = [c.lower() for c in categories if c]
        
        # Priority categories
        priority_order = [
            'layer-1', 'defi', 'nft', 'gaming', 'stablecoin', 'meme', 'dao', 
            'exchange', 'privacy', 'real-world-assets'
        ]
        
        # First try direct match with priority categories
        for category in priority_order:
            if category in normalized_categories:
                return category
        
        # Try normalized matching with mappings
        for category_name, aliases in self.category_mappings.items():
            for alias in aliases:
                for cat in normalized_categories:
                    if alias in cat or cat == alias:
                        return category_name
        
        # Fallback to first category
        if categories and categories[0]:
            return categories[0].lower()
        
        return 'unknown'
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive handling of missing values
        
        Args:
            df: DataFrame to process
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        logger.info("Handling missing values")
        
        # Make a copy
        result_df = df.copy()
        
        # Handle numerical columns
        numerical_cols = [
            'market_cap', 'total_volume', 'current_price', 'price_change_percentage_24h',
            'price_change_percentage_7d_in_currency', 'price_change_percentage_30d_in_currency',
            'reddit_subscribers', 'twitter_followers', 'github_stars'
        ]
        
        for col in numerical_cols:
            if col in result_df.columns:
                # First replace infinity and other invalid values
                result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                
                # Then fill missing values with appropriate defaults
                if col in ['market_cap', 'total_volume']:
                    result_df[col] = result_df[col].fillna(0)
                elif 'price_change' in col:
                    result_df[col] = result_df[col].fillna(0)
                else:
                    # For social metrics, use 0
                    result_df[col] = result_df[col].fillna(0)
                
                # Ensure proper data type
                try:
                    if 'price_change' in col:
                        result_df[col] = result_df[col].astype(float)
                    else:
                        result_df[col] = result_df[col].astype(float)
                except Exception as e:
                    logger.warning(f"Error converting {col} to numeric: {e}")
                    # Try more aggressive cleaning
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)
        
        # Handle categorical columns
        if 'chain' in result_df.columns:
            result_df['chain'] = result_df['chain'].fillna('unknown')
            result_df['chain'] = result_df['chain'].astype(str)
        
        if 'primary_category' in result_df.columns:
            result_df['primary_category'] = result_df['primary_category'].fillna('unknown')
            result_df['primary_category'] = result_df['primary_category'].astype(str)
        
        # Handle date columns
        if 'genesis_date' in result_df.columns:
            result_df['genesis_date'] = pd.to_datetime(result_df['genesis_date'], errors='coerce')
        
        return result_df
    
    def _calculate_enhanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced metrics for better recommendation modeling
        
        Args:
            df: DataFrame to enhance
            
        Returns:
            pd.DataFrame: DataFrame with enhanced metrics
        """
        logger.info("Calculating enhanced metrics")
        
        # Make a copy
        result_df = df.copy()
        
        try:
            # 1. Enhanced Popularity Score
            # Using log scale for market cap and volume to handle large ranges
            market_cap = np.log1p(result_df['market_cap'].fillna(0)) / 30
            volume = np.log1p(result_df['total_volume'].fillna(0)) / 25
            
            # Social metrics with balanced weighting
            reddit = np.log1p(result_df['reddit_subscribers'].fillna(0)) / 15
            twitter = np.log1p(result_df['twitter_followers'].fillna(0)) / 15
            github = np.log1p(result_df['github_stars'].fillna(0)) / 10
            
            # Combined popularity score
            popularity_score = (
                0.35 * market_cap + 
                0.25 * volume + 
                0.15 * reddit + 
                0.15 * twitter +
                0.10 * github
            )
            
            # Scale to 0-100
            result_df['popularity_score'] = popularity_score * 100
            
            # 2. Enhanced Trend Score
            if 'price_change_percentage_24h' in result_df.columns:
                # Normalize price changes to -1 to 1 scale with sigmoid-like transform for outliers
                price_24h = result_df['price_change_percentage_24h'].fillna(0) / 100
                price_24h = price_24h.clip(-1, 1)  # Clip extreme values
                
                # Get 7d and 30d changes if available
                price_7d = pd.Series(0, index=result_df.index)
                price_30d = pd.Series(0, index=result_df.index)
                
                if 'price_change_percentage_7d_in_currency' in result_df.columns:
                    price_7d = result_df['price_change_percentage_7d_in_currency'].fillna(0) / 100
                    price_7d = price_7d.clip(-1, 1)
                
                if 'price_change_percentage_30d_in_currency' in result_df.columns:
                    price_30d = result_df['price_change_percentage_30d_in_currency'].fillna(0) / 100
                    price_30d = price_30d.clip(-1, 1)
                
                # Weighted trend score with decay (recent changes matter more)
                trend_score = (
                    0.6 * price_24h + 
                    0.3 * price_7d + 
                    0.1 * price_30d
                )
                
                # Scale to 0-100 with 50 as neutral
                result_df['trend_score'] = 50 + (trend_score * 50)
            else:
                # Default trend score
                result_df['trend_score'] = 50
            
            # 3. Social Engagement Score
            # Calculate engagement as ratio of followers to market cap
            if 'market_cap' in result_df.columns and result_df['market_cap'].max() > 0:
                # Social followers normalized by market cap
                social_sum = result_df['reddit_subscribers'] + result_df['twitter_followers']
                market_cap_millions = result_df['market_cap'] / 1_000_000
                
                # Avoid division by zero
                market_cap_norm = market_cap_millions.replace(0, np.nan)
                
                # Calculate engagement ratio
                engagement_ratio = social_sum / market_cap_norm
                
                # Fill NaN values with median
                median_ratio = engagement_ratio.median()
                engagement_ratio = engagement_ratio.fillna(median_ratio)
                
                # Apply log and normalize
                engagement_score = np.log1p(engagement_ratio)
                max_score = engagement_score.quantile(0.95)  # Use 95th percentile to avoid outliers
                engagement_score = engagement_score / max_score
                
                # Scale to 0-100
                result_df['social_engagement_score'] = (engagement_score * 100).clip(0, 100)
            else:
                result_df['social_engagement_score'] = 50
            
            # 4. Market Maturity Score
            # Based on age, market cap, volume stability
            if 'genesis_date' in result_df.columns:
                # Calculate age in days
                now = pd.Timestamp.now().normalize()
                result_df['age_days'] = (now - result_df['genesis_date']).dt.days
                
                # Fill missing values with median
                median_age = result_df['age_days'].median()
                result_df['age_days'] = result_df['age_days'].fillna(median_age)
                
                # Normalize age (older = more mature)
                max_age = result_df['age_days'].quantile(0.95)
                age_score = (result_df['age_days'] / max_age).clip(0, 1)
                
                # Combine with market cap for maturity score
                market_cap_score = market_cap
                
                # Calculate maturity score
                maturity_score = (0.6 * age_score + 0.4 * market_cap_score)
                
                # Scale to 0-100
                result_df['maturity_score'] = (maturity_score * 100).clip(0, 100)
            else:
                result_df['maturity_score'] = 50
            
            # 5. Developer Activity Score
            if 'github_stars' in result_df.columns and 'github_forks' in result_df.columns:
                github_stats = np.log1p(result_df['github_stars']) + np.log1p(result_df['github_forks'])
                max_stats = github_stats.quantile(0.95)
                
                if max_stats > 0:
                    dev_score = (github_stats / max_stats).clip(0, 1)
                    result_df['developer_activity_score'] = (dev_score * 100).clip(0, 100)
                else:
                    result_df['developer_activity_score'] = 0
            else:
                result_df['developer_activity_score'] = 0
            
        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {e}")
            logger.error(traceback.format_exc())
            
            # Set defaults in case of error
            result_df['popularity_score'] = 50
            result_df['trend_score'] = 50
            result_df['social_engagement_score'] = 50
            result_df['maturity_score'] = 50
            result_df['developer_activity_score'] = 0
        
        return result_df
    
    def _add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from text descriptions using NLP techniques
        
        Args:
            df: DataFrame with description column
            
        Returns:
            pd.DataFrame: DataFrame with added text features
        """
        if 'description' not in df.columns:
            return df
        
        logger.info("Adding text features from descriptions")
        
        # Make a copy
        result_df = df.copy()
        
        try:
            # Clean descriptions
            descriptions = result_df['description'].fillna('').astype(str)
            
            # Count lengths as basic feature
            result_df['description_length'] = descriptions.apply(len)
            
            # Skip further processing if most descriptions are empty
            if (descriptions == '').mean() > 0.8:
                logger.warning("Over 80% of descriptions are empty, skipping NLP processing")
                return result_df
            
            # Use TF-IDF to extract features from text
            tfidf = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                min_df=5,
                max_df=0.9
            )
            
            try:
                # Transform descriptions to TF-IDF features
                tfidf_matrix = tfidf.fit_transform(descriptions)
                
                # Use dimensionality reduction to get manageable number of features
                svd = TruncatedSVD(n_components=5)
                text_features = svd.fit_transform(tfidf_matrix)
                
                # Add features to dataframe
                for i in range(text_features.shape[1]):
                    result_df[f'text_feature_{i+1}'] = text_features[:, i]
                
                logger.info(f"Added {text_features.shape[1]} text features")
                
            except ValueError as e:
                logger.warning(f"Error in TF-IDF processing: {e}")
        
        except Exception as e:
            logger.error(f"Error adding text features: {e}")
            logger.error(traceback.format_exc())
        
        return result_df
    
    def _verify_data_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Verify and fix data integrity issues
        
        Args:
            df: DataFrame to verify
            
        Returns:
            pd.DataFrame: Verified DataFrame
        """
        logger.info("Verifying data integrity")
        
        # Copy to avoid modifying original
        result_df = df.copy()
        
        # 1. Check if required columns exist
        required_cols = ['id', 'name', 'symbol', 'market_cap', 'total_volume', 
                         'chain', 'primary_category', 'popularity_score', 'trend_score']
        
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            
            # Add missing columns with default values
            for col in missing_cols:
                if col in ['id', 'name', 'symbol', 'chain', 'primary_category']:
                    result_df[col] = 'unknown'
                else:
                    result_df[col] = 0
        
        # 2. Verify all numerical columns have valid values
        num_cols = ['market_cap', 'total_volume', 'current_price', 
                   'popularity_score', 'trend_score']
        
        for col in num_cols:
            if col in result_df.columns:
                # Check if column has invalid values
                invalid_mask = ~np.isfinite(result_df[col])
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    logger.warning(f"Column {col} has {invalid_count} invalid values")
                    result_df.loc[invalid_mask, col] = 0
        
        # 3. Verify platforms and categories are the correct types
        if 'platforms' in result_df.columns:
            platform_types = result_df['platforms'].apply(type).value_counts()
            logger.info(f"Platform column types: {platform_types}")
            
            # Fix any rows where platforms isn't a dict
            for idx, row in result_df.iterrows():
                if not isinstance(row['platforms'], dict):
                    result_df.at[idx, 'platforms'] = {}
        
        if 'categories' in result_df.columns:
            category_types = result_df['categories'].apply(type).value_counts()
            logger.info(f"Category column types: {category_types}")
            
            # Fix any rows where categories isn't a list
            for idx, row in result_df.iterrows():
                if not isinstance(row['categories'], list):
                    result_df.at[idx, 'categories'] = []
        
        # 4. Verify chain and category values are normalized
        if 'chain' in result_df.columns:
            chain_counts = result_df['chain'].value_counts()
            logger.info(f"Found {len(chain_counts)} unique chains")
            
            # Normalize chain values for consistency
            result_df['chain'] = result_df['chain'].str.lower()
            
            # Map similar chains to canonical names
            chain_mapping = {
                'bsc': 'binance-smart-chain',
                'bnb chain': 'binance-smart-chain',
                'binance': 'binance-smart-chain',
                'eth': 'ethereum',
                'matic': 'polygon-pos',
                'polygon': 'polygon-pos',
                'sol': 'solana'
            }
            
            result_df['chain'] = result_df['chain'].map(
                lambda x: chain_mapping.get(x, x) if x in chain_mapping else x
            )
        
        if 'primary_category' in result_df.columns:
            category_counts = result_df['primary_category'].value_counts()
            logger.info(f"Found {len(category_counts)} unique categories")
            
            # Normalize category values for consistency
            result_df['primary_category'] = result_df['primary_category'].str.lower()
            
            # Map similar categories to canonical names
            category_mapping = {
                'defi': 'defi',
                'decentralized-finance': 'defi',
                'decentralized-finance-defi': 'defi',
                'nft': 'nft',
                'non-fungible-tokens': 'nft',
                'non-fungible-tokens-nft': 'nft',
                'layer1': 'layer-1',
                'layer 1': 'layer-1',
                'l1': 'layer-1',
                'layer2': 'layer-2',
                'layer 2': 'layer-2',
                'l2': 'layer-2',
                'gaming': 'gaming',
                'play-to-earn': 'gaming',
                'p2e': 'gaming',
                'metaverse': 'metaverse',
                'meme': 'meme',
                'meme-token': 'meme',
                'exchange': 'exchange',
                'exchange-token': 'exchange'
            }
            
            result_df['primary_category'] = result_df['primary_category'].map(
                lambda x: category_mapping.get(x, x) if x in category_mapping else x
            )
        
        # Log integrity verification results
        logger.info("Data integrity verification completed")
        
        return result_df
    
    def convert_json_columns(self, df, json_columns):
        """
        Converts JSON string columns to Python objects safely
        
        Args:
            df: DataFrame with JSON string columns
            json_columns: List of column names containing JSON strings
            
        Returns:
            pd.DataFrame: DataFrame with converted columns
        """
        # Pastikan input adalah DataFrame yang valid
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Input is not a DataFrame: {type(df)}")
            return df
        
        result_df = df.copy()
        
        for col in json_columns:
            if col not in result_df.columns:
                continue
                
            # Buat list baru untuk nilai yang dikonversi
            converted_values = []
            
            # Gunakan pendekatan berbasis list untuk menghindari masalah pandas
            for i in range(len(result_df)):
                try:
                    val = result_df.iloc[i][col]
                    
                    # Handle None/NaN case
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        converted_values.append({} if col == 'platforms' else [])
                        continue
                    
                    # Already correct type
                    if (col == 'platforms' and isinstance(val, dict)) or \
                    (col == 'categories' and isinstance(val, list)):
                        converted_values.append(val)
                        continue
                    
                    # Handle string case
                    if isinstance(val, str):
                        if val == '' or val == '[]' or val == '{}':
                            converted_values.append({} if col == 'platforms' else [])
                        else:
                            try:
                                # Try standard JSON parse
                                parsed = json.loads(val)
                                converted_values.append(parsed)
                            except:
                                try:
                                    # Clean and try again
                                    cleaned = val.replace('\"\"', '\"').replace('\\"', '"')
                                    parsed = json.loads(cleaned)
                                    converted_values.append(parsed)
                                except:
                                    # Set default value
                                    converted_values.append({} if col == 'platforms' else [])
                    else:
                        # Not a recognized format
                        converted_values.append({} if col == 'platforms' else [])
                except Exception as e:
                    logger.warning(f"Error processing {col} at index {i}: {str(e)}")
                    converted_values.append({} if col == 'platforms' else [])
            
            # Replace column with converted values in one operation
            result_df[col] = converted_values
            logger.info(f"Conversion complete for {col} column")
        
        return result_df
    
    def create_interaction_data(self, projects_df: pd.DataFrame, n_users: int = 500, seed: int = 42) -> pd.DataFrame:
        """
        Create realistic synthetic user-item interactions with improved persona modeling
        
        Args:
            projects_df: DataFrame of projects
            n_users: Number of synthetic users
            seed: Random seed for reproducibility
            
        Returns:
            pd.DataFrame: DataFrame of user interactions
        """
        logger.info(f"Creating synthetic user interaction data for {n_users} users")
        
        # Set random seed
        np.random.seed(seed)
        rng = np.random.default_rng(seed)  # Use modern RNG
        
        # Ensure required columns exist
        for col in ['chain', 'primary_category', 'popularity_score']:
            if col not in projects_df.columns:
                if col in ['chain', 'primary_category']:
                    projects_df[col] = 'unknown'
                elif col == 'popularity_score':
                    projects_df[col] = 50.0
        
        # Create interactions
        interactions = []
        
        # Distribution of user personas (more diverse for better training)
        personas = list(self.user_persona_weights.keys())
        
        # Generate users with consistent patterns
        for user_id in range(1, n_users + 1):
            try:
                # Assign a persona to this user
                user_persona = personas[user_id % len(personas)]
                persona_weights = self.user_persona_weights[user_persona]
                
                # Determine interaction count based on activity level
                activity_level = rng.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])
                
                if activity_level == 'low':
                    n_interactions = rng.integers(3, 10)
                elif activity_level == 'medium':
                    n_interactions = rng.integers(10, 25)
                else:  # high
                    n_interactions = rng.integers(25, 50)
                
                # Preferred categories and chains based on persona
                preferred_categories = list(persona_weights['categories'].keys())
                category_weights = list(persona_weights['categories'].values())
                
                preferred_chains = list(persona_weights['chains'].keys())
                chain_weights = list(persona_weights['chains'].values())
                
                # Select preferred category and chain for this user
                primary_category = rng.choice(preferred_categories, p=category_weights)
                primary_chain = rng.choice(preferred_chains, p=chain_weights)
                
                # Create composite filter mask
                chain_mask = projects_df['chain'] == primary_chain
                category_mask = projects_df['primary_category'] == primary_category
                
                # Combined mask with OR logic
                preferred_mask = np.logical_or(chain_mask, category_mask)
                preferred_projects = projects_df[preferred_mask]
                
                # Fallback if no matches
                if preferred_projects.empty:
                    preferred_projects = projects_df
                
                # Calculate weighted selection probabilities
                popularity_boost = preferred_projects['popularity_score'].values.copy()
                # Handle non-numeric values
                popularity_boost = np.nan_to_num(popularity_boost, nan=50.0)
                
                # Different strategies for different personas
                if user_persona == 'defi_enthusiast':
                    # DeFi enthusiasts favor higher market cap projects
                    market_cap_boost = np.log1p(preferred_projects['market_cap'].fillna(0))
                    weights = popularity_boost * market_cap_boost
                elif user_persona == 'nft_collector':
                    # NFT collectors are less price sensitive
                    weights = popularity_boost
                elif user_persona == 'trader':
                    # Traders favor projects with high volatility
                    if 'price_change_percentage_24h' in preferred_projects.columns:
                        volatility = np.abs(preferred_projects['price_change_percentage_24h'].fillna(0))
                        weights = popularity_boost * (1 + volatility/100)
                    else:
                        weights = popularity_boost
                elif user_persona == 'gamer':
                    # Gamers favor newer, trending projects
                    if 'trend_score' in preferred_projects.columns:
                        weights = popularity_boost * preferred_projects['trend_score'].fillna(50)
                    else:
                        weights = popularity_boost
                elif user_persona == 'meme_investor':
                    # Meme investors heavily favor trending tokens
                    if 'trend_score' in preferred_projects.columns:
                        trend_boost = preferred_projects['trend_score'].fillna(50)
                        weights = popularity_boost * trend_boost
                    else:
                        weights = popularity_boost
                else:
                    # Default weighting
                    weights = popularity_boost
                
                # Normalize weights
                weights = np.maximum(weights, 0.001)  # Ensure positive
                weights = weights / weights.sum()
                
                # Select projects
                try:
                    indices = np.arange(len(preferred_projects))
                    if len(indices) > 0:
                        selections = min(n_interactions, len(indices))
                        selected_indices = rng.choice(indices, size=selections, p=weights, replace=False)
                        selected_positions = preferred_projects.index[selected_indices]
                        selected_projects = preferred_projects.loc[selected_positions]
                        
                        # Create interactions for each selected project
                        for _, project in selected_projects.iterrows():
                            # Determine interaction type with probabilities customized by persona
                            if user_persona == 'defi_enthusiast':
                                interaction_probs = [0.3, 0.3, 0.3, 0.1]  # More portfolio adds
                            elif user_persona == 'nft_collector':
                                interaction_probs = [0.3, 0.4, 0.2, 0.1]  # More favorites
                            elif user_persona == 'trader':
                                interaction_probs = [0.2, 0.2, 0.4, 0.2]  # More portfolio adds
                            elif user_persona == 'gamer':
                                interaction_probs = [0.4, 0.3, 0.2, 0.1]  # More views
                            elif user_persona == 'meme_investor':
                                interaction_probs = [0.3, 0.4, 0.2, 0.1]  # More favorites
                            else:
                                interaction_probs = [0.4, 0.3, 0.2, 0.1]
                            
                            interaction_type = rng.choice(
                                ['view', 'favorite', 'portfolio_add', 'research'],
                                p=interaction_probs
                            )
                            
                            # Weight based on interaction type
                            if interaction_type == 'view':
                                weight = rng.integers(1, 3)
                            elif interaction_type == 'favorite':
                                weight = rng.integers(3, 5)
                            elif interaction_type == 'portfolio_add':
                                weight = rng.integers(4, 6)
                            else:  # research
                                weight = rng.integers(2, 4)
                            
                            # Add to interactions
                            interactions.append({
                                'user_id': f"user_{user_id}",
                                'project_id': project['id'],
                                'interaction_type': interaction_type,
                                'weight': weight,
                                'timestamp': datetime.now().isoformat()
                            })
                except Exception as e:
                    logger.warning(f"Error selecting projects for user {user_id}: {e}")
                    
            except Exception as e:
                logger.warning(f"Error generating interactions for user {user_id}: {e}")
                continue
        
        # Create DataFrame
        interactions_df = pd.DataFrame(interactions) if interactions else pd.DataFrame(
            columns=['user_id', 'project_id', 'interaction_type', 'weight', 'timestamp']
        )
        
        logger.info(f"Created {len(interactions_df)} user interactions across {len(set(interactions_df['user_id']))} users")
        
        # Add some additional analysis
        if not interactions_df.empty:
            projects_per_user = interactions_df.groupby('user_id')['project_id'].nunique().mean()
            interactions_per_project = interactions_df.groupby('project_id')['user_id'].nunique().mean()
            logger.info(f"Average projects per user: {projects_per_user:.2f}")
            logger.info(f"Average users per project: {interactions_per_project:.2f}")
        
        return interactions_df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced feature preparation for machine learning models
        
        Args:
            df: DataFrame of projects
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        logger.info("Preparing enhanced features for modeling")
        
        # Make a copy
        processed_df = df.copy()
        
        # Ensure all required columns exist
        for feat in NUMERICAL_FEATURES:
            if feat not in processed_df.columns:
                logger.warning(f"Numerical feature {feat} not found, adding with default 0")
                processed_df[feat] = 0
                
        for feat in CATEGORICAL_FEATURES:
            if feat not in processed_df.columns:
                logger.warning(f"Categorical feature {feat} not found, adding with default 'unknown'")
                processed_df[feat] = 'unknown'
        
        # Process numerical features with scaling
        num_features = []
        
        # Process regular numerical features
        regular_num_features = [f for f in NUMERICAL_FEATURES if f not in 
                               ['market_cap', 'volume_24h', 'reddit_subscribers', 'twitter_followers']]
        
        if regular_num_features:
            reg_num_df = processed_df[regular_num_features].copy()
            
            # Handle missing and invalid values
            reg_num_df = reg_num_df.fillna(0)
            reg_num_df = reg_num_df.replace([np.inf, -np.inf], 0)
            
            # Apply robust scaling to handle outliers
            scaled_reg_num = pd.DataFrame(
                self.num_scaler.fit_transform(reg_num_df),
                columns=reg_num_df.columns,
                index=reg_num_df.index
            )
            num_features.append(scaled_reg_num)
        
        # Process market cap and volume with separate scaling (large range)
        if 'market_cap' in processed_df.columns:
            market_cap_df = processed_df[['market_cap']].copy()
            market_cap_df = market_cap_df.fillna(0)
            
            # Apply log transform before scaling
            market_cap_log = np.log1p(market_cap_df)
            
            # Scale to 0-1
            scaled_market_cap = pd.DataFrame(
                self.market_cap_scaler.fit_transform(market_cap_log),
                columns=['market_cap_scaled'],
                index=market_cap_df.index
            )
            num_features.append(scaled_market_cap)
        
        if 'volume_24h' in processed_df.columns:
            volume_df = processed_df[['volume_24h']].copy()
            volume_df = volume_df.fillna(0)
            
            # Apply log transform before scaling
            volume_log = np.log1p(volume_df)
            
            # Scale to 0-1
            scaled_volume = pd.DataFrame(
                self.volume_scaler.fit_transform(volume_log),
                columns=['volume_24h_scaled'],
                index=volume_df.index
            )
            num_features.append(scaled_volume)
        
        # Process social metrics with separate scaling
        social_cols = [col for col in ['reddit_subscribers', 'twitter_followers', 'github_stars'] 
                      if col in processed_df.columns]
        
        if social_cols:
            social_df = processed_df[social_cols].copy()
            social_df = social_df.fillna(0)
            
            # Apply log transform before scaling
            social_log = np.log1p(social_df)
            
            # Scale to 0-1
            scaled_social = pd.DataFrame(
                self.social_scaler.fit_transform(social_log),
                columns=[f"{col}_scaled" for col in social_cols],
                index=social_df.index
            )
            num_features.append(scaled_social)
        
        # Add computed metrics if available
        computed_metrics = [col for col in [
            'popularity_score', 'trend_score', 'social_engagement_score',
            'maturity_score', 'developer_activity_score', 'sentiment_positive'
        ] if col in processed_df.columns]
        
        if computed_metrics:
            metrics_df = processed_df[computed_metrics].copy()
            metrics_df = metrics_df.fillna(50)  # Default to neutral
            
            # Scale all to 0-1
            metrics_df = metrics_df / 100
            
            num_features.append(metrics_df)
        
        # Process categorical features - chain
        if 'chain' in processed_df.columns:
            # Focus on most common chains to avoid dimensionality explosion
            chain_counts = processed_df['chain'].value_counts()
            top_chains = chain_counts.head(15).index.tolist()
            
            # One-hot encode chains
            chain_dummies = pd.get_dummies(
                processed_df['chain'].apply(lambda x: x if x in top_chains else 'other'),
                prefix='chain'
            )
            num_features.append(chain_dummies)
        
        # Process categorical features - category
        if 'primary_category' in processed_df.columns:
            # Focus on most common categories
            category_counts = processed_df['primary_category'].value_counts()
            top_categories = category_counts.head(15).index.tolist()
            
            # One-hot encode categories
            category_dummies = pd.get_dummies(
                processed_df['primary_category'].apply(lambda x: x if x in top_categories else 'other'),
                prefix='category'
            )
            num_features.append(category_dummies)
        
        # Add text features if available
        text_features = [col for col in processed_df.columns if col.startswith('text_feature_')]
        if text_features:
            text_df = processed_df[text_features].copy()
            text_df = text_df.fillna(0)
            num_features.append(text_df)
        
        # Combine all features
        if not num_features:
            logger.error("No features were created!")
            return pd.DataFrame(index=processed_df.index)
            
        all_features = pd.concat(num_features, axis=1)
        
        # Ensure no NaN values in final feature matrix
        all_features = all_features.fillna(0)
        
        logger.info(f"Created feature matrix with shape: {all_features.shape}")
        return all_features
    
    
    def process_categorical_features(self, df):
        """
        Memproses fitur kategorikal dalam dataframe

        Args:
            df: DataFrame dengan data projects

        Returns:
            DataFrame: DataFrame dengan fitur kategorikal yang sudah diproses
        """
        self.logger.info("Processing categorical features")

        # Pastikan kolom kategori dan platform/chain tersedia
        if 'category' not in df.columns:
            self.logger.warning("'category' column not found in dataframe, creating empty column")
            df['category'] = None

        if 'platforms' not in df.columns and 'chain' not in df.columns:
            self.logger.warning("No blockchain platform/chain column found, creating empty column")
            df['chain'] = None
        elif 'platforms' in df.columns and 'chain' not in df.columns:
            # Rename platforms to chain for consistency
            df['chain'] = df['platforms']

        # Pastikan primary_category tersedia
        if 'primary_category' not in df.columns:
            self.logger.info("Creating primary_category from category")
            # Extract first category if category is a list
            def extract_primary(cat):
                if isinstance(cat, list) and len(cat) > 0:
                    return cat[0]
                return cat

            df['primary_category'] = df['category'].apply(extract_primary)

        # Pastikan semua kolom kategori tersedia untuk config
        from config.config import CATEGORICAL_FEATURES
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                self.logger.warning(f"Categorical column {col} not found, creating empty column")
                df[col] = None

        return df
    def process_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
            """
            Process all data and save it in a format ready for modeling
            
            Returns:
                tuple: (projects_df, interactions_df, feature_matrix)
            """
            try:
                logger.info("Starting data processing pipeline")
                
                # Load data
                projects_df, categories_df, trending_df = self.load_latest_data()
                
                if projects_df is None or projects_df.empty:
                    logger.error("No project data available for processing")
                    # Process categorical features
                    projects_df = self.process_categorical_features(projects_df)
                    return None, None, None
                
                # Ensure consistent data types
                projects_df = self.ensure_valid_data_types(projects_df)
                
                # Clean and enhance project data
                logger.info("Cleaning and enhancing projects data")
                cleaned_projects = self.clean_projects_data(projects_df)
                logger.info(f"Processed {len(cleaned_projects)} projects")
                
                # Verify requirements after cleaning
                if cleaned_projects is None or cleaned_projects.empty:
                    logger.error("Cleaned projects data is empty")
                    return None, None, None
                
                # Verify required columns
                required_columns = ['id', 'name', 'symbol', 'market_cap', 'total_volume', 
                                    'chain', 'primary_category', 'popularity_score', 'trend_score',
                                    'platforms', 'categories']
                
                missing_columns = [col for col in required_columns if col not in cleaned_projects.columns]
                if missing_columns:
                    logger.error(f"Missing required columns after cleaning: {missing_columns}")
                    # Add default values for missing columns
                    for col in missing_columns:
                        if col in ['id', 'name', 'symbol', 'chain', 'primary_category']:
                            cleaned_projects[col] = 'unknown'
                        elif col in ['platforms']:
                            cleaned_projects[col] = [{} for _ in range(len(cleaned_projects))]
                        elif col in ['categories']:
                            cleaned_projects[col] = [[] for _ in range(len(cleaned_projects))]
                        else:
                            cleaned_projects[col] = 0
                
                # Create synthetic user interactions
                logger.info("Creating synthetic user interactions")
                interactions_df = self.create_interaction_data(cleaned_projects, n_users=500)
                logger.info(f"Created {len(interactions_df)} user interactions")
                
                # Prepare feature matrix for modeling
                logger.info("Preparing feature matrix")
                feature_matrix = self.prepare_features(cleaned_projects)
                logger.info(f"Created feature matrix with shape {feature_matrix.shape}")
                
                # Save processed data
                logger.info("Saving processed data")
                self._save_processed_data(cleaned_projects, interactions_df, feature_matrix)
                logger.info("All data saved successfully")
                
                return cleaned_projects, interactions_df, feature_matrix
                
            except Exception as e:
                logger.error(f"Unexpected error in process_data: {e}")
                logger.error(traceback.format_exc())
                return None, None, None
    
    def ensure_valid_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all columns have the correct data types
        
        Args:
            df: DataFrame to validate
            
        Returns:
            pd.DataFrame: DataFrame with corrected data types
        """
        logger.info("Ensuring valid data types")
        
        # Copy to avoid modifying original
        validated_df = df.copy()
        
        # Fix platforms column
        if 'platforms' in validated_df.columns:
            for idx, row in validated_df.iterrows():
                platforms = row['platforms']
                if isinstance(platforms, str):
                    try:
                        # Parse JSON string
                        validated_df.at[idx, 'platforms'] = json.loads(platforms)
                    except json.JSONDecodeError:
                        try:
                            # Try eval as fallback (secure since we control the data)
                            validated_df.at[idx, 'platforms'] = eval(platforms)
                        except:
                            validated_df.at[idx, 'platforms'] = {}
                elif not isinstance(platforms, dict):
                    validated_df.at[idx, 'platforms'] = {}
        
        # Fix categories column
        if 'categories' in validated_df.columns:
            for idx, row in validated_df.iterrows():
                categories = row['categories']
                if isinstance(categories, str):
                    try:
                        # Parse JSON string
                        validated_df.at[idx, 'categories'] = json.loads(categories)
                    except json.JSONDecodeError:
                        try:
                            # Try eval as fallback
                            validated_df.at[idx, 'categories'] = eval(categories)
                        except:
                            validated_df.at[idx, 'categories'] = []
                elif not isinstance(categories, list):
                    validated_df.at[idx, 'categories'] = []
        
        # Ensure numerical columns are actually numeric
        numeric_columns = [
            'market_cap', 'total_volume', 'current_price', 
            'price_change_percentage_24h', 'reddit_subscribers', 
            'twitter_followers', 'github_stars'
        ]
        
        for col in numeric_columns:
            if col in validated_df.columns:
                # Convert to numeric, force errors to NaN
                validated_df[col] = pd.to_numeric(validated_df[col], errors='coerce')
                
                # Fill NaN with appropriate default
                validated_df[col] = validated_df[col].fillna(0)
        
        logger.info("Data types validation completed")
        return validated_df
    
    def _save_processed_data(self, projects_df: pd.DataFrame, interactions_df: pd.DataFrame, feature_matrix: pd.DataFrame) -> None:
        """
        Save processed data with improved formats and integrity
        
        Args:
            projects_df: DataFrame of processed projects
            interactions_df: DataFrame of user interactions
            feature_matrix: Feature matrix for modeling
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure directories exist
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        
        # 1. First save full data as JSON to preserve complex structures
        json_projects_path = os.path.join(PROCESSED_DATA_PATH, f"processed_projects_{timestamp}.json")
        
        # Prepare projects for JSON serialization
        def prepare_for_json(df):
            result = []
            for _, row in df.iterrows():
                # Convert row to dict
                row_dict = row.to_dict()
                
                # Handle numpy types
                for key, value in row_dict.items():
                    if isinstance(value, np.integer):
                        row_dict[key] = int(value)
                    elif isinstance(value, np.floating):
                        row_dict[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        row_dict[key] = value.tolist()
                
                result.append(row_dict)
            return result
        
        try:
            with open(json_projects_path, 'w', encoding='utf-8') as f:
                json.dump(prepare_for_json(projects_df), f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)
            logger.info(f"Saved complete project data as JSON to {json_projects_path}")
        except Exception as e:
            logger.error(f"Error saving projects JSON: {e}")
        
        # 2. Save CSV version with flattened structures for easier inspection
        try:
            # Prepare CSV-friendly version
            csv_df = projects_df.copy()
            
            # Convert JSON columns to string representation
            if 'platforms' in csv_df.columns:
                csv_df['platforms'] = csv_df['platforms'].apply(
                    lambda x: ', '.join(list(x.keys())) if isinstance(x, dict) and x else ""
                )
            
            if 'categories' in csv_df.columns:
                csv_df['categories'] = csv_df['categories'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else ""
                )
            
            # Save to CSV with quotes only when needed
            projects_path = os.path.join(PROCESSED_DATA_PATH, f"processed_projects_{timestamp}.csv")
            csv_df.to_csv(projects_path, index=False, quoting=csv.QUOTE_MINIMAL)
            logger.info(f"Saved processed projects CSV to {projects_path}")
            
            # Also save standard filename version
            standard_path = os.path.join(PROCESSED_DATA_PATH, "processed_projects.csv")
            csv_df.to_csv(standard_path, index=False, quoting=csv.QUOTE_MINIMAL)
        except Exception as e:
            logger.error(f"Error saving projects CSV: {e}")
        
        # 3. Save interactions data
        try:
            interactions_path = os.path.join(PROCESSED_DATA_PATH, f"user_interactions_{timestamp}.csv")
            interactions_df.to_csv(interactions_path, index=False)
            logger.info(f"Saved user interactions to {interactions_path}")
            
            # Standard filename
            standard_int_path = os.path.join(PROCESSED_DATA_PATH, "user_interactions.csv")
            interactions_df.to_csv(standard_int_path, index=False)
        except Exception as e:
            logger.error(f"Error saving interactions: {e}")
        
        # 4. Save feature matrix
        try:
            feature_path = os.path.join(PROCESSED_DATA_PATH, f"feature_matrix_{timestamp}.csv")
            feature_matrix.to_csv(feature_path, index_label="project_id")
            logger.info(f"Saved feature matrix to {feature_path}")

            # Standard filename
            standard_feat_path = os.path.join(PROCESSED_DATA_PATH, "feature_matrix.csv")
            feature_matrix.to_csv(standard_feat_path, index_label="project_id")
        except Exception as e:
            logger.error(f"Error saving feature matrix: {e}")
    
    def load_latest_processed_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load latest processed data from processed directory
        
        Returns:
            tuple: (projects_df, interactions_df, feature_matrix)
        """
        logger.info("Loading latest processed data")
        
        # Look for processed projects files
        projects_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("processed_projects_") and f.endswith(".csv")
        ]
        
        projects_df = None
        if projects_files:
            try:
                latest_projects_file = max(projects_files)
                projects_path = os.path.join(PROCESSED_DATA_PATH, latest_projects_file)
                projects_df = pd.read_csv(projects_path, low_memory=False)
                logger.info(f"Loaded {len(projects_df)} projects from {latest_projects_file}")
                
                # Convert JSON columns
                if 'platforms' in projects_df.columns and 'categories' in projects_df.columns:
                    projects_df = self.convert_json_columns(projects_df, ['platforms', 'categories'])
            except Exception as e:
                logger.error(f"Error loading processed projects: {e}")
        
        if projects_df is None:
            # Try standard filename
            standard_path = os.path.join(PROCESSED_DATA_PATH, "processed_projects.csv")
            if os.path.exists(standard_path):
                try:
                    projects_df = pd.read_csv(standard_path, low_memory=False)
                    logger.info(f"Loaded {len(projects_df)} projects from standard file")
                    
                    # Convert JSON columns
                    if 'platforms' in projects_df.columns and 'categories' in projects_df.columns:
                        projects_df = self.convert_json_columns(projects_df, ['platforms', 'categories'])
                except Exception as e:
                    logger.error(f"Error loading standard projects file: {e}")
            else:
                logger.error("No processed projects data found")
        
        # Look for interaction files
        interactions_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("user_interactions_") and f.endswith(".csv")
        ]
        
        interactions_df = None
        if interactions_files:
            try:
                latest_interactions_file = max(interactions_files)
                interactions_path = os.path.join(PROCESSED_DATA_PATH, latest_interactions_file)
                interactions_df = pd.read_csv(interactions_path)
                logger.info(f"Loaded {len(interactions_df)} interactions from {latest_interactions_file}")
            except Exception as e:
                logger.error(f"Error loading interactions: {e}")
        
        if interactions_df is None:
            # Try standard filename
            standard_path = os.path.join(PROCESSED_DATA_PATH, "user_interactions.csv")
            if os.path.exists(standard_path):
                try:
                    interactions_df = pd.read_csv(standard_path)
                    logger.info(f"Loaded {len(interactions_df)} interactions from standard file")
                except Exception as e:
                    logger.error(f"Error loading standard interactions file: {e}")
            else:
                logger.warning("No interactions data found")
        
        # Look for feature matrix files
        feature_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("feature_matrix_") and f.endswith(".csv")
        ]
        
        feature_matrix = None
        if feature_files:
            try:
                latest_feature_file = max(feature_files)
                feature_path = os.path.join(PROCESSED_DATA_PATH, latest_feature_file)
                feature_matrix = pd.read_csv(feature_path, index_col="project_id")
                logger.info(f"Loaded feature matrix from {latest_feature_file}")
            except Exception as e:
                logger.error(f"Error loading feature matrix: {e}")
        
        if feature_matrix is None:
            # Try standard filename
            standard_path = os.path.join(PROCESSED_DATA_PATH, "feature_matrix.csv")
            if os.path.exists(standard_path):
                try:
                    feature_matrix = pd.read_csv(standard_path, index_col="project_id")
                    logger.info("Loaded feature matrix from standard file")
                except Exception as e:
                    logger.error(f"Error loading standard feature matrix: {e}")
            else:
                logger.warning("No feature matrix found")
        
        return projects_df, interactions_df, feature_matrix


if __name__ == "__main__":
    # Set up command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Process Web3 project data")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--load-only", action="store_true", help="Only load and validate data without processing")
    parser.add_argument("--users", type=int, default=500, help="Number of synthetic users to generate")
    args = parser.parse_args()
    
    # Create processor
    processor = DataProcessor(debug_mode=args.debug)
    
    if args.load_only:
        # Just load and validate
        projects_df, _, _ = processor.load_latest_data()
        if projects_df is not None:
            print(f"Successfully loaded {len(projects_df)} projects")
            print(f"Sample columns: {projects_df.columns[:10].tolist()}")
            
            # Check some key fields
            print("\nSample project details:")
            if len(projects_df) > 0:
                sample = projects_df.iloc[0]
                print(f"Name: {sample.get('name', 'N/A')}")
                print(f"Symbol: {sample.get('symbol', 'N/A')}")
                print(f"Market Cap: {sample.get('market_cap', 'N/A')}")
                
                # Show platforms and categories if available
                if 'platforms' in sample:
                    print(f"Platforms: {sample['platforms']}")
                if 'categories' in sample:
                    print(f"Categories: {sample['categories']}")
        else:
            print("Failed to load project data")
    else:
        # Process full pipeline
        projects_df, interactions_df, feature_matrix = processor.process_data()
        
        if projects_df is not None:
            print(f"Successfully processed {len(projects_df)} projects")
            print(f"Created {len(interactions_df)} synthetic interactions")
            print(f"Feature matrix shape: {feature_matrix.shape}")
        else:
            print("Data processing failed")