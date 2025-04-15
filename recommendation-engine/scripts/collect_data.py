"""
Script untuk mengumpulkan data dari CoinGecko API
"""

import os
import sys
import argparse
import logging
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import dan setup central logging
from central_logging import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# Import modules
from src.collectors.coingecko_collector import CoinGeckoCollector
from src.processors.data_processor import DataProcessor
from src.utils.api_utils import cache_api_response
from config.config import (
    COINGECKO_API_URL, 
    COINGECKO_API_KEY, 
    RAW_DATA_PATH, 
    TOP_COINS_LIMIT, 
    CATEGORIES, 
    RATE_LIMIT_DELAY, 
    REQUEST_TIMEOUT
)

def collect_historical_prices(args, collector, projects_df):
    """
    Collect historical price data for top projects
    
    Args:
        args: Command line arguments
        collector: CoinGeckoCollector instance
        projects_df: DataFrame of projects
    """
    logger.info("Collecting historical price data for top projects")
    
    # Get top projects by market cap
    if projects_df is not None and not projects_df.empty:
        top_projects = projects_df.sort_values('market_cap', ascending=False).head(args.detail_limit)
        
        # Create database connection
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Collect historical data for each project
            for idx, project in top_projects.iterrows():
                project_id = project['id']
                logger.info(f"Collecting historical prices for {project_id} ({idx+1}/{len(top_projects)})")
                
                # Get market chart data
                market_data = collector.get_market_chart(
                    project_id, 
                    days=7,  # Last 7 days data
                    interval='hourly'
                )
                
                if market_data and 'prices' in market_data:
                    # Clear old data to avoid duplicates (optional)
                    cursor.execute(
                        "DELETE FROM historical_prices WHERE project_id = %s AND timestamp >= NOW() - INTERVAL '7 DAY'", 
                        (project_id,)
                    )
                    
                    # Insert new data
                    for timestamp_ms, price in market_data['prices']:
                        timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                        
                        # Get volume and market cap if available
                        volume = None
                        market_cap = None
                        
                        for vol_ts, vol in market_data.get('total_volumes', []):
                            if vol_ts == timestamp_ms:
                                volume = vol
                                break
                                
                        for cap_ts, cap in market_data.get('market_caps', []):
                            if cap_ts == timestamp_ms:
                                market_cap = cap
                                break
                        
                        # Insert data
                        try:
                            cursor.execute(
                                """
                                INSERT INTO historical_prices 
                                (project_id, timestamp, price, volume, market_cap)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (project_id, timestamp) 
                                DO UPDATE SET price = %s, volume = %s, market_cap = %s
                                """,
                                (project_id, timestamp, price, volume, market_cap, price, volume, market_cap)
                            )
                        except Exception as e:
                            logger.warning(f"Error inserting data point: {e}")
                    
                    conn.commit()
                    logger.info(f"Saved {len(market_data['prices'])} data points for {project_id}")
                
                # Respect rate limits
                time.sleep(args.rate_limit)
            
            cursor.close()
            conn.close()
            logger.info("Historical price collection completed")
            
        except Exception as e:
            logger.error(f"Error collecting historical prices: {e}")
    else:
        logger.warning("No projects data available for historical price collection")

def collect_all_data(args) -> bool:
    """
    Mengumpulkan semua data yang diperlukan dari CoinGecko API
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    logger.info("Starting data collection")
    
    # Inisialisasi collector dengan timeout dan rate limit dari config
    collector = CoinGeckoCollector(timeout=args.timeout or REQUEST_TIMEOUT)
    
    # Check API status
    if not args.skip_ping and not collector.ping_api():
        logger.error("CoinGecko API is not available. Aborting.")
        return False
    
    # Collect coins list
    if not args.skip_coins_list:
        logger.info("Collecting coins list...")
        coins_list_df = collector.get_coins_list()
        if coins_list_df is None or coins_list_df.empty:
            logger.warning("Failed to collect coins list")
    
    # Collect coins categories
    if not args.skip_categories:
        logger.info("Collecting coins categories...")
        categories_df = collector.get_coins_categories()
        if categories_df is None or categories_df.empty:
            logger.warning("Failed to collect coins categories")
        
        # Also collect categories list for better mapping
        logger.info("Collecting coins categories list...")
        _ = collector.get_coins_categories_list()
    
    # Collect trending coins
    if not args.skip_trending:
        logger.info("Collecting trending coins...")
        trending_data = collector.get_trending_coins()
        if not trending_data:
            logger.warning("Failed to collect trending coins")
    
    # Collect top coins market data
    all_market_data = []
    if not args.skip_markets:
        logger.info("Collecting top coins market data...")
        
        # General top coins
        limit = args.limit if args.limit else TOP_COINS_LIMIT
        pages = (limit // 250) + (1 if limit % 250 > 0 else 0)
        
        for page in range(1, pages + 1):
            logger.info(f"Collecting market data page {page}/{pages}...")
            market_df = collector.get_coins_markets(page=page)
            if market_df is None or market_df.empty:
                logger.warning(f"Failed to collect market data for page {page}")
                break
            all_market_data.append(market_df)
            if len(market_df) < 250:
                break
            
            # Respect API rate limits
            rate_limit = args.rate_limit if args.rate_limit is not None else RATE_LIMIT_DELAY
            time.sleep(rate_limit)
        
        # Top coins by category
        if not args.skip_categories_markets:
            for category in CATEGORIES:
                logger.info(f"Collecting market data for category {category}...")
                category_df = collector.get_coins_markets(category=category)
                if category_df is not None and not category_df.empty:
                    all_market_data.append(category_df)
                else:
                    logger.warning(f"No data returned for category: {category}")
                
                # Respect API rate limits
                rate_limit = args.rate_limit if args.rate_limit is not None else RATE_LIMIT_DELAY
                time.sleep(rate_limit)
    
    # Collect detailed data for top coins
    if not args.skip_details and all_market_data:
        logger.info("Collecting detailed coin data...")
        
        # Combine all market data and get top coins
        market_df = pd.concat(all_market_data, ignore_index=True).drop_duplicates(subset='id')
        
        # Limit number of coins for detailed data
        detail_limit = min(args.detail_limit, len(market_df))
        top_coins = market_df.sort_values('market_cap', ascending=False).head(detail_limit)['id'].tolist()
        
        logger.info(f"Collecting detailed data for top {detail_limit} coins...")
        successful_details = 0
        
        for i, coin_id in enumerate(top_coins, 1):
            logger.info(f"Collecting details for coin {i}/{detail_limit}: {coin_id}...")
            coin_details = collector.get_coin_details(coin_id)
            
            if coin_details:
                successful_details += 1
            else:
                logger.warning(f"Failed to collect details for coin {coin_id}")
            
            # Progress update every 50 coins
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{detail_limit} coins processed. Success rate: {successful_details/i:.1%}")
            
            # Respect API rate limits
            rate_limit = args.rate_limit if args.rate_limit is not None else RATE_LIMIT_DELAY
            time.sleep(rate_limit)
        
        logger.info(f"Detailed data collection completed. Collected {successful_details}/{detail_limit} coins.")

    # Collect historical prices
    if not args.skip_historical:
        market_df = pd.concat(all_market_data, ignore_index=True).drop_duplicates(subset='id')
        collect_historical_prices(args, collector, market_df)
    
    # Process the collected data
    if args.process:
        logger.info("Processing collected data...")
        
        # Wait a bit to ensure file system operations are complete
        time.sleep(2)
        
        # Verify CSV files exist
        try:
            files = os.listdir(RAW_DATA_PATH)
            csv_files = [f for f in files if f.endswith('.csv')]
            logger.info(f"CSV files in RAW_DATA_PATH: {len(csv_files)} files found")
        except Exception as e:
            logger.warning(f"Error checking RAW_DATA_PATH: {e}")
        
        # Run processor with better error handling
        try:
            processor = DataProcessor()
            processed_data = processor.process_data()
            
            # Check if processing was successful
            if processed_data[0] is None:
                logger.warning("Failed to process data")
            else:
                project_count = len(processed_data[0]) if processed_data[0] is not None else 0
                interaction_count = len(processed_data[1]) if processed_data[1] is not None else 0
                feature_count = len(processed_data[2]) if processed_data[2] is not None else 0
                
                logger.info("Data processing completed successfully:")
                logger.info(f"- Processed {project_count} projects")
                logger.info(f"- Generated {interaction_count} synthetic interactions")
                logger.info(f"- Created feature matrix with {feature_count} entries")
        except Exception as e:
            logger.error(f"Error during data processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("Data collection completed successfully")
    return True

def main() -> int:
    """
    Main function
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(description="Collect data from CoinGecko API")
    
    # Add arguments
    parser.add_argument("--skip-ping", action="store_true", help="Skip API availability check")
    parser.add_argument("--skip-coins-list", action="store_true", help="Skip collecting coins list")
    parser.add_argument("--skip-categories", action="store_true", help="Skip collecting coins categories")
    parser.add_argument("--skip-trending", action="store_true", help="Skip collecting trending coins")
    parser.add_argument("--skip-markets", action="store_true", help="Skip collecting market data")
    parser.add_argument("--skip-categories-markets", action="store_true", help="Skip collecting market data by category")
    parser.add_argument("--skip-details", action="store_true", help="Skip collecting detailed coin data")
    parser.add_argument("--limit", type=int, default=TOP_COINS_LIMIT, help=f"Limit number of top coins (default: {TOP_COINS_LIMIT})")
    parser.add_argument("--detail-limit", type=int, default=100, help="Limit number of coins for detailed data collection (default: 100)")
    parser.add_argument("--rate-limit", type=float, default=RATE_LIMIT_DELAY, help=f"Delay between API requests in seconds (default: {RATE_LIMIT_DELAY})")
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT, help=f"API request timeout in seconds (default: {REQUEST_TIMEOUT})")
    parser.add_argument("--process", action="store_true", help="Process collected data")
    
    args = parser.parse_args()
    
    try:
        success = collect_all_data(args)
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.warning("Data collection interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Error during data collection: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())