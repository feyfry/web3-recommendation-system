"""
Module untuk mengumpulkan data dari CoinGecko API
"""

import os
import time
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import logging
import traceback
import sys
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import config
from config.config import (
    COINGECKO_API_URL, 
    COINGECKO_API_KEY, 
    TOP_COINS_LIMIT, 
    CATEGORIES,
    RAW_DATA_PATH,
    RATE_LIMIT_DELAY,
    REQUEST_TIMEOUT
)

# Setup logging
from central_logging import get_logger
logger = get_logger(__name__)

# Import data utilities
from src.utils.data_utils import DateTimeEncoder, safe_convert


class CoinGeckoCollector:
    """
    Class untuk mengumpulkan data dari CoinGecko API
    """
    
    def __init__(self, api_url: str = COINGECKO_API_URL, api_key: str = COINGECKO_API_KEY, 
                 timeout: int = REQUEST_TIMEOUT, rate_limit: float = RATE_LIMIT_DELAY):
        """
        Inisialisasi collector dengan URL API dan API key
        
        Args:
            api_url: URL base API CoinGecko
            api_key: API key untuk CoinGecko API
            timeout: Timeout untuk API request dalam detik
            rate_limit: Delay antara API request dalam detik
        """
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {"accept": "application/json"}
        self.timeout = timeout
        self.rate_limit = rate_limit
        
        # Buat direktori untuk data mentah jika belum ada
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        
    def _make_api_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Membuat request ke CoinGecko API dengan rate limiting
        
        Args:
            endpoint: Endpoint API
            params: Parameter query. Default None.
            
        Returns:
            dict: Response dari API atau None jika gagal
        """
        if params is None:
            params = {}
            
        # Tambahkan API key ke parameter
        params["x_cg_demo_api_key"] = self.api_key
        
        url = f"{self.api_url}/{endpoint}"
        
        try:
            logger.debug(f"Making API request to {url}")
            response = requests.get(url, headers=self.headers, params=params, timeout=self.timeout)
            
            # Check rate limits - Demo API: 10-30 calls/minute
            if response.status_code == 429:  # Too Many Requests
                logger.warning("Rate limit exceeded. Waiting for 60 seconds...")
                time.sleep(60)
                return self._make_api_request(endpoint, params)
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making API request to {url}: {str(e)}")
            return None
            
    def ping_api(self) -> bool:
        """
        Memeriksa status API
        
        Returns:
            bool: True jika API tersedia, False jika tidak
        """
        response = self._make_api_request("ping")
        return response is not None and "gecko_says" in response
        
    def get_coins_list(self) -> Optional[pd.DataFrame]:
        """
        Mendapatkan daftar semua koin yang tersedia di CoinGecko
        
        Returns:
            pd.DataFrame: DataFrame dengan informasi koin atau None jika gagal
        """
        response = self._make_api_request("coins/list", {"include_platform": "true"})
        
        if not response:
            logger.error("Failed to get coins list")
            return None
            
        logger.info(f"Retrieved information for {len(response)} coins")
        
        # Simpan data mentah
        self._save_raw_data("coins_list.json", response)
        
        # Convert ke DataFrame
        try:
            df = pd.DataFrame(response)
            return df
        except Exception as e:
            logger.error(f"Error converting coins list to DataFrame: {e}")
            return None
        
    def get_coins_markets(self, vs_currency: str = "usd", category: Optional[str] = None, 
                         per_page: int = 250, page: int = 1) -> Optional[pd.DataFrame]:
        """
        Mendapatkan data market untuk koin
        
        Args:
            vs_currency: Mata uang untuk nilai market. Default "usd".
            category: Filter berdasarkan kategori. Default None.
            per_page: Jumlah item per halaman. Default 250.
            page: Nomor halaman. Default 1.
            
        Returns:
            pd.DataFrame: DataFrame dengan data market koin atau None jika gagal
        """
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": "true",
            "price_change_percentage": "1h,24h,7d,30d",
            "locale": "en",
            "precision": "full"
        }
        
        if category:
            # Coba beberapa format kategori
            category_formats = [
                category,
                f"{category}s",  # Coba dengan plural
                f"{category}-token",
                f"{category}-tokens",
                # Format lain yang mungkin
            ]
            
            for cat_format in category_formats:
                params["category"] = cat_format
                response = self._make_api_request("coins/markets", params)
                if response:  # Jika berhasil
                    break
        else:
            response = self._make_api_request("coins/markets", params)
        
        if not response:
            logger.error(f"Failed to get coins markets for category: {category}")
            return None
            
        logger.info(f"Retrieved market data for {len(response)} coins")
        
        # Simpan data mentah
        filename = f"coins_markets_{category or 'all'}_page{page}.json"
        self._save_raw_data(filename, response)
        
        # Convert ke DataFrame
        try:
            df = pd.DataFrame(response)
            return df
        except Exception as e:
            logger.error(f"Error converting market data to DataFrame: {e}")
            return None
        
    def get_coin_details(self, coin_id: str) -> Optional[Dict[str, Any]]:
        """
        Mendapatkan detail lengkap untuk koin tertentu
        
        Args:
            coin_id: ID koin di CoinGecko
            
        Returns:
            dict: Detail koin atau None jika gagal
        """
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false"
        }
        
        response = self._make_api_request(f"coins/{coin_id}", params)
        
        if not response:
            logger.error(f"Failed to get details for coin: {coin_id}")
            return None
            
        logger.info(f"Retrieved details for coin: {coin_id}")
        
        # Simpan data mentah
        self._save_raw_data(f"coin_details_{coin_id}.json", response)
        
        return response
        
    def get_coins_categories(self) -> Optional[pd.DataFrame]:
        """
        Mendapatkan daftar kategori koin
        
        Returns:
            pd.DataFrame: DataFrame dengan informasi kategori atau None jika gagal
        """
        response = self._make_api_request("coins/categories")
        
        if not response:
            logger.error("Failed to get coins categories")
            return None
            
        logger.info(f"Retrieved {len(response)} coin categories")
        
        # Simpan data mentah
        self._save_raw_data("coins_categories.json", response)
        
        # Convert ke DataFrame
        try:
            df = pd.DataFrame(response)
            return df
        except Exception as e:
            logger.error(f"Error converting categories to DataFrame: {e}")
            return None
    
    def get_coins_categories_list(self) -> Optional[pd.DataFrame]:
        """
        Mendapatkan daftar semua kategori koin yang tersedia di CoinGecko
        
        Returns:
            pd.DataFrame: DataFrame dengan informasi kategori atau None jika gagal
        """
        response = self._make_api_request("coins/categories/list")
        
        if not response:
            logger.error("Failed to get coins categories list")
            return None
            
        logger.info(f"Retrieved {len(response)} coin category IDs")
        
        # Simpan data mentah
        self._save_raw_data("coins_categories_list.json", response)
        
        # Convert ke DataFrame
        try:
            df = pd.DataFrame(response)
            return df
        except Exception as e:
            logger.error(f"Error converting categories list to DataFrame: {e}")
            return None
        
    def get_trending_coins(self) -> Optional[Dict[str, Any]]:
        """
        Mendapatkan daftar koin yang sedang trending
        
        Returns:
            dict: Informasi koin trending atau None jika gagal
        """
        response = self._make_api_request("search/trending")
        
        if not response:
            logger.error("Failed to get trending coins")
            return None
            
        logger.info("Retrieved trending coins data")
        
        # Simpan data mentah
        self._save_raw_data("trending_coins.json", response)
        
        return response
        
    def collect_all_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Mengumpulkan semua data yang diperlukan dan menyimpannya
        
        Returns:
            tuple: (projects_df, categories_df, trending_df) atau (None, None, None) jika gagal
        """
        logger.info("Starting data collection from CoinGecko API")
        
        # Check API status
        if not self.ping_api():
            logger.error("CoinGecko API is not available")
            return None, None, None
            
        # 1. Collect coins list
        coins_list_df = self.get_coins_list()
        
        # 2. Collect coins categories
        categories_df = self.get_coins_categories()

        # 2b. Collect coins categories list (ID Map)
        categories_list_df = self.get_coins_categories_list()
        
        # 3. Collect trending coins
        trending_data = self.get_trending_coins()
        trending_df = None
        if trending_data and 'coins' in trending_data:
            trending_coins = [item['item'] for item in trending_data['coins']]
            trending_df = pd.DataFrame(trending_coins)
        
        # 4. Get valid categories
        valid_categories = self._get_valid_categories()
        
        # 5. Collect detailed market data for top coins (by market cap)
        all_market_data = []
        
        # Collect general top coins
        for page in range(1, (TOP_COINS_LIMIT // 250) + 2):
            market_df = self.get_coins_markets(page=page)
            if market_df is None or market_df.empty:
                break
            all_market_data.append(market_df)
            if len(market_df) < 250:
                break
                
            # Pause to respect rate limits
            time.sleep(self.rate_limit)
        
        # Collect top coins by category
        for category in CATEGORIES:
            # Validasi kategori
            if category not in valid_categories:
                logger.warning(f"Category '{category}' is not valid, skipping...")
                continue
                
            logger.info(f"Collecting market data for category {category}...")
            category_df = self.get_coins_markets(category=category)
            if category_df is not None and not category_df.empty:
                all_market_data.append(category_df)
            else:
                logger.warning(f"No data returned for category: {category}")
            
            # Pause to respect rate limits
            time.sleep(self.rate_limit)
        
        # Combine all market data and remove duplicates
        if not all_market_data:
            logger.error("No market data collected")
            return None, None, None
            
        market_df = pd.concat(all_market_data, ignore_index=True)
        market_df = market_df.drop_duplicates(subset='id')
        
        # 6. Collect detailed data for top coins
        detailed_data = []
        detail_limit = min(100, len(market_df))  # Limit to top 100 to avoid rate limits
        for idx, coin_id in enumerate(market_df['id'].head(detail_limit).tolist(), 1):
            logger.info(f"Collecting details for coin {idx}/{detail_limit}: {coin_id}...")
            coin_details = self.get_coin_details(coin_id)
            
            if coin_details:
                # Extract relevant data
                try:
                    # Get platforms with proper type handling
                    platforms = coin_details.get('platforms', {})
                    if isinstance(platforms, str):
                        try:
                            platforms = json.loads(platforms)
                        except json.JSONDecodeError:
                            platforms = {}
                    if not isinstance(platforms, dict):
                        platforms = {}
                    
                    # Get categories with proper type handling
                    categories = coin_details.get('categories', [])
                    if isinstance(categories, str):
                        try:
                            categories = json.loads(categories)
                        except json.JSONDecodeError:
                            categories = []
                    if not isinstance(categories, list):
                        categories = []
                    
                    # Get social and developer data
                    community_data = coin_details.get('community_data', {}) or {}
                    developer_data = coin_details.get('developer_data', {}) or {}
                    
                    # Create detail record
                    detailed_info = {
                        'id': coin_id,
                        'platforms': platforms,
                        'categories': categories,
                        'reddit_subscribers': int(safe_convert(community_data.get('reddit_subscribers'), int, 0)),
                        'twitter_followers': int(safe_convert(community_data.get('twitter_followers'), int, 0)),
                        'github_stars': int(safe_convert(developer_data.get('stars'), int, 0)),
                        'description': coin_details.get('description', {}).get('en', '')
                    }
                    detailed_data.append(detailed_info)
                    
                    # Log for important coins
                    if coin_id in ['bitcoin', 'ethereum', 'tether']:
                        logger.info(f"Details for {coin_id}: categories={len(categories)}, platforms={len(platforms)}")
                except Exception as e:
                    logger.error(f"Error processing details for {coin_id}: {e}")
            
            # Pause to respect rate limits
            time.sleep(self.rate_limit)
        
        # Create detailed DataFrame
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            
            # 7. Merge market data with detailed data
            projects_df = market_df.merge(detailed_df, on='id', how='left', suffixes=('', '_detailed'))
        else:
            projects_df = market_df
            logger.warning("No detailed data collected, using market data only")
        
        # 8. Save processed data
        self._save_processed_data(projects_df, categories_df, trending_df)
        
        logger.info("Data collection completed successfully")
        
        return projects_df, categories_df, trending_df
    
    def _get_valid_categories(self) -> Set[str]:
        """
        Mendapatkan daftar kategori yang valid dari CoinGecko API
        
        Returns:
            set: Set kategori yang valid
        """
        logger.info("Fetching valid categories from CoinGecko API")
        
        # Ambil daftar kategori dari API
        response = self._make_api_request("coins/categories/list")
        
        if not response:
            logger.warning("Failed to fetch categories list")
            return set()
            
        # Ekstrak category_id dari respons
        valid_categories = {item['category_id'] for item in response}
        
        logger.info(f"Found {len(valid_categories)} valid categories")
        
        return valid_categories
            
    def _save_raw_data(self, filename: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Menyimpan data mentah dalam format JSON
        
        Args:
            filename: Nama file
            data: Data untuk disimpan
        """
        try:
            # Pastikan data dapat dikonversi ke JSON
            processed_data = self._prepare_json_data(data)
            
            filepath = os.path.join(RAW_DATA_PATH, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)
            logger.info(f"Raw data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving raw data: {e}")
            logger.debug(traceback.format_exc())
    
    def _prepare_json_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Persiapkan data untuk disimpan sebagai JSON dengan memastikan tipe data yang kompatibel
        
        Args:
            data: Data untuk diproses
            
        Returns:
            Data yang telah diproses
        """
        if isinstance(data, dict):
            # Handle single item
            result = {}
            for key, value in data.items():
                if key == 'platforms' and not isinstance(value, str):
                    result[key] = json.dumps(value) if value else "{}"
                elif key == 'categories' and not isinstance(value, str):
                    result[key] = json.dumps(value) if value else "[]"
                elif isinstance(value, (dict, list)):
                    result[key] = self._prepare_json_data(value)
                else:
                    result[key] = value
            return result
        elif isinstance(data, list):
            # Handle list of items
            return [self._prepare_json_data(item) if isinstance(item, (dict, list)) else item for item in data]
        else:
            return data
    
    def _save_processed_data(self, projects_df: pd.DataFrame, categories_df: Optional[pd.DataFrame],
                           trending_df: Optional[pd.DataFrame]) -> None:
        """
        Menyimpan data yang sudah diproses dalam format CSV
        
        Args:
            projects_df: DataFrame proyek
            categories_df: DataFrame kategori
            trending_df: DataFrame trending
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Pastikan kolom JSON terproses dengan benar sebelum disimpan
        # Buat salinan DataFrame untuk manipulasi
        projects_df_csv = projects_df.copy()
        
        # Persiapkan kolom platforms dan categories untuk CSV
        if 'platforms' in projects_df_csv.columns:
            projects_df_csv['platforms'] = projects_df_csv['platforms'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else 
                          (json.dumps({}) if pd.isna(x) else x)
            )
        
        if 'categories' in projects_df_csv.columns:
            projects_df_csv['categories'] = projects_df_csv['categories'].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else 
                          (json.dumps([]) if pd.isna(x) else x)
            )
        
        # Pastikan kedua direktori ada
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        
        # Debug information
        logger.info(f"Projects DataFrame has {len(projects_df)} rows")
        if categories_df is not None:
            logger.info(f"Categories DataFrame has {len(categories_df)} rows")
        if trending_df is not None:
            logger.info(f"Trending DataFrame has {len(trending_df)} rows")
        
        # Simpan data proyek
        if projects_df is not None and not projects_df.empty:
            # Simpan di RAW_DATA_PATH
            projects_path_raw = os.path.join(RAW_DATA_PATH, f"web3_projects_{timestamp}.csv")
            try:
                projects_df_csv.to_csv(projects_path_raw, index=False)
                logger.info(f"Projects data saved to {projects_path_raw}")
                
                # Simpan juga sebagai JSON untuk mempertahankan struktur kompleks
                projects_json_path = os.path.join(RAW_DATA_PATH, f"web3_projects_{timestamp}.json")
                
                # Konversi DataFrame ke records dan simpan sebagai JSON
                projects_records = projects_df.to_dict(orient='records')
                with open(projects_json_path, 'w', encoding='utf-8') as f:
                    json.dump(projects_records, f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)
                logger.info(f"Projects data also saved as JSON to {projects_json_path}")
                
            except Exception as e:
                logger.error(f"Error saving projects data: {e}")
                logger.debug(traceback.format_exc())
        else:
            logger.warning("Projects DataFrame is empty or None, not saving")
        
        # Simpan data kategori
        if categories_df is not None and not categories_df.empty:
            categories_path = os.path.join(RAW_DATA_PATH, f"categories_{timestamp}.csv")
            try:
                categories_df.to_csv(categories_path, index=False)
                logger.info(f"Categories data saved to {categories_path}")
            except Exception as e:
                logger.error(f"Error saving categories data: {e}")
                logger.debug(traceback.format_exc())
        else:
            logger.warning("Categories DataFrame is empty or None, not saving")
        
        # Simpan data trending
        if trending_df is not None and not trending_df.empty:
            trending_path = os.path.join(RAW_DATA_PATH, f"trending_{timestamp}.csv")
            try:
                trending_df.to_csv(trending_path, index=False)
                logger.info(f"Trending data saved to {trending_path}")
            except Exception as e:
                logger.error(f"Error saving trending data: {e}")
                logger.debug(traceback.format_exc())
        else:
            logger.warning("Trending DataFrame is empty or None, not saving")


if __name__ == "__main__":
    # Test function
    collector = CoinGeckoCollector()
    
    # Check API status
    if collector.ping_api():
        print("API is available")
        
        # Collect data sample for testing
        projects_df, categories_df, trending_df = collector.collect_all_data()
        
        if projects_df is not None:
            print(f"Collected data for {len(projects_df)} projects")
        else:
            print("Failed to collect project data")
    else:
        print("API is not available")