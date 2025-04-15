"""
Script untuk mengambil data dari CoinGecko API dan menyimpannya ke database
"""

import os
import sys
import logging
import pandas as pd
import psycopg2
import json
import requests
from datetime import datetime, timedelta

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import dan setup central logging
from central_logging import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# Import modules
from config.config import (
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    COINGECKO_API_URL, COINGECKO_API_KEY, RATE_LIMIT_DELAY
)
from src.collectors.coingecko_collector import CoinGeckoCollector

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    conn.autocommit = True
    return conn

def update_coingecko_data(update_type='all'):
    """
    Ambil data dari CoinGecko API dan simpan ke database
    
    Args:
        update_type (str): Tipe update ('all', 'market', 'trending', 'full', 'historical')
    """
    logger.info(f"Starting CoinGecko data update: {update_type}")
    
    # Inisialisasi collector
    collector = CoinGeckoCollector()
    
    # Periksa koneksi API
    if not collector.ping_api():
        logger.error("CoinGecko API is not available")
        return False
    
    # Update berdasarkan tipe
    if update_type in ['all', 'market', 'full']:
        # Collect data
        projects_df, _, _ = collector.collect_all_data()
        
        if projects_df is None or projects_df.empty:
            logger.error("Failed to collect data from CoinGecko")
            return False
        
        # Simpan ke database
        save_projects_to_db(projects_df)
    
    if update_type in ['all', 'trending']:
        # Collect trending data
        trending_data = collector.get_trending_coins()
        
        if trending_data and 'coins' in trending_data:
            trending_coins = [item['item'] for item in trending_data['coins']]
            update_trending_score(trending_coins)
    
    if update_type in ['all', 'full', 'historical']:
        # Update historical price data for top projects
        update_historical_prices(collector)
    
    logger.info(f"CoinGecko data update ({update_type}) completed successfully")
    return True

def save_projects_to_db(projects_df):
    """
    Simpan data proyek ke database
    
    Args:
        projects_df (pd.DataFrame): DataFrame proyek
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Untuk setiap proyek, simpan atau perbarui di database
        count_insert = 0
        count_update = 0
        
        for _, project in projects_df.iterrows():
            # Cek apakah proyek sudah ada
            cursor.execute("SELECT COUNT(*) FROM projects WHERE id = %s", (project['id'],))
            exists = cursor.fetchone()[0] > 0
            
            # Konversi JSON ke format JSONB PostgreSQL jika perlu
            categories = project.get('categories', [])
            platforms = project.get('platforms', {})
            
            if isinstance(categories, str):
                try:
                    categories = json.loads(categories)
                except:
                    categories = []
            
            if isinstance(platforms, str):
                try:
                    platforms = json.loads(platforms)
                except:
                    platforms = {}
            
            # Hitung popularity score
            market_cap = float(project.get('market_cap', 0) or 0)
            volume_24h = float(project.get('total_volume', 0) or 0)
            reddit = int(project.get('reddit_subscribers', 0) or 0)
            twitter = int(project.get('twitter_followers', 0) or 0)
            
            # Log transform
            import numpy as np
            market_cap_log = np.log1p(market_cap) / 30 if market_cap > 0 else 0
            volume_log = np.log1p(volume_24h) / 25 if volume_24h > 0 else 0
            reddit_log = np.log1p(reddit) / 15 if reddit > 0 else 0
            twitter_log = np.log1p(twitter) / 15 if twitter > 0 else 0
            
            # Calculate popularity score (0-100)
            popularity_score = (
                0.4 * market_cap_log + 
                0.3 * volume_log + 
                0.15 * reddit_log + 
                0.15 * twitter_log
            ) * 100
            
            # Hitung social score
            social_score = (reddit_log + twitter_log) * 100 / 2
            
            # Ekstrak primary_category
            primary_category = 'unknown'
            if categories and len(categories) > 0:
                if isinstance(categories[0], str):
                    primary_category = categories[0].lower()
            
            # Ekstrak chain
            chain = 'unknown'
            if platforms:
                if 'ethereum' in platforms:
                    chain = 'ethereum'
                elif 'binance-smart-chain' in platforms:
                    chain = 'binance-smart-chain'
                elif len(platforms) > 0:
                    chain = list(platforms.keys())[0]
            
            if exists:
                # Update existing project
                query = """
                UPDATE projects SET 
                    name = %s, 
                    symbol = %s, 
                    categories = %s, 
                    platforms = %s, 
                    market_cap = %s, 
                    volume_24h = %s, 
                    price_usd = %s, 
                    price_change_24h = %s,
                    price_change_percentage_24h = %s,
                    price_change_percentage_7d = %s,
                    price_change_percentage_1h = %s,
                    price_change_percentage_30d = %s,
                    image_url = %s, 
                    popularity_score = %s, 
                    social_score = %s, 
                    reddit_subscribers = %s, 
                    twitter_followers = %s, 
                    github_stars = %s,
                    primary_category = %s,
                    chain = %s,
                    updated_at = NOW()
                WHERE id = %s
                """
                
                cursor.execute(query, (
                    project.get('name', ''),
                    project.get('symbol', ''),
                    json.dumps(categories),
                    json.dumps(platforms),
                    project.get('market_cap', None),
                    project.get('total_volume', None),
                    project.get('current_price', None),
                    project.get('price_change_24h', None),
                    project.get('price_change_percentage_24h', None),
                    project.get('price_change_percentage_7d_in_currency', None),
                    project.get('price_change_percentage_1h_in_currency', None),
                    project.get('price_change_percentage_30d_in_currency', None),
                    project.get('image', None),
                    popularity_score,
                    social_score,
                    project.get('reddit_subscribers', 0),
                    project.get('twitter_followers', 0),
                    project.get('github_stars', 0),
                    primary_category,
                    chain,
                    project['id']
                ))
                
                count_update += 1
            else:
                # Insert new project
                query = """
                INSERT INTO projects (
                    id, name, symbol, categories, platforms, market_cap, volume_24h, 
                    price_usd, price_change_24h, price_change_percentage_24h,
                    price_change_percentage_7d, price_change_percentage_1h, 
                    price_change_percentage_30d, image_url, popularity_score, 
                    social_score, reddit_subscribers, twitter_followers, 
                    github_stars, primary_category, chain
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(query, (
                    project['id'],
                    project.get('name', ''),
                    project.get('symbol', ''),
                    json.dumps(categories),
                    json.dumps(platforms),
                    project.get('market_cap', None),
                    project.get('total_volume', None),
                    project.get('current_price', None),
                    project.get('price_change_24h', None),
                    project.get('price_change_percentage_24h', None),
                    project.get('price_change_percentage_7d_in_currency', None),
                    project.get('price_change_percentage_1h_in_currency', None),
                    project.get('price_change_percentage_30d_in_currency', None),
                    project.get('image', None),
                    popularity_score,
                    social_score,
                    project.get('reddit_subscribers', 0),
                    project.get('twitter_followers', 0),
                    project.get('github_stars', 0),
                    primary_category,
                    chain
                ))
                
                count_insert += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Saved {count_insert} new projects and updated {count_update} existing projects")
    
    except Exception as e:
        logger.error(f"Error saving projects to database: {e}")
        # Menentukan kelas exception yang lebih spesifik atau re-raise exception
        raise RuntimeError(f"Database operation failed: {e}") from e

def update_trending_score(trending_coins):
    """
    Perbarui trend_score untuk koin yang sedang trending
    
    Args:
        trending_coins (list): List koin trending
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Reset semua trend score
        cursor.execute("UPDATE projects SET trend_score = 50")
        
        # Perbarui trend score untuk koin trending
        for i, coin in enumerate(trending_coins):
            if 'id' in coin:
                # Skor berdasarkan rank trending (1-7 skala)
                trend_boost = 30 + (8 - min(8, i)) * 5
                
                cursor.execute("""
                UPDATE projects 
                SET trend_score = popularity_score + %s, 
                    updated_at = NOW()
                WHERE id = %s
                """, (trend_boost, coin['id']))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Updated trend score for {len(trending_coins)} trending coins")
    
    except Exception as e:
        logger.error(f"Error updating trending score: {e}")

def update_historical_prices(collector):
    """
    Perbarui data harga historis untuk proyek teratas
    
    Args:
        collector: CoinGeckoCollector instance
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ambil top 100 proyek berdasarkan market cap
        cursor.execute("""
        SELECT id FROM projects 
        ORDER BY market_cap DESC NULLS LAST 
        LIMIT 100
        """)
        
        top_projects = [row[0] for row in cursor.fetchall()]
        
        # Perbarui data historis untuk setiap proyek
        for project_id in top_projects:
            try:
                collect_historical_prices(project_id, conn)
                # Delay untuk menghindari rate limiting
                time.sleep(RATE_LIMIT_DELAY)
            except Exception as e:
                logger.error(f"Error updating historical prices for {project_id}: {e}")
        
        conn.close()
        
        logger.info(f"Updated historical prices for {len(top_projects)} projects")
    
    except Exception as e:
        logger.error(f"Error updating historical prices: {e}")

def collect_historical_prices(coin_id, conn=None, days=7):
    """
    Collect historical price data for a specific coin
    
    Args:
        coin_id: Coin ID
        conn: Database connection (optional)
        days: Number of days of historical data
    """
    logger.info(f"Collecting historical prices for {coin_id}")
    
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "hourly",
        "x_cg_demo_api_key": COINGECKO_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'prices' not in data:
            logger.error(f"No price data available for {coin_id}")
            return
        
        # Buat koneksi DB jika tidak diberikan
        close_conn = False
        if conn is None:
            conn = get_db_connection()
            close_conn = True
            
        cursor = conn.cursor()
        
        # Hapus data lama untuk menghindari duplikasi
        cursor.execute(
            "DELETE FROM historical_prices WHERE project_id = %s AND timestamp >= NOW() - INTERVAL %s DAY",
            (coin_id, days)
        )
        
        # Simpan data baru
        for timestamp_ms, price in data['prices']:
            # Konversi timestamp dari miliseconds ke datetime
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            
            # Ambil volume dan market cap untuk timestamp ini jika tersedia
            volume = None
            market_cap = None
            
            if 'total_volumes' in data:
                for vol_timestamp_ms, vol in data['total_volumes']:
                    if vol_timestamp_ms == timestamp_ms:
                        volume = vol
                        break
            
            if 'market_caps' in data:
                for cap_timestamp_ms, cap in data['market_caps']:
                    if cap_timestamp_ms == timestamp_ms:
                        market_cap = cap
                        break
            
            # Simpan ke database
            try:
                cursor.execute(
                    """
                    INSERT INTO historical_prices 
                    (project_id, timestamp, price, volume, market_cap)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (coin_id, timestamp, price, volume, market_cap)
                )
            except psycopg2.Error as e:
                # Jika ada error duplikasi, lewati saja
                logger.warning(f"Error inserting historical price: {e}")
                continue
        
        conn.commit()
        
        if close_conn:
            cursor.close()
            conn.close()
            
        logger.info(f"Saved {len(data['prices'])} historical price points for {coin_id}")
        
    except Exception as e:
        logger.error(f"Error collecting historical prices for {coin_id}: {e}")
        
if __name__ == "__main__":
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description="Update CoinGecko data in database")
    parser.add_argument('--type', default='all', choices=['all', 'market', 'trending', 'full', 'historical'],
                      help='Type of update to perform')
    args = parser.parse_args()
    
    start_time = time.time()
    update_coingecko_data(args.type)
    elapsed_time = time.time() - start_time
    
    print(f"Update completed in {elapsed_time:.2f} seconds")