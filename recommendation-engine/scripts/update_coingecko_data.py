"""
Script untuk mengambil data dari CoinGecko API dan menyimpannya ke database
"""

import os
import sys
import logging
import pandas as pd
import psycopg2
import json
from datetime import datetime

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import dan setup central logging
from central_logging import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# Import modules
from config.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
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

def update_coingecko_data():
    """
    Ambil data dari CoinGecko API dan simpan ke database
    """
    logger.info("Starting CoinGecko data update")
    
    # Inisialisasi collector
    collector = CoinGeckoCollector()
    
    # Collect data - menggunakan _ untuk variabel yang tidak digunakan
    projects_df, _, _ = collector.collect_all_data()
    
    if projects_df is None or projects_df.empty:
        logger.error("Failed to collect data from CoinGecko")
        return False
    
    # Simpan ke database
    save_projects_to_db(projects_df)
    
    logger.info("CoinGecko data update completed successfully")
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
                    image_url = %s, 
                    popularity_score = %s, 
                    social_score = %s, 
                    reddit_subscribers = %s, 
                    twitter_followers = %s, 
                    github_stars = %s,
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
                    project.get('price_change_percentage_24h', None),
                    project.get('image', None),
                    popularity_score,
                    social_score,
                    project.get('reddit_subscribers', 0),
                    project.get('twitter_followers', 0),
                    project.get('github_stars', 0),
                    project['id']
                ))
                
                count_update += 1
            else:
                # Insert new project
                query = """
                INSERT INTO projects (
                    id, name, symbol, categories, platforms, market_cap, volume_24h, 
                    price_usd, price_change_24h, image_url, popularity_score, social_score,
                    reddit_subscribers, twitter_followers, github_stars
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    project.get('price_change_percentage_24h', None),
                    project.get('image', None),
                    popularity_score,
                    social_score,
                    project.get('reddit_subscribers', 0),
                    project.get('twitter_followers', 0),
                    project.get('github_stars', 0)
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

if __name__ == "__main__":
    update_coingecko_data()