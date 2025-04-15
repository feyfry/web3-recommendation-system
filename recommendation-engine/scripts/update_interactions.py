"""
Script untuk mengambil interaksi user dari database dan memperbarui model
"""

import os
import sys
import logging
import pandas as pd
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import numpy as np
import json

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import dan setup central logging
from central_logging import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# Import modules
from config.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
from src.models.matrix_builder import MatrixBuilder
from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.feature_enhanced_cf import FeatureEnhancedCF
from src.models.technical_analysis import generate_trading_signals, personalize_signals

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

def get_interactions_from_db():
    """
    Ambil data interaksi dari database
    
    Returns:
        pd.DataFrame: DataFrame interaksi user
    """
    try:
        conn = get_db_connection()
        
        # Query untuk mengambil interaksi
        query = """
        SELECT user_id, project_id, interaction_type, weight, created_at
        FROM interactions
        ORDER BY created_at DESC
        """
        
        # Baca ke DataFrame
        interactions_df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(interactions_df)} interactions from database")
        return interactions_df
    
    except Exception as e:
        logger.error(f"Error loading interactions from database: {e}")
        return pd.DataFrame()

def get_projects_from_db():
    """
    Ambil data proyek dari database
    
    Returns:
        pd.DataFrame: DataFrame proyek
    """
    try:
        conn = get_db_connection()
        
        # Query untuk mengambil proyek
        query = """
        SELECT id, name, symbol, market_cap, volume_24h, price_usd, 
               price_change_24h, price_change_percentage_24h, price_change_percentage_7d,
               price_change_percentage_1h, price_change_percentage_30d,
               image_url, popularity_score, trend_score, social_score,
               categories, platforms, reddit_subscribers, twitter_followers, 
               github_stars, primary_category, chain, genesis_date, 
               sentiment_positive, description
        FROM projects
        """
        
        # Baca ke DataFrame
        projects_df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(projects_df)} projects from database")
        return projects_df
    
    except Exception as e:
        logger.error(f"Error loading projects from database: {e}")
        return pd.DataFrame()

def update_recommendation_model():
    """
    Update model rekomendasi berdasarkan interaksi terbaru
    """
    logger.info("Starting recommendation model update")
    
    # Ambil data dari database
    interactions_df = get_interactions_from_db()
    projects_df = get_projects_from_db()
    
    if interactions_df.empty or projects_df.empty:
        logger.error("Failed to retrieve data from database")
        return False
    
    # Bangun matriks
    logger.info("Building matrices")
    matrix_builder = MatrixBuilder()
    matrices = matrix_builder.build_matrices_from_df(interactions_df, projects_df)
    
    if matrices[0] is None:
        logger.error("Failed to build matrices")
        return False
    
    user_item_df, _, _, item_similarity_df, _, _, _ = matrices
    
    # Simpan similarity matrix ke database (opsional)
    save_similarity_matrix(item_similarity_df, 'item-based')
    
    # Generate rekomendasi untuk semua user
    generate_recommendations_for_all_users(user_item_df, item_similarity_df, projects_df)
    
    # Generate rekomendasi teknikal berdasarkan analisis harga
    generate_technical_recommendations()
    
    logger.info("Recommendation model update completed successfully")
    return True

def save_similarity_matrix(similarity_df, sim_type):
    """
    Simpan matriks kesamaan ke database
    
    Args:
        similarity_df (pd.DataFrame): Matriks kesamaan
        sim_type (str): Tipe kesamaan
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Hapus matriks lama
        cursor.execute("DELETE FROM similarity_matrix WHERE similarity_type = %s", (sim_type,))
        
        # Simpan matriks baru (simpan hanya nilai yang cukup tinggi untuk menghemat ruang)
        threshold = 0.1
        insert_query = """
        INSERT INTO similarity_matrix 
        (project_id_1, project_id_2, similarity_score, similarity_type)
        VALUES (%s, %s, %s, %s)
        """
        
        count = 0
        for proj1 in similarity_df.index:
            for proj2 in similarity_df.columns:
                if proj1 != proj2:
                    score = float(similarity_df.loc[proj1, proj2])
                    if score > threshold:
                        cursor.execute(insert_query, (proj1, proj2, score, sim_type))
                        count += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Saved {count} similarity scores to database")
    
    except Exception as e:
        logger.error(f"Error saving similarity matrix to database: {e}")

def generate_recommendations_for_all_users(user_item_df, item_similarity_df, projects_df):
    """
    Generate rekomendasi untuk semua user dan simpan ke database
    
    Args:
        user_item_df (pd.DataFrame): Matriks user-item
        item_similarity_df (pd.DataFrame): Matriks kesamaan item
        projects_df (pd.DataFrame): DataFrame proyek
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ambil semua user dari database
        cursor.execute("SELECT user_id FROM users")
        users = [row[0] for row in cursor.fetchall()]
        
        # Inisialisasi model
        cf = CollaborativeFiltering()
        cf.user_item_df = user_item_df
        cf.item_similarity_df = item_similarity_df
        cf.projects_df = projects_df
        
        feature_cf = FeatureEnhancedCF()
        
        # Generate rekomendasi trending untuk semua user (hanya sekali)
        trending_recs = cf.get_trending_projects(n=20)
        popular_recs = cf.get_popular_projects(n=20)
        
        # Hapus rekomendasi trending dan popular lama
        cursor.execute("DELETE FROM recommendations WHERE recommendation_type IN ('trending', 'popular')")
        
        # Simpan rekomendasi trending dan popular untuk semua user
        for user_id in users:
            # Simpan trending
            for i, rec in enumerate(trending_recs):
                project_id = rec.get('id')
                score = rec.get('recommendation_score', 0)
                
                cursor.execute("""
                INSERT INTO recommendations (user_id, project_id, score, rank, recommendation_type)
                VALUES (%s, %s, %s, %s, 'trending')
                """, (user_id, project_id, score, i+1))
            
            # Simpan popular
            for i, rec in enumerate(popular_recs):
                project_id = rec.get('id')
                score = rec.get('recommendation_score', 0)
                
                cursor.execute("""
                INSERT INTO recommendations (user_id, project_id, score, rank, recommendation_type)
                VALUES (%s, %s, %s, %s, 'popular')
                """, (user_id, project_id, score, i+1))
        
        # Generate rekomendasi personal untuk setiap user
        for user_id in users:
            if user_id in user_item_df.index:
                logger.info(f"Generating recommendations for user {user_id}")
                
                # Hapus rekomendasi lama
                cursor.execute("""
                DELETE FROM recommendations 
                WHERE user_id = %s AND recommendation_type IN ('hybrid', 'user-based', 'item-based', 'feature-enhanced')
                """, (user_id,))
                
                # Generate hybrid recommendations
                hybrid_recs = cf.hybrid_recommendations(user_id, n=20, feature_cf=feature_cf)
                for i, rec in enumerate(hybrid_recs):
                    project_id = rec.get('id')
                    score = rec.get('recommendation_score', 0)
                    
                    cursor.execute("""
                    INSERT INTO recommendations (user_id, project_id, score, rank, recommendation_type)
                    VALUES (%s, %s, %s, %s, 'hybrid')
                    """, (user_id, project_id, score, i+1))
                
                # Generate user-based recommendations
                user_recs = cf.user_based_cf(user_id, n=20)
                for i, rec in enumerate(user_recs):
                    project_id = rec[0]
                    score = rec[1]
                    
                    cursor.execute("""
                    INSERT INTO recommendations (user_id, project_id, score, rank, recommendation_type)
                    VALUES (%s, %s, %s, %s, 'user-based')
                    """, (user_id, project_id, score, i+1))
                
                # Generate item-based recommendations
                item_recs = cf.item_based_cf(user_id, n=20)
                for i, rec in enumerate(item_recs):
                    project_id = rec[0]
                    score = rec[1]
                    
                    cursor.execute("""
                    INSERT INTO recommendations (user_id, project_id, score, rank, recommendation_type)
                    VALUES (%s, %s, %s, %s, 'item-based')
                    """, (user_id, project_id, score, i+1))
                
                # Generate feature-enhanced recommendations
                feature_recs = feature_cf.recommend_projects(user_id, user_item_df, item_similarity_df, projects_df, n=20)
                for i, rec in enumerate(feature_recs):
                    project_id = rec.get('id')
                    score = rec.get('recommendation_score', 0)
                    
                    cursor.execute("""
                    INSERT INTO recommendations (user_id, project_id, score, rank, recommendation_type)
                    VALUES (%s, %s, %s, %s, 'feature-enhanced')
                    """, (user_id, project_id, score, i+1))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Recommendations generated and saved to database")
    
    except Exception as e:
        logger.error(f"Error generating recommendations for users: {e}")

def generate_technical_recommendations():
    """
    Generate rekomendasi teknikal berdasarkan analisis harga
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ambil daftar proyek teratas berdasarkan market cap
        cursor.execute("""
        SELECT id FROM projects 
        WHERE price_usd IS NOT NULL AND price_usd > 0
        ORDER BY market_cap DESC NULLS LAST
        LIMIT 50
        """)
        
        top_projects = [row[0] for row in cursor.fetchall()]
        
        # Ambil daftar user
        cursor.execute("SELECT user_id, risk_tolerance FROM users")
        users = cursor.fetchall()
        
        # Hapus rekomendasi teknikal lama
        cursor.execute("DELETE FROM recommendations WHERE recommendation_type = 'technical'")
        
        # Generate rekomendasi untuk setiap proyek
        for project_id in top_projects:
            try:
                # Ambil data harga historis
                cursor.execute("""
                SELECT timestamp, price, volume, market_cap
                FROM historical_prices
                WHERE project_id = %s
                ORDER BY timestamp DESC
                LIMIT 168  -- 7 days with hourly data
                """, (project_id,))
                
                price_history = cursor.fetchall()
                
                if not price_history or len(price_history) < 24:
                    logger.warning(f"Insufficient price history for {project_id}")
                    continue
                
                # Konversi ke DataFrame
                df = pd.DataFrame(price_history, columns=['timestamp', 'price', 'volume', 'market_cap'])
                
                # Ambil informasi proyek
                cursor.execute("""
                SELECT * FROM projects WHERE id = %s
                """, (project_id,))
                
                project_info = cursor.fetchone()
                project_dict = {}
                if project_info:
                    columns = [desc[0] for desc in cursor.description]
                    project_dict = dict(zip(columns, project_info))
                
                # Generate sinyal trading
                signals = generate_trading_signals(df, project_dict)
                
                # Simpan rekomendasi untuk setiap user dengan personalisasi
                for user_id, risk_tolerance in users:
                    # Personalisasi sinyal berdasarkan profil risiko
                    personalized = personalize_signals(signals, risk_tolerance or 'medium')
                    
                    # Simpan rekomendasi dengan detail teknikal
                    cursor.execute("""
                    INSERT INTO recommendations (
                        user_id, project_id, score, rank, recommendation_type, 
                        action_type, confidence_score, target_price, explanation,
                        created_at, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '24 hours')
                    """, (
                        user_id,
                        project_id,
                        personalized.get('confidence', 0.5) * 100,  # Score between 0-100
                        1,  # Default rank
                        'technical',
                        personalized.get('action', 'hold'),
                        personalized.get('confidence', 0.5),
                        personalized.get('target_price'),
                        personalized.get('reason', 'Technical analysis recommendation')
                    ))
                
                logger.info(f"Generated technical recommendations for {project_id}")
            
            except Exception as e:
                logger.error(f"Error generating technical recommendation for {project_id}: {e}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Technical recommendations generated and saved to database")
    
    except Exception as e:
        logger.error(f"Error generating technical recommendations: {e}")

def update_trading_signals():
    """
    Update trading signals for projects in user portfolios
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get projects in user portfolios
        cursor.execute("""
        SELECT DISTINCT p.user_id, p.project_id 
        FROM portfolios p
        JOIN projects pr ON p.project_id = pr.id
        WHERE p.amount > 0 
        """)
        
        portfolio_items = cursor.fetchall()
        
        # Generate signals for each portfolio item
        for user_id, project_id in portfolio_items:
            try:
                # Get historical price data
                cursor.execute("""
                SELECT timestamp, price, volume, market_cap
                FROM historical_prices
                WHERE project_id = %s
                ORDER BY timestamp DESC
                LIMIT 168  -- 7 days with hourly data
                """, (project_id,))
                
                price_history = cursor.fetchall()
                
                if not price_history or len(price_history) < 24:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(price_history, columns=['timestamp', 'price', 'volume', 'market_cap'])
                
                # Get project info
                cursor.execute("""
                SELECT * FROM projects WHERE id = %s
                """, (project_id,))
                
                project_info = cursor.fetchone()
                project_dict = {}
                if project_info:
                    columns = [desc[0] for desc in cursor.description]
                    project_dict = dict(zip(columns, project_info))
                
                # Get user risk profile
                cursor.execute("SELECT risk_tolerance FROM users WHERE user_id = %s", (user_id,))
                user_data = cursor.fetchone()
                risk_tolerance = user_data[0] if user_data else 'medium'
                
                # Generate and personalize signals
                signals = generate_trading_signals(df, project_dict)
                personalized = personalize_signals(signals, risk_tolerance)
                
                # Check if there's a significant change from previous recommendation
                cursor.execute("""
                SELECT action_type FROM recommendations 
                WHERE user_id = %s AND project_id = %s AND recommendation_type = 'technical'
                ORDER BY created_at DESC LIMIT 1
                """, (user_id, project_id))
                
                prev_action = cursor.fetchone()
                prev_action_type = prev_action[0] if prev_action else None
                
                # If action changed or no previous recommendation exists
                if prev_action_type != personalized.get('action') or prev_action_type is None:
                    # Save new recommendation
                    cursor.execute("""
                    INSERT INTO recommendations (
                        user_id, project_id, score, rank, recommendation_type, 
                        action_type, confidence_score, target_price, explanation,
                        created_at, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '24 hours')
                    """, (
                        user_id,
                        project_id,
                        personalized.get('confidence', 0.5) * 100,
                        1,
                        'technical',
                        personalized.get('action', 'hold'),
                        personalized.get('confidence', 0.5),
                        personalized.get('target_price'),
                        personalized.get('reason', 'Technical analysis recommendation')
                    ))
                    
                    # Send notification if action changed significantly
                    if prev_action_type is not None and prev_action_type != personalized.get('action'):
                        cursor.execute("""
                        INSERT INTO notifications (
                            user_id, notification_type, title, content, data, created_at
                        ) VALUES (%s, %s, %s, %s, %s, NOW())
                        """, (
                            user_id,
                            'trading_signal',
                            f"New {personalized.get('action').upper()} signal for {project_dict.get('name', project_id)}",
                            personalized.get('reason', 'Technical analysis recommendation'),
                            json.dumps({
                                'project_id': project_id,
                                'action_type': personalized.get('action'),
                                'confidence': personalized.get('confidence'),
                                'target_price': personalized.get('target_price')
                            })
                        ))
            
            except Exception as e:
                logger.error(f"Error updating trading signal for {user_id}, {project_id}: {e}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Trading signals updated for portfolio items")
    
    except Exception as e:
        logger.error(f"Error updating trading signals: {e}")

def update_price_alerts():
    """
    Periksa dan proses price alerts
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ambil semua alert aktif
        cursor.execute("""
        SELECT a.id, a.user_id, a.project_id, a.target_price, a.alert_type, p.price_usd
        FROM price_alerts a
        JOIN projects p ON a.project_id = p.id
        WHERE a.is_triggered = FALSE AND p.price_usd IS NOT NULL
        """)
        
        alerts = cursor.fetchall()
        
        # Periksa setiap alert
        for alert_id, user_id, project_id, target_price, alert_type, current_price in alerts:
            try:
                # Cek apakah alert tercapai
                is_triggered = False
                
                if alert_type == 'above' and current_price >= target_price:
                    is_triggered = True
                elif alert_type == 'below' and current_price <= target_price:
                    is_triggered = True
                
                if is_triggered:
                    # Update alert status
                    cursor.execute("""
                    UPDATE price_alerts
                    SET is_triggered = TRUE, triggered_at = NOW()
                    WHERE id = %s
                    """, (alert_id,))
                    
                    # Ambil nama proyek
                    cursor.execute("SELECT name FROM projects WHERE id = %s", (project_id,))
                    project_name = cursor.fetchone()[0] if cursor.fetchone() else project_id
                    
                    # Buat notifikasi
                    cursor.execute("""
                    INSERT INTO notifications (
                        user_id, notification_type, title, content, data, created_at
                    ) VALUES (%s, %s, %s, %s, %s, NOW())
                    """, (
                        user_id,
                        'price_alert',
                        f"Price Alert: {project_name} is now {alert_type} ${target_price}",
                        f"Current price: ${current_price}",
                        json.dumps({
                            'project_id': project_id,
                            'target_price': target_price,
                            'current_price': current_price,
                            'alert_type': alert_type
                        })
                    ))
                    
                    logger.info(f"Triggered price alert {alert_id} for {user_id}, {project_id}")
            
            except Exception as e:
                logger.error(f"Error processing price alert {alert_id}: {e}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Processed {len(alerts)} price alerts")
    
    except Exception as e:
        logger.error(f"Error updating price alerts: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update recommendation models and technical signals")
    parser.add_argument('--full', action='store_true', help='Run full update including CF models')
    parser.add_argument('--technical', action='store_true', help='Update only technical signals')
    parser.add_argument('--alerts', action='store_true', help='Process price alerts')
    args = parser.parse_args()
    
    if args.technical:
        # Hanya update sinyal teknikal
        generate_technical_recommendations()
        update_trading_signals()
    elif args.alerts:
        # Hanya proses price alerts
        update_price_alerts()
    elif args.full:
        # Update semua
        update_recommendation_model()
        update_price_alerts()
    else:
        # Default - update semua
        update_recommendation_model()