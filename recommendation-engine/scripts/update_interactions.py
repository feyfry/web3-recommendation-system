"""
Script untuk mengambil interaksi user dari database dan memperbarui model
"""

import os
import sys
import logging
import pandas as pd
import psycopg2
import psycopg2.extras
from datetime import datetime

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
               price_change_24h, image_url, popularity_score, social_score,
               categories, platforms, reddit_subscribers, twitter_followers, github_stars
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
                hybrid_recs = cf.hybrid_recommendations(user_id, n=20)
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

if __name__ == "__main__":
    update_recommendation_model()