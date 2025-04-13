"""
Module untuk implementasi algoritma Collaborative Filtering
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from src.models.feature_enhanced_cf import FeatureEnhancedCF

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Setup logging
from central_logging import get_logger
logger = get_logger(__name__)

from config.config import (
    PROCESSED_DATA_PATH,
    MODELS_PATH,
    SIMILARITY_THRESHOLD,
    NUM_RECOMMENDATIONS,
    USER_BASED_WEIGHT,
    ITEM_BASED_WEIGHT,
    POPULARITY_WEIGHT,
    TREND_WEIGHT,
    FEATURE_WEIGHT
)

class CollaborativeFiltering:
    """
    Class untuk mengimplementasikan algoritma Collaborative Filtering
    dan merekomendasikan proyek Web3
    """
    
    def __init__(self):
        """
        Inisialisasi collaborative filtering engine
        """
        # Load latest matrices and data
        (
            self.user_item_df,
            self.user_similarity_df,
            self.item_similarity_df,
            self.feature_similarity_df,
            self.combined_similarity_df,
            self.projects_df
        ) = self._load_latest_data()
        
    def _load_latest_data(self) -> Tuple[
            Optional[pd.DataFrame], 
            Optional[pd.DataFrame], 
            Optional[pd.DataFrame], 
            Optional[pd.DataFrame], 
            Optional[pd.DataFrame], 
            Optional[pd.DataFrame]
        ]:
        """
        Memuat data dan matrices terbaru
        
        Returns:
            tuple: (user_item_df, user_similarity_df, item_similarity_df, 
                   feature_similarity_df, combined_similarity_df, projects_df)
        """
        logger.info("Loading latest matrices and data")
        
        # Load user-item matrix
        user_item_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("user_item_matrix_") and f.endswith(".csv")
        ]
        
        user_item_df = None
        if user_item_files:
            latest_user_item_file = max(user_item_files)
            try:
                user_item_df = pd.read_csv(
                    os.path.join(PROCESSED_DATA_PATH, latest_user_item_file),
                    index_col=0
                )
                logger.info(f"Loaded user-item matrix from {latest_user_item_file}")
            except Exception as e:
                logger.error(f"Error loading user-item matrix: {e}")
        else:
            logger.warning("No user-item matrix files found")
        
        # Load user similarity matrix
        user_sim_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("user_similarity_") and f.endswith(".csv")
        ]
        
        user_similarity_df = None
        if user_sim_files:
            latest_user_sim_file = max(user_sim_files)
            try:
                user_similarity_df = pd.read_csv(
                    os.path.join(PROCESSED_DATA_PATH, latest_user_sim_file),
                    index_col=0
                )
                logger.info(f"Loaded user similarity matrix from {latest_user_sim_file}")
            except Exception as e:
                logger.error(f"Error loading user similarity matrix: {e}")
        else:
            logger.warning("No user similarity matrix files found")
        
        # Load item similarity matrix
        item_sim_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("item_similarity_") and f.endswith(".csv")
        ]
        
        item_similarity_df = None
        if item_sim_files:
            latest_item_sim_file = max(item_sim_files)
            try:
                item_similarity_df = pd.read_csv(
                    os.path.join(PROCESSED_DATA_PATH, latest_item_sim_file),
                    index_col=0
                )
                logger.info(f"Loaded item similarity matrix from {latest_item_sim_file}")
            except Exception as e:
                logger.error(f"Error loading item similarity matrix: {e}")
        else:
            logger.warning("No item similarity matrix files found")
        
        # Load feature similarity matrix
        feature_sim_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("feature_similarity_") and f.endswith(".csv")
        ]
        
        feature_similarity_df = None
        if feature_sim_files:
            latest_feature_sim_file = max(feature_sim_files)
            try:
                feature_similarity_df = pd.read_csv(
                    os.path.join(PROCESSED_DATA_PATH, latest_feature_sim_file),
                    index_col=0
                )
                logger.info(f"Loaded feature similarity matrix from {latest_feature_sim_file}")
            except Exception as e:
                logger.error(f"Error loading feature similarity matrix: {e}")
        else:
            logger.warning("No feature similarity matrix files found")
        
        # Load combined similarity matrix
        combined_sim_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("combined_similarity_") and f.endswith(".csv")
        ]
        
        combined_similarity_df = None
        if combined_sim_files:
            latest_combined_sim_file = max(combined_sim_files)
            try:
                combined_similarity_df = pd.read_csv(
                    os.path.join(PROCESSED_DATA_PATH, latest_combined_sim_file),
                    index_col=0
                )
                logger.info(f"Loaded combined similarity matrix from {latest_combined_sim_file}")
            except Exception as e:
                logger.error(f"Error loading combined similarity matrix: {e}")
        else:
            logger.warning("No combined similarity matrix files found")
        
        # Load processed projects data
        projects_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("processed_projects_") and f.endswith(".csv")
        ]
        
        projects_df = None
        if projects_files:
            latest_projects_file = max(projects_files)
            try:
                projects_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, latest_projects_file))
                logger.info(f"Loaded processed projects data from {latest_projects_file}")
            except Exception as e:
                logger.error(f"Error loading processed projects data: {e}")
        else:
            # Try standard filename
            standard_path = os.path.join(PROCESSED_DATA_PATH, "processed_projects.csv")
            if os.path.exists(standard_path):
                try:
                    projects_df = pd.read_csv(standard_path)
                    logger.info("Loaded processed projects data from standard file")
                except Exception as e:
                    logger.error(f"Error loading processed projects data from standard file: {e}")
            else:
                logger.warning("No processed projects data found")
        
        return (
            user_item_df,
            user_similarity_df,
            item_similarity_df,
            feature_similarity_df,
            combined_similarity_df,
            projects_df
        )
    
    def user_based_cf(self, user_id: str, n: int = NUM_RECOMMENDATIONS, threshold: float = None) -> List[Tuple[str, float]]:
        """
        Implementasi User-Based Collaborative Filtering

        Args:
            user_id: ID user
            n: Jumlah rekomendasi
            threshold: Nilai threshold similarity (override nilai config)

        Returns:
            list: List proyek yang direkomendasikan dengan skornya
        """
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD

        # Khusus untuk evaluasi, gunakan threshold yang lebih rendah
        if threshold == SIMILARITY_THRESHOLD and n > NUM_RECOMMENDATIONS:
            threshold = SIMILARITY_THRESHOLD / 2

        if (self.user_item_df is None or self.user_similarity_df is None or 
                user_id not in self.user_item_df.index):
            logger.error(f"Cannot generate user-based recommendations for {user_id}")
            return []
            
        logger.info(f"Generating user-based recommendations for {user_id}")
        
        try:
            # Dapatkan rating dari user target
            user_ratings = self.user_item_df.loc[user_id]
            
            # Dapatkan proyek yang belum diinteraksi oleh user
            unrated_projects = user_ratings[user_ratings == 0].index.tolist()
            
            # Dapatkan similarity dengan user lain
            user_similarities = self.user_similarity_df.loc[user_id]
            
            # Filter user yang similar di atas threshold
            similar_users = user_similarities[user_similarities > SIMILARITY_THRESHOLD].index.tolist()
            if user_id in similar_users:
                similar_users.remove(user_id)  # Hapus diri sendiri
            
            # Jika tidak ada user yang similar, return empty
            if not similar_users:
                logger.warning(f"No similar users found for {user_id}")
                return []
            
            # Hitung weighted rating untuk setiap proyek yang belum diinteraksi
            recommendations = {}
            
            for project_id in unrated_projects:
                # Ambil rating dan similarity dari semua user yang sudah berinteraksi dengan proyek ini
                ratings = []
                sim_weights = []
                
                for sim_user in similar_users:
                    # Cek jika user sudah berinteraksi dengan proyek
                    rating = self.user_item_df.loc[sim_user, project_id]
                    if rating > 0:
                        sim_weight = user_similarities[sim_user]
                        ratings.append(rating)
                        sim_weights.append(sim_weight)
                
                # Jika ada rating, hitung weighted average
                if ratings:
                    weighted_rating = np.average(ratings, weights=sim_weights)
                    recommendations[project_id] = weighted_rating
            
            # Sort berdasarkan rating dan ambil top n
            sorted_recommendations = sorted(
                recommendations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:n]
            
            return sorted_recommendations
            
        except Exception as e:
            logger.error(f"Error in user_based_cf for user {user_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def item_based_cf(self, user_id: str, n: int = NUM_RECOMMENDATIONS) -> List[Tuple[str, float]]:
        """
        Implementasi Item-Based Collaborative Filtering
        
        Args:
            user_id: ID user
            n: Jumlah rekomendasi
            
        Returns:
            list: List proyek yang direkomendasikan dengan skornya
        """
        if (self.user_item_df is None or self.item_similarity_df is None or 
                user_id not in self.user_item_df.index):
            logger.error(f"Cannot generate item-based recommendations for {user_id}")
            return []
            
        logger.info(f"Generating item-based recommendations for {user_id}")
        
        try:
            # Dapatkan rating dari user target
            user_ratings = self.user_item_df.loc[user_id]
            
            # Dapatkan proyek yang sudah diinteraksi oleh user
            rated_projects = user_ratings[user_ratings > 0].index.tolist()
            
            # Dapatkan proyek yang belum diinteraksi oleh user
            unrated_projects = user_ratings[user_ratings == 0].index.tolist()
            
            # Jika tidak ada proyek yang sudah diinteraksi, return empty
            if not rated_projects:
                logger.warning(f"No rated projects found for {user_id}")
                return []
            
            # Hitung weighted rating untuk setiap proyek yang belum diinteraksi
            recommendations = {}
            
            for unrated_proj in unrated_projects:
                weighted_sum = 0
                similarity_sum = 0
                
                for rated_proj in rated_projects:
                    # Cek similarity antara proyek yang sudah dan belum diinteraksi
                    if rated_proj in self.item_similarity_df.index and unrated_proj in self.item_similarity_df.columns:
                        similarity = self.item_similarity_df.loc[rated_proj, unrated_proj]
                        
                        # Gunakan similarity jika di atas threshold
                        if similarity > SIMILARITY_THRESHOLD:
                            rating = user_ratings[rated_proj]
                            weighted_sum += similarity * rating
                            similarity_sum += similarity
                
                # Jika ada similarity yang signifikan, hitung weighted score
                if similarity_sum > 0:
                    recommendations[unrated_proj] = weighted_sum / similarity_sum
            
            # Sort berdasarkan rating dan ambil top n
            sorted_recommendations = sorted(
                recommendations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:n]
            
            return sorted_recommendations
            
        except Exception as e:
            logger.error(f"Error in item_based_cf for user {user_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def feature_enhanced_cf(self, user_id: str, n: int = NUM_RECOMMENDATIONS) -> List[Tuple[str, float]]:
        """
        Implementasi Feature-Enhanced Collaborative Filtering
        
        Args:
            user_id: ID user
            n: Jumlah rekomendasi
            
        Returns:
            list: List proyek yang direkomendasikan dengan skornya
        """
        if (self.user_item_df is None or self.combined_similarity_df is None or 
                user_id not in self.user_item_df.index):
            logger.error(f"Cannot generate feature-enhanced recommendations for {user_id}")
            return []
            
        logger.info(f"Generating feature-enhanced recommendations for {user_id}")
        
        try:
            # Dapatkan rating dari user target
            user_ratings = self.user_item_df.loc[user_id]
            
            # Dapatkan proyek yang sudah diinteraksi oleh user
            rated_projects = user_ratings[user_ratings > 0].index.tolist()
            
            # Dapatkan proyek yang belum diinteraksi oleh user
            unrated_projects = user_ratings[user_ratings == 0].index.tolist()
            
            # Jika tidak ada proyek yang sudah diinteraksi, return empty
            if not rated_projects:
                logger.warning(f"No rated projects found for {user_id}")
                return []
            
            # Hitung weighted rating untuk setiap proyek yang belum diinteraksi
            recommendations = {}
            
            for unrated_proj in unrated_projects:
                weighted_sum = 0
                similarity_sum = 0
                
                for rated_proj in rated_projects:
                    # Cek apakah proyek ada di combined similarity matrix
                    if (rated_proj in self.combined_similarity_df.index and 
                            unrated_proj in self.combined_similarity_df.columns):
                        # Gunakan combined similarity (CF + content based)
                        similarity = self.combined_similarity_df.loc[rated_proj, unrated_proj]
                        
                        # Gunakan similarity jika di atas threshold
                        if similarity > SIMILARITY_THRESHOLD:
                            rating = user_ratings[rated_proj]
                            weighted_sum += similarity * rating
                            similarity_sum += similarity
                
                # Jika ada similarity yang signifikan, hitung weighted score
                if similarity_sum > 0:
                    recommendations[unrated_proj] = weighted_sum / similarity_sum
            
            # Sort berdasarkan rating dan ambil top n
            sorted_recommendations = sorted(
                recommendations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:n]
            
            return sorted_recommendations
            
        except Exception as e:
            logger.error(f"Error in feature_enhanced_cf for user {user_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def hybrid_recommendations(self, user_id: str, n: int = NUM_RECOMMENDATIONS, feature_cf = None) -> List[Dict[str, Any]]:
        """
        Implementasi Hybrid Recommendation yang menggabungkan user-based CF,
        item-based CF, popularitas, dan trend

        Args:
            user_id: User ID
            n: Jumlah rekomendasi
            feature_cf: Object FeatureEnhancedCF (opsional)

        Returns:
            list: List proyek yang direkomendasikan dengan skornya dan detail
        """
        logger.info(f"Generating hybrid recommendations for user {user_id}")

        try:
            all_recommendations = defaultdict(float)

            # 1. User-based CF (personality-based)
            user_based_recs = self.user_based_cf(user_id, n=n*2)  # Get more to combine
            if user_based_recs:  # Pastikan ada rekomendasi
                for project_id, score in user_based_recs:
                    # Normalize score to 0-1 range (assuming original is 0-5)
                    norm_score = score / 5.0
                    all_recommendations[project_id] += USER_BASED_WEIGHT * norm_score

            # 2. Item-based CF (project-similarity)
            item_based_recs = self.item_based_cf(user_id, n=n*2)
            if item_based_recs:  # Pastikan ada rekomendasi
                for project_id, score in item_based_recs:
                    # Normalize score to 0-1 range
                    norm_score = score / 5.0
                    all_recommendations[project_id] += ITEM_BASED_WEIGHT * norm_score

            # 3. PERBAIKAN: Feature-enhanced CF jika feature_cf object disediakan
            if feature_cf is not None and hasattr(feature_cf, 'recommend_projects'):
                try:
                    feature_recs = feature_cf.recommend_projects(
                        user_id,
                        self.user_item_df,
                        self.item_similarity_df,
                        self.projects_df,
                        n=n*2
                    )
                    
                    if feature_recs:
                        for rec in feature_recs:
                            if isinstance(rec, dict) and 'id' in rec:
                                proj_id = rec['id']
                                score = rec.get('recommendation_score', 0.5)
                                all_recommendations[proj_id] += FEATURE_WEIGHT * score
                                
                except Exception as e:
                    logger.warning(f"Error using feature_enhanced_cf in hybrid recommendations: {e}")
                    # Fallback silently

            # 4. Add Popularity and Trend factors
            if self.projects_df is not None:
                # Ensure we have the necessary columns
                required_cols = ['popularity_score', 'trend_score']
                if all(col in self.projects_df.columns for col in required_cols):
                    # Create project_id to index mapping for quick lookups
                    id_to_idx = {id: i for i, id in enumerate(self.projects_df['id'])}

                    # Normalize popularity and trend scores
                    max_popularity = self.projects_df['popularity_score'].max() if len(self.projects_df) > 0 else 1.0
                    max_trend = self.projects_df['trend_score'].max() if len(self.projects_df) > 0 else 1.0

                    # Add popularity and trend bonuses to recommendations
                    for project_id in all_recommendations.keys():
                        if project_id in id_to_idx:
                            idx = id_to_idx[project_id]
                            
                            # Get popularity score (normalized to 0-1)
                            popularity_score = self.projects_df.iloc[idx]['popularity_score'] / max_popularity if max_popularity > 0 else 0
                            all_recommendations[project_id] += POPULARITY_WEIGHT * popularity_score
                            
                            # Get trend score (normalized to 0-1)
                            trend_score = self.projects_df.iloc[idx]['trend_score'] / max_trend if max_trend > 0 else 0
                            all_recommendations[project_id] += TREND_WEIGHT * trend_score

            # Jika tidak ada rekomendasi yang terkumpul, tambahkan rekomendasi populer sebagai fallback
            if not all_recommendations:
                logger.warning(f"No hybrid recommendations collected for user {user_id}, using popular as fallback")
                popular_projects = self.get_popular_projects(n=n)
                if popular_projects:
                    for proj in popular_projects:
                        if isinstance(proj, dict) and 'id' in proj:
                            proj_id = proj.get('id')
                            if proj_id:
                                all_recommendations[proj_id] = proj.get('recommendation_score', 0.5)
            
            # Sort based on final combined score
            sorted_recommendations = sorted(
                all_recommendations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n]
            
            # Add project details
            final_recommendations = []
            if self.projects_df is not None and not self.projects_df.empty:
                for project_id, score in sorted_recommendations:
                    project_data = self.projects_df[self.projects_df['id'] == project_id]
                    if not project_data.empty:
                        project_info = project_data.iloc[0].to_dict()
                        
                        # Add the recommendation score
                        project_info['recommendation_score'] = float(score)
                        
                        # Convert numpy values to native Python
                        for k, v in project_info.items():
                            if isinstance(v, (np.int64, np.int32)):
                                project_info[k] = int(v)
                            elif isinstance(v, (np.float64, np.float32)):
                                project_info[k] = float(v)
                        
                        # Add to final recommendations
                        final_recommendations.append(project_info)
                    else:
                        # If project not found in DataFrame, still include basic info
                        final_recommendations.append({
                            'id': project_id,
                            'recommendation_score': float(score)
                        })
            else:
                # If projects data is not available, just return the IDs and scores
                final_recommendations = [
                    {'id': project_id, 'recommendation_score': float(score)}
                    for project_id, score in sorted_recommendations
                ]
            
            # Use a fallback if no recommendations were generated
            if not final_recommendations:
                logger.warning(f"No final recommendations generated for user {user_id}")
                popular_projects = self.get_popular_projects(n=n)
                if popular_projects:
                    return popular_projects
                return [{'id': 'fallback_recommendation', 'recommendation_score': 0.5}]
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error in hybrid_recommendations for user {user_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Return popular recommendations as fallback in case of errors
            try:
                popular_projects = self.get_popular_projects(n=n)
                if popular_projects:
                    return popular_projects
            except:
                pass
            
            # Return minimal fallback if popular recommendations also fail
            return [{'id': 'error_fallback', 'recommendation_score': 0.1}]
        
    def get_cold_start_recommendations(self, user_id: str, user_interests: Optional[List[str]] = None, feature_cf = None, n: int = NUM_RECOMMENDATIONS) -> List[Dict[str, Any]]:
        """
        Mendapatkan rekomendasi untuk pengguna baru dengan penekanan pada content-based

        Args:
            user_id: User ID
            user_interests: Daftar kategori yang diminati user (optional)
            feature_cf: FeatureEnhancedCF instance
            n: Jumlah rekomendasi

        Returns:
            list: List rekomendasi proyek
        """
        logger.info(f"Generating cold-start recommendations for new user {user_id}")
        
        # Inisialisasi feature_cf jika tidak disediakan
        if feature_cf is None:
            from src.models.feature_enhanced_cf import FeatureEnhancedCF
            feature_cf = FeatureEnhancedCF()
        
        # Strategi 1: Jika terdapat informasi minat pengguna, prioritaskan proyek dari kategori tersebut
        if user_interests and self.projects_df is not None:
            # Membuat dict untuk menampung proyek dari masing-masing kategori
            category_recommendations = {}
            
            # Batasi jumlah rekomendasi per kategori
            per_category = max(3, n // len(user_interests))
            
            # Untuk tiap kategori minat, ambil proyek top dari kategori tersebut
            for category in user_interests:
                # Filter proyek berdasarkan kategori
                category_df = self.projects_df[
                    (self.projects_df['primary_category'] == category) | 
                    (self.projects_df['categories'].apply(lambda x: category in x if isinstance(x, list) else False))
                ]
                
                if not category_df.empty:
                    # Urutkan berdasarkan popularity score
                    top_category_projects = category_df.sort_values('popularity_score', ascending=False).head(per_category)
                    
                    # Tambahkan ke dict rekomendasi
                    category_recommendations[category] = []
                    for _, project in top_category_projects.iterrows():
                        project_dict = project.to_dict()
                        # Beri skor rekomendasi berdasarkan popularity
                        project_dict['recommendation_score'] = float(project_dict.get('popularity_score', 50)) / 100.0
                        category_recommendations[category].append(project_dict)
            
            # Gabungkan semua rekomendasi dari berbagai kategori
            all_recommendations = []
            for category, projects in category_recommendations.items():
                all_recommendations.extend(projects)
            
            # Deduplikasi berdasarkan ID
            seen_ids = set()
            unique_recommendations = []
            for rec in all_recommendations:
                if rec['id'] not in seen_ids:
                    seen_ids.add(rec['id'])
                    unique_recommendations.append(rec)
                    
            # Jika mendapat cukup rekomendasi, return
            if len(unique_recommendations) >= n:
                return unique_recommendations[:n]
        
        # Strategi 2: Simulasikan interaksi terbatas dengan proyek populer
        
        # Clone user-item matrix
        if (self.user_item_df is not None and 
            user_id not in self.user_item_df.index and 
            self.projects_df is not None):
            
            # Buat copy matrix untuk menghindari modifikasi asli
            test_matrix = self.user_item_df.copy()
            
            # Tambahkan user baru dengan nilai 0
            test_matrix.loc[user_id] = np.zeros(len(test_matrix.columns))
            
            # Pilih beberapa proyek populer
            popular_projects = self.projects_df.sort_values('popularity_score', ascending=False).head(3)
            
            # Tambahkan rating sintetis untuk proyek populer
            rng = np.random.default_rng(42)
            for _, project in popular_projects.iterrows():
                project_id = project['id']
                if project_id in test_matrix.columns:
                    test_matrix.loc[user_id, project_id] = rng.integers(3, 6)  # Rating 3-5
            
            # Update model
            self.user_item_df = test_matrix
        
        # Strategi 3: Gunakan hybrid recommendation dengan bobot tinggi pada komponen content-based
        global FEATURE_WEIGHT, USER_BASED_WEIGHT, ITEM_BASED_WEIGHT
        
        # Simpan bobot asli
        orig_feature_weight = FEATURE_WEIGHT
        orig_user_weight = USER_BASED_WEIGHT
        orig_item_weight = ITEM_BASED_WEIGHT
        
        try:
            # Modifikasi bobot untuk cold start: prioritaskan content-based
            FEATURE_WEIGHT = 0.9  # Penekanan tinggi pada fitur
            USER_BASED_WEIGHT = 0.05
            ITEM_BASED_WEIGHT = 0.05
            
            # Generate hybrid recommendations
            recommendations = self.hybrid_recommendations(user_id, n=n, feature_cf=feature_cf)
            
            # Jika terlalu sedikit rekomendasi, tambahkan proyek trending
            if len(recommendations) < n // 2:
                trending = self.get_trending_projects(n=n)
                recommendations.extend(trending)
                
                # Deduplikasi
                seen_ids = set()
                unique_recs = []
                for rec in recommendations:
                    if isinstance(rec, dict) and 'id' in rec:
                        rec_id = rec['id']
                        if rec_id not in seen_ids:
                            seen_ids.add(rec_id)
                            unique_recs.append(rec)
                
                recommendations = unique_recs[:n]
            
            return recommendations
        finally:
            # Kembalikan bobot ke nilai asli
            FEATURE_WEIGHT = orig_feature_weight
            USER_BASED_WEIGHT = orig_user_weight
            ITEM_BASED_WEIGHT = orig_item_weight
    
    def get_trending_projects(self, n: int = NUM_RECOMMENDATIONS) -> List[Dict[str, Any]]:
        """
        Mendapatkan proyek trending berdasarkan score
        
        Args:
            n: Jumlah proyek trending
            
        Returns:
            list: List proyek trending dengan detailnya
        """
        if self.projects_df is None:
            logger.error("Projects data not available")
            return []
            
        logger.info(f"Getting top {n} trending projects")
        
        try:
            # Ensure we have the necessary columns
            if 'trend_score' not in self.projects_df.columns:
                logger.error("Trend score not available in projects data")
                return []
            
            # Get top trending projects
            trending_projects = self.projects_df.sort_values(
                by='trend_score', 
                ascending=False
            ).head(n)
            
            # Convert to list of dicts
            trending_list = trending_projects.to_dict('records')
            
            # Convert numpy types to Python types for JSON serialization
            for project in trending_list:
                for key, value in project.items():
                    if isinstance(value, (np.int64, np.int32)):
                        project[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        project[key] = float(value)
                    
                # Add recommendation_score based on trend_score for consistency
                if 'trend_score' in project:
                    project['recommendation_score'] = float(project['trend_score']) / 100.0
            
            return trending_list
            
        except Exception as e:
            logger.error(f"Error in get_trending_projects: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def get_popular_projects(self, n: int = NUM_RECOMMENDATIONS) -> List[Dict[str, Any]]:
        """
        Mendapatkan proyek populer berdasarkan popularity score
        
        Args:
            n: Jumlah proyek populer
            
        Returns:
            list: List proyek populer dengan detailnya
        """
        if self.projects_df is None:
            logger.error("Projects data not available")
            return []
            
        logger.info(f"Getting top {n} popular projects")
        
        try:
            # Ensure we have the necessary columns
            if 'popularity_score' not in self.projects_df.columns:
                logger.error("Popularity score not available in projects data")
                return []
            
            # Get top popular projects
            popular_projects = self.projects_df.sort_values(
                by='popularity_score', 
                ascending=False
            ).head(n)
            
            # Convert to list of dicts
            popular_list = popular_projects.to_dict('records')
            
            # Convert numpy types to Python types for JSON serialization
            for project in popular_list:
                for key, value in project.items():
                    if isinstance(value, (np.int64, np.int32)):
                        project[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        project[key] = float(value)
                    
                # Add recommendation_score based on popularity_score for consistency
                if 'popularity_score' in project:
                    project['recommendation_score'] = float(project['popularity_score']) / 100.0
            
            return popular_list
            
        except Exception as e:
            logger.error(f"Error in get_popular_projects: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def get_recommendations_by_category(self, user_id: str, category: str, n: int = NUM_RECOMMENDATIONS, feature_cf=None) -> List[Dict[str, Any]]:
        """
        Mendapatkan rekomendasi berdasarkan kategori tertentu
        
        Args:
            user_id: ID user
            category: Kategori proyek
            n: Jumlah rekomendasi
            
        Returns:
            list: List proyek dalam kategori dengan skornya
        """
        if self.projects_df is None:
            logger.error("Projects data not available")
            return []
            
        logger.info(f"Getting recommendations in category '{category}' for {user_id}")
        
        try:
            if feature_cf is None:
                # Buat instance baru jika diperlukan
                from src.models.feature_enhanced_cf import FeatureEnhancedCF
                feature_cf = FeatureEnhancedCF()

            # Get hybrid recommendations first
            all_recs = self.hybrid_recommendations(user_id, n=n*3, feature_cf=feature_cf)  # Get more to filter
            
            # Filter by category
            category_recs = []
            for project in all_recs:
                # Check primary_category field
                if 'primary_category' in project and project['primary_category'] == category:
                    category_recs.append(project)
                    continue
                    
                # Check also in categories json if available
                if 'categories' in project:
                    categories = project['categories']
                    # Handle both list and string (JSON) formats
                    if isinstance(categories, str):
                        try:
                            categories = eval(categories)  # Convert string to list/dict
                        except:
                            categories = []
                    
                    if isinstance(categories, list) and category in categories:
                        category_recs.append(project)
            
            # Return top n
            return category_recs[:n]
            
        except Exception as e:
            logger.error(f"Error in get_recommendations_by_category for user {user_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def get_recommendations_by_chain(self, user_id: str, chain: str, n: int = NUM_RECOMMENDATIONS, feature_cf=None) -> List[Dict[str, Any]]:
        """
        Mendapatkan rekomendasi berdasarkan blockchain tertentu
        
        Args:
            user_id: ID user
            chain: Blockchain (e.g., ethereum, solana)
            n: Jumlah rekomendasi
            
        Returns:
            list: List proyek dalam blockchain dengan skornya
        """
        if self.projects_df is None:
            logger.error("Projects data not available")
            return []
            
        logger.info(f"Getting recommendations in chain '{chain}' for {user_id}")
        
        try:
            if feature_cf is None:
                # Buat instance baru jika diperlukan
                from src.models.feature_enhanced_cf import FeatureEnhancedCF
                feature_cf = FeatureEnhancedCF()

            # Get hybrid recommendations first
            all_recs = self.hybrid_recommendations(user_id, n=n*3, feature_cf=feature_cf)  # Get more to filter
            
            # Filter by chain
            chain_recs = []
            for project in all_recs:
                if 'chain' in project and project['chain'] == chain:
                    chain_recs.append(project)
            
            # Return top n
            return chain_recs[:n]
            
        except Exception as e:
            logger.error(f"Error in get_recommendations_by_chain for user {user_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def save_recommendations(self, user_id: str, recommendations: List[Dict[str, Any]], rec_type: str = 'hybrid') -> None:
        """
        Menyimpan rekomendasi untuk analisis lebih lanjut
        
        Args:
            user_id: ID user
            recommendations: List rekomendasi
            rec_type: Tipe rekomendasi
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recommendations_{user_id}_{rec_type}_{timestamp}.csv"
        
        try:
            # Convert recommendations to DataFrame
            if recommendations and isinstance(recommendations[0], dict):
                recs_df = pd.DataFrame(recommendations)
            else:
                recs_df = pd.DataFrame(recommendations, columns=['project_id', 'score'])
            
            # Add user_id and recommendation type
            recs_df['user_id'] = user_id
            recs_df['recommendation_type'] = rec_type
            
            # Ensure PROCESSED_DATA_PATH exists
            os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
            
            # Save to processed data folder
            filepath = os.path.join(PROCESSED_DATA_PATH, filename)
            recs_df.to_csv(filepath, index=False)
            
            logger.info(f"Recommendations saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving recommendations: {e}")


if __name__ == "__main__":
    # Test with a sample user
    cf = CollaborativeFiltering()
    
    # Check if initialization was successful
    if cf.user_item_df is None:
        print("Error: No user-item matrix found. Please run data collection and processing first.")
        sys.exit(1)
    
    # Test with a sample user
    sample_user = cf.user_item_df.index[0] if cf.user_item_df is not None and len(cf.user_item_df) > 0 else None
    
    if sample_user:
        print(f"Testing with sample user: {sample_user}")
        
        # Inisialisasi FeatureEnhancedCF
        feature_cf = FeatureEnhancedCF()
        
        # Get and save recommendations
        recommendations = cf.hybrid_recommendations(sample_user, n=10, feature_cf=feature_cf)
        cf.save_recommendations(sample_user, recommendations, 'hybrid')
        
        # Print top 5 recommendations
        print(f"\nTop 5 hybrid recommendations for {sample_user}:")
        for i, rec in enumerate(recommendations[:5], 1):
            if isinstance(rec, dict):
                print(f"{i}. {rec.get('name', rec.get('id', 'Unknown'))} - Score: {rec.get('recommendation_score', 0):.4f}")
            else:
                print(f"{i}. {rec[0]} - Score: {rec[1]:.4f}")
        
        # Test user-based CF
        user_recs = cf.user_based_cf(sample_user, n=5)
        print(f"\nTop 5 user-based recommendations for {sample_user}:")
        for i, (item, score) in enumerate(user_recs, 1):
            print(f"{i}. {item} - Score: {score:.4f}")
        
        # Test trending projects
        trending = cf.get_trending_projects(n=5)
        print("\nTop 5 trending projects:")
        for i, rec in enumerate(trending, 1):
            print(f"{i}. {rec.get('name', rec.get('id', 'Unknown'))} - Score: {rec.get('recommendation_score', 0):.4f}")
    else:
        print("No sample user found. Please run data collection and preprocessing first.")