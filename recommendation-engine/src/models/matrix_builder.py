"""
Module untuk membuat user-item matrix dan similarity matrix
"""

import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import logging
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Set

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Setup logging
from central_logging import get_logger
logger = get_logger(__name__)

from config.config import (
    PROCESSED_DATA_PATH,
    MODELS_PATH,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORY_SIMILARITY_WEIGHT
)

class MatrixBuilder:
    """
    Class untuk membangun user-item matrix dan similarity matrix
    untuk collaborative filtering
    """
    
    def __init__(self):
        """
        Inisialisasi matrix builder
        """
        # Ensure directories exist
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        os.makedirs(MODELS_PATH, exist_ok=True)
    
    def load_latest_processed_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Memuat data terproses terbaru
        
        Returns:
            tuple: (projects_df, interactions_df, feature_matrix)
        """
        logger.info("Loading latest processed data")
        
        # Find latest processed_projects CSV
        projects_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("processed_projects_") and f.endswith(".csv")
        ]
        
        projects_df = None
        if projects_files:
            try:
                latest_projects_file = max(projects_files)
                projects_path = os.path.join(PROCESSED_DATA_PATH, latest_projects_file)
                projects_df = pd.read_csv(projects_path)
                logger.info(f"Loaded projects data from {latest_projects_file}")
            except Exception as e:
                logger.error(f"Error loading processed projects data: {e}")
                # Try standard filename
                standard_path = os.path.join(PROCESSED_DATA_PATH, "processed_projects.csv")
                if os.path.exists(standard_path):
                    try:
                        projects_df = pd.read_csv(standard_path)
                        logger.info("Loaded projects data from standard file")
                    except Exception as e:
                        logger.error(f"Error loading processed projects from standard file: {e}")
        else:
            # Try standard filename
            standard_path = os.path.join(PROCESSED_DATA_PATH, "processed_projects.csv")
            if os.path.exists(standard_path):
                try:
                    projects_df = pd.read_csv(standard_path)
                    logger.info("Loaded projects data from standard file")
                except Exception as e:
                    logger.error(f"Error loading processed projects from standard file: {e}")
            else:
                logger.error("No processed projects data found")
        
        # Fix JSON columns for platforms and categories
        if projects_df is not None and 'platforms' in projects_df.columns and 'categories' in projects_df.columns:
            try:
                projects_df = self.convert_json_columns(projects_df, ['platforms', 'categories'])
                logger.info("Converted JSON columns in projects data")
            except Exception as e:
                logger.warning(f"Error converting JSON columns: {e}")
        
        # Find latest user_interactions CSV
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
                logger.info(f"Loaded interactions data from {latest_interactions_file}")
            except Exception as e:
                logger.error(f"Error loading interactions data: {e}")
        else:
            # Try standard filename
            standard_path = os.path.join(PROCESSED_DATA_PATH, "user_interactions.csv")
            if os.path.exists(standard_path):
                try:
                    interactions_df = pd.read_csv(standard_path)
                    logger.info("Loaded interactions data from standard file")
                except Exception as e:
                    logger.error(f"Error loading interactions from standard file: {e}")
            else:
                logger.warning("No interactions data found")
        
        # Find latest feature matrix CSV
        feature_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("feature_matrix_") and f.endswith(".csv")
        ]
        
        feature_matrix = None
        if feature_files:
            try:
                latest_feature_file = max(feature_files)
                feature_path = os.path.join(PROCESSED_DATA_PATH, latest_feature_file)
                feature_matrix = pd.read_csv(feature_path, index_col=0)
                logger.info(f"Loaded feature matrix from {latest_feature_file}")
            except Exception as e:
                logger.error(f"Error loading feature matrix: {e}")
        else:
            # Try standard filename
            standard_path = os.path.join(PROCESSED_DATA_PATH, "feature_matrix.csv")
            if os.path.exists(standard_path):
                try:
                    feature_matrix = pd.read_csv(standard_path, index_col=0)
                    logger.info("Loaded feature matrix from standard file")
                except Exception as e:
                    logger.error(f"Error loading feature matrix from standard file: {e}")
            else:
                logger.warning("No feature matrix found")
        
        return projects_df, interactions_df, feature_matrix

    def convert_json_columns(self, df: pd.DataFrame, json_columns: List[str]) -> pd.DataFrame:
        """
        Converts JSON string columns back to Python objects
        
        Args:
            df: DataFrame with JSON string columns
            json_columns: List of column names containing JSON strings
            
        Returns:
            pd.DataFrame: DataFrame with converted columns
        """
        result_df = df.copy()
        
        for col in json_columns:
            if col not in result_df.columns:
                continue
                
            def parse_json(val):
                if pd.isna(val) or val == '' or val == '[]' or val == '{}':
                    return {} if col == 'platforms' else []
                    
                try:
                    if isinstance(val, str):
                        try:
                            # Standard JSON parsing
                            parsed = json.loads(val)
                            return parsed
                        except json.JSONDecodeError:
                            try:
                                # Clean up excessive quotes and try again
                                cleaned = val.replace('\"\"', '\"').replace('\\"', '"')
                                parsed = json.loads(cleaned)
                                return parsed
                            except:
                                try:
                                    # Last resort - strip quotes and use eval (with caution)
                                    if val.startswith('"[') and val.endswith(']"'):
                                        val = val[1:-1]
                                    if (col == 'categories' and val.startswith('[')) or \
                                    (col == 'platforms' and val.startswith('{')):
                                        return eval(val)
                                    else:
                                        return {} if col == 'platforms' else []
                                except:
                                    return {} if col == 'platforms' else []
                    elif isinstance(val, dict) and col == 'platforms':
                        return val
                    elif isinstance(val, list) and col == 'categories':
                        return val
                    else:
                        return {} if col == 'platforms' else []
                except Exception as e:
                    logger.debug(f"JSON parse error detail: {val}")
                    logger.warning(f"Error parsing JSON in column {col}: {e}")
                    return {} if col == 'platforms' else []
                    
            result_df[col] = result_df[col].apply(parse_json)
            logger.info(f"Conversion complete for {col} column")
            
        return result_df
    
    def build_user_item_matrix(self, interactions_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Dict[str, int], Dict[str, int]]:
        """
        Membuat user-item matrix dari data interaksi
        
        Args:
            interactions_df: DataFrame interaksi
            
        Returns:
            tuple: (user_item_df, user_indices, item_indices)
        """
        if interactions_df is None or interactions_df.empty:
            logger.error("Interactions DataFrame is empty or None")
            return None, {}, {}
            
        logger.info("Building user-item matrix")
        
        try:
            # Mapping user dan item ke indeks
            user_ids = interactions_df['user_id'].unique()
            project_ids = interactions_df['project_id'].unique()
            
            user_indices = {user: i for i, user in enumerate(user_ids)}
            item_indices = {item: i for i, item in enumerate(project_ids)}
            
            # Membuat matrix dengan nilai sebagai bobot interaksi
            # Jika ada beberapa interaksi untuk pasangan user-item yang sama, ambil nilai tertinggi
            user_item_data = interactions_df.groupby(['user_id', 'project_id'])['weight'].max().reset_index()
            
            # Membuat sparse matrix
            rows = [user_indices[user] for user in user_item_data['user_id']]
            cols = [item_indices[item] for item in user_item_data['project_id']]
            values = user_item_data['weight'].values
            
            sparse_matrix = csr_matrix((values, (rows, cols)), 
                                    shape=(len(user_indices), len(item_indices)))
            
            # Convert to DataFrame for easier handling
            user_item_df = pd.DataFrame(
                sparse_matrix.toarray(),
                index=list(user_indices.keys()),
                columns=list(item_indices.keys())
            )
            
            logger.info(f"Built user-item matrix with shape {user_item_df.shape}")
            return user_item_df, user_indices, item_indices
            
        except Exception as e:
            logger.error(f"Error building user-item matrix: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, {}, {}
    
    def build_item_similarity_matrix(self, user_item_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Membuat item similarity matrix berdasarkan collaborative filtering
        
        Args:
            user_item_df: User-item matrix
            
        Returns:
            pd.DataFrame: Item similarity matrix
        """
        if user_item_df is None or user_item_df.empty:
            logger.error("User-item DataFrame is empty or None")
            return None
            
        logger.info("Building item similarity matrix")
        
        try:
            # Transpose matrix untuk mendapatkan item-user matrix
            item_user = user_item_df.T
            
            # Hitung cosine similarity antar item
            item_similarity = cosine_similarity(item_user)
            
            # Buat DataFrame untuk item similarity
            item_similarity_df = pd.DataFrame(
                item_similarity,
                index=item_user.index,
                columns=item_user.index
            )
            
            logger.info(f"Built item similarity matrix with shape {item_similarity_df.shape}")
            return item_similarity_df
            
        except Exception as e:
            logger.error(f"Error building item similarity matrix: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def build_user_similarity_matrix(self, user_item_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Membuat user similarity matrix berdasarkan collaborative filtering
        
        Args:
            user_item_df: User-item matrix
            
        Returns:
            pd.DataFrame: User similarity matrix
        """
        if user_item_df is None or user_item_df.empty:
            logger.error("User-item DataFrame is empty or None")
            return None
            
        logger.info("Building user similarity matrix")
        
        try:
            # Hitung cosine similarity antar user
            user_similarity = cosine_similarity(user_item_df)
            
            # Buat DataFrame untuk user similarity
            user_similarity_df = pd.DataFrame(
                user_similarity,
                index=user_item_df.index,
                columns=user_item_df.index
            )
            
            logger.info(f"Built user similarity matrix with shape {user_similarity_df.shape}")
            return user_similarity_df
            
        except Exception as e:
            logger.error(f"Error building user similarity matrix: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def build_feature_similarity_matrix(self, projects_df: pd.DataFrame, feature_matrix: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Membuat item similarity matrix berdasarkan fitur item
        
        Args:
            projects_df: DataFrame proyek
            feature_matrix: Matrix fitur
            
        Returns:
            pd.DataFrame: Feature-based similarity matrix
        """
        if feature_matrix is None or feature_matrix.empty:
            logger.error("Feature matrix is empty or None")
            return None
            
        logger.info("Building feature-based similarity matrix")
        
        try:
            # Hitung cosine similarity berdasarkan fitur
            feature_similarity = cosine_similarity(feature_matrix)
            
            # Dapatkan ID proyek dari projects_df
            project_ids = projects_df['id'].tolist()
            
            # Buat DataFrame untuk feature similarity
            feature_similarity_df = pd.DataFrame(
                feature_similarity,
                index=project_ids,
                columns=project_ids
            )
            
            logger.info(f"Built feature similarity matrix with shape {feature_similarity_df.shape}")
            return feature_similarity_df
            
        except Exception as e:
            logger.error(f"Error building feature similarity matrix: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def build_combined_similarity_matrix(
            self, 
            item_sim_matrix: pd.DataFrame, 
            feature_sim_matrix: pd.DataFrame, 
            alpha: float = 0.5
        ) -> Optional[pd.DataFrame]:
        """
        Menggabungkan item similarity matrix dan feature similarity matrix
        
        Args:
            item_sim_matrix: Item similarity matrix
            feature_sim_matrix: Feature similarity matrix
            alpha: Bobot untuk item similarity (1-alpha untuk feature similarity)
            
        Returns:
            pd.DataFrame: Combined similarity matrix
        """
        if item_sim_matrix is None or feature_sim_matrix is None:
            logger.error("One or both similarity matrices are None")
            return None
            
        logger.info(f"Building combined similarity matrix with alpha={alpha}")
        
        try:
            # Make sure indexes match
            common_indices = set(item_sim_matrix.index).intersection(set(feature_sim_matrix.index))
            
            # Convert set to list for pandas indexing
            common_indices_list = list(common_indices)
            
            if not common_indices_list:
                logger.error("No common indices between matrices")
                return None
            
            # Use list, not set, for indexing
            item_sim_subset = item_sim_matrix.loc[common_indices_list, common_indices_list]
            feature_sim_subset = feature_sim_matrix.loc[common_indices_list, common_indices_list]
            
            # Combine matrices
            combined_sim_matrix = alpha * item_sim_subset + (1 - alpha) * feature_sim_subset
            
            logger.info(f"Built combined similarity matrix with shape {combined_sim_matrix.shape}")
            return combined_sim_matrix
            
        except Exception as e:
            logger.error(f"Error building combined similarity matrix: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def build_matrices(self) -> Tuple[
        Optional[pd.DataFrame], 
        Optional[Dict[str, int]], 
        Optional[Dict[str, int]], 
        Optional[pd.DataFrame], 
        Optional[pd.DataFrame], 
        Optional[pd.DataFrame], 
        Optional[pd.DataFrame]
    ]:
        """
        Build all matrices needed for the recommendation system
        
        Returns:
            tuple: (user_item_df, user_indices, item_indices, item_similarity_df, 
                   user_similarity_df, feature_similarity_df, combined_similarity_df)
        """
        try:
            # Load data
            projects_df, interactions_df, feature_matrix = self.load_latest_processed_data()
            
            if projects_df is None or interactions_df is None:
                logger.error("Required data not available")
                return None, None, None, None, None, None, None
            
            # Build user-item matrix
            user_item_df, user_indices, item_indices = self.build_user_item_matrix(interactions_df)
            
            if user_item_df is None:
                logger.error("Failed to build user-item matrix")
                return None, None, None, None, None, None, None
            
            # Build item similarity matrix (collaborative filtering based)
            item_similarity_df = self.build_item_similarity_matrix(user_item_df)
            
            # Build user similarity matrix (collaborative filtering based)
            user_similarity_df = self.build_user_similarity_matrix(user_item_df)
            
            # Build or use feature matrix
            if feature_matrix is None:
                logger.info("Feature matrix not found, preparing from projects data")
                feature_matrix = self.prepare_feature_matrix(projects_df)
            
            # Build feature similarity matrix (content-based)
            feature_similarity_df = self.build_feature_similarity_matrix(projects_df, feature_matrix)
            
            # Build combined similarity matrix
            combined_similarity_df = self.build_combined_similarity_matrix(
                item_similarity_df, feature_similarity_df, alpha=0.6
            )
            
            # Save all matrices
            self._save_matrices(
                user_item_df, item_similarity_df, user_similarity_df, 
                feature_similarity_df, combined_similarity_df
            )
            
            return (user_item_df, user_indices, item_indices, item_similarity_df, 
                    user_similarity_df, feature_similarity_df, combined_similarity_df)
                    
        except Exception as e:
            logger.error(f"Error building matrices: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None, None, None, None, None, None
    
    def _save_matrices(
            self, 
            user_item_df: pd.DataFrame, 
            item_similarity_df: pd.DataFrame, 
            user_similarity_df: pd.DataFrame, 
            feature_similarity_df: pd.DataFrame, 
            combined_similarity_df: pd.DataFrame
        ) -> None:
        """
        Menyimpan semua matrix
        
        Args:
            user_item_df: User-item matrix
            item_similarity_df: Item similarity matrix
            user_similarity_df: User similarity matrix
            feature_similarity_df: Feature similarity matrix
            combined_similarity_df: Combined similarity matrix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save user-item matrix
        if user_item_df is not None:
            try:
                user_item_path = os.path.join(PROCESSED_DATA_PATH, f"user_item_matrix_{timestamp}.csv")
                user_item_df.to_csv(user_item_path, index_label="user_id")
                logger.info(f"User-item matrix saved to {user_item_path}")
                
                # Save a copy with standard name for easy access
                standard_path = os.path.join(PROCESSED_DATA_PATH, "user_item_matrix.csv")
                user_item_df.to_csv(standard_path, index_label="user_id")
            except Exception as e:
                logger.error(f"Error saving user-item matrix: {e}")
        
        # Save item similarity matrix
        if item_similarity_df is not None:
            try:
                item_sim_path = os.path.join(PROCESSED_DATA_PATH, f"item_similarity_{timestamp}.csv")
                item_similarity_df.to_csv(item_sim_path, index_label="item_id")
                logger.info(f"Item similarity matrix saved to {item_sim_path}")
                
                # Save a copy with standard name for easy access
                standard_path = os.path.join(PROCESSED_DATA_PATH, "item_similarity.csv")
                item_similarity_df.to_csv(standard_path, index_label="item_id")
            except Exception as e:
                logger.error(f"Error saving item similarity matrix: {e}")
        
        # Save user similarity matrix
        if user_similarity_df is not None:
            try:
                user_sim_path = os.path.join(PROCESSED_DATA_PATH, f"user_similarity_{timestamp}.csv")
                user_similarity_df.to_csv(user_sim_path, index_label="user_id")
                logger.info(f"User similarity matrix saved to {user_sim_path}")
                
                # Save a copy with standard name for easy access
                standard_path = os.path.join(PROCESSED_DATA_PATH, "user_similarity.csv")
                user_similarity_df.to_csv(standard_path, index_label="user_id")
            except Exception as e:
                logger.error(f"Error saving user similarity matrix: {e}")
        
        # Save feature similarity matrix
        if feature_similarity_df is not None:
            try:
                feature_sim_path = os.path.join(PROCESSED_DATA_PATH, f"feature_similarity_{timestamp}.csv")
                feature_similarity_df.to_csv(feature_sim_path, index_label="project_id")
                logger.info(f"Feature similarity matrix saved to {feature_sim_path}")
                
                # Save a copy with standard name for easy access
                standard_path = os.path.join(PROCESSED_DATA_PATH, "feature_similarity.csv")
                feature_similarity_df.to_csv(standard_path, index_label="project_id")
            except Exception as e:
                logger.error(f"Error saving feature similarity matrix: {e}")
        
        # Save combined similarity matrix
        if combined_similarity_df is not None:
            try:
                combined_sim_path = os.path.join(PROCESSED_DATA_PATH, f"combined_similarity_{timestamp}.csv")
                combined_similarity_df.to_csv(combined_sim_path, index_label="project_id")
                logger.info(f"Combined similarity matrix saved to {combined_sim_path}")
                
                # Save a copy with standard name for easy access
                standard_path = os.path.join(PROCESSED_DATA_PATH, "combined_similarity.csv")
                combined_similarity_df.to_csv(standard_path, index_label="project_id")
            except Exception as e:
                logger.error(f"Error saving combined similarity matrix: {e}")

    def _load_latest_data(self) -> Tuple[
        Optional[pd.DataFrame], 
        Optional[pd.DataFrame], 
        Optional[pd.DataFrame], 
        Optional[pd.DataFrame], 
        Optional[pd.DataFrame], 
        Optional[pd.DataFrame]
    ]:
        """
        Load the latest matrix data from processed directory.
        
        Returns:
            tuple: (user_item_df, user_similarity_df, item_similarity_df, feature_similarity_df, combined_similarity_df, projects_df)
        """
        logger.info("Loading latest matrix data")
        
        # Initialize variables
        user_item_df = None
        user_similarity_df = None
        item_similarity_df = None
        feature_similarity_df = None
        combined_similarity_df = None
        projects_df = None
        
        # Load user-item matrix
        user_item_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("user_item_matrix_") and f.endswith(".csv")
        ]
        
        if user_item_files:
            try:
                latest_user_item_file = max(user_item_files)
                user_item_df = pd.read_csv(
                    os.path.join(PROCESSED_DATA_PATH, latest_user_item_file),
                    index_col="user_id"
                )
                logger.info(f"Loaded user-item matrix from {latest_user_item_file}")
            except Exception as e:
                logger.error(f"Error loading user-item matrix: {e}")
        else:
            # Check for standard file
            standard_path = os.path.join(PROCESSED_DATA_PATH, "user_item_matrix.csv", index_col=0)
            if os.path.exists(standard_path):
                try:
                    user_item_df = pd.read_csv(standard_path, index_col="user_id")
                    logger.info("Loaded user-item matrix from standard file")
                except Exception as e:
                    logger.error(f"Error loading user-item matrix from standard file: {e}")
        
        # Load user similarity matrix
        user_sim_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("user_similarity_") and f.endswith(".csv")
        ]
        
        if user_sim_files:
            try:
                latest_user_sim_file = max(user_sim_files)
                user_similarity_df = pd.read_csv(
                    os.path.join(PROCESSED_DATA_PATH, latest_user_sim_file),
                    index_col="user_id"
                )
                logger.info(f"Loaded user similarity matrix from {latest_user_sim_file}")
            except Exception as e:
                logger.error(f"Error loading user similarity matrix: {e}")
        else:
            # Check for standard file
            standard_path = os.path.join(PROCESSED_DATA_PATH, "user_similarity.csv", index_col=0)
            if os.path.exists(standard_path):
                try:
                    user_similarity_df = pd.read_csv(standard_path, index_col="user_id")
                    logger.info("Loaded user similarity matrix from standard file")
                except Exception as e:
                    logger.error(f"Error loading user similarity matrix from standard file: {e}")
        
        # Load item similarity matrix
        item_sim_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("item_similarity_") and f.endswith(".csv")
        ]
        
        if item_sim_files:
            try:
                latest_item_sim_file = max(item_sim_files)
                item_similarity_df = pd.read_csv(
                    os.path.join(PROCESSED_DATA_PATH, latest_item_sim_file),
                    index_col="item_id"
                )
                logger.info(f"Loaded item similarity matrix from {latest_item_sim_file}")
            except Exception as e:
                logger.error(f"Error loading item similarity matrix: {e}")
        else:
            # Check for standard file
            standard_path = os.path.join(PROCESSED_DATA_PATH, "item_similarity.csv")
            if os.path.exists(standard_path):
                try:
                    item_similarity_df = pd.read_csv(standard_path, index_col="item_id")
                    logger.info("Loaded item similarity matrix from standard file")
                except Exception as e:
                    logger.error(f"Error loading item similarity matrix from standard file: {e}")
        
        # Load feature similarity matrix
        feature_sim_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("feature_similarity_") and f.endswith(".csv")
        ]
        
        if feature_sim_files:
            try:
                latest_feature_sim_file = max(feature_sim_files)
                feature_similarity_df = pd.read_csv(
                    os.path.join(PROCESSED_DATA_PATH, latest_feature_sim_file),
                    index_col="project_id"
                )
                logger.info(f"Loaded feature similarity matrix from {latest_feature_sim_file}")
            except Exception as e:
                logger.error(f"Error loading feature similarity matrix: {e}")
        else:
            # Check for standard file
            standard_path = os.path.join(PROCESSED_DATA_PATH, "feature_similarity.csv")
            if os.path.exists(standard_path):
                try:
                    feature_similarity_df = pd.read_csv(standard_path, index_col="project_id")
                    logger.info("Loaded feature similarity matrix from standard file")
                except Exception as e:
                    logger.error(f"Error loading feature similarity matrix from standard file: {e}")
        
        # Load combined similarity matrix
        combined_sim_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("combined_similarity_") and f.endswith(".csv")
        ]
        
        if combined_sim_files:
            try:
                latest_combined_sim_file = max(combined_sim_files)
                combined_similarity_df = pd.read_csv(
                    os.path.join(PROCESSED_DATA_PATH, latest_combined_sim_file),
                    index_col="project_id"
                )
                logger.info(f"Loaded combined similarity matrix from {latest_combined_sim_file}")
            except Exception as e:
                logger.error(f"Error loading combined similarity matrix: {e}")
        else:
            # Check for standard file
            standard_path = os.path.join(PROCESSED_DATA_PATH, "combined_similarity.csv")
            if os.path.exists(standard_path):
                try:
                    combined_similarity_df = pd.read_csv(standard_path, index_col="project_id")
                    logger.info("Loaded combined similarity matrix from standard file")
                except Exception as e:
                    logger.error(f"Error loading combined similarity matrix from standard file: {e}")
        
        # Load processed projects data
        projects_files = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("processed_projects_") and f.endswith(".csv")
        ]
        
        if projects_files:
            try:
                latest_projects_file = max(projects_files)
                projects_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, latest_projects_file))
                logger.info(f"Loaded processed projects data from {latest_projects_file}")
            except Exception as e:
                logger.error(f"Error loading processed projects data: {e}")
        else:
            # Check for standard file
            standard_path = os.path.join(PROCESSED_DATA_PATH, "processed_projects.csv")
            if os.path.exists(standard_path):
                try:
                    projects_df = pd.read_csv(standard_path)
                    logger.info("Loaded processed projects data from standard file")
                except Exception as e:
                    logger.error(f"Error loading processed projects data from standard file: {e}")
        
        return (
            user_item_df,
            user_similarity_df,
            item_similarity_df,
            feature_similarity_df,
            combined_similarity_df,
            projects_df
        )
    
    def build_matrices_from_df(
            self, 
            interactions_df: pd.DataFrame, 
            projects_df: pd.DataFrame
        ) -> Tuple[
            Optional[pd.DataFrame], 
            Optional[Dict[str, int]], 
            Optional[Dict[str, int]], 
            Optional[pd.DataFrame], 
            Optional[pd.DataFrame], 
            Optional[pd.DataFrame], 
            Optional[pd.DataFrame]
        ]:
        """
        Membangun matriks dari DataFrame interaksi dan proyek
        
        Args:
            interactions_df: DataFrame interaksi
            projects_df: DataFrame proyek
            
        Returns:
            tuple: (user_item_df, user_indices, item_indices, item_similarity_df, 
                user_similarity_df, feature_similarity_df, combined_similarity_df)
        """
        logger.info("Building matrices from DataFrames")
        
        try:
            if interactions_df.empty or projects_df.empty:
                logger.error("Empty DataFrames provided")
                return None, None, None, None, None, None, None
            
            # Pastikan kolom yang diperlukan ada
            required_cols = ['user_id', 'project_id', 'weight']
            if not all(col in interactions_df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in interactions_df.columns]
                logger.error(f"Missing required columns in interactions_df: {missing}")
                return None, None, None, None, None, None, None
            
            # Build user-item matrix
            user_ids = interactions_df['user_id'].unique()
            project_ids = projects_df['id'].unique()
            
            user_indices = {user: i for i, user in enumerate(user_ids)}
            item_indices = {item: i for i, item in enumerate(project_ids)}
            
            # Agregasikan bobot interaksi (jika ada beberapa untuk pasangan user-project)
            user_item_data = interactions_df.groupby(['user_id', 'project_id'])['weight'].max().reset_index()
            
            # Only keep valid items that are in the project_ids
            valid_interactions = user_item_data[user_item_data['project_id'].isin(project_ids)]
            
            if valid_interactions.empty:
                logger.error("No valid interactions found after filtering")
                return None, None, None, None, None, None, None
            
            # Build the matrix
            rows = [user_indices[user] for user in valid_interactions['user_id']]
            cols = [item_indices[item] for item in valid_interactions['project_id']]
            values = valid_interactions['weight'].values
            
            # Create sparse matrix
            sparse_matrix = csr_matrix((values, (rows, cols)), 
                                    shape=(len(user_indices), len(item_indices)))
            
            # Convert to DataFrame
            user_item_df = pd.DataFrame(
                sparse_matrix.toarray(),
                index=list(user_indices.keys()),
                columns=list(item_indices.keys())
            )
            
            # Build similarity matrices
            item_similarity_df = self.build_item_similarity_matrix(user_item_df)
            user_similarity_df = self.build_user_similarity_matrix(user_item_df)
            
            # Build feature similarity matrix
            feature_matrix = self.prepare_feature_matrix(projects_df)
            feature_similarity_df = self.build_feature_similarity_matrix(projects_df, feature_matrix)
            
            # Build combined similarity matrix
            combined_similarity_df = self.build_combined_similarity_matrix(
                item_similarity_df, feature_similarity_df, alpha=0.6
            )
            
            # Save the matrices
            self._save_matrices(
                user_item_df, item_similarity_df, user_similarity_df, 
                feature_similarity_df, combined_similarity_df
            )
            
            return (user_item_df, user_indices, item_indices, item_similarity_df, 
                    user_similarity_df, feature_similarity_df, combined_similarity_df)
                    
        except Exception as e:
            logger.error(f"Error building matrices from DataFrames: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None, None, None, None, None, None
    
    def prepare_feature_matrix(self, projects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Menyiapkan matriks fitur dari DataFrame proyek
        
        Args:
            projects_df: DataFrame proyek
            
        Returns:
            pd.DataFrame: Matriks fitur
        """
        logger.info("Preparing feature matrix from projects data")
        
        try:
            # Buat salinan untuk menghindari modifikasi DataFrame asli
            projects_df = projects_df.copy()
            
            # Convert JSON columns to Python objects if needed
            if 'categories' in projects_df.columns or 'platforms' in projects_df.columns:
                json_cols = []
                if 'categories' in projects_df.columns:
                    json_cols.append('categories')
                if 'platforms' in projects_df.columns:
                    json_cols.append('platforms')
                    
                projects_df = self.convert_json_columns(projects_df, json_cols)
            
            # Ambil fitur numerik
            numeric_features = [col for col in NUMERICAL_FEATURES if col in projects_df.columns]
            
            if not numeric_features:
                logger.warning("No numeric features found in projects DataFrame")
                # Buat dataframe kosong dengan indeks yang sama
                empty_df = pd.DataFrame(index=projects_df['id'])
                return empty_df
            
            numeric_df = projects_df[numeric_features].copy()
            
            # Handle missing values
            numeric_df = numeric_df.fillna(0)
            
            # Log transform untuk fitur dengan range besar
            for col in ['market_cap', 'volume_24h', 'reddit_subscribers', 'twitter_followers']:
                if col in numeric_df.columns:
                    numeric_df[col] = np.log1p(numeric_df[col])
            
            # Normalize fitur numerik
            scaler = MinMaxScaler()
            numeric_scaled = scaler.fit_transform(numeric_df)
            numeric_scaled_df = pd.DataFrame(
                numeric_scaled, 
                columns=numeric_df.columns,
                index=projects_df['id']
            )
            
            # Fitur kategorikal - Categories
            # Ekstrak dan one-hot encode kategori utama
            logger.info("Processing categorical features - categories")
            
            # Define common cryptocurrency categories
            common_categories = [
                'layer-1', 'smart-contract-platform', 'defi', 'nft', 
                'gaming', 'meme', 'stablecoin', 'metaverse', 'dao', 
                'privacy', 'exchange', 'yield-farming', 'lending', 'scaling'
            ]
            
            # Create category features
            category_features = pd.DataFrame(0, index=projects_df['id'], columns=common_categories)
            
            # Populate category features
            for idx, row in projects_df.iterrows():
                if 'categories' in row and row['categories']:
                    categories = row['categories']
                    if not categories or not isinstance(categories, (list, tuple)):
                        continue
                        
                    for cat in categories:
                        cat_lower = str(cat).lower() if cat else ""
                        for common_cat in common_categories:
                            if common_cat in cat_lower or cat_lower == common_cat:
                                category_features.loc[row['id'], common_cat] = 1
                                break
            
            # Fitur kategorikal - Platforms/Chains
            logger.info("Processing categorical features - platforms")
            
            # Define common blockchain platforms
            common_platforms = [
                'ethereum', 'binance-smart-chain', 'solana', 'polygon', 
                'avalanche', 'tron', 'fantom', 'arbitrum', 'optimism', 'cosmos'
            ]
            
            # Create platform features
            platform_features = pd.DataFrame(0, index=projects_df['id'], columns=common_platforms)
            
            # Populate platform features
            for idx, row in projects_df.iterrows():
                if 'platforms' in row and row['platforms']:
                    platforms = row['platforms']
                    if not platforms or not isinstance(platforms, dict):
                        continue
                        
                    platform_keys = list(platforms.keys())
                    for platform in platform_keys:
                        platform_lower = str(platform).lower() if platform else ""
                        for common_platform in common_platforms:
                            if common_platform in platform_lower or platform_lower == common_platform:
                                platform_features.loc[row['id'], common_platform] = 1
                                break
            
            # Combine all features
            logger.info("Combining all features")
            all_features = pd.concat([numeric_scaled_df, category_features, platform_features], axis=1)
            
            # Ensure no NaN values
            all_features = all_features.fillna(0)
            
            # Save feature matrix for reuse
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feature_matrix_path = os.path.join(PROCESSED_DATA_PATH, f"feature_matrix_{timestamp}.csv")
            all_features.to_csv(feature_matrix_path)
            logger.info(f"Feature matrix saved to {feature_matrix_path}")
            
            # Also save with standard name
            all_features.to_csv(os.path.join(PROCESSED_DATA_PATH, "feature_matrix.csv"))
            
            logger.info(f"Feature matrix created with shape: {all_features.shape}")
            return all_features
            
        except Exception as e:
            logger.error(f"Error preparing feature matrix: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return pd.DataFrame(index=projects_df['id'] if 'id' in projects_df.columns else projects_df.index)


if __name__ == "__main__":
    # Test functionality
    try:
        matrix_builder = MatrixBuilder()
        
        # Load data
        projects_df, interactions_df, feature_matrix = matrix_builder.load_latest_processed_data()
        
        if projects_df is not None and interactions_df is not None:
            print(f"Loaded {len(projects_df)} projects and {len(interactions_df)} interactions")
            
            # Build matrices
            print("Building matrices...")
            matrices = matrix_builder.build_matrices()
            
            if matrices[0] is not None:
                user_item_df, user_indices, item_indices, item_similarity_df, user_similarity_df, feature_similarity_df, combined_similarity_df = matrices
                
                print(f"User-item matrix shape: {user_item_df.shape}")
                print(f"Item similarity matrix shape: {item_similarity_df.shape}")
                print(f"User similarity matrix shape: {user_similarity_df.shape}")
                print(f"Feature similarity matrix shape: {feature_similarity_df.shape}")
                print(f"Combined similarity matrix shape: {combined_similarity_df.shape}")
                
                print("Matrices built and saved successfully!")
            else:
                print("Failed to build matrices")
        else:
            print("Required data not available. Please run data collection and processing first.")
    except Exception as e:
        import traceback
        print(f"Error in matrix builder: {e}")
        print(traceback.format_exc())