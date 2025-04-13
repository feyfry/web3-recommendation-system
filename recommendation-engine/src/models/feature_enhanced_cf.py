"""
Module untuk implementasi Feature-Enhanced Collaborative Filtering
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
import sys
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from sklearn.metrics.pairwise import cosine_similarity

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Setup logging
from central_logging import get_logger
logger = get_logger(__name__)

from config.config import (
    PROCESSED_DATA_PATH,
    SIMILARITY_THRESHOLD,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    MARKET_CAP_WEIGHT,
    VOLUME_WEIGHT,
    PRICE_CHANGE_WEIGHT,
    SOCIAL_SCORE_WEIGHT,
    DEVELOPER_SCORE_WEIGHT,
    CATEGORY_SIMILARITY_WEIGHT
)

class FeatureEnhancedCF:
    """
    Class untuk implementasi Feature-Enhanced Collaborative Filtering
    dengan menggabungkan collaborative filtering dan content-based filtering
    """
    
    def __init__(self):
        """
        Inisialisasi Feature-Enhanced CF
        """
        pass
        
    def calculate_category_similarity(self, projects_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Menghitung similarity antar proyek berdasarkan kategori
        
        Args:
            projects_df: DataFrame proyek
            
        Returns:
            pd.DataFrame: Category similarity matrix atau None jika gagal
        """
        logger.info("Calculating category similarity matrix")
        
        try:
            # Check required columns
            categorical_columns = [col for col in CATEGORICAL_FEATURES if col in projects_df.columns]
            if not categorical_columns:
                logger.error("No categorical features found in projects data")
                return None
            
            # Create one-hot encoded matrix for categorical features
            categorical_data = pd.get_dummies(
                projects_df[categorical_columns],
                columns=categorical_columns,
                prefix=categorical_columns
            )
            
            # Calculate cosine similarity
            category_similarity = cosine_similarity(categorical_data)
            
            # Create DataFrame
            category_sim_df = pd.DataFrame(
                category_similarity,
                index=projects_df['id'],
                columns=projects_df['id']
            )
            
            return category_sim_df
            
        except Exception as e:
            logger.error(f"Error calculating category similarity: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
        
    def calculate_numerical_feature_similarity(self, projects_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Menghitung similarity antar proyek berdasarkan fitur numerik
        
        Args:
            projects_df: DataFrame proyek
            
        Returns:
            pd.DataFrame: Numerical feature similarity matrix atau None jika gagal
        """
        logger.info("Calculating numerical feature similarity matrix")
        
        try:
            # Check required columns
            numerical_columns = [col for col in NUMERICAL_FEATURES if col in projects_df.columns]
            if not numerical_columns:
                logger.error("No numerical features found in projects data")
                return None
            
            # Create numerical features DataFrame
            numerical_data = projects_df[numerical_columns].copy()
            
            # Handle missing values and infinities
            numerical_data = numerical_data.fillna(0)
            numerical_data = numerical_data.replace([np.inf, -np.inf], 0)
            
            # Apply log1p to features with large ranges
            for col in ['market_cap', 'volume_24h', 'reddit_subscribers', 'twitter_followers']:
                if col in numerical_data.columns:
                    numerical_data[col] = np.log1p(numerical_data[col])
            
            # Scale features to 0-1 range
            for col in numerical_data.columns:
                min_val = numerical_data[col].min()
                max_val = numerical_data[col].max()
                
                if max_val > min_val:
                    numerical_data[col] = (numerical_data[col] - min_val) / (max_val - min_val)
            
            # Calculate cosine similarity
            numerical_similarity = cosine_similarity(numerical_data)
            
            # Use project ids for index and columns
            if 'id' in projects_df.columns:
                indices = projects_df['id']
            else:
                indices = projects_df.index
            
            # Create DataFrame
            numerical_sim_df = pd.DataFrame(
                numerical_similarity,
                index=indices,
                columns=indices
            )
            
            return numerical_sim_df
            
        except Exception as e:
            logger.error(f"Error calculating numerical feature similarity: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
        
    def calculate_weighted_feature_similarity(self, projects_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Menghitung similarity antar proyek dengan pembobotan fitur
        
        Args:
            projects_df: DataFrame proyek
            
        Returns:
            pd.DataFrame: Weighted feature similarity matrix atau None jika gagal
        """
        logger.info("Calculating weighted feature similarity matrix")
        
        try:
            # Calculate category similarity
            category_sim_df = self.calculate_category_similarity(projects_df)
            
            # Calculate numerical feature similarity
            numerical_sim_df = self.calculate_numerical_feature_similarity(projects_df)
            
            if category_sim_df is None or numerical_sim_df is None:
                logger.error("Could not calculate one or both similarity matrices")
                return None
            
            # Ensure indices match
            common_indices = set(category_sim_df.index).intersection(set(numerical_sim_df.index))
            
            # Convert set to list for indexing
            common_indices_list = list(common_indices)
            
            if not common_indices:
                logger.error("No common indices between similarity matrices")
                return None
            
            # Subset to common indices
            category_sim_subset = category_sim_df.loc[common_indices_list, common_indices_list]
            numerical_sim_subset = numerical_sim_df.loc[common_indices_list, common_indices_list]
            
            # Calculate weighted similarity using the constant from config
            weighted_sim = pd.DataFrame(
                CATEGORY_SIMILARITY_WEIGHT * category_sim_subset.values + 
                (1 - CATEGORY_SIMILARITY_WEIGHT) * numerical_sim_subset.values,
                index=common_indices_list,
                columns=common_indices_list
            )
            
            return weighted_sim
            
        except Exception as e:
            logger.error(f"Error calculating weighted feature similarity: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
        
    def enhance_collaborative_filtering(
            self, 
            user_item_matrix: pd.DataFrame, 
            item_similarity_matrix: pd.DataFrame, 
            projects_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Menambahkan bobot fitur ke collaborative filtering
        
        Args:
            user_item_matrix: User-item matrix
            item_similarity_matrix: Item similarity matrix dari CF
            projects_df: DataFrame proyek
            
        Returns:
            pd.DataFrame: Feature-enhanced similarity matrix
        """
        logger.info("Enhancing collaborative filtering with feature weights")
        
        try:
            # Calculate feature similarity
            feature_sim_df = self.calculate_weighted_feature_similarity(projects_df)
            
            if feature_sim_df is None:
                logger.error("Could not calculate feature similarity")
                return item_similarity_matrix
            
            # Get common items
            common_items = set(item_similarity_matrix.index).intersection(set(feature_sim_df.index))
            
            # Convert set to list for indexing
            common_items_list = list(common_items)
            
            if not common_items:
                logger.error("No common items between similarity matrices")
                return item_similarity_matrix
            
            # Subset matrices to common items
            item_sim_subset = item_similarity_matrix.loc[common_items_list, common_items_list]
            feature_sim_subset = feature_sim_df.loc[common_items_list, common_items_list]
            
            # Calculate enhanced similarity matrix (70% CF, 30% feature-based)
            enhanced_similarity = pd.DataFrame(
                0.7 * item_sim_subset.values + 0.3 * feature_sim_subset.values,
                index=common_items_list,
                columns=common_items_list
            )
            
            return enhanced_similarity
            
        except Exception as e:
            logger.error(f"Error enhancing collaborative filtering: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return item_similarity_matrix
        
    def predict_ratings(
            self, 
            user_id: str, 
            user_item_matrix: pd.DataFrame, 
            enhanced_similarity: pd.DataFrame, 
            n: int = 10
        ) -> List[Tuple[str, float]]:
        """
        Memprediksi rating untuk item yang belum dirating
        
        Args:
            user_id: ID user
            user_item_matrix: User-item matrix
            enhanced_similarity: Enhanced similarity matrix
            n: Jumlah rekomendasi
            
        Returns:
            list: List item yang direkomendasikan dengan skornya
        """
        logger.info(f"Predicting ratings for user {user_id}")
        
        try:
            # Check if user exists
            if user_id not in user_item_matrix.index:
                logger.error(f"User {user_id} not found in user-item matrix")
                return []
            
            # Get user ratings
            user_ratings = user_item_matrix.loc[user_id]
            
            # Get items user has rated and not rated
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            unrated_items = user_ratings[user_ratings == 0].index.tolist()
            
            # Check if user has rated any items
            if not rated_items:
                logger.warning(f"User {user_id} has not rated any items")
                return []
            
            # Predict ratings for unrated items
            predictions = {}
            
            for item in unrated_items:
                # Check if item is in enhanced similarity matrix
                if item not in enhanced_similarity.columns:
                    continue
                    
                # Calculate weighted sum of ratings based on similarity
                weighted_sum = 0
                similarity_sum = 0
                
                for rated_item in rated_items:
                    # Check if rated item is in enhanced similarity matrix
                    if rated_item not in enhanced_similarity.index:
                        continue
                        
                    # Get similarity between items
                    similarity = enhanced_similarity.loc[rated_item, item]
                    
                    # Only consider significant similarities
                    if similarity > SIMILARITY_THRESHOLD:
                        weighted_sum += similarity * user_ratings[rated_item]
                        similarity_sum += similarity
                
                # If we found similar items, calculate predicted rating
                if similarity_sum > 0:
                    predictions[item] = weighted_sum / similarity_sum
            
            # Sort predictions and get top n
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
            
            return sorted_predictions
            
        except Exception as e:
            logger.error(f"Error predicting ratings for user {user_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
        
    def recommend_projects(
        self,
        user_id: str,
        user_item_matrix: pd.DataFrame,
        item_similarity_matrix: Optional[pd.DataFrame],
        projects_df: pd.DataFrame,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Memberikan rekomendasi proyek untuk user berdasarkan feature-enhanced CF
        
        Args:
            user_id: ID user
            user_item_matrix: User-item matrix
            item_similarity_matrix: Item similarity matrix dari CF (dapat None jika akan dikomputasi)
            projects_df: DataFrame proyek
            n: Jumlah rekomendasi
            
        Returns:
            list: List proyek yang direkomendasikan dengan detailnya
        """
        logger.info(f"Generating feature-enhanced recommendations for user {user_id}")
        
        try:
            # Pastikan user_id ada dalam matrix
            if user_id not in user_item_matrix.index:
                logger.warning(f"User ID {user_id} not found in user-item matrix")
                return []
                
            # Compute item similarity matrix if not provided
            if item_similarity_matrix is None:
                logger.info("Item similarity matrix not provided, computing from user-item matrix")
                from src.models.matrix_builder import MatrixBuilder
                matrix_builder = MatrixBuilder()
                item_similarity_matrix = matrix_builder.build_item_similarity_matrix(user_item_matrix)
                
            if item_similarity_matrix is None:
                logger.error("Failed to build item similarity matrix")
                return []
                
            # Enhance item similarity with features
            enhanced_similarity = self.enhance_collaborative_filtering(
                user_item_matrix, item_similarity_matrix, projects_df
            )
            
            # Predict ratings
            predicted_ratings = self.predict_ratings(
                user_id, user_item_matrix, enhanced_similarity, n=n
            )
            
            if not predicted_ratings:
                logger.warning(f"No ratings predicted for user {user_id}")
                return []
                
            # Get project details for recommended items
            recommendations = []
            
            for project_id, score in predicted_ratings:
                project_data = projects_df[projects_df['id'] == project_id]
                
                if not project_data.empty:
                    project_info = project_data.iloc[0].to_dict()
                    
                    # Convert numpy types to Python native types for JSON serialization
                    for key, value in project_info.items():
                        if isinstance(value, (np.int64, np.int32)):
                            project_info[key] = int(value)
                        elif isinstance(value, (np.float64, np.float32)):
                            project_info[key] = float(value)
                            
                    # Add recommendation score
                    project_info['recommendation_score'] = float(score)
                    recommendations.append(project_info)
                else:
                    # If project details not found, add minimal info
                    recommendations.append({
                        'id': project_id,
                        'recommendation_score': float(score),
                        'name': project_id,  # Use ID as fallback name
                        'symbol': '',
                        'primary_category': 'unknown',
                        'chain': 'unknown'
                    })
                    
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def evaluate_recommendations(
            self, 
            user_item_matrix: pd.DataFrame, 
            item_similarity_matrix: pd.DataFrame, 
            projects_df: pd.DataFrame, 
            test_users: Optional[List[str]] = None, 
            n: int = 10
        ) -> Dict[str, float]:
        """
        Mengevaluasi performa rekomendasi dengan metrik precision dan recall
        
        Args:
            user_item_matrix: User-item matrix
            item_similarity_matrix: Item similarity matrix
            projects_df: DataFrame proyek
            test_users: List user untuk evaluasi (default: 5 random)
            n: Jumlah rekomendasi
            
        Returns:
            dict: Metrics evaluasi (precision, recall, F1)
        """
        logger.info("Evaluating recommendation performance")
        
        try:
            # If no test users provided, select random users
            if test_users is None or not test_users:
                # Use numpy's modern random generator
                rng = np.random.default_rng(42)
                
                if len(user_item_matrix) < 5:
                    test_users = user_item_matrix.index.tolist()
                else:
                    test_users = rng.choice(user_item_matrix.index, 5, replace=False).tolist()
            
            # Enhance similarity matrix
            enhanced_similarity = self.enhance_collaborative_filtering(
                user_item_matrix, item_similarity_matrix, projects_df
            )
            
            # Metrics
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for user_id in test_users:
                # Create train/test split
                user_ratings = user_item_matrix.loc[user_id].copy()
                
                # Get items user has rated
                rated_items = user_ratings[user_ratings > 0].index.tolist()
                
                # If user has rated less than 5 items, skip
                if len(rated_items) < 5:
                    continue
                    
                # Hold out 20% of ratings for testing
                # Use numpy's modern random generator
                rng = np.random.default_rng(42)
                rng.shuffle(rated_items)
                split_idx = max(1, int(len(rated_items) * 0.2))
                
                test_items = rated_items[:split_idx]
                train_items = rated_items[split_idx:]
                
                # Create training matrix
                train_ratings = user_ratings.copy()
                train_ratings[test_items] = 0
                
                # Update user-item matrix for training
                train_matrix = user_item_matrix.copy()
                train_matrix.loc[user_id] = train_ratings
                
                # Get recommendations
                predictions = self.predict_ratings(
                    user_id, train_matrix, enhanced_similarity, n=n
                )
                
                recommended_items = [item for item, _ in predictions]
                
                # Calculate precision and recall
                true_positives = set(recommended_items) & set(test_items)
                
                if recommended_items:
                    precision = len(true_positives) / len(recommended_items)
                    precision_scores.append(precision)
                
                if test_items:
                    recall = len(true_positives) / len(test_items)
                    recall_scores.append(recall)
                
                # Calculate F1
                if precision_scores and recall_scores:
                    prec = precision_scores[-1]
                    rec = recall_scores[-1]
                    
                    if prec + rec > 0:
                        f1 = 2 * (prec * rec) / (prec + rec)
                        f1_scores.append(f1)
            
            # Average metrics
            avg_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
            avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
            avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
            
            metrics = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'num_users_evaluated': len(precision_scores)
            }
            
            logger.info(f"Evaluation results: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating recommendations: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'num_users_evaluated': 0,
                'error': str(e)
            }
    
    def get_similar_projects(
            self, 
            project_id: str, 
            projects_df: pd.DataFrame, 
            n: int = 10
        ) -> List[Dict[str, Any]]:
        """
        Mendapatkan proyek yang similar berdasarkan fitur
        
        Args:
            project_id: ID proyek
            projects_df: DataFrame proyek
            n: Jumlah proyek similar
            
        Returns:
            list: List proyek yang similar dengan detailnya
        """
        logger.info(f"Finding similar projects to {project_id}")
        
        try:
            # Check if project exists
            if project_id not in projects_df['id'].values:
                logger.error(f"Project {project_id} not found in projects data")
                return []
            
            # Calculate feature similarity
            feature_sim_df = self.calculate_weighted_feature_similarity(projects_df)
            
            if feature_sim_df is None or project_id not in feature_sim_df.index:
                logger.error(f"Could not calculate similarity for project {project_id}")
                return []
            
            # Get similarity scores for the target project
            project_similarities = feature_sim_df.loc[project_id]
            
            # Sort by similarity and get top n (excluding self)
            similar_projects = project_similarities.drop(project_id).sort_values(ascending=False).head(n)
            
            # Get project details
            recommendations = []
            
            for sim_project_id, similarity_score in similar_projects.items():
                project_data = projects_df[projects_df['id'] == sim_project_id]
                
                if not project_data.empty:
                    project_info = project_data.iloc[0].to_dict()
                    
                    # Convert numpy types to Python native types for JSON serialization
                    for key, value in project_info.items():
                        if isinstance(value, (np.int64, np.int32)):
                            project_info[key] = int(value)
                        elif isinstance(value, (np.float64, np.float32)):
                            project_info[key] = float(value)
                    
                    # Add similarity score
                    project_info['similarity_score'] = float(similarity_score)
                    
                    recommendations.append(project_info)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error finding similar projects for {project_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []


if __name__ == "__main__":
    # Example usage
    import os
    import pandas as pd
    from src.models.matrix_builder import MatrixBuilder
    
    try:
        # Initialize feature-enhanced CF
        feature_cf = FeatureEnhancedCF()
        
        # Load data
        matrix_builder = MatrixBuilder()
        projects_df, interactions_df, feature_matrix = matrix_builder.load_latest_processed_data()
        
        if projects_df is not None and interactions_df is not None:
            # Build user-item matrix
            user_item_df, user_indices, item_indices = matrix_builder.build_user_item_matrix(interactions_df)
            
            # Build item similarity matrix
            item_similarity_df = matrix_builder.build_item_similarity_matrix(user_item_df)
            
            # Get a sample user
            sample_user = user_item_df.index[0] if len(user_item_df) > 0 else None
            
            if sample_user and item_similarity_df is not None:
                print(f"Testing with sample user: {sample_user}")
                
                # Generate recommendations
                recommendations = feature_cf.recommend_projects(
                    sample_user, user_item_df, item_similarity_df, projects_df, n=5
                )
                
                # Print recommendations
                print(f"\nTop 5 feature-enhanced recommendations for {sample_user}:")
                for i, rec in enumerate(recommendations, 1):
                    name = rec.get('name', rec.get('id', 'Unknown'))
                    score = rec.get('recommendation_score', 0.0)
                    print(f"{i}. {name} - Score: {score:.4f}")
                
                # Evaluate recommendations
                metrics = feature_cf.evaluate_recommendations(user_item_df, item_similarity_df, projects_df)
                print(f"\nEvaluation metrics: {metrics}")
                
                # Get similar projects for a sample project
                if len(projects_df) > 0:
                    sample_project = projects_df['id'].iloc[0]
                    similar_projects = feature_cf.get_similar_projects(sample_project, projects_df, n=3)
                    
                    print(f"\nProjects similar to {sample_project}:")
                    for i, proj in enumerate(similar_projects, 1):
                        name = proj.get('name', proj.get('id', 'Unknown'))
                        score = proj.get('similarity_score', 0.0)
                        print(f"{i}. {name} - Similarity: {score:.4f}")
            else:
                print("No sample user or item similarity matrix available")
        else:
            print("Required data not available. Please run data collection and preprocessing first.")
    except Exception as e:
        import traceback
        print(f"Error in example: {e}")
        print(traceback.format_exc())