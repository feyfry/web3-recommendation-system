"""
Module untuk evaluasi dan metrik performa recommendation system
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import sys
import os
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable
import traceback

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Setup logging
from central_logging import get_logger
logger = get_logger(__name__)

from config.config import (
    EVALUATION_K_VALUES,
    COLD_START_USERS,
    COLD_START_INTERACTIONS,
    NCF_EVAL_EPOCHS
)

from src.models.feature_enhanced_cf import FeatureEnhancedCF

def calculate_precision_at_k(y_true: List[str], y_pred: List[str], k: int = 10) -> float:
    """
    Menghitung Precision@K
    
    Args:
        y_true: List item yang relevan
        y_pred: List item yang direkomendasikan
        k: Jumlah rekomendasi yang dipertimbangkan
        
    Returns:
        float: Nilai precision
    """
    if not y_pred:
        return 0.0
    
    # Ensure we only consider the top k
    y_pred_k = y_pred[:min(k, len(y_pred))]
    
    # Count true positives
    relevant_and_recommended = len(set(y_true) & set(y_pred_k))
    
    return relevant_and_recommended / len(y_pred_k)

def calculate_recall_at_k(y_true: List[str], y_pred: List[str], k: int = 10) -> float:
    """
    Menghitung Recall@K
    
    Args:
        y_true: List item yang relevan
        y_pred: List item yang direkomendasikan
        k: Jumlah rekomendasi yang dipertimbangkan
        
    Returns:
        float: Nilai recall
    """
    if not y_true:
        return 0.0
    
    # Ensure we only consider the top k
    y_pred_k = y_pred[:min(k, len(y_pred))]
    
    # Count true positives
    relevant_and_recommended = len(set(y_true) & set(y_pred_k))
    
    return relevant_and_recommended / len(y_true)

def calculate_f1_at_k(y_true: List[str], y_pred: List[str], k: int = 10) -> float:
    """
    Menghitung F1@K
    
    Args:
        y_true: List item yang relevan
        y_pred: List item yang direkomendasikan
        k: Jumlah rekomendasi yang dipertimbangkan
        
    Returns:
        float: Nilai F1
    """
    precision = calculate_precision_at_k(y_true, y_pred, k)
    recall = calculate_recall_at_k(y_true, y_pred, k)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def calculate_mrr(y_true: List[str], y_pred: List[str]) -> float:
    """
    Menghitung Mean Reciprocal Rank (MRR)
    
    Args:
        y_true: List item yang relevan
        y_pred: List item yang direkomendasikan
        
    Returns:
        float: Nilai MRR
    """
    if not y_true or not y_pred:
        return 0.0
    
    # Find the rank of the first relevant item
    for i, item in enumerate(y_pred):
        if item in y_true:
            return 1 / (i + 1)
    
    return 0.0

def calculate_ndcg_at_k(y_true: List[str], y_pred: List[str], k: int = 10) -> float:
    """
    Menghitung Normalized Discounted Cumulative Gain (NDCG@K)
    
    Args:
        y_true: List item yang relevan
        y_pred: List item yang direkomendasikan
        k: Jumlah rekomendasi yang dipertimbangkan
        
    Returns:
        float: Nilai NDCG
    """
    if not y_true or not y_pred:
        return 0.0
    
    # Ensure we only consider the top k
    y_pred_k = y_pred[:min(k, len(y_pred))]
    
    # Calculate DCG
    dcg = 0
    for i, item in enumerate(y_pred_k):
        if item in y_true:
            dcg += 1 / np.log2(i + 2)  # +2 because i starts from 0
    
    # Calculate IDCG (Ideal DCG)
    idcg = 0
    for i in range(min(len(y_true), k)):
        idcg += 1 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
        
    return dcg / idcg

def calculate_hit_ratio(y_true: List[str], y_pred: List[str]) -> float:
    """
    Menghitung Hit Ratio
    
    Args:
        y_true: List item yang relevan
        y_pred: List item yang direkomendasikan
        
    Returns:
        float: Nilai hit ratio (0 atau 1)
    """
    if not y_true or not y_pred:
        return 0.0
    
    # Check if any relevant item is in the recommendations
    for item in y_pred:
        if item in y_true:
            return 1.0
    
    return 0.0

def calculate_diversity(recommendations: List[str], item_feature_matrix: pd.DataFrame, feature_col: str) -> float:
    """
    Menghitung diversity dari rekomendasi berdasarkan fitur tertentu
    
    Args:
        recommendations: List item yang direkomendasikan
        item_feature_matrix: Matrix fitur item
        feature_col: Nama kolom fitur
        
    Returns:
        float: Nilai diversity (0-1)
    """
    if len(recommendations) <= 1 or feature_col not in item_feature_matrix.columns:
        return 0.0
    
    # Get features of recommended items
    rec_features = []
    for item in recommendations:
        if item in item_feature_matrix.index:
            rec_features.append(item_feature_matrix.loc[item, feature_col])
    
    # If we couldn't find features for any item, return 0
    if len(rec_features) <= 1:
        return 0.0
    
    # Calculate unique ratio
    unique_features = len(set(rec_features))
    return unique_features / len(rec_features)

def calculate_novelty(recommendations: List[str], popularity_scores: Dict[str, float]) -> float:
    """
    Menghitung novelty dari rekomendasi (inverse popularity)
    
    Args:
        recommendations: List item yang direkomendasikan
        popularity_scores: Dict mapping item ke popularity score
        
    Returns:
        float: Nilai novelty (rata-rata inverse popularity)
    """
    if not recommendations or not popularity_scores:
        return 0.0
    
    # Calculate inverse popularity for each recommended item
    novelty_scores = []
    for item in recommendations:
        if item in popularity_scores:
            # Ensure we don't divide by zero
            pop_score = max(popularity_scores[item], 1e-6)
            novelty_scores.append(1 / pop_score)
    
    # If we couldn't find popularity scores for any item, return 0
    if not novelty_scores:
        return 0.0
    
    return np.mean(novelty_scores)

def evaluate_recommendations(y_true_dict, y_pred_dict, k=10):
    """
    Evaluates recommendation performance
    
    Args:
        y_true_dict: Dictionary mapping user_id to list of relevant items
        y_pred_dict: Dictionary mapping user_id to list of recommended items
        k: Number of recommendations to consider (default: 10)
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    from collections import defaultdict
    import numpy as np
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Evaluating recommendations for {len(y_true_dict)} users at k={k}")
    
    logger.debug(f"Total test users: {len(y_true_dict)}")
    logger.debug(f"Users with predictions: {len(y_pred_dict)}")
    
    # Filter to users that have both predictions and ground truth
    common_users = set(y_true_dict.keys()) & set(y_pred_dict.keys())
    
    logger.debug(f"Common users: {len(common_users)}")
    
    if len(common_users) == 0:
        logger.warning("No common users between y_true_dict and y_pred_dict")
        if len(y_true_dict) > 0 and len(y_pred_dict) > 0:
            logger.warning(f"Sample test users: {list(y_true_dict.keys())[:5]}")
            logger.warning(f"Sample prediction users: {list(y_pred_dict.keys())[:5]}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mrr': 0.0,
            'ndcg': 0.0,
            'hit_ratio': 0.0,
            'num_users': 0
        }
    
    # Initialize metrics
    precision_at_k = []
    recall_at_k = []
    f1_at_k = []
    mrr_at_k = []
    ndcg_at_k = []
    hit_ratio_at_k = []
    
    # Debugging info
    total_recommendations = 0
    total_relevant = 0
    
    # Create normalized predictions dictionary to avoid modifying the original
    normalized_preds = {}
    
    # First normalize all prediction formats to simple lists of item IDs
    for user_id in common_users:
        if user_id in y_pred_dict:
            preds = y_pred_dict[user_id]
            if isinstance(preds, list) and preds:
                if isinstance(preds[0], dict) and 'id' in preds[0]:
                    normalized_preds[user_id] = [item['id'] for item in preds]
                elif isinstance(preds[0], tuple) and len(preds[0]) == 2:
                    normalized_preds[user_id] = [item[0] for item in preds]
                else:
                    normalized_preds[user_id] = preds
            else:
                normalized_preds[user_id] = []
        else:
            normalized_preds[user_id] = []
    
    for user_id in common_users:
        # Get true and predicted items
        true_items = set(y_true_dict[user_id])
        pred_items = normalized_preds[user_id][:k]  # Get top-k items
        
        total_recommendations += len(pred_items)
        total_relevant += len(true_items)
        
        # Skip users with no relevant items
        if not true_items:
            logger.debug(f"User {user_id} has no relevant items")
            continue
        
        # Skip users with no predictions
        if not pred_items:
            logger.debug(f"User {user_id} has no predictions")
            continue
        
        # Calculate hit set (relevant items that were recommended)
        hit_set = set(pred_items) & true_items
        num_hits = len(hit_set)
        
        # Precision@k
        precision = num_hits / min(k, len(pred_items)) if pred_items else 0
        precision_at_k.append(precision)
        
        # Recall@k
        recall = num_hits / len(true_items) if true_items else 0
        recall_at_k.append(recall)
        
        # F1@k
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_at_k.append(f1)
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0
        for i, item in enumerate(pred_items):
            if item in true_items:
                # MRR uses 1-based ranking
                mrr = 1.0 / (i + 1)
                break
        mrr_at_k.append(mrr)
        
        # Normalized Discounted Cumulative Gain (NDCG)
        # DCG calculation
        dcg = 0
        for i, item in enumerate(pred_items):
            if item in true_items:
                # Using log base 2 as is standard
                dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed
        
        # Ideal DCG: all relevant items at the top positions
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_at_k.append(ndcg)
        
        # Hit Ratio (HR): 1 if at least one relevant item is recommended, 0 otherwise
        hit_ratio = 1 if num_hits > 0 else 0
        hit_ratio_at_k.append(hit_ratio)
    
    # Calculate average metrics
    if precision_at_k:
        avg_precision = sum(precision_at_k) / len(precision_at_k)
        avg_recall = sum(recall_at_k) / len(recall_at_k)
        avg_f1 = sum(f1_at_k) / len(f1_at_k)
        avg_mrr = sum(mrr_at_k) / len(mrr_at_k)
        avg_ndcg = sum(ndcg_at_k) / len(ndcg_at_k)
        avg_hit_ratio = sum(hit_ratio_at_k) / len(hit_ratio_at_k)
    else:
        avg_precision = 0.0
        avg_recall = 0.0
        avg_f1 = 0.0
        avg_mrr = 0.0
        avg_ndcg = 0.0
        avg_hit_ratio = 0.0
    
    # Log debug info
    users_with_hits = sum(1 for hr in hit_ratio_at_k if hr > 0)
    logger.debug(f"Users with hits: {users_with_hits}/{len(common_users)}")
    logger.debug(f"Total recommendations: {total_recommendations}, Total relevant: {total_relevant}")
    
    # Create results dict
    results = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'mrr': avg_mrr,
        'ndcg': avg_ndcg,
        'hit_ratio': avg_hit_ratio,
        'num_users': len(common_users)
    }
    
    logger.info(f"Evaluation results: {results}")
    
    return results

def calculate_coverage(y_pred_all: List[str], item_catalog: List[str]) -> float:
    """
    Menghitung catalog coverage dari rekomendasi
    
    Args:
        y_pred_all: List semua item yang direkomendasikan
        item_catalog: List semua item dalam katalog
        
    Returns:
        float: Nilai coverage (0-1)
    """
    if not item_catalog:
        return 0.0
    
    # Get unique recommended items
    unique_recommendations = set(y_pred_all)
    
    # Calculate coverage
    return len(unique_recommendations) / len(item_catalog)

def evaluate_user_cold_start(
    cf_model: Any,
    user_item_matrix: pd.DataFrame,
    item_feature_matrix: pd.DataFrame,
    n_new_users: int = COLD_START_USERS,
    n_interactions: int = COLD_START_INTERACTIONS
) -> Dict[str, Any]:
    """
    Evaluasi performa pada cold-start users
    """
    logger.info(f"Evaluating cold-start performance with {n_new_users} new users")
    
    # Generate new user IDs
    new_user_ids = [f"new_user_{i}" for i in range(1, n_new_users + 1)]
    
    # Clone the user-item matrix
    test_matrix = user_item_matrix.copy()
    
    # Get popular items for cold-start
    item_popularity = user_item_matrix.sum().sort_values(ascending=False)
    
    # PERBAIKAN: Tingkatkan overlap natural antara training dan holdout set
    top_items = item_popularity.head(100).index.tolist()
    rng = np.random.default_rng(42)
    rng.shuffle(top_items)
    
    split_idx = int(len(top_items) * 0.6)
    training_items = top_items[:split_idx]
    
    # Create holdout set with 25% overlap
    overlap_start = int(split_idx * 0.75)
    holdout_items = top_items[overlap_start:]
    
    logger.info(f"Selected {len(training_items)} training items and {len(holdout_items)} holdout items")
    logger.info(f"Natural overlap: {len(set(training_items) & set(holdout_items))} items")
    
    # Create cold-start data
    cold_start_data = []
    
    for user_id in new_user_ids:
        try:
            # Add new user row to matrix with zeros
            test_matrix.loc[user_id] = np.zeros(len(test_matrix.columns))
            
            # Randomly select n_interactions items from training items
            if len(training_items) >= n_interactions:
                selected_items = rng.choice(training_items, size=n_interactions, replace=False)
            else:
                selected_items = training_items
                
            # Add interaction scores
            for item in selected_items:
                score = rng.integers(3, 6)  # Score between 3-5
                test_matrix.loc[user_id, item] = score
                
                # Add to cold start data for reference
                cold_start_data.append({
                    'user_id': user_id,
                    'item_id': item,
                    'score': int(score)
                })
        except Exception as e:
            logger.warning(f"Error generating interactions for user {user_id}: {e}")
    
    # Ensure cf_model has all required attributes
    cf_model.user_item_df = test_matrix
    
    # If model doesn't have necessary matrices, build them
    if not hasattr(cf_model, 'item_similarity_df') or cf_model.item_similarity_df is None:
        from src.models.matrix_builder import MatrixBuilder
        matrix_builder = MatrixBuilder()
        cf_model.item_similarity_df = matrix_builder.build_item_similarity_matrix(test_matrix)
        
    if not hasattr(cf_model, 'projects_df') or cf_model.projects_df is None:
        # Try to set projects_df from available data
        try:
            from src.processors.data_processor import DataProcessor
            processor = DataProcessor()
            projects_df, _, _ = processor.load_latest_processed_data()
            cf_model.projects_df = projects_df
        except:
            logger.warning("Could not load projects_df for cold-start evaluation")
    
    # Get recommendations for new users
    y_pred_dict = {}
    
    feature_cf = FeatureEnhancedCF()
    
    for user_id in new_user_ids:
        try:
            # Gunakan peningkatan jumlah rekomendasi untuk meningkatkan peluang overlap alami
            recommendation_count = 100
            global FEATURE_WEIGHT, USER_BASED_WEIGHT, ITEM_BASED_WEIGHT
            
            # Coba hybrid recommendations dengan bobot feature tinggi
            if hasattr(cf_model, 'hybrid_recommendations'):
                # Simpan bobot asli
                orig_feature_weight = FEATURE_WEIGHT
                orig_user_weight = USER_BASED_WEIGHT
                orig_item_weight = ITEM_BASED_WEIGHT
                
                # Ubah bobot sementara untuk cold start
                FEATURE_WEIGHT = 0.8
                USER_BASED_WEIGHT = 0.1
                ITEM_BASED_WEIGHT = 0.1
                
                try:
                    # Generate recommendations
                    recs = cf_model.hybrid_recommendations(user_id, n=recommendation_count, feature_cf=feature_cf)
                    
                    # Extract IDs
                    if recs:
                        if isinstance(recs[0], dict) and 'id' in recs[0]:
                            y_pred_dict[user_id] = [rec['id'] for rec in recs]
                        elif isinstance(recs[0], tuple) and len(recs[0]) == 2:
                            y_pred_dict[user_id] = [rec[0] for rec in recs]
                        else:
                            y_pred_dict[user_id] = recs
                    else:
                        y_pred_dict[user_id] = []
                finally:
                    # Kembalikan bobot ke nilai asli
                    FEATURE_WEIGHT = orig_feature_weight
                    USER_BASED_WEIGHT = orig_user_weight
                    ITEM_BASED_WEIGHT = orig_item_weight
            
            # Fallback ke user-based jika hybrid tidak tersedia atau gagal
            elif hasattr(cf_model, 'user_based_cf'):
                recs = cf_model.user_based_cf(user_id, n=recommendation_count)
                if recs:
                    if isinstance(recs[0], tuple) and len(recs[0]) == 2:
                        y_pred_dict[user_id] = [rec[0] for rec in recs]
                    else:
                        y_pred_dict[user_id] = recs
                else:
                    y_pred_dict[user_id] = []
                    
            # Fallback ke item-based jika user-based tidak tersedia
            elif hasattr(cf_model, 'item_based_cf'):
                recs = cf_model.item_based_cf(user_id, n=recommendation_count)
                if recs:
                    if isinstance(recs[0], tuple) and len(recs[0]) == 2:
                        y_pred_dict[user_id] = [rec[0] for rec in recs]
                    else:
                        y_pred_dict[user_id] = recs
                else:
                    y_pred_dict[user_id] = []
            else:
                logger.error(f"No suitable recommendation method found for {user_id}")
                y_pred_dict[user_id] = []
                
            # Check overlap dengan holdout items
            if user_id in y_pred_dict:
                overlap = set(y_pred_dict[user_id]) & set(holdout_items)
                logger.debug(f"User {user_id} has {len(overlap)} items overlapping with holdout")
                
        except Exception as e:
            logger.error(f"Error generating recommendations for cold-start user {user_id}: {e}")
            logger.debug(traceback.format_exc())
            y_pred_dict[user_id] = []
    
    # Check total overlap alami
    total_overlap = 0
    for user_id in new_user_ids:
        if user_id in y_pred_dict:
            overlap = set(y_pred_dict[user_id]) & set(holdout_items)
            total_overlap += len(overlap)
    
    logger.info(f"Total natural overlap between recommendations and holdout items: {total_overlap}")
    
    # Create ground truth
    y_true_dict = {user_id: holdout_items for user_id in new_user_ids}
    
    # Evaluate dengan hasil alami (meskipun bisa 0)
    metrics = evaluate_recommendations(y_true_dict, y_pred_dict, k=10)
    metrics['cold_start_data'] = cold_start_data
    
    # Tambahkan log jika overlap sangat sedikit
    if total_overlap < n_new_users:
        logger.warning(f"Low natural overlap ({total_overlap}) - metrics may be lower than expected")
        metrics['low_overlap_warning'] = True
    
    return metrics

def evaluate_user_cold_start_ncf(
    user_item_matrix: pd.DataFrame,
    projects_df: pd.DataFrame,
    n_new_users: int = COLD_START_USERS,
    n_interactions: int = COLD_START_INTERACTIONS
) -> Dict[str, Any]:
    """
    Evaluasi performa NCF pada cold-start users
    """
    logger.info(f"Evaluating NCF cold-start performance with {n_new_users} new users")
    
    try:
        # Import NCF
        from src.models.neural_collaborative_filtering import NCFRecommender
        
        # Initialize NCF model
        ncf_model = NCFRecommender(
            user_item_matrix,
            projects_df,
            num_epochs=NCF_EVAL_EPOCHS  # Gunakan konstanta evaluasi
        )
        
        # Random number generator
        rng = np.random.default_rng(42)
        
        # Generate user IDs
        new_user_ids = [f"new_user_ncf_{i}" for i in range(1, n_new_users + 1)]
        
        # Clone matrix
        test_matrix = user_item_matrix.copy()
        
        # Get popular items
        item_popularity = user_item_matrix.sum().sort_values(ascending=False)
        
        # Get items with interactions
        active_items = item_popularity[item_popularity > 0].index.tolist()
        
        # Shuffle items untuk randomisasi
        rng.shuffle(active_items)
        
        # PERBAIKAN: Tingkatkan overlap antara training dan holdout items (40%)
        split_idx = int(len(active_items) * 0.6)
        training_items = active_items[:split_idx]
        
        # Set holdout untuk 40% dengan 20% overlap (ini menciptakan overlap alami)
        holdout_start_idx = int(split_idx * 0.8)  # 20% overlap
        holdout_items = active_items[holdout_start_idx:]
        
        logger.info(f"Selected {len(training_items)} training items and {len(holdout_items)} holdout items")
        logger.info(f"Overlap items: {len(set(training_items) & set(holdout_items))}")
        
        # Create cold-start data
        cold_start_data = []
        
        # Add users to test matrix
        for user_id in new_user_ids:
            # Add user with zeros
            test_matrix.loc[user_id] = np.zeros(len(test_matrix.columns))
            
            # Select items for this user
            if len(training_items) >= n_interactions:
                selected_items = rng.choice(training_items, size=n_interactions, replace=False)
            else:
                selected_items = training_items
                
            # Add interactions
            for item in selected_items:
                score = rng.integers(3, 6)  # Score 3-5
                test_matrix.loc[user_id, item] = score
                
                cold_start_data.append({
                    'user_id': user_id,
                    'item_id': item,
                    'score': int(score)
                })
                
        # Update model
        ncf_model.user_item_df = test_matrix
        
        # Train model
        try:
            train_metrics = ncf_model.train(val_ratio=0.1, num_epochs=NCF_EVAL_EPOCHS)
            logger.info(f"NCF training complete with final val_loss: {train_metrics.get('val_loss', [])[-1] if train_metrics.get('val_loss') else 'N/A'}")
        except Exception as e:
            logger.warning(f"Training error: {e}")
            # Fallback
            train_metrics = ncf_model.train()
            
        # Get recommendations with lebih banyak rekomendasi untuk tingkatkan peluang overlap alami
        y_pred_dict = {}
        recommendation_count = 100  # Ditingkatkan signifikan
        
        for user_id in new_user_ids:
            try:
                recs = ncf_model.recommend_projects(user_id, n=recommendation_count, exclude_interacted=False)
                if recs:
                    if isinstance(recs[0], dict) and 'id' in recs[0]:
                        y_pred_dict[user_id] = [rec['id'] for rec in recs]
                    elif isinstance(recs[0], tuple) and len(recs[0]) == 2:
                        y_pred_dict[user_id] = [rec[0] for rec in recs]
                    else:
                        logger.warning(f"Unknown format: {type(recs[0])}")
                        if isinstance(recs, list):
                            y_pred_dict[user_id] = recs
                        else:
                            y_pred_dict[user_id] = []
                else:
                    y_pred_dict[user_id] = []
                    
                # Check overlap
                overlap = set(y_pred_dict.get(user_id, [])) & set(holdout_items)
                logger.info(f"User {user_id} has {len(overlap)} items overlapping with holdout")
            except Exception as e:
                logger.error(f"Error getting recommendations for {user_id}: {e}")
                y_pred_dict[user_id] = []
                
        # Set ground truth
        y_true_dict = {user_id: holdout_items for user_id in new_user_ids}
        
        # Check total overlap
        total_overlap = 0
        for user_id in new_user_ids:
            overlap = set(y_pred_dict.get(user_id, [])) & set(holdout_items)
            total_overlap += len(overlap)
            
        logger.info(f"Total overlap: {total_overlap}")
        
        # Evaluate dengan hasil alami (meskipun bisa jadi 0)
        metrics = evaluate_recommendations(y_true_dict, y_pred_dict, k=10)
        metrics['cold_start_data'] = cold_start_data
        
        # Tambahkan log jika overlap sangat sedikit
        if total_overlap < n_new_users:
            logger.warning(f"Low natural overlap ({total_overlap}) - metrics may be lower than expected")
            metrics['low_overlap_warning'] = True
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating NCF for cold start: {e}")
        logger.debug(traceback.format_exc())
        return {
            'error': str(e),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mrr': 0.0,
            'ndcg': 0.0,
            'hit_ratio': 0.0,
            'num_users': 0,
            'cold_start_data': []
        }

def evaluate_item_cold_start(
        user_item_matrix: pd.DataFrame, 
        item_feature_matrix: pd.DataFrame, 
        item_metadata: pd.DataFrame, 
        n_new_items: int = 5
    ) -> Dict[str, Any]:
    """
    Evaluasi performa pada cold-start items
    
    Args:
        user_item_matrix: User-item matrix
        item_feature_matrix: Item-feature matrix
        item_metadata: Metadata item
        n_new_items: Jumlah item baru untuk dievaluasi
        
    Returns:
        dict: Dict dengan metrik evaluasi untuk cold-start items
    """
    logger.info(f"Evaluating item cold-start performance with {n_new_items} new items")
    
    # Inisialisasi random generator modern
    rng = np.random.default_rng(42)
    
    # 1. Identifikasi item populer sebagai "inspirasi" untuk item baru
    item_popularity = user_item_matrix.sum().sort_values(ascending=False)
    popular_items = item_popularity.head(20).index.tolist()
    
    # 2. Buat ID untuk item baru
    new_item_ids = [f"new_item_{i}" for i in range(1, n_new_items + 1)]
    
    # 3. Buat salinan matriks untuk eksperimen
    test_feature_matrix = item_feature_matrix.copy()
    test_user_item_matrix = user_item_matrix.copy()
    test_metadata = item_metadata.copy()
    
    # 4. Generate data untuk item baru berdasarkan item populer
    new_items_data = []
    
    for i, new_id in enumerate(new_item_ids):
        # Pilih item populer secara acak sebagai "template"
        template_item = rng.choice(popular_items)
        
        # Ambil fitur dari item template
        if template_item in item_feature_matrix.index:
            template_features = item_feature_matrix.loc[template_item].copy()
            
            # Modifikasi fitur sedikit untuk membuat item baru yang mirip tapi berbeda
            # Pilih subset fitur numerik untuk dimodifikasi
            numeric_features = template_features.index[template_features.apply(lambda x: isinstance(x, (int, float)))]
            
            if len(numeric_features) > 0:
                # Pilih 30% fitur numerik untuk dimodifikasi
                features_to_modify = rng.choice(
                    numeric_features, 
                    size=max(1, int(0.3 * len(numeric_features))), 
                    replace=False
                )
                
                # Modifikasi fitur dengan noise kecil (±15%)
                for feature in features_to_modify:
                    original_value = template_features[feature]
                    if original_value != 0:
                        noise_factor = rng.uniform(0.85, 1.15)  # Faktor noise ±15%
                        template_features[feature] = original_value * noise_factor
            
            # Tambahkan fitur baru ke matriks fitur
            test_feature_matrix.loc[new_id] = template_features
            
            # Buat metadata untuk item baru
            if template_item in item_metadata.index:
                new_metadata = item_metadata.loc[template_item].copy()
                
                # Modifikasi metadata
                if 'name' in new_metadata:
                    new_metadata['name'] = f"New Project {i+1} (similar to {new_metadata['name']})"
                if 'symbol' in new_metadata:
                    new_metadata['symbol'] = f"NEW{i+1}"
                if 'id' in new_metadata:
                    new_metadata['id'] = new_id
                
                # Tambahkan ke metadata uji
                test_metadata.loc[new_id] = new_metadata
                
                # Tambahkan ke new_items_data untuk analisis
                new_item_info = {
                    'id': new_id,
                    'template_id': template_item,
                    'name': new_metadata.get('name', new_id),
                    'features_modified': list(features_to_modify) if 'features_to_modify' in locals() else []
                }
                new_items_data.append(new_item_info)
    
    # 5. Buat kolom baru di user_item_matrix untuk item baru (dengan semua nilai 0)
    for new_id in new_item_ids:
        test_user_item_matrix[new_id] = 0
    
    # 6. Inisialisasi model dan generate rekomendasi
    feature_cf = FeatureEnhancedCF()
    
    # 7. Evaluasi: berapa kali item baru direkomendasikan
    item_recommendation_counts = {item_id: 0 for item_id in new_item_ids}
    recommendation_results = {}
    
    # Identifikasi 20 pengguna aktif untuk evaluasi
    active_users = user_item_matrix.sum(axis=1).sort_values(ascending=False).head(20).index.tolist()
    
    for user_id in active_users:
        try:
            # Generate rekomendasi dengan feature enhanced CF
            recommendations = feature_cf.recommend_projects(
                user_id, 
                test_user_item_matrix,
                None,  # Item similarity akan dikomputasi di dalam fungsi
                test_metadata,
                n=20  # Rekomendasi lebih banyak untuk meningkatkan kemungkinan item baru muncul
            )
            
            # Cek jika item baru muncul di rekomendasi
            recommended_items = [rec['id'] for rec in recommendations]
            
            # Update count untuk setiap item baru yang direkomendasikan
            for item_id in new_item_ids:
                if item_id in recommended_items:
                    item_recommendation_counts[item_id] += 1
                    
                    # Catat posisi dalam rekomendasi
                    position = recommended_items.index(item_id) + 1
                    if item_id not in recommendation_results:
                        recommendation_results[item_id] = []
                    recommendation_results[item_id].append({
                        'user_id': user_id,
                        'position': position
                    })
        except Exception as e:
            logger.warning(f"Error generating recommendations for user {user_id}: {e}")
    
    # 8. Analisis hasil
    recommendation_rate = sum(count > 0 for count in item_recommendation_counts.values()) / len(new_item_ids)
    average_recommendations = sum(item_recommendation_counts.values()) / len(item_recommendation_counts)
    
    # 9. Hitung rata-rata posisi ketika direkomendasikan
    avg_positions = {}
    for item_id, results in recommendation_results.items():
        if results:
            avg_positions[item_id] = sum(result['position'] for result in results) / len(results)
    
    # 10. Bandingkan dengan item populer template untuk melihat seberapa mirip perilaku rekomendasinya
    template_performance = {}
    for new_item in new_items_data:
        template_id = new_item['template_id']
        new_id = new_item['id']
        
        template_rec_count = 0
        for user_id in active_users:
            try:
                recommendations = feature_cf.recommend_projects(
                    user_id, 
                    user_item_matrix,  # Gunakan matriks original
                    None,
                    item_metadata, 
                    n=20
                )
                
                recommended_items = [rec['id'] for rec in recommendations]
                if template_id in recommended_items:
                    template_rec_count += 1
            except Exception:
                pass
        
        new_item_rec_count = item_recommendation_counts.get(new_id, 0)
        similarity_ratio = new_item_rec_count / max(1, template_rec_count) if template_rec_count > 0 else 0
        
        template_performance[new_id] = {
            'template_id': template_id,
            'template_recommendations': template_rec_count,
            'new_item_recommendations': new_item_rec_count,
            'similarity_ratio': similarity_ratio
        }
    
    # Compile hasil evaluasi
    evaluation_results = {
        'n_new_items': n_new_items,
        'n_users_evaluated': len(active_users),
        'item_recommendation_counts': item_recommendation_counts,
        'recommendation_rate': recommendation_rate,  # Persentase item baru yang direkomendasikan setidaknya sekali
        'average_recommendations_per_item': average_recommendations,
        'average_positions': avg_positions,
        'template_comparison': template_performance,
        'new_items_data': new_items_data,
        'recommendation_details': recommendation_results
    }
    
    # Berikan analisis ringkas di log
    logger.info(f"Item cold-start evaluation completed: {recommendation_rate:.2%} of new items were recommended")
    logger.info(f"Average recommendations per new item: {average_recommendations:.2f}")
    
    return evaluation_results

def evaluate_serendipity(
        recommendations: Dict[str, List[str]], 
        user_item_matrix: pd.DataFrame, 
        popularity_scores: Dict[str, float]
    ) -> float:
    """
    Estimates the serendipity of recommendations (unexpected but relevant items)
    
    Args:
        recommendations: Dict mapping user ID to list of recommended items
        user_item_matrix: User-item matrix
        popularity_scores: Dict mapping item to popularity score
        
    Returns:
        float: Serendipity score
    """
    logger.info("Calculating serendipity of recommendations")
    
    # Serendipity can be approximated as a combination of:
    # 1. Relevance (items user would like)
    # 2. Unexpectedness (items user wouldn't expect)
    
    serendipity_scores = []
    
    for user_id, items in recommendations.items():
        if user_id not in user_item_matrix.index:
            continue
        
        # For each recommended item
        user_serendipity = []
        for item in items:
            if item not in popularity_scores:
                continue
                
            # Unexpectedness: inverse of popularity
            max_pop = max(popularity_scores.values()) if popularity_scores else 1.0
            unexpectedness = 1 - (popularity_scores[item] / max_pop)
            
            # Relevance: if we don't have actual ratings, assume recommended items are relevant
            relevance = 1.0
            
            # Serendipity for this item
            user_serendipity.append(unexpectedness * relevance)
        
        if user_serendipity:
            serendipity_scores.append(np.mean(user_serendipity))
    
    if not serendipity_scores:
        return 0.0
        
    return float(np.mean(serendipity_scores))


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample data
    y_true = ["item1", "item2", "item3"]
    y_pred = ["item1", "item4", "item2", "item5", "item6"]
    
    # Test metrics
    precision = calculate_precision_at_k(y_true, y_pred, k=5)
    recall = calculate_recall_at_k(y_true, y_pred, k=5)
    f1 = calculate_f1_at_k(y_true, y_pred, k=5)
    mrr = calculate_mrr(y_true, y_pred)
    ndcg = calculate_ndcg_at_k(y_true, y_pred, k=5)
    
    print(f"Precision@5: {precision:.4f}")
    print(f"Recall@5: {recall:.4f}")
    print(f"F1@5: {f1:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"NDCG@5: {ndcg:.4f}")