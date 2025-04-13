"""
Script untuk training model recommendation system
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import dan setup central logging
from central_logging import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# Import modules
from src.processors.data_processor import DataProcessor
from src.models.matrix_builder import MatrixBuilder
from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.feature_enhanced_cf import FeatureEnhancedCF
from src.models.neural_collaborative_filtering import NCFRecommender
from src.evaluation.metrics import evaluate_recommendations, evaluate_user_cold_start, evaluate_user_cold_start_ncf
from config.config import (
    PROCESSED_DATA_PATH,
    MODELS_PATH,
    EVALUATION_SPLIT,
    EVALUATION_RANDOM_SEED,
    NUM_RECOMMENDATIONS,
    COLD_START_USERS,
    COLD_START_INTERACTIONS,
    NCF_EMBEDDING_SIZE,
    NCF_LAYERS,
    NCF_LEARNING_RATE,
    NCF_BATCH_SIZE,
    NCF_NUM_EPOCHS,
    NCF_VALIDATION_RATIO
)

def train_model(args):
    """
    Melatih model recommendation system
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    # Ensure all required attributes exist with defaults
    if not hasattr(args, 'cold_start_users'):
        args.cold_start_users = 50
    if not hasattr(args, 'cold_start_interactions'):
        args.cold_start_interactions = 5
    if not hasattr(args, 'min_interactions'):
        args.min_interactions = 5
        
    # Tambahan: Pastikan atribut yang diperlukan untuk metode evaluasi tersedia
    if not hasattr(args, 'include_user_cf'):
        args.include_user_cf = getattr(args, 'include_all', False)
    if not hasattr(args, 'include_item_cf'):
        args.include_item_cf = getattr(args, 'include_all', False)
    if not hasattr(args, 'include_feature_cf'):
        args.include_feature_cf = getattr(args, 'include_all', False)
    if not hasattr(args, 'include_hybrid'):
        args.include_hybrid = getattr(args, 'include_all', False)
    if not hasattr(args, 'include_ncf'):
        args.include_ncf = getattr(args, 'include_all', False)
        
    logger.info("Starting model training")
    
    logger.info(f"Training with attributes: user_cf={args.include_user_cf}, item_cf={args.include_item_cf}, "
              f"feature_cf={args.include_feature_cf}, hybrid={args.include_hybrid}, ncf={args.include_ncf}")
    
    # Step 1: Load processed data
    if args.reprocess:
        logger.info("Reprocessing data...")
        processor = DataProcessor()
        projects_df, interactions_df, feature_matrix = processor.process_data()
    else:
        logger.info("Loading processed data...")
        processor = DataProcessor()
        projects_df, interactions_df, feature_matrix = processor.load_latest_processed_data()
        
    if projects_df is None or interactions_df is None:
        logger.error("Required data not available. Please run data collection and processing first.")
        return False
        
    logger.info(f"Loaded {len(projects_df)} projects and {len(interactions_df)} interactions")
    
    # Step 2: Build matrices
    if args.rebuild_matrices:
        logger.info("Building matrices...")
        matrix_builder = MatrixBuilder()
        matrices = matrix_builder.build_matrices()
        
        if matrices[0] is None:
            logger.error("Failed to build matrices")
            return False
            
        # Use _ for unused variables
        user_item_df, _, _, item_similarity_df, user_similarity_df, feature_similarity_df, combined_similarity_df = matrices
    else:
        logger.info("Loading existing matrices...")
        matrix_builder = MatrixBuilder()
        user_item_df, user_similarity_df, item_similarity_df, feature_similarity_df, combined_similarity_df, _ = matrix_builder._load_latest_data()
        
        if user_item_df is None or item_similarity_df is None:
            logger.error("Required matrices not available. Please rebuild matrices.")
            return False
            
    logger.info(f"User-item matrix shape: {user_item_df.shape}")
    
    # Alias user_item_df as user_item_matrix for consistent naming
    user_item_matrix = user_item_df
    
    # Step 3: Create and evaluate models
    logger.info("Training and evaluating models...")
    
    # Dictionary to store model performances
    model_performances = {}
    
    # Prepare data for evaluation
    if args.eval_split > 0:
        logger.info(f"Creating evaluation split with {args.eval_split*100}% of data...")
        
        # Create random test users using numpy's new random generator
        all_users = user_item_df.index.tolist()
        rng = np.random.default_rng(args.random_seed)  # New random generator
        test_users = rng.choice(all_users, int(len(all_users) * args.eval_split), replace=False).tolist()
        
        # Create ground truth for evaluation
        y_true_dict = {}
        train_matrix = user_item_df.copy()
        
        for user_id in test_users:
            # Get items user has rated
            user_ratings = user_item_df.loc[user_id].copy()
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            
            # Hanya masukkan ke test jika memiliki cukup ratings
            if len(rated_items) >= args.min_interactions:
                # Hold out 20% of ratings for testing
                rng.shuffle(rated_items)
                split_idx = max(1, int(len(rated_items) * 0.2))
                test_items = rated_items[:split_idx]
                
                # Update training matrix
                train_ratings = user_ratings.copy()
                train_ratings[test_items] = 0
                train_matrix.loc[user_id] = train_ratings
                
                # Add to ground truth
                y_true_dict[user_id] = test_items
            else:
                logger.warning(f"User {user_id} has too few interactions ({len(rated_items)}) for testing")
                
        logger.info(f"Created test set with {len(test_users)} users, avg {sum(len(y_true_dict[u]) for u in y_true_dict)/len(y_true_dict):.2f} test items per user")
    else:
        train_matrix = user_item_df
        test_users = []
        y_true_dict = {}
        
    # Evaluate User-Based Collaborative Filtering
    if args.include_user_cf:
        logger.info("Evaluating User-Based Collaborative Filtering...")
        
        cf = CollaborativeFiltering()
        cf.user_item_df = train_matrix  # Tetapkan train_matrix
        
        logger.info(f"Train matrix shape: {train_matrix.shape}")
        
        y_pred_dict = {}
        for user_id in test_users:
            if user_id in y_true_dict:  # Only evaluate users with ground truth
                recommendations = cf.user_based_cf(user_id, n=args.top_n)
                if recommendations:
                    y_pred_dict[user_id] = [item for item, _ in recommendations]
                else:
                    logger.warning(f"No recommendations generated for user {user_id} using user-based CF")
                    
        # Periksa overlap dengan ground truth
        common_users = set(y_true_dict.keys()) & set(y_pred_dict.keys())
        logger.info(f"Common users between ground truth and predictions: {len(common_users)}/{len(test_users)}")
        
        if len(y_pred_dict) > 0:
            user_cf_metrics = evaluate_recommendations(y_true_dict, y_pred_dict, k=args.top_n)
            model_performances['user_based_cf'] = user_cf_metrics
            logger.info(f"User-Based CF performance: {user_cf_metrics}")
        else:
            logger.warning("No predictions generated for User-Based CF")
            
    # Evaluate Item-Based Collaborative Filtering
    if args.include_item_cf:
        logger.info("Evaluating Item-Based Collaborative Filtering...")
        
        cf = CollaborativeFiltering()
        cf.user_item_df = train_matrix  # Tetapkan train_matrix
        
        logger.info(f"Evaluating with {len(test_users)} test users")
        
        y_pred_dict = {}
        for user_id in test_users:
            if user_id in y_true_dict:  # Only evaluate users with ground truth
                recommendations = cf.item_based_cf(user_id, n=args.top_n)
                if recommendations:
                    y_pred_dict[user_id] = [item for item, _ in recommendations]
                else:
                    logger.warning(f"No recommendations generated for user {user_id} using item-based CF")
                    
        # Periksa overlap dengan ground truth
        common_users = set(y_true_dict.keys()) & set(y_pred_dict.keys())
        logger.info(f"Common users between ground truth and predictions: {len(common_users)}/{len(test_users)}")
        
        if len(y_pred_dict) > 0:
            item_cf_metrics = evaluate_recommendations(y_true_dict, y_pred_dict, k=args.top_n)
            model_performances['item_based_cf'] = item_cf_metrics
            logger.info(f"Item-Based CF performance: {item_cf_metrics}")
        else:
            logger.warning("No predictions generated for Item-Based CF")
            
    # Evaluate Feature-Enhanced Collaborative Filtering
    if args.include_feature_cf:
        logger.info("Evaluating Feature-Enhanced Collaborative Filtering...")
        
        feature_cf = FeatureEnhancedCF()
        
        logger.info(f"Evaluating with {len(test_users)} test users")
        
        y_pred_dict = {}
        for user_id in test_users:
            if user_id in y_true_dict:  # Only evaluate users with ground truth
                recommendations = feature_cf.recommend_projects(
                    user_id, train_matrix, item_similarity_df, projects_df, n=args.top_n
                )
                
                if recommendations:
                    y_pred_dict[user_id] = [rec['id'] for rec in recommendations]
                else:
                    logger.warning(f"No recommendations generated for user {user_id} using feature-enhanced CF")
                    
        # Periksa overlap dengan ground truth
        common_users = set(y_true_dict.keys()) & set(y_pred_dict.keys())
        logger.info(f"Common users between ground truth and predictions: {len(common_users)}/{len(test_users)}")
        
        if len(y_pred_dict) > 0:
            feature_cf_metrics = evaluate_recommendations(y_true_dict, y_pred_dict, k=args.top_n)
            model_performances['feature_enhanced_cf'] = feature_cf_metrics
            logger.info(f"Feature-Enhanced CF performance: {feature_cf_metrics}")
        else:
            logger.warning("No predictions generated for Feature-Enhanced CF")
            
    # Evaluate Hybrid Recommendations
    if args.include_hybrid:
        logger.info("Evaluating Hybrid Recommendations...")

        cf = CollaborativeFiltering()
        cf.user_item_df = train_matrix
        cf.item_similarity_df = item_similarity_df
        cf.projects_df = projects_df

        # Inisialisasi feature_cf untuk hybrid
        feature_cf_for_hybrid = FeatureEnhancedCF()
        
        logger.info(f"Evaluating with {len(test_users)} test users")
        
        y_pred_dict = {}
        for user_id in test_users:
            if user_id in y_true_dict:  # Only evaluate users with ground truth
                recommendations = cf.hybrid_recommendations(user_id, n=args.top_n, feature_cf=feature_cf_for_hybrid)
                
                if recommendations:
                    # Extract IDs based on format
                    if isinstance(recommendations[0], dict) and 'id' in recommendations[0]:
                        y_pred_dict[user_id] = [rec['id'] for rec in recommendations]
                    elif isinstance(recommendations[0], tuple):
                        y_pred_dict[user_id] = [rec[0] for rec in recommendations]
                    else:
                        logger.warning(f"Unknown recommendation format for hybrid: {type(recommendations[0])}")
                        y_pred_dict[user_id] = []
                else:
                    logger.warning(f"No recommendations generated for user {user_id} using hybrid recommendations")
                    
        # Periksa overlap dengan ground truth
        common_users = set(y_true_dict.keys()) & set(y_pred_dict.keys())
        logger.info(f"Common users between ground truth and predictions: {len(common_users)}/{len(test_users)}")
        
        if len(y_pred_dict) > 0:
            hybrid_metrics = evaluate_recommendations(y_true_dict, y_pred_dict, k=args.top_n)
            model_performances['hybrid'] = hybrid_metrics
            logger.info(f"Hybrid performance: {hybrid_metrics}")
        else:
            logger.warning("No predictions generated for Hybrid Recommendations")
            
    # Evaluate Neural Collaborative Filtering
    if args.include_ncf:
        logger.info("Evaluating Neural Collaborative Filtering...")
        logger.info(f"Evaluating with {len(test_users)} test users")
        
        ncf = NCFRecommender(
            train_matrix,
            projects_df,
            embedding_size=NCF_EMBEDDING_SIZE,
            layers=NCF_LAYERS,
            learning_rate=NCF_LEARNING_RATE,
            batch_size=NCF_BATCH_SIZE,
            num_epochs=NCF_NUM_EPOCHS
        )

        # Train model
        train_metrics = ncf.train(val_ratio=NCF_VALIDATION_RATIO)
        
        # PERBAIKAN: Buat model_performances untuk ncf_training dengan struktur lengkap
        model_performances['ncf_training'] = {
            'train_loss': train_metrics.get('train_loss', []),
            'val_loss': train_metrics.get('val_loss', []),
            'precision': 0.0,  # Akan diupdate jika evaluasi berhasil
            'recall': 0.0,
            'f1': 0.0,
            'mrr': 0.0,
            'ndcg': 0.0,
            'hit_ratio': 0.0,
            'num_users': 0
        }

        # Generate recommendations
        y_pred_dict = {}
        for user_id in test_users:
            if user_id in y_true_dict:  # Only evaluate users with ground truth
                recommendations = ncf.recommend_projects(user_id, n=args.top_n)
                if recommendations and isinstance(recommendations[0], dict) and 'id' in recommendations[0]:
                    y_pred_dict[user_id] = [rec['id'] for rec in recommendations]
                else:
                    logger.warning(f"No valid recommendations for user {user_id} from NCF")

        # Check overlap with ground truth
        common_users = set(y_true_dict.keys()) & set(y_pred_dict.keys())
        logger.info(f"Common users between ground truth and predictions: {len(common_users)}/{len(test_users)}")

        if len(y_pred_dict) > 0:
            ncf_metrics = evaluate_recommendations(y_true_dict, y_pred_dict, k=args.top_n)
            model_performances['ncf'] = ncf_metrics
            logger.info(f"Neural CF performance: {ncf_metrics}")
            
            for key in ['precision', 'recall', 'f1', 'mrr', 'ndcg', 'hit_ratio', 'num_users']:
                if key in ncf_metrics:
                    model_performances['ncf_training'][key] = ncf_metrics[key]
            
            # Cold-start evaluation for NCF
            if args.eval_cold_start:
                cold_start_metrics = evaluate_user_cold_start_ncf(
                    user_item_matrix,  # Using aliased variable
                    projects_df,
                    n_new_users=args.cold_start_users,
                    n_interactions=args.cold_start_interactions
                )
                model_performances['cold_start_ncf'] = cold_start_metrics
                logger.info(f"NCF Cold-start performance: {cold_start_metrics}")
                
            # Save model if requested
            if args.save_model:
                # Ensure MODELS_PATH exists
                os.makedirs(MODELS_PATH, exist_ok=True)
                
                # Using the global timestamp defined at the top
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(MODELS_PATH, f"ncf_model_{timestamp}.pkl")
                
                ncf.save_model(model_path)
                logger.info(f"NCF model saved to {model_path}")
                
    # Evaluate Cold-Start Scenarios
    if args.eval_cold_start:
        logger.info("Evaluating cold-start scenarios...")
        
        # Initialize CF and Feature-CF models with required data
        cf = CollaborativeFiltering()
        cf.user_item_df = user_item_df
        cf.projects_df = projects_df
        cf.item_similarity_df = item_similarity_df
        
        feature_cf = FeatureEnhancedCF()
        
        cold_start_metrics = evaluate_user_cold_start(
            cf,  # Pass CF model with all required attributes
            user_item_df,
            feature_matrix,
            n_new_users=getattr(args, "cold_start_users", 50),
            n_interactions=getattr(args, "cold_start_interactions", 5)
        )
        
        model_performances['cold_start'] = cold_start_metrics
        logger.info(f"Cold-start performance: {cold_start_metrics}")
        
    # Step 4: Save model performance results
    if model_performances:
        logger.info("Saving model performance results...")
        
        # Ensure MODELS_PATH exists
        os.makedirs(MODELS_PATH, exist_ok=True)
        
        # Using timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(MODELS_PATH, f"model_performance_{timestamp}.json")
        
        # Convert any numpy values to regular Python types for JSON serialization
        serializable_performances = {}
        for model_name, metrics in model_performances.items():
            serializable_performances[model_name] = {}
            for metric_name, value in metrics.items():
                if isinstance(value, (np.int64, np.int32)):
                    serializable_performances[model_name][metric_name] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    serializable_performances[model_name][metric_name] = float(value)
                else:
                    serializable_performances[model_name][metric_name] = value
        
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_performances, f, ensure_ascii=False, indent=2)
            logger.info(f"Model performance results saved to {results_path}")
        except Exception as e:
            logger.error(f"Error saving model performance results: {e}")
            
    # Step 5: Save the feature-enhanced model weights if requested
    if args.save_model:
        logger.info("Saving model...")
        
        # Ensure MODELS_PATH exists
        os.makedirs(MODELS_PATH, exist_ok=True)
        
        # For simplicity, we'll save the similarity matrices since they're the core of the model
        model_data = {
            'timestamp': datetime.now().isoformat(),
            'item_similarity_matrix': item_similarity_df.to_dict() if item_similarity_df is not None else None,
            'user_similarity_matrix': user_similarity_df.to_dict() if user_similarity_df is not None else None,
            'feature_similarity_matrix': feature_similarity_df.to_dict() if feature_similarity_df is not None else None,
            'combined_similarity_matrix': combined_similarity_df.to_dict() if combined_similarity_df is not None else None,
            'config': {
                'top_n': args.top_n,
                'eval_split': args.eval_split,
                'random_seed': args.random_seed
            }
        }
        
        # Using timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODELS_PATH, f"recommendation_model_{timestamp}.pkl")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            
    logger.info("Model training and evaluation completed successfully")
    return True

def main() -> int:
    """
    Main function
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(description="Train and evaluate recommendation models")
    
    # Add arguments
    parser.add_argument("--reprocess", action="store_true", help="Reprocess data before training")
    parser.add_argument("--rebuild-matrices", action="store_true", help="Rebuild matrices before training")
    parser.add_argument("--eval-split", type=float, default=EVALUATION_SPLIT, 
                       help=f"Proportion of users to use for evaluation (default: {EVALUATION_SPLIT})")
    parser.add_argument("--random-seed", type=int, default=EVALUATION_RANDOM_SEED, 
                       help=f"Random seed for reproducibility (default: {EVALUATION_RANDOM_SEED})")
    parser.add_argument("--top-n", type=int, default=NUM_RECOMMENDATIONS, 
                       help=f"Number of recommendations to generate (default: {NUM_RECOMMENDATIONS})")
    parser.add_argument("--include-user-cf", action="store_true", help="Include User-Based CF evaluation")
    parser.add_argument("--include-item-cf", action="store_true", help="Include Item-Based CF evaluation")
    parser.add_argument("--include-feature-cf", action="store_true", help="Include Feature-Enhanced CF evaluation")
    parser.add_argument("--include-hybrid", action="store_true", help="Include Hybrid recommendations evaluation")
    parser.add_argument("--include-ncf", action="store_true", help="Include Neural CF evaluation")
    parser.add_argument("--include-all", action="store_true", help="Include all recommendation methods for evaluation")
    parser.add_argument("--eval-cold-start", action="store_true", help="Evaluate cold-start scenarios")
    parser.add_argument("--cold-start-users", type=int, default=COLD_START_USERS, 
                       help=f"Number of cold-start users for evaluation (default: {COLD_START_USERS})")
    parser.add_argument("--cold-start-interactions", type=int, default=COLD_START_INTERACTIONS, 
                       help=f"Number of interactions for cold-start users (default: {COLD_START_INTERACTIONS})")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model")
    parser.add_argument("--output-dir", type=str, default=MODELS_PATH,
                       help=f"Directory to save models and results (default: {MODELS_PATH})")
    
    args = parser.parse_args()
    
    # If include_all is set, enable all recommendation methods
    if args.include_all:
        args.include_user_cf = True
        args.include_item_cf = True
        args.include_feature_cf = True
        args.include_hybrid = True
        args.include_ncf = True
    
    # If no specific method is selected, show warning
    if not any([args.include_user_cf, args.include_item_cf, args.include_feature_cf,
               args.include_hybrid, args.include_ncf]):
        print("Warning: No recommendation method selected for evaluation.")
        print("         Use --include-all to evaluate all methods or select specific methods.")
        print("         Continuing with Feature-Enhanced CF as default.")
        args.include_feature_cf = True
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        success = train_model(args)
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())