"""
Script untuk menghasilkan rekomendasi proyek Web3
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
from typing import Dict, List, Any, Optional, Tuple, Union

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
from config.config import (
    PROCESSED_DATA_PATH, 
    MODELS_PATH,
    NUM_RECOMMENDATIONS,
    NCF_QUICK_EPOCHS
)

def load_model(model_path: Optional[str] = None) -> Tuple[CollaborativeFiltering, FeatureEnhancedCF]:
    """
    Load saved model or create a new one
    
    Args:
        model_path (str, optional): Path to saved model
        
    Returns:
        tuple: (CollaborativeFiltering, FeatureEnhancedCF) instances
    """
    if model_path and os.path.exists(model_path):
        # Load saved model
        logger.info(f"Loading model from {model_path}")
        try:
            with open(model_path, 'rb') as f:
                # We load the model data but directly use it to initialize our models
                similarity_matrices = pickle.load(f)
                
            # Create model objects and use the matrices
            cf = CollaborativeFiltering()
            feature_cf = FeatureEnhancedCF()
            
            # If matrices are available in the loaded data, set them
            if 'item_similarity_matrix' in similarity_matrices and similarity_matrices['item_similarity_matrix']:
                cf.item_similarity_df = pd.DataFrame.from_dict(similarity_matrices['item_similarity_matrix'])
                
            if 'user_similarity_matrix' in similarity_matrices and similarity_matrices['user_similarity_matrix']:
                cf.user_similarity_df = pd.DataFrame.from_dict(similarity_matrices['user_similarity_matrix'])
                
            if 'combined_similarity_matrix' in similarity_matrices and similarity_matrices['combined_similarity_matrix']:
                cf.combined_similarity_df = pd.DataFrame.from_dict(similarity_matrices['combined_similarity_matrix'])
            
            logger.info("Model loaded successfully")
            return cf, feature_cf
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Creating new model instances as fallback")
            return CollaborativeFiltering(), FeatureEnhancedCF()
    else:
        # Create new models
        logger.info("Creating new model instances")
        return CollaborativeFiltering(), FeatureEnhancedCF()

def generate_recommendations(args) -> bool:
    """
    Generate recommendations based on the specified method
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if successful, False if failed
    """
    logger.info("Starting recommendations generation")
    
    # Load models
    cf, feature_cf = load_model(args.model_path)
    
    # Check if user ID is provided
    if args.user_id is None and not args.popular and not args.trending:
        logger.error("User ID is required for user-based recommendations")
        return False
    
    # Generate recommendations based on the specified method
    recommendations = []
    
    try:
        if args.popular:
            # Generate popular recommendations
            logger.info("Generating popular recommendations")
            popular_recs = cf.get_popular_projects(n=args.num)
            recommendations = popular_recs
            method = "popular"
            
        elif args.trending:
            # Generate trending recommendations
            logger.info("Generating trending recommendations")
            trending_recs = cf.get_trending_projects(n=args.num)
            recommendations = trending_recs
            method = "trending"
            
        elif args.method == "user-based":
            # Generate user-based recommendations
            logger.info(f"Generating user-based recommendations for user {args.user_id}")
            user_recs = cf.user_based_cf(args.user_id, n=args.num)
            
            # Convert to list of dicts with more details
            if cf.projects_df is not None:
                detailed_recs = []
                for project_id, score in user_recs:
                    project_data = cf.projects_df[cf.projects_df['id'] == project_id]
                    if not project_data.empty:
                        project_info = project_data.iloc[0].to_dict()
                        project_info['recommendation_score'] = score
                        detailed_recs.append(project_info)
                recommendations = detailed_recs
            else:
                recommendations = [{"id": item, "score": score} for item, score in user_recs]
                
            method = "user-based"
            
        elif args.method == "item-based":
            # Generate item-based recommendations
            logger.info(f"Generating item-based recommendations for user {args.user_id}")
            item_recs = cf.item_based_cf(args.user_id, n=args.num)
            
            # Convert to list of dicts with more details
            if cf.projects_df is not None:
                detailed_recs = []
                for project_id, score in item_recs:
                    project_data = cf.projects_df[cf.projects_df['id'] == project_id]
                    if not project_data.empty:
                        project_info = project_data.iloc[0].to_dict()
                        project_info['recommendation_score'] = score
                        detailed_recs.append(project_info)
                recommendations = detailed_recs
            else:
                recommendations = [{"id": item, "score": score} for item, score in item_recs]
                
            method = "item-based"
            
        elif args.method == "feature-enhanced":
            # Generate feature-enhanced recommendations
            logger.info(f"Generating feature-enhanced recommendations for user {args.user_id}")
            
            # We need to get matrices from the CF object since we're using shared data
            if cf.user_item_df is not None and cf.item_similarity_df is not None and cf.projects_df is not None:
                feature_recs = feature_cf.recommend_projects(
                    args.user_id, 
                    cf.user_item_df, 
                    cf.item_similarity_df, 
                    cf.projects_df, 
                    n=args.num
                )
                recommendations = feature_recs
            else:
                logger.error("Required matrices not available for feature-enhanced recommendations")
                return False
                
            method = "feature-enhanced"

        elif args.method == 'ncf':
            # Generate NCF recommendations
            logger.info(f"Generating NCF recommendations for user {args.user_id}")
            
            # Initialize DataProcessor first
            processor = DataProcessor()
            projects_df, interactions_df, _ = processor.load_latest_processed_data()
            
            if interactions_df is None or projects_df is None:
                logger.error("Required data not available for NCF recommendations")
                return False
                
            # Load data
            matrix_builder = MatrixBuilder()
            user_item_df, _, _ = matrix_builder.build_user_item_matrix(interactions_df)
            
            # Create NCF instance
            ncf = NCFRecommender(user_item_df, projects_df)
            
            # Load trained model if available
            model_loaded = False
            if args.model_path and os.path.exists(args.model_path) and args.model_path.endswith('.pkl'):
                try:
                    ncf.load_model(args.model_path)
                    model_loaded = True
                    logger.info(f"Successfully loaded NCF model from {args.model_path}")
                except Exception as e:
                    logger.error(f"Error loading NCF model: {e}")
                    model_loaded = False
                    
            # Quick training if no model loaded
            if not model_loaded:
                logger.info("No valid NCF model provided, performing quick training")
                ncf.train(val_ratio=0.1, num_epochs=NCF_QUICK_EPOCHS)
                
            # Generate recommendations
            recommendations = ncf.recommend_projects(args.user_id, n=args.num)
            method = "ncf"
            
        elif args.method == "hybrid":
            # Generate hybrid recommendations
            logger.info(f"Generating hybrid recommendations for user {args.user_id}")
            hybrid_recs = cf.hybrid_recommendations(args.user_id, n=args.num, feature_cf=feature_cf)
            recommendations = hybrid_recs
            method = "hybrid"
            
        elif args.method == "category":
            # Generate category-based recommendations
            if args.category is None:
                logger.error("Category is required for category-based recommendations")
                return False
                
            logger.info(f"Generating {args.category} category recommendations for user {args.user_id}")
            category_recs = cf.get_recommendations_by_category(args.user_id, args.category, n=args.num)
            recommendations = category_recs
            method = f"category_{args.category}"
            
        elif args.method == "chain":
            # Generate chain-based recommendations
            if args.chain is None:
                logger.error("Chain is required for chain-based recommendations")
                return False
                
            logger.info(f"Generating {args.chain} chain recommendations for user {args.user_id}")
            chain_recs = cf.get_recommendations_by_chain(args.user_id, args.chain, n=args.num)
            recommendations = chain_recs
            method = f"chain_{args.chain}"
        
        else:
            logger.error(f"Unknown recommendation method: {args.method}")
            return False
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Check if recommendations were generated
    if not recommendations:
        logger.warning("No recommendations generated")
        print("No recommendations generated")
        return False
    
    # Print recommendations
    print(f"\nTop {min(args.num, len(recommendations))} recommendations:")
    print(f"Method: {method}")
    
    for i, rec in enumerate(recommendations[:args.num], 1):
        if isinstance(rec, dict):
            name = rec.get('name', rec.get('id', 'Unknown'))
            symbol = rec.get('symbol', '')
            category = rec.get('primary_category', rec.get('category', ''))
            chain = rec.get('chain', '')
            
            if 'recommendation_score' in rec:
                score = rec['recommendation_score']
                print(f"{i}. {name} ({symbol}) - Score: {score:.4f}")
            elif 'similarity_score' in rec:
                score = rec['similarity_score']
                print(f"{i}. {name} ({symbol}) - Similarity: {score:.4f}")
            else:
                print(f"{i}. {name} ({symbol})")
                
            if category or chain:
                print(f"   Category: {category}, Chain: {chain}")
        else:
            print(f"{i}. {rec[0]} - Score: {rec[1]:.4f}")
    
    # Save recommendations if requested
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.user_id:
            filename = f"recommendations_{args.user_id}_{method}_{timestamp}"
        else:
            filename = f"recommendations_{method}_{timestamp}"
        
        if args.format == 'json':
            filepath = os.path.join(PROCESSED_DATA_PATH, f"{filename}.json")
            
            # Convert recommendations to serializable format
            serializable_recs = []
            for rec in recommendations:
                if isinstance(rec, dict):
                    # Convert numpy values to Python native types
                    serializable_rec = {}
                    for key, value in rec.items():
                        if isinstance(value, (np.int64, np.int32)):
                            serializable_rec[key] = int(value)
                        elif isinstance(value, (np.float64, np.float32)):
                            serializable_rec[key] = float(value)
                        else:
                            serializable_rec[key] = value
                    serializable_recs.append(serializable_rec)
                else:
                    serializable_recs.append({
                        "id": rec[0],
                        "score": float(rec[1])
                    })
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(serializable_recs, f, ensure_ascii=False, indent=2)
                logger.info(f"Recommendations saved to {filepath}")
            except Exception as e:
                logger.error(f"Error saving JSON recommendations: {e}")
                
        elif args.format == 'csv':
            filepath = os.path.join(PROCESSED_DATA_PATH, f"{filename}.csv")
            
            try:
                # Convert recommendations to DataFrame
                if recommendations and isinstance(recommendations[0], dict):
                    recs_df = pd.DataFrame(recommendations)
                else:
                    recs_df = pd.DataFrame(recommendations, columns=['project_id', 'score'])
                
                # Add method and timestamp
                recs_df['method'] = method
                recs_df['timestamp'] = timestamp
                if args.user_id:
                    recs_df['user_id'] = args.user_id
                
                recs_df.to_csv(filepath, index=False)
                logger.info(f"Recommendations saved to {filepath}")
            except Exception as e:
                logger.error(f"Error saving CSV recommendations: {e}")
        
        print(f"\nRecommendations saved to {filepath}")
    
    logger.info("Recommendations generation completed successfully")
    return True

def main() -> int:
    """
    Main function
    
    Returns:
        int: Exit code (0: Success, 1: Failure)
    """
    parser = argparse.ArgumentParser(description="Generate Web3 project recommendations")
    
    # Add arguments
    parser.add_argument("--user-id", type=str, help="User ID for recommendations")
    parser.add_argument("--method", type=str, choices=[
        "user-based", "item-based", "feature-enhanced", "hybrid", "category", "chain", "ncf"
    ], default="hybrid", help="Recommendation method (default: hybrid)")
    parser.add_argument("--num", type=int, default=NUM_RECOMMENDATIONS, 
                       help=f"Number of recommendations to generate (default: {NUM_RECOMMENDATIONS})")
    parser.add_argument("--model-path", type=str, help="Path to saved model")
    parser.add_argument("--popular", action="store_true", help="Generate popular recommendations")
    parser.add_argument("--trending", action="store_true", help="Generate trending recommendations")
    parser.add_argument("--category", type=str, help="Category for category-based recommendations")
    parser.add_argument("--chain", type=str, help="Blockchain for chain-based recommendations")
    parser.add_argument("--save", action="store_true", help="Save recommendations to file")
    parser.add_argument("--format", type=str, choices=["json", "csv"], default="json", help="Output format (default: json)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.popular and not args.trending and not args.user_id:
        print("Error: User ID is required for user-based recommendations")
        print("       Use --popular or --trending for non-user-specific recommendations")
        return 1
    
    # Check if model path exists
    if args.model_path and not os.path.exists(args.model_path):
        print(f"Warning: Model path {args.model_path} does not exist.")
        logger.warning(f"Model path {args.model_path} does not exist.")
        
        # Check in MODELS_PATH as fallback
        model_name = os.path.basename(args.model_path)
        potential_path = os.path.join(MODELS_PATH, model_name)
        
        if os.path.exists(potential_path):
            args.model_path = potential_path
            print(f"Found model at {potential_path} instead.")
            logger.info(f"Found model at {potential_path} instead.")
    
    try:
        success = generate_recommendations(args)
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.warning("Recommendations generation interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Error generating recommendations: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())