"""
Main entry point untuk Web3 Recommendation Engine
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import sys
import time
from datetime import datetime
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import dan setup central logging
from central_logging import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# Import modules
from config.config import (
    PROCESSED_DATA_PATH, 
    RAW_DATA_PATH, 
    MODELS_PATH, 
    NUM_RECOMMENDATIONS,
    COLD_START_USERS,
    COLD_START_INTERACTIONS,
    NCF_QUICK_EPOCHS
)

# Lazy loading untuk modules yang berat
_cf_model = None
_feature_cf = None
_ncf_model = None

def get_models():
    """
    Inisialisasi model rekomendasi dengan lazy loading
    
    Returns:
        tuple: (cf_model, feature_cf, ncf_model)
    """
    global _cf_model, _feature_cf, _ncf_model
    
    if _cf_model is None:
        logger.info("Initializing CollaborativeFiltering model")
        from src.models.collaborative_filtering import CollaborativeFiltering
        _cf_model = CollaborativeFiltering()
    
    if _feature_cf is None:
        logger.info("Initializing FeatureEnhancedCF model")
        from src.models.feature_enhanced_cf import FeatureEnhancedCF
        _feature_cf = FeatureEnhancedCF()
    
    return _cf_model, _feature_cf, _ncf_model

def collect_data(args):
    """
    Mengumpulkan data dari CoinGecko API
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting data collection process")
    
    try:
        from src.collectors.coingecko_collector import CoinGeckoCollector
        from scripts.collect_data import collect_all_data
        
        # Extract arguments
        if hasattr(args, 'limit'):
            limit = args.limit
        else:
            limit = 500  # Default
            
        if hasattr(args, 'detail_limit'):
            detail_limit = args.detail_limit
        else:
            detail_limit = 100  # Default
            
        if hasattr(args, 'rate_limit'):
            rate_limit = args.rate_limit
        else:
            rate_limit = 2  # Default
            
        if hasattr(args, 'skip_categories'):
            skip_categories = args.skip_categories
        else:
            skip_categories = False
            
        # Set up collection args
        class CollectionArgs:
            def __init__(self):
                self.limit = limit
                self.detail_limit = detail_limit
                self.skip_ping = False
                self.skip_coins_list = False
                self.skip_categories = skip_categories
                self.skip_trending = False
                self.skip_markets = False
                self.skip_categories_markets = False
                self.skip_details = False
                self.rate_limit = rate_limit
                self.timeout = 30
                self.process = True
        
        collection_args = CollectionArgs()
        
        # Check if data directory exists
        if not os.path.exists(RAW_DATA_PATH):
            os.makedirs(RAW_DATA_PATH, exist_ok=True)
            logger.info(f"Created raw data directory: {RAW_DATA_PATH}")
            
        # Collect data
        start_time = time.time()
        success = collect_all_data(collection_args)
        
        if success:
            elapsed_time = time.time() - start_time
            logger.info(f"Data collection completed successfully in {elapsed_time:.2f} seconds")
            print(f"✅ Data collection completed successfully in {elapsed_time:.2f} seconds")
        else:
            logger.error("Data collection failed")
            print("❌ Data collection failed. Check logs for details.")
            
        return success
    
    except ImportError as e:
        logger.error(f"Import error during data collection: {e}")
        print(f"❌ Error: Missing required module - {e}")
        return False
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Error during data collection: {e}")
        return False

def process_data(args):
    """
    Memproses data yang sudah dikumpulkan
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting data processing")
    
    try:
        from src.processors.data_processor import DataProcessor
        
        # Check if raw data exists
        if not os.path.exists(RAW_DATA_PATH) or not os.listdir(RAW_DATA_PATH):
            logger.error("No raw data found. Please collect data first.")
            print("❌ No raw data found. Please collect data first with: python main.py collect")
            return False
            
        # Create processed directory if it doesn't exist
        if not os.path.exists(PROCESSED_DATA_PATH):
            os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
            
        # Process data
        start_time = time.time()
        processor = DataProcessor()
        
        # Get advanced options if specified
        users = getattr(args, 'users', 500)  # Default 500 synthetic users
        
        if hasattr(args, 'verbose') and args.verbose:
            processor.debug_mode = True
        
        # Process with progress logging
        print("Processing data...")
        projects_df, interactions_df, feature_matrix = processor.process_data()
        
        if projects_df is not None and interactions_df is not None:
            elapsed_time = time.time() - start_time
            logger.info(f"Data processing completed successfully in {elapsed_time:.2f} seconds")
            
            # Print summary
            print(f"✅ Data processing completed successfully in {elapsed_time:.2f} seconds")
            print(f"   Processed {len(projects_df)} projects")
            print(f"   Generated {len(interactions_df)} synthetic user interactions")
            print(f"   Created feature matrix with shape {feature_matrix.shape if feature_matrix is not None else '(None)'}")
            
            return True
        else:
            logger.error("Data processing failed")
            print("❌ Data processing failed. Check logs for details.")
            return False
    
    except ImportError as e:
        logger.error(f"Import error during data processing: {e}")
        print(f"❌ Error: Missing required module - {e}")
        return False
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Error during data processing: {e}")
        return False

def build_matrices(args):
    """
    Membangun matrices untuk collaborative filtering
    
    Args:
        args: Command line arguments
    """
    logger.info("Building matrices for collaborative filtering")
    
    try:
        from src.models.matrix_builder import MatrixBuilder
        
        # Check if processed data exists
        if not os.path.exists(PROCESSED_DATA_PATH):
            logger.error("No processed data found. Please process data first.")
            print("❌ No processed data found. Please process data first with: python main.py process")
            return False
            
        # Build matrices
        start_time = time.time()
        matrix_builder = MatrixBuilder()
        
        print("Building matrices...")
        matrices = matrix_builder.build_matrices()
        
        if matrices[0] is not None:
            elapsed_time = time.time() - start_time
            logger.info(f"Matrix building completed successfully in {elapsed_time:.2f} seconds")
            
            # Unpack matrices
            user_item_df, user_indices, item_indices, item_similarity_df, user_similarity_df, feature_similarity_df, combined_similarity_df = matrices
            
            # Print summary
            print(f"✅ Matrix building completed successfully in {elapsed_time:.2f} seconds")
            print(f"   User-item matrix shape: {user_item_df.shape}")
            print(f"   Item similarity matrix shape: {item_similarity_df.shape}")
            print(f"   User similarity matrix shape: {user_similarity_df.shape}")
            print(f"   Feature similarity matrix shape: {feature_similarity_df.shape if feature_similarity_df is not None else '(None)'}")
            print(f"   Combined similarity matrix shape: {combined_similarity_df.shape if combined_similarity_df is not None else '(None)'}")
            
            return True
        else:
            logger.error("Matrix building failed")
            print("❌ Matrix building failed. Check logs for details.")
            return False
    
    except ImportError as e:
        logger.error(f"Import error during matrix building: {e}")
        print(f"❌ Error: Missing required module - {e}")
        return False
    except Exception as e:
        logger.error(f"Error during matrix building: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Error during matrix building: {e}")
        return False

def train_models(args):
    """
    Melatih model rekomendasi
    
    Args:
        args: Command line arguments
    """
    logger.info("Training recommendation models")
    
    try:
        from scripts.train_model import train_model
        
        # Check if matrices exist
        matrix_files = [f for f in os.listdir(PROCESSED_DATA_PATH) if f.startswith("user_item_matrix_") and f.endswith(".csv")]
        if not matrix_files:
            logger.error("No matrices found. Please build matrices first.")
            print("❌ No matrices found. Please build matrices first with: python main.py build")
            return False
            
        # Create models directory if it doesn't exist
        models_path = MODELS_PATH
        if not os.path.exists(models_path):
            os.makedirs(models_path, exist_ok=True)
            
        # Extract options
        training_args = args

        # PENTING: Periksa apakah include_all diaktifkan, dan aktifkan semua metode jika iya
        if getattr(training_args, 'include_all', False):
            training_args.include_user_cf = True
            training_args.include_item_cf = True 
            training_args.include_feature_cf = True
            training_args.include_hybrid = True
            training_args.include_ncf = True

        # Set atribut secara eksplisit dengan nilai default jika belum ada
        if not hasattr(training_args, 'include_user_cf'):
            training_args.include_user_cf = False
        if not hasattr(training_args, 'include_item_cf'):
            training_args.include_item_cf = False
        if not hasattr(training_args, 'include_feature_cf'):
            training_args.include_feature_cf = False
        if not hasattr(training_args, 'include_hybrid'):
            training_args.include_hybrid = False
        if not hasattr(training_args, 'include_ncf'):
            training_args.include_ncf = False
            
        # Set default untuk parameter lainnya
        training_args.reprocess = getattr(args, 'reprocess', False)
        training_args.rebuild_matrices = getattr(args, 'rebuild_matrices', False)
        training_args.eval_split = getattr(args, 'eval_split', 0.2)
        training_args.random_seed = getattr(args, 'random_seed', 42)
        training_args.top_n = getattr(args, 'top_n', 10)
        training_args.eval_cold_start = getattr(args, 'eval_cold_start', False)
        training_args.min_interactions = getattr(args, 'min_interactions', 5)
        training_args.cold_start_users = getattr(args, 'cold_start_users', 50)
        training_args.cold_start_interactions = getattr(args, 'cold_start_interactions', 5)
        training_args.save_model = getattr(args, 'save_model', False)

        # Debug log
        logger.info(f"Training with include_all={getattr(training_args, 'include_all', False)}")
        logger.info(f"Training with include_user_cf={training_args.include_user_cf}")
        logger.info(f"Training with include_item_cf={training_args.include_item_cf}")
        logger.info(f"Training with include_feature_cf={training_args.include_feature_cf}")
        logger.info(f"Training with include_hybrid={training_args.include_hybrid}")
        logger.info(f"Training with include_ncf={training_args.include_ncf}")
            
        # Train models
        start_time = time.time()
        print("Training recommendation models...")
        success = train_model(training_args)
        
        if success:
            elapsed_time = time.time() - start_time
            logger.info(f"Model training completed successfully in {elapsed_time:.2f} seconds")
            print(f"✅ Model training completed successfully in {elapsed_time:.2f} seconds")
            
            # Reset models to force reloading
            global _cf_model, _feature_cf, _ncf_model
            _cf_model = None
            _feature_cf = None
            _ncf_model = None
            
            return True
        else:
            logger.error("Model training failed")
            print("❌ Model training failed. Check logs for details.")
            return False
    
    except ImportError as e:
        logger.error(f"Import error during model training: {e}")
        print(f"❌ Error: Missing required module - {e}")
        return False
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Error during model training: {e}")
        return False

def recommend(args):
    """
    Memberikan rekomendasi untuk user
    
    Args:
        args: Command line arguments
    """
    user_id = args.user_id
    num_recommendations = args.num or NUM_RECOMMENDATIONS
    category = args.category
    chain = args.chain
    rec_type = args.type or 'hybrid'
    save = getattr(args, 'save', True)
    format = getattr(args, 'format', 'json')
    
    logger.info(f"Generating recommendations for user '{user_id}'")
    logger.info(f"Parameters: type={rec_type}, num={num_recommendations}, category={category}, chain={chain}")
    
    try:
        # Get models
        cf, feature_cf, ncf = get_models()
        
        if cf is None:
            logger.error("Failed to initialize recommendation models")
            print("❌ Failed to initialize recommendation models. Check logs for details.")
            return False

        # Deteksi pengguna cold-start
        is_cold_start = False
        if user_id and cf.user_item_df is not None:
            if user_id not in cf.user_item_df.index:
                is_cold_start = True
            else:
                # Cek jumlah interaksi
                user_interactions = cf.user_item_df.loc[user_id]
                if user_interactions[user_interactions > 0].count() < 3:  # Kurang dari 3 interaksi
                    is_cold_start = True

        # Handle pengguna cold-start
        if is_cold_start and rec_type not in ['popular', 'trending']:
            logger.info(f"Detected cold-start user: {user_id}")
            print(f"Detected cold-start user: {user_id}. Using specialized recommendations strategy.")
            
            # Anda bisa menambahkan parameter untuk minat pengguna
            user_interests = getattr(args, 'interests', None)
            if user_interests and isinstance(user_interests, str):
                user_interests = user_interests.split(',')
            
            # Gunakan strategi cold-start
            recommendations = cf.get_cold_start_recommendations(
                user_id, 
                user_interests=user_interests,
                feature_cf=feature_cf,
                n=num_recommendations
            )
        
        # Initialize NCF model if requested
        if rec_type == 'ncf':
            logger.info("Initializing NCF model")
            from src.models.neural_collaborative_filtering import NCFRecommender
            print("Initializing Neural Collaborative Filtering model...")
            global _ncf_model
            _ncf_model = NCFRecommender(cf.user_item_df, cf.projects_df)
            
            # Load pre-trained model if available
            model_path = os.path.join(MODELS_PATH, "ncf_model.pkl")
            if os.path.exists(model_path):
                print(f"Loading pre-trained NCF model from: {model_path}")
                _ncf_model.load_model(model_path)
            else:
                # Try to find model in directory
                model_files = [f for f in os.listdir(MODELS_PATH) if f.startswith("ncf_model_") and f.endswith(".pkl")]
                if model_files:
                    latest_model = os.path.join(MODELS_PATH, max(model_files))
                    print(f"Loading pre-trained NCF model from: {latest_model}")
                    _ncf_model.load_model(latest_model)
                else:
                    # Gunakan konstanta NCF_QUICK_EPOCHS
                    print(f"No pre-trained NCF model found. Training a quick model ({NCF_QUICK_EPOCHS} epochs)...")
                    _ncf_model.train(val_ratio=0.1, num_epochs=NCF_QUICK_EPOCHS)
                    
            ncf = _ncf_model
            
        # Generate recommendations based on type
        print(f"Generating {rec_type} recommendations for user '{user_id}'...")
        
        if rec_type == 'user-based':
            recommendations = cf.user_based_cf(user_id, n=num_recommendations)
            recommendations = convert_recommendations_format(recommendations, cf.projects_df)
        elif rec_type == 'item-based':
            recommendations = cf.item_based_cf(user_id, n=num_recommendations)
            recommendations = convert_recommendations_format(recommendations, cf.projects_df)
        elif rec_type == 'feature-enhanced':
            recommendations = feature_cf.recommend_projects(
                user_id, 
                cf.user_item_df, 
                cf.item_similarity_df, 
                cf.projects_df, 
                n=num_recommendations
            )
        elif rec_type == 'ncf':
            if ncf is None:
                logger.error("NCF model not initialized")
                print("❌ NCF model not initialized")
                return False
            recommendations = ncf.recommend_projects(user_id, n=num_recommendations)
        elif rec_type == 'trending':
            recommendations = cf.get_trending_projects(n=num_recommendations)
        elif rec_type == 'popular':
            recommendations = cf.get_popular_projects(n=num_recommendations)
        elif category:
            recommendations = cf.get_recommendations_by_category(user_id, category, n=num_recommendations)
            rec_type = f"category_{category}"
        elif chain:
            recommendations = cf.get_recommendations_by_chain(user_id, chain, n=num_recommendations)
            rec_type = f"chain_{chain}"
        else:  # hybrid
            recommendations = cf.hybrid_recommendations(user_id, n=num_recommendations, feature_cf=feature_cf)
        
        # Save recommendations if requested
        if save:
            save_recommendations(user_id, recommendations, rec_type, format)
        
        # Print recommendations
        print(f"\nTop {min(num_recommendations, len(recommendations))} recommendations for user '{user_id}' ({rec_type}):")
        print("=" * 80)
        
        for i, rec in enumerate(recommendations[:num_recommendations], 1):
            if isinstance(rec, dict):
                name = rec.get('name', rec.get('id', 'Unknown'))
                score = rec.get('recommendation_score', 0)
                symbol = rec.get('symbol', '')
                category = rec.get('primary_category', rec.get('category', ''))
                chain = rec.get('chain', '')
                
                print(f"{i}. {name} ({symbol}) - Score: {score:.4f}")
                print(f"   Category: {category}, Chain: {chain}")
                if 'price_usd' in rec and rec['price_usd'] is not None:
                    price = rec['price_usd']
                    price_change = rec.get('price_change_24h', 0) or 0
                    print(f"   Price: ${price:.4f} ({price_change:.2f}%)")
                print()
            else:
                print(f"{i}. {rec[0]} - Score: {rec[1]:.4f}")
        
        logger.info("Recommendations generated successfully")
        return True
    
    except KeyError as e:
        logger.error(f"KeyError during recommendation: {e}")
        print(f"❌ Error: Item/user not found - {e}")
        return False
    except ImportError as e:
        logger.error(f"Import error during recommendation: {e}")
        print(f"❌ Error: Missing required module - {e}")
        return False
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Error generating recommendations: {e}")
        return False

def save_recommendations(
    user_id: str, 
    recommendations: List[Dict[str, Any]], 
    rec_type: str,
    format: str = 'json'
) -> None:
    """
    Save recommendations to file
    
    Args:
        user_id: User ID
        recommendations: List of recommendations
        rec_type: Recommendation type
        format: Output format (json or csv)
    """
    import json
    import csv
    from datetime import datetime
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recommendations_{user_id}_{rec_type}_{timestamp}"
        
        if format.lower() == 'json':
            # Prepare for JSON serialization
            json_recs = []
            for rec in recommendations:
                if isinstance(rec, dict):
                    # Convert numpy values to regular Python types
                    cleaned_rec = {}
                    for key, value in rec.items():
                        if hasattr(value, 'dtype'):  # Check if it's a numpy type
                            if np.issubdtype(value.dtype, np.integer):
                                cleaned_rec[key] = int(value)
                            elif np.issubdtype(value.dtype, np.floating):
                                cleaned_rec[key] = float(value)
                            else:
                                cleaned_rec[key] = str(value)
                        else:
                            cleaned_rec[key] = value
                    json_recs.append(cleaned_rec)
                elif isinstance(rec, tuple) and len(rec) == 2:
                    json_recs.append({
                        'id': rec[0],
                        'recommendation_score': float(rec[1])
                    })
                else:
                    json_recs.append(rec)
            
            # Save as JSON
            filepath = os.path.join(PROCESSED_DATA_PATH, f"{filename}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_recs, f, ensure_ascii=False, indent=2)
                
        elif format.lower() == 'csv':
            # Prepare for CSV serialization
            if recommendations and isinstance(recommendations[0], dict):
                # Extract fields
                fieldnames = list(recommendations[0].keys())
                rows = []
                
                for rec in recommendations:
                    if isinstance(rec, dict):
                        # Convert numpy values
                        row = {}
                        for key, value in rec.items():
                            if hasattr(value, 'dtype'):  # Check if it's a numpy type
                                if np.issubdtype(value.dtype, np.integer):
                                    row[key] = int(value)
                                elif np.issubdtype(value.dtype, np.floating):
                                    row[key] = float(value)
                                else:
                                    row[key] = str(value)
                            else:
                                row[key] = value
                        rows.append(row)
                    elif isinstance(rec, tuple) and len(rec) == 2:
                        rows.append({
                            'id': rec[0],
                            'recommendation_score': float(rec[1])
                        })
                
                # Save as CSV
                filepath = os.path.join(PROCESSED_DATA_PATH, f"{filename}.csv")
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                # Simple tuple format
                filepath = os.path.join(PROCESSED_DATA_PATH, f"{filename}.csv")
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['project_id', 'score'])
                    for rec in recommendations:
                        if isinstance(rec, tuple) and len(rec) == 2:
                            writer.writerow([rec[0], float(rec[1])])
                        elif isinstance(rec, dict):
                            writer.writerow([rec.get('id', ''), rec.get('recommendation_score', 0)])
        else:
            logger.warning(f"Unknown format: {format}")
            return
            
        print(f"Recommendations saved to: {filepath}")
        logger.info(f"Recommendations saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving recommendations: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Error saving recommendations: {e}")

def analyze_results(args):
    """
    Menganalisis hasil rekomendasi dan metrics
    
    Args:
        args: Command line arguments
    """
    logger.info("Analyzing recommendation results")
    
    try:
        # Load all recommendation files
        rec_files_json = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("recommendations_") and f.endswith(".json")
        ]
        
        rec_files_csv = [
            f for f in os.listdir(PROCESSED_DATA_PATH) 
            if f.startswith("recommendations_") and f.endswith(".csv")
        ]
        
        all_rec_files = rec_files_json + rec_files_csv
        
        if not all_rec_files:
            logger.error("No recommendation files found for analysis")
            print("❌ No recommendation files found for analysis")
            return False
        
        # Load model performance metrics
        perf_files = [
            f for f in os.listdir(MODELS_PATH) 
            if f.startswith("model_performance_") and f.endswith(".json")
        ]
        
        # Analysis parameters
        detailed = getattr(args, 'detailed', False)
        output_file = getattr(args, 'output', None)
        
        # Create report
        report = []
        report.append("# Web3 Recommendation System Analysis")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model Performance Analysis
        if perf_files:
            latest_perf_file = max(perf_files)
            perf_path = os.path.join(MODELS_PATH, latest_perf_file)
            
            with open(perf_path, 'r') as f:
                import json
                performance = json.load(f)
            
            report.append("## Model Performance")
            report.append(f"File: {latest_perf_file}")
            report.append("")
            
            # Table header
            report.append("| Model | Precision | Recall | F1 | NDCG | Hit Ratio | Users |")
            report.append("|-------|-----------|--------|----|----|-----------|-------|")
            
            # Add each model's performance
            for model, metrics in performance.items():
                if model == 'cold_start_data':
                    continue
                    
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1', 0)
                ndcg = metrics.get('ndcg', 0)
                hit_ratio = metrics.get('hit_ratio', 0)
                num_users = metrics.get('num_users', 0)
                
                report.append(
                    f"| {model} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {ndcg:.4f} | {hit_ratio:.4f} | {num_users} |"
                )
            
            report.append("")
            
            # Cold start analysis
            if 'cold_start' in performance:
                report.append("### Cold Start Performance")
                cold_start = performance['cold_start']
                
                precision = cold_start.get('precision', 0)
                recall = cold_start.get('recall', 0)
                f1 = cold_start.get('f1', 0)
                
                report.append(f"- Precision: {precision:.4f}")
                report.append(f"- Recall: {recall:.4f}")
                report.append(f"- F1 Score: {f1:.4f}")
                report.append(f"- Number of users: {cold_start.get('num_users', 0)}")
                report.append("")
        
        # Recommendations Analysis
        report.append("## Recommendation Analysis")
        
        # Process recommendation files
        all_recs = []
        rec_types = set()
        user_counts = {}
        project_counts = {}
        
        for file in all_rec_files[:10]:  # Limit to 10 most recent files to avoid memory issues
            filepath = os.path.join(PROCESSED_DATA_PATH, file)
            
            # Extract metadata from filename
            parts = file.replace('.json', '').replace('.csv', '').split('_')
            if len(parts) >= 3:
                user_id = parts[1]
                rec_type = parts[2]
                
                rec_types.add(rec_type)
                
                if user_id not in user_counts:
                    user_counts[user_id] = 0
                user_counts[user_id] += 1
            
            # Load file
            if file.endswith('.json'):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        try:
                            recs = json.load(f)
                            
                            # Count projects
                            for rec in recs:
                                if isinstance(rec, dict) and 'id' in rec:
                                    project_id = rec['id']
                                    if project_id not in project_counts:
                                        project_counts[project_id] = 0
                                    project_counts[project_id] += 1
                            
                            all_recs.append(recs)
                        except json.JSONDecodeError as json_err:
                            logger.warning(f"JSON parsing error in {filepath}: {json_err}")
                except Exception as e:
                    logger.warning(f"Error loading JSON file {filepath}: {str(e)}")
            elif file.endswith('.csv'):
                try:
                    recs_df = pd.read_csv(filepath)
                    
                    # Count projects
                    for _, row in recs_df.iterrows():
                        if 'project_id' in row:
                            project_id = row['project_id']
                        elif 'id' in row:
                            project_id = row['id']
                        else:
                            continue
                            
                        if project_id not in project_counts:
                            project_counts[project_id] = 0
                        project_counts[project_id] += 1
                    
                    all_recs.append(recs_df.to_dict('records'))
                except Exception as e:
                    logger.warning(f"Error loading CSV file {filepath}: {str(e)}")
        
        # Generate report
        report.append(f"Total recommendation files analyzed: {len(all_rec_files)}")
        report.append(f"Recommendation types: {', '.join(sorted(rec_types))}")
        report.append(f"Unique users: {len(user_counts)}")
        report.append("")
        
        # Top users
        report.append("### Top Users by Recommendation Count")
        top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for user_id, count in top_users:
            report.append(f"- User {user_id}: {count} recommendation sets")
        
        report.append("")
        
        # Top recommended projects
        report.append("### Most Frequently Recommended Projects")
        top_projects = sorted(project_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Get project details if available
        project_details = {}
        try:
            cf, _, _ = get_models()
            if cf and cf.projects_df is not None:
                for project_id, _ in top_projects:
                    project_row = cf.projects_df[cf.projects_df['id'] == project_id]
                    if not project_row.empty:
                        name = project_row.iloc[0].get('name', project_id)
                        symbol = project_row.iloc[0].get('symbol', '')
                        category = project_row.iloc[0].get('primary_category', '')
                        project_details[project_id] = {
                            'name': name,
                            'symbol': symbol,
                            'category': category
                        }
        except Exception as e:
            logger.warning(f"Could not load project details: {str(e)}")
        
        for project_id, count in top_projects:
            if project_id in project_details:
                details = project_details[project_id]
                report.append(f"- {details['name']} ({details['symbol']}) [{details['category']}]: {count} recommendations")
            else:
                report.append(f"- {project_id}: {count} recommendations")
        
        # Add detailed analysis if requested
        if detailed:
            report.append("")
            report.append("## Detailed Category Analysis")
            
            # Category distribution
            category_counts = {}
            for project_id, count in project_counts.items():
                if project_id in project_details:
                    category = project_details[project_id].get('category', 'unknown')
                    if category not in category_counts:
                        category_counts[category] = 0
                    category_counts[category] += count
            
            # Sort categories by count
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            
            report.append("### Category Distribution in Recommendations")
            for category, count in sorted_categories:
                report.append(f"- {category}: {count} recommendations")
            
            report.append("")
            report.append("## Recommendation Type Analysis")
            
            # Analysis by recommendation type
            for rec_type in sorted(rec_types):
                type_files = [f for f in all_rec_files if f"_{rec_type}_" in f]
                report.append(f"### {rec_type.title()} Recommendations")
                report.append(f"- Number of files: {len(type_files)}")
                report.append("")
        
        # Print report
        print("\n".join(report))
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write("\n".join(report))
            print(f"Analysis report saved to: {output_file}")
        
        logger.info("Analysis completed")
        return True
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"❌ Error during analysis: {str(e)}")
        return False

def run_pipeline(args):
    """
    Menjalankan seluruh pipeline recommendation engine
    
    Args:
        args: Command line arguments
    """
    logger.info("Running complete recommendation engine pipeline")
    print("Starting complete recommendation engine pipeline...")
    
    # Steps with progress tracking
    steps = [
        ("Data Collection", collect_data),
        ("Data Processing", process_data),
        ("Building Matrices", build_matrices),
        ("Training Models", train_models)
    ]
    
    step_results = []
    for step_name, step_func in steps:
        print(f"\n{'-'*80}")
        print(f"Step: {step_name}")
        print(f"{'-'*80}")
        
        success = step_func(args)
        step_results.append((step_name, success))
        
        if not success:
            print(f"\n❌ Pipeline failed at step: {step_name}")
            print("Continuing with next steps anyway...")
    
    # Generate recommendations for sample users if requested
    if not args.skip_recommendations:
        print(f"\n{'-'*80}")
        print("Generating Sample Recommendations")
        print(f"{'-'*80}")
        
        # Use models that were loaded or create new ones
        cf, feature_cf, _ = get_models()
        
        if cf and cf.user_item_df is not None and not cf.user_item_df.empty:
            # Get sample users (first 5 or less)
            sample_count = min(5, len(cf.user_item_df.index))
            sample_users = cf.user_item_df.index[:sample_count].tolist()
            
            print(f"Generating recommendations for {sample_count} sample users...")
            
            for i, user in enumerate(sample_users, 1):
                print(f"\nSample User {i}: {user}")
                
                # Create args for recommend function
                class RecArgs:
                    def __init__(self, user_id):
                        self.user_id = user_id
                        self.num = 5  # Show only 5 recommendations per user
                        self.type = 'hybrid'
                        self.category = None
                        self.chain = None
                        self.save = False  # Don't save these sample recommendations
                
                rec_args = RecArgs(user)
                recommend(rec_args)
                
        else:
            print("❌ Could not generate sample recommendations: models or data not available")
    
    # Run analysis if requested
    if not args.skip_analysis:
        print(f"\n{'-'*80}")
        print("Analyzing Results")
        print(f"{'-'*80}")
        
        analyze_results(args)
    
    # Print summary
    print(f"\n{'-'*80}")
    print("Pipeline Summary")
    print(f"{'-'*80}")
    
    all_success = all([success for _, success in step_results])
    
    for step_name, success in step_results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{step_name}: {status}")
    
    if all_success:
        print("\n🎉 Complete pipeline executed successfully!")
    else:
        print("\n⚠️ Pipeline completed with some errors. Check logs for details.")
    
    logger.info("Pipeline execution completed")
    return all_success

def run_interactive_mode():
    """
    Run in interactive mode with menu
    """
    print("\nWeb3 Recommendation Engine Interactive Mode")
    print("=" * 40)
    
    while True:
        print("\nSelect an option:")
        print("1. Collect Data from CoinGecko")
        print("2. Process Data")
        print("3. Build Matrices")
        print("4. Train Models")
        print("5. Generate Recommendations")
        print("6. Analyze Results")
        print("7. Run Complete Pipeline")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-7): ")
        
        if choice == '0':
            print("Exiting...")
            break
            
        elif choice == '1':
            limit = input("Enter coin limit (default: 500): ") or 500
            detail_limit = input("Enter detail limit (default: 100): ") or 100
            
            class Args:
                pass
                
            args = Args()
            args.limit = int(limit)
            args.detail_limit = int(detail_limit)
            args.rate_limit = 2
            
            collect_data(args)
            
        elif choice == '2':
            class Args:
                pass
                
            args = Args()
            args.users = int(input("Enter number of synthetic users (default: 500): ") or 500)
            
            process_data(args)
            
        elif choice == '3':
            class Args:
                pass
                
            args = Args()
            
            build_matrices(args)
            
        elif choice == '4':
            class Args:
                pass
            args = Args()
            args.include_all = True
            args.eval_cold_start = True
            args.save_model = True
            args.cold_start_users = COLD_START_USERS
            args.cold_start_interactions = COLD_START_INTERACTIONS
            args.min_interactions = 5
            
            # Tambahkan argumen yang hilang
            args.reprocess = False
            args.rebuild_matrices = False
            args.eval_split = 0.2
            args.random_seed = 42
            args.top_n = 10
            args.include_user_cf = True
            args.include_item_cf = True
            args.include_feature_cf = True
            args.include_hybrid = True
            args.include_ncf = True
            
            train_models(args)
            
        elif choice == '5':
            user_id = input("Enter user ID: ")
            rec_type = input("Enter recommendation type (hybrid, user-based, item-based, feature-enhanced, ncf, popular, trending): ") or 'hybrid'
            num = input("Enter number of recommendations (default: 10): ") or 10
            
            class Args:
                pass
                
            args = Args()
            args.user_id = user_id
            args.type = rec_type
            args.num = int(num)
            args.category = None
            args.chain = None
            
            # Ask for category or chain if needed
            if rec_type == 'category':
                args.category = input("Enter category: ")
                args.type = None
            elif rec_type == 'chain':
                args.chain = input("Enter blockchain: ")
                args.type = None
            
            recommend(args)
            
        elif choice == '6':
            class Args:
                pass
                
            args = Args()
            args.detailed = input("Show detailed analysis? (y/n, default: n): ").lower() == 'y'
            
            analyze_results(args)
            
        elif choice == '7':
            class Args:
                pass
                
            args = Args()
            args.skip_recommendations = False
            args.skip_analysis = False
            
            run_pipeline(args)
            
        else:
            print("Invalid choice. Please try again.")

def convert_recommendations_format(
    recommendations: List[Tuple[str, float]], 
    projects_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Convert recommendations from (id, score) tuples to dictionaries with project details
    
    Args:
        recommendations: List of recommendation tuples
        projects_df: DataFrame with project details
        
    Returns:
        list: List of recommendation dictionaries
    """
    result = []
    
    for project_id, score in recommendations:
        # Find project in DataFrame
        project_data = projects_df[projects_df['id'] == project_id]
        
        if not project_data.empty:
            # Create dictionary with project details
            project_info = project_data.iloc[0].to_dict()
            
            # Convert numpy types to Python types for JSON serialization
            for key, value in project_info.items():
                if hasattr(value, 'dtype'):
                    if np.issubdtype(value.dtype, np.integer):
                        project_info[key] = int(value)
                    elif np.issubdtype(value.dtype, np.floating):
                        project_info[key] = float(value)
            
            # Add recommendation score
            project_info['recommendation_score'] = float(score)
            
            result.append(project_info)
        else:
            # Fallback if project not found in DataFrame
            result.append({
                'id': project_id,
                'recommendation_score': float(score)
            })
    
    return result

def main():
    """
    Main function to parse command line arguments and run appropriate function
    """
    parser = argparse.ArgumentParser(
        description="Web3 Project Recommendation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data from CoinGecko
  python main.py collect --limit 500 --detail-limit 100
  
  # Process collected data
  python main.py process
  
  # Build matrices for collaborative filtering
  python main.py build
  
  # Train recommendation models
  python main.py train --include-all --save-model
  
  # Generate recommendations for a user
  python main.py recommend --user_id user_1 --type hybrid --num 10
  
  # Analyze recommendation results
  python main.py analyze --detailed
  
  # Run complete pipeline
  python main.py run
  
  # Run in interactive mode
  python main.py interactive
"""
    )
    
    subparsers = parser.add_subparsers(help="Commands", dest="command")
    
    # collect command
    collect_parser = subparsers.add_parser("collect", help="Collect data from CoinGecko API")
    collect_parser.add_argument("--limit", type=int, default=500, help="Number of top coins to collect (default: 500)")
    collect_parser.add_argument("--detail-limit", type=int, default=100, help="Number of coins to get detailed data for (default: 100)")
    collect_parser.add_argument("--rate-limit", type=float, default=2, help="Delay between API requests (default: 2)")
    collect_parser.add_argument("--skip-categories", action="store_true", help="Skip collecting category data")
    
    # process command
    process_parser = subparsers.add_parser("process", help="Process collected data")
    process_parser.add_argument("--users", type=int, default=500, help="Number of synthetic users to generate (default: 500)")
    process_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # build command
    subparsers.add_parser("build", help="Build matrices for collaborative filtering")
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train recommendation models")
    train_parser.add_argument("--include-all", action="store_true", help="Include all recommendation methods")
    train_parser.add_argument("--include-user-cf", action="store_true", help="Include User-Based CF")
    train_parser.add_argument("--include-item-cf", action="store_true", help="Include Item-Based CF")
    train_parser.add_argument("--include-feature-cf", action="store_true", help="Include Feature-Enhanced CF")
    train_parser.add_argument("--include-hybrid", action="store_true", help="Include Hybrid recommendations")
    train_parser.add_argument("--include-ncf", action="store_true", help="Include Neural CF")
    train_parser.add_argument("--eval-cold-start", action="store_true", help="Evaluate cold-start scenarios")
    train_parser.add_argument("--save-model", action="store_true", help="Save the trained model")
    train_parser.add_argument("--reprocess", action="store_true", help="Reprocess data before training")
    train_parser.add_argument("--rebuild-matrices", action="store_true", help="Rebuild matrices before training")
    train_parser.add_argument("--eval-split", type=float, default=0.2, help="Evaluation split ratio")
    train_parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    train_parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations for evaluation")
    train_parser.add_argument("--min-interactions", type=int, default=5, help="Minimum interactions for cold start")
    
    # recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Generate recommendations")
    recommend_parser.add_argument("--user_id", required=True, help="User ID to recommend for")
    recommend_parser.add_argument("--type", choices=[
        'hybrid', 'user-based', 'item-based', 'feature-enhanced', 'ncf', 'trending', 'popular'
    ], help="Recommendation type")
    recommend_parser.add_argument("--num", type=int, help="Number of recommendations")
    recommend_parser.add_argument("--category", help="Filter by category")
    recommend_parser.add_argument("--chain", help="Filter by blockchain")
    recommend_parser.add_argument("--save", action="store_true", help="Save recommendations to file")
    recommend_parser.add_argument("--format", choices=['json', 'csv'], default='json', help="Output format")
    recommend_parser.add_argument("--interests", help="Comma-separated list of user interests (categories)")
    
    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze recommendation results")
    analyze_parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    analyze_parser.add_argument("--output", help="Save analysis to file")
    
    # run command (full pipeline)
    run_parser = subparsers.add_parser("run", help="Run the complete pipeline")
    run_parser.add_argument("--skip-recommendations", action="store_true", help="Skip generating sample recommendations")
    run_parser.add_argument("--skip-analysis", action="store_true", help="Skip analyzing results")
    
    # interactive command
    subparsers.add_parser("interactive", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Call appropriate function based on command
    try:
        if args.command == "collect":
            collect_data(args)
        elif args.command == "process":
            process_data(args)
        elif args.command == "build":
            build_matrices(args)
        elif args.command == "train":
            train_models(args)
        elif args.command == "recommend":
            recommend(args)
        elif args.command == "analyze":
            analyze_results(args)
        elif args.command == "run":
            run_pipeline(args)
        elif args.command == "interactive":
            run_interactive_mode()
        else:
            parser.print_help()
    except Exception as e:
        logger.exception(f"Unhandled error in main: {e}")
        print(f"❌ An error occurred: {e}")
        print("Check logs for more details.")
        return 1
        
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error in main: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)