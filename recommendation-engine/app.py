"""
Main API service untuk recommendation engine
"""
from flask import Flask, request, jsonify, make_response
import psycopg2
import psycopg2.extras
import json
import logging
import sys
import os
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import modules
from config.config import (
    PROCESSED_DATA_PATH, 
    DB_HOST, 
    DB_PORT, 
    DB_NAME, 
    DB_USER, 
    DB_PASSWORD,
    API_SECRET_KEY, 
    CACHE_ENABLED, 
    CACHE_TTL, 
    RECOMMENDATION_CACHE_TTL,
    NCF_QUICK_EPOCHS
)
from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.feature_enhanced_cf import FeatureEnhancedCF
from src.models.neural_collaborative_filtering import NCFRecommender
from src.models.technical_analysis import generate_trading_signals, personalize_signals
from src.utils.data_utils import DateTimeEncoder

# Import dan setup central logging untuk API service
from central_logging import setup_logging, get_logger
setup_logging(service_name="api_service")  # Membuat file log terpisah untuk API
logger = get_logger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = API_SECRET_KEY

# Simple in-memory cache
cache = {}

# Inisialisasi model rekomendasi secara lazy loading
cf_model = None
feature_cf = None
ncf_model = None

def get_models():
    """
    Lazy loading untuk model rekomendasi
    
    Returns:
        tuple: (cf_model, feature_cf, ncf_model)
    """
    global cf_model, feature_cf, ncf_model
    
    if cf_model is None:
        logger.info("Initializing CollaborativeFiltering model")
        cf_model = CollaborativeFiltering()
    
    if feature_cf is None:
        logger.info("Initializing FeatureEnhancedCF model")
        feature_cf = FeatureEnhancedCF()
    
    return cf_model, feature_cf, ncf_model

def handle_cold_start_user(user_id: str, user_interests: Optional[List[str]] = None, n: int = 10) -> List[Dict[str, Any]]:
    """
    Handle cold-start user dengan strategi khusus
    
    Args:
        user_id: ID pengguna
        user_interests: Daftar kategori yang diminati (optional)
        n: Jumlah rekomendasi
        
    Returns:
        list: Daftar rekomendasi
    """
    logger.info(f"Handling cold-start user {user_id}")
    
    # Get models
    cf, feature_cf, _ = get_models()
    
    # Gunakan strategi cold-start khusus
    return cf.get_cold_start_recommendations(user_id, user_interests, feature_cf, n)

# API key authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != API_SECRET_KEY:
            logger.warning(f"Unauthorized access attempt with API key: {api_key}")
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# Performance monitoring decorator
def monitor_performance(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log performance for endpoints taking more than 0.5s
        if execution_time > 0.5:
            logger.info(f"Performance: {request.path} took {execution_time:.2f}s")
        
        return result
    return decorated

# Database connection with context manager
class DatabaseConnection:
    def __init__(self):
        self.conn = None
        self.cursor = None
    
    def __enter__(self):
        self.conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        self.conn.autocommit = True
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        return self.cursor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

# Cache management functions
def get_cache_key(prefix: str, **kwargs) -> str:
    """Generate a cache key from arguments"""
    key_parts = [prefix]
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    return ":".join(key_parts)

def get_from_cache(key: str) -> Optional[Dict[str, Any]]:
    """Get item from cache if valid"""
    if not CACHE_ENABLED:
        return None
        
    if key in cache:
        item = cache[key]
        if datetime.now() < item['expires']:
            logger.debug(f"Cache hit for key: {key}")
            return item['data']
        else:
            # Expired, remove from cache
            del cache[key]
            
    logger.debug(f"Cache miss for key: {key}")
    return None

def set_in_cache(key: str, data: Any, ttl: int) -> None:
    """Store item in cache with expiration"""
    if not CACHE_ENABLED:
        return
        
    expires = datetime.now() + timedelta(seconds=ttl)
    cache[key] = {
        'data': data,
        'expires': expires
    }
    logger.debug(f"Stored in cache with key: {key}, expires: {expires}")

# API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Check database connection
    try:
        with DatabaseConnection() as cursor:
            cursor.execute("SELECT 1")
            db_status = "up"
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        db_status = "down"
    
    # Check if models can be loaded
    try:
        cf, feature, ncf = get_models()
        models_status = "up" if cf and feature else "partial"
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        models_status = "down"
    
    status = "healthy" if db_status == "up" and models_status == "up" else "degraded"
    
    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": db_status,
            "models": models_status
        }
    })

@app.route('/api/recommendations', methods=['GET'])
@monitor_performance
def get_recommendations():
    """
    Get recommendations for a user
    
    Query Parameters:
        user_id (str): User ID
        type (str): Recommendation type (hybrid, user-based, item-based, feature-enhanced, ncf, popular, trending)
        limit (int): Number of recommendations to return
        category (str, optional): Filter by category
        chain (str, optional): Filter by blockchain
    """
    try:
        # Get parameters and validate
        user_id = request.args.get('user_id', '')
        rec_type = request.args.get('type', 'hybrid')
        limit = min(int(request.args.get('limit', 10)), 100)  # Limit to max 100
        category = request.args.get('category')
        chain = request.args.get('chain')
        
        # Validate user_id
        if not user_id and rec_type not in ['popular', 'trending']:
            return jsonify({"error": "User ID is required"}), 400
        
        # Cek apakah user adalah cold-start user
        is_cold_start = False
        if user_id:
            # Load models untuk mengakses user_item_df
            cf, feature_cf, _ = get_models()
            
            # Cek dari user-item matrix
            if cf.user_item_df is not None and user_id not in cf.user_item_df.index:
                is_cold_start = True
            else:
                # Cek dari tabel interaksi di database
                with DatabaseConnection() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM interactions WHERE user_id = %s", (user_id,))
                    count = cursor.fetchone()[0]
                    is_cold_start = count < 3  # User dengan <3 interaksi dianggap cold-start

        # Tangani cold-start user dengan strategi khusus
        if is_cold_start:
            # Extract user interests dari query parameter jika ada
            user_interests = request.args.get('interests', '').split(',') if request.args.get('interests') else None
            recommendations = handle_cold_start_user(user_id, user_interests, limit)
            
            # Skip cache dan langsung kembalikan hasil
            return jsonify(recommendations)
        
        # Check cache first
        cache_key = get_cache_key(
            "recommendations", 
            user_id=user_id, 
            type=rec_type, 
            limit=limit,
            category=category,
            chain=chain
        )
        
        cached_result = get_from_cache(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Get recommendations from DB if available
        recommendations = get_recommendations_from_db(user_id, rec_type, category, chain)
        
        # If not available or old, generate new recommendations
        if not recommendations:
            # Load models
            cf, feature, ncf = get_models()
            
            # Verify models are loaded
            if cf is None:
                return jsonify({"error": "Recommendation engine is not initialized"}), 500
            
            # Generate recommendations based on type
            if rec_type == 'popular':
                recommendations = cf.get_popular_projects(n=limit)
            elif rec_type == 'trending':
                recommendations = cf.get_trending_projects(n=limit)
            elif rec_type == 'user-based':
                recommendations = cf.user_based_cf(user_id, n=limit)
                recommendations = convert_recommendations_format(recommendations, cf.projects_df)
            elif rec_type == 'item-based':
                recommendations = cf.item_based_cf(user_id, n=limit)
                recommendations = convert_recommendations_format(recommendations, cf.projects_df)
            elif rec_type == 'feature-enhanced':
                recommendations = feature.recommend_projects(
                    user_id, 
                    cf.user_item_df, 
                    cf.item_similarity_df, 
                    cf.projects_df, 
                    n=limit
                )
            elif rec_type == 'ncf':
                # Lazy loading NCF model
                global ncf_model
                if ncf_model is None:
                    logger.info("Initializing NCF model")
                    ncf_model = NCFRecommender(cf.user_item_df, cf.projects_df)
                    
                    # Load pre-trained model if available
                    model_path = os.path.join(PROCESSED_DATA_PATH, "ncf_model.pkl")
                    if os.path.exists(model_path):
                        ncf_model.load_model(model_path)
                        logger.info(f"Loaded NCF model from {model_path}")
                    else:
                        # Quick training menggunakan konstanta standar
                        logger.info("Pre-trained NCF model not found, training new model")
                        ncf_model.train(val_ratio=0.1, num_epochs=NCF_QUICK_EPOCHS)
                        
                # Generate recommendations
                recommendations = ncf_model.recommend_projects(user_id, n=limit)
            elif category:
                recommendations = cf.get_recommendations_by_category(user_id, category, n=limit)
            elif chain:
                recommendations = cf.get_recommendations_by_chain(user_id, chain, n=limit)
            else:  # hybrid (default)
                recommendations = cf.hybrid_recommendations(user_id, n=limit, feature_cf=feature_cf)
            
            # Save recommendations to database for future use
            save_recommendations_to_db(user_id, recommendations, rec_type, category, chain)
            
            # Apply post-processing (limit results, clean format)
            recommendations = recommendations[:limit]
            
            # Ensure consistent format for all recommendation types
            if rec_type in ['user-based', 'item-based'] and isinstance(recommendations, list) and recommendations and isinstance(recommendations[0], tuple):
                recommendations = convert_recommendations_format(recommendations, cf.projects_df)
        
        # Store in cache
        set_in_cache(cache_key, recommendations, RECOMMENDATION_CACHE_TTL)
        
        # Return recommendations
        return jsonify(recommendations)
    
    except ValueError as e:
        logger.warning(f"Invalid parameter: {e}")
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/interactions', methods=['POST'])
@monitor_performance
def record_interaction():
    """
    Record user interaction with a project
    
    Request Body:
        user_id (str): User ID
        project_id (str): Project ID
        interaction_type (str): Type of interaction (view, favorite, portfolio_add)
        weight (int): Weight of interaction
    """
    try:
        data = request.json
        
        # Validate required fields
        user_id = data.get('user_id')
        project_id = data.get('project_id')
        
        if not user_id or not project_id:
            return jsonify({"error": "User ID and Project ID are required"}), 400
        
        interaction_type = data.get('interaction_type', 'view')
        weight = int(data.get('weight', 1))
        
        # Validate interaction type
        valid_types = ['view', 'favorite', 'portfolio_add', 'research', 'click']
        if interaction_type not in valid_types:
            return jsonify({"error": f"Invalid interaction type. Must be one of: {', '.join(valid_types)}"}), 400
        
        # Validate weight
        if not (1 <= weight <= 5):
            return jsonify({"error": "Weight must be between 1 and 5"}), 400
        
        # Record interaction in database
        with DatabaseConnection() as cursor:
            # Check if project exists
            cursor.execute("SELECT id FROM projects WHERE id = %s", (project_id,))
            project = cursor.fetchone()
            
            if not project:
                return jsonify({"error": f"Project ID {project_id} not found"}), 404
                
            # Check if user exists, create if not
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
            user = cursor.fetchone()
            
            if not user:
                cursor.execute(
                    "INSERT INTO users (user_id, created_at) VALUES (%s, NOW())",
                    (user_id,)
                )
            
            # Record interaction
            query = """
            INSERT INTO interactions (user_id, project_id, interaction_type, weight, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            """
            
            cursor.execute(query, (user_id, project_id, interaction_type, weight))
            
        # Invalidate cache for this user
        invalidate_user_cache(user_id)
        
        return jsonify({
            "success": True, 
            "message": "Interaction recorded successfully",
            "data": {
                "user_id": user_id,
                "project_id": project_id,
                "interaction_type": interaction_type,
                "weight": weight,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    except ValueError as e:
        logger.warning(f"Invalid parameter in interaction: {e}")
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error recording interaction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/projects', methods=['GET'])
@monitor_performance
def get_projects():
    """
    Get list of projects with filtering options
    
    Query Parameters:
        category (str, optional): Filter by category
        chain (str, optional): Filter by blockchain
        search (str, optional): Search term for name or symbol
        sort_by (str, optional): Field to sort by
        sort_dir (str, optional): Sort direction (asc/desc)
        limit (int, optional): Number of results to return
        offset (int, optional): Offset for pagination
    """
    try:
        # Get filter parameters
        category = request.args.get('category')
        chain = request.args.get('chain')
        search = request.args.get('search')
        sort_by = request.args.get('sort_by', 'popularity_score')
        sort_dir = request.args.get('sort_dir', 'desc').lower()
        limit = min(int(request.args.get('limit', 50)), 200)  # Limit to max 200
        offset = int(request.args.get('offset', 0))
        
        # Validate sort parameters
        valid_sort_fields = [
            'popularity_score', 'trend_score', 'market_cap', 'volume_24h', 
            'price_usd', 'price_change_24h', 'name', 'symbol'
        ]
        if sort_by not in valid_sort_fields:
            return jsonify({"error": f"Invalid sort field. Must be one of: {', '.join(valid_sort_fields)}"}), 400
            
        if sort_dir not in ['asc', 'desc']:
            return jsonify({"error": "Sort direction must be 'asc' or 'desc'"}), 400
        
        # Check cache
        cache_key = get_cache_key(
            "projects", 
            category=category,
            chain=chain,
            search=search,
            sort_by=sort_by,
            sort_dir=sort_dir,
            limit=limit,
            offset=offset
        )
        
        cached_result = get_from_cache(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Build query
        query = """
        SELECT id, name, symbol, market_cap, volume_24h, price_usd, 
               price_change_24h, image_url, popularity_score, trend_score,
               primary_category AS category, chain
        FROM projects
        WHERE 1=1
        """
        params = []
        
        if category:
            query += " AND (primary_category = %s OR categories::text ILIKE %s)"
            params.extend([category, f'%"{category}"%'])
            
        if chain:
            query += " AND chain = %s"
            params.append(chain)
            
        if search:
            query += " AND (name ILIKE %s OR symbol ILIKE %s OR id ILIKE %s)"
            search_param = f"%{search}%"
            params.extend([search_param, search_param, search_param])
        
        # Add sorting
        query += f" ORDER BY {sort_by} {sort_dir}"
        
        # Add pagination
        query += " LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        # Execute query
        with DatabaseConnection() as cursor:
            cursor.execute(query, params)
            projects = cursor.fetchall()
            
            # Get total count for pagination
            count_query = """
            SELECT COUNT(*)
            FROM projects
            WHERE 1=1
            """
            
            # Re-use the WHERE conditions without ORDER BY, LIMIT, OFFSET
            if category:
                count_query += " AND (primary_category = %s OR categories::text ILIKE %s)"
                
            if chain:
                count_query += " AND chain = %s"
                
            if search:
                count_query += " AND (name ILIKE %s OR symbol ILIKE %s OR id ILIKE %s)"
            
            cursor.execute(count_query, params[:-2] if params else [])  # Remove limit and offset params
            total_count = cursor.fetchone()[0]
        
        # Prepare response
        result = {
            "projects": [dict(project) for project in projects],
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            },
            "filters": {
                "category": category,
                "chain": chain,
                "search": search
            }
        }
        
        # Store in cache
        set_in_cache(cache_key, result, CACHE_TTL)
        
        return jsonify(result)
    
    except ValueError as e:
        logger.warning(f"Invalid parameter in projects query: {e}")
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error getting projects: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/projects/<project_id>', methods=['GET'])
@monitor_performance
def get_project_details(project_id):
    """
    Get detailed information about a project
    
    Path Parameters:
        project_id (str): Project ID
        
    Query Parameters:
        include_similar (bool, optional): Include similar projects
    """
    try:
        # Get parameters
        include_similar = request.args.get('include_similar', 'false').lower() == 'true'
        
        # Check cache
        cache_key = get_cache_key("project_details", project_id=project_id, include_similar=include_similar)
        cached_result = get_from_cache(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Get project details
        with DatabaseConnection() as cursor:
            query = """
            SELECT *
            FROM projects
            WHERE id = %s
            """
            
            cursor.execute(query, (project_id,))
            project = cursor.fetchone()
            
            if not project:
                return jsonify({"error": f"Project with ID {project_id} not found"}), 404
            
            # Convert to dict and handle JSON columns
            project_dict = dict(project)
            
            # Convert JSON string fields to Python objects
            for json_field in ['platforms', 'categories']:
                if json_field in project_dict and isinstance(project_dict[json_field], str):
                    try:
                        project_dict[json_field] = json.loads(project_dict[json_field])
                    except json.JSONDecodeError:
                        project_dict[json_field] = {} if json_field == 'platforms' else []
            
            # Get similar projects if requested
            similar_projects = []
            if include_similar:
                cf, feature, _ = get_models()
                
                if feature:
                    similar_projects = feature.get_similar_projects(
                        project_id, 
                        cf.projects_df, 
                        n=5
                    )
        
        # Prepare response
        result = {
            "project": project_dict
        }
        
        if include_similar:
            result["similar_projects"] = similar_projects
        
        # Store in cache
        set_in_cache(cache_key, result, CACHE_TTL)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error getting project details: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<user_id>/interactions', methods=['GET'])
@monitor_performance
def get_user_interactions(user_id):
    """
    Get user's interaction history
    
    Path Parameters:
        user_id (str): User ID
        
    Query Parameters:
        limit (int, optional): Number of results to return
        offset (int, optional): Offset for pagination
    """
    try:
        # Get parameters
        limit = min(int(request.args.get('limit', 20)), 100)
        offset = int(request.args.get('offset', 0))
        
        # Get interactions
        with DatabaseConnection() as cursor:
            query = """
            SELECT i.project_id, i.interaction_type, i.weight, i.created_at,
                   p.name, p.symbol, p.image_url, p.primary_category, p.chain
            FROM interactions i
            JOIN projects p ON i.project_id = p.id
            WHERE i.user_id = %s
            ORDER BY i.created_at DESC
            LIMIT %s OFFSET %s
            """
            
            cursor.execute(query, (user_id, limit, offset))
            interactions = cursor.fetchall()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM interactions WHERE user_id = %s", (user_id,))
            total_count = cursor.fetchone()[0]
        
        # Prepare response
        result = {
            "interactions": [dict(interaction) for interaction in interactions],
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
        
        return jsonify(result)
    
    except ValueError as e:
        logger.warning(f"Invalid parameter in interactions query: {e}")
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error getting user interactions: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/categories', methods=['GET'])
@monitor_performance
def get_categories():
    """Get all available categories with project counts"""
    try:
        # Check cache
        cache_key = "categories_list"
        cached_result = get_from_cache(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Get categories
        with DatabaseConnection() as cursor:
            query = """
            SELECT primary_category AS category, COUNT(*) AS project_count
            FROM projects
            WHERE primary_category IS NOT NULL AND primary_category != 'unknown'
            GROUP BY primary_category
            ORDER BY project_count DESC
            """
            
            cursor.execute(query)
            categories = cursor.fetchall()
        
        # Convert to list of dicts
        result = [dict(category) for category in categories]
        
        # Store in cache
        set_in_cache(cache_key, result, CACHE_TTL * 2)  # Longer TTL for categories
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error getting categories: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/chains', methods=['GET'])
@monitor_performance
def get_chains():
    """Get all available blockchain chains with project counts"""
    try:
        # Check cache
        cache_key = "chains_list"
        cached_result = get_from_cache(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Get chains
        with DatabaseConnection() as cursor:
            query = """
            SELECT chain, COUNT(*) AS project_count
            FROM projects
            WHERE chain IS NOT NULL AND chain != 'unknown'
            GROUP BY chain
            ORDER BY project_count DESC
            """
            
            cursor.execute(query)
            chains = cursor.fetchall()
        
        # Convert to list of dicts
        result = [dict(chain) for chain in chains]
        
        # Store in cache
        set_in_cache(cache_key, result, CACHE_TTL * 2)  # Longer TTL for chains
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error getting chains: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/trading-signals', methods=['GET'])
@monitor_performance
def get_trading_signals():
    """
    Get trading signals for a specific project
    
    Query Parameters:
        project_id (str): Project ID
        user_id (str, optional): User ID for personalized signals
    """
    try:
        project_id = request.args.get('project_id')
        user_id = request.args.get('user_id', '')
        
        if not project_id:
            return jsonify({"error": "Project ID is required"}), 400
        
        # Load data
        cf, feature_cf, _ = get_models()
        
        # Get price history
        with DatabaseConnection() as cursor:
            query = """
            SELECT timestamp, price, volume, market_cap
            FROM historical_prices
            WHERE project_id = %s
            ORDER BY timestamp DESC
            LIMIT 168  -- Last 7 days hourly data
            """
            
            cursor.execute(query, (project_id,))
            price_history = cursor.fetchall()
            
            if not price_history:
                return jsonify({"error": "No price data available"}), 404
        
        # Create DataFrame from price data
        price_df = pd.DataFrame(price_history)
        
        # Get project info
        project_info = None
        if cf and cf.projects_df is not None:
            project_data = cf.projects_df[cf.projects_df['id'] == project_id]
            if not project_data.empty:
                project_info = project_data.iloc[0].to_dict()
        
        # Generate trading signals
        signals = generate_trading_signals(price_df, project_info)
        
        # If user_id provided, personalize recommendations
        if user_id:
            # Get user risk profile
            with DatabaseConnection() as cursor:
                cursor.execute("SELECT risk_tolerance FROM users WHERE user_id = %s", (user_id,))
                user_data = cursor.fetchone()
                risk_tolerance = user_data['risk_tolerance'] if user_data else 'medium'
            
            # Personalize based on risk profile
            signals = personalize_signals(signals, risk_tolerance)
        
        return jsonify(signals)
    
    except Exception as e:
        logger.error(f"Error generating trading signals: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/projects/<project_id>/history', methods=['GET'])
@monitor_performance
def get_project_price_history(project_id):
    """
    Get historical price data for a project
    
    Path Parameters:
        project_id (str): Project ID
        
    Query Parameters:
        period (str, optional): Time period (1d, 7d, 30d, 90d, 1y, all)
        interval (str, optional): Data interval (1h, 4h, 1d, 1w)
    """
    try:
        # Get parameters
        period = request.args.get('period', '7d').lower()
        interval = request.args.get('interval', '1h').lower()
        
        # Map period to hours
        period_mapping = {
            '1d': 24,
            '7d': 168,
            '30d': 720,
            '90d': 2160,
            '1y': 8760,
            'all': 100000  # Very large number to get all data
        }
        
        limit = period_mapping.get(period, 168)  # Default to 7d
        
        # Build query based on interval
        if interval == '1h':
            # Hourly data
            query = """
            SELECT timestamp, price, volume, market_cap
            FROM historical_prices
            WHERE project_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """
        elif interval == '4h':
            # 4-hour intervals
            query = """
            SELECT 
                timestamp,
                price,
                volume,
                market_cap
            FROM historical_prices
            WHERE 
                project_id = %s AND
                EXTRACT(HOUR FROM timestamp) % 4 = 0
            ORDER BY timestamp DESC
            LIMIT %s
            """
        elif interval == '1d':
            # Daily data
            query = """
            SELECT 
                DATE_TRUNC('day', timestamp) AS timestamp,
                AVG(price) AS price,
                SUM(volume) AS volume,
                MAX(market_cap) AS market_cap
            FROM historical_prices
            WHERE project_id = %s
            GROUP BY DATE_TRUNC('day', timestamp)
            ORDER BY timestamp DESC
            LIMIT %s
            """
        elif interval == '1w':
            # Weekly data
            query = """
            SELECT 
                DATE_TRUNC('week', timestamp) AS timestamp,
                AVG(price) AS price,
                SUM(volume) AS volume,
                MAX(market_cap) AS market_cap
            FROM historical_prices
            WHERE project_id = %s
            GROUP BY DATE_TRUNC('week', timestamp)
            ORDER BY timestamp DESC
            LIMIT %s
            """
        else:
            # Default to hourly
            query = """
            SELECT timestamp, price, volume, market_cap
            FROM historical_prices
            WHERE project_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """
        
        # Execute query
        with DatabaseConnection() as cursor:
            cursor.execute(query, (project_id, limit))
            history = cursor.fetchall()
            
            if not history:
                return jsonify({"error": "No historical data found"}), 404
        
        # Convert to list of dicts
        result = []
        for record in history:
            record_dict = dict(record)
            # Convert datetime to string
            record_dict['timestamp'] = record_dict['timestamp'].isoformat()
            # Convert numeric values
            for key in ['price', 'volume', 'market_cap']:
                if key in record_dict and record_dict[key] is not None:
                    record_dict[key] = float(record_dict[key])
            result.append(record_dict)
        
        # Sort by timestamp (ascending)
        result.sort(key=lambda x: x['timestamp'])
        
        return jsonify({
            "project_id": project_id,
            "period": period,
            "interval": interval,
            "data": result
        })
    
    except Exception as e:
        logger.error(f"Error getting price history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/train-model', methods=['POST'])
@require_api_key
@monitor_performance
def train_model_endpoint():
    """Retrain recommendation models (protected endpoint)"""
    try:
        # Get parameters
        include_all = request.json.get('include_all', True)
        eval_cold_start = request.json.get('eval_cold_start', True)
        save_model = request.json.get('save_model', True)
        
        # Import dynamically to avoid circular imports
        from scripts.train_model import train_model
        
        # Create argument object (similar to argparse)
        class Args:
            pass
            
        args = Args()
        args.include_all = include_all
        args.eval_cold_start = eval_cold_start
        args.save_model = save_model
        args.include_user_cf = include_all
        args.include_item_cf = include_all
        args.include_feature_cf = include_all
        args.include_hybrid = include_all
        args.include_ncf = include_all
        args.reprocess = False
        args.rebuild_matrices = True
        args.eval_split = 0.3
        args.random_seed = 42
        args.top_n = 10
        
        # Train model in a separate process to avoid blocking the API
        import multiprocessing
        process = multiprocessing.Process(target=train_model, args=(args,))
        process.start()
        
        # Reset models to force reloading after training
        global cf_model, feature_cf, ncf_model
        cf_model = None
        feature_cf = None
        ncf_model = None
        
        # Clear cache
        global cache
        cache = {}
        
        return jsonify({
            "success": True, 
            "message": "Model training started in background",
            "job_id": str(process.pid)
        })
    
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/update-data', methods=['POST'])
@require_api_key
@monitor_performance
def update_data_endpoint():
    """Update project data from CoinGecko API (protected endpoint)"""
    try:
        # Get parameters
        limit = min(int(request.json.get('limit', 250)), 1000)
        detail_limit = min(int(request.json.get('detail_limit', 100)), 500)
        
        # Import dynamically 
        from scripts.collect_data import collect_all_data
        
        # Create argument object
        class Args:
            pass
            
        args = Args()
        args.limit = limit
        args.detail_limit = detail_limit
        args.skip_ping = False
        args.skip_coins_list = False
        args.skip_categories = False
        args.skip_trending = False
        args.skip_markets = False
        args.skip_categories_markets = False
        args.skip_details = False
        args.rate_limit = 2
        args.timeout = 30
        args.process = True
        
        # Update data in a separate process
        import multiprocessing
        process = multiprocessing.Process(target=collect_all_data, args=(args,))
        process.start()
        
        # Reset models to force reloading after data update
        global cf_model, feature_cf, ncf_model
        cf_model = None
        feature_cf = None
        ncf_model = None
        
        # Clear cache
        global cache
        cache = {}
        
        return jsonify({
            "success": True, 
            "message": "Data update started in background",
            "job_id": str(process.pid)
        })
    
    except ValueError as e:
        logger.warning(f"Invalid parameter in update data: {e}")
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error updating data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Helper Functions
def get_recommendations_from_db(
    user_id: str, 
    rec_type: str,
    category: Optional[str] = None,
    chain: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get recommendations from database
    
    Args:
        user_id: User ID
        rec_type: Recommendation type
        category: Category filter
        chain: Chain filter
        
    Returns:
        list: List of recommendation dictionaries
    """
    try:
        with DatabaseConnection() as cursor:
            # Add category or chain condition if provided
            category_condition = ""
            chain_condition = ""
            params = [user_id, rec_type]
            
            if category:
                category_condition = " AND p.primary_category = %s"
                params.append(category)
                
            if chain:
                chain_condition = " AND p.chain = %s"
                params.append(chain)
            
            # Find recent recommendations (less than 12 hours old)
            query = f"""
            SELECT r.project_id, r.score, r.rank,
                   p.name, p.symbol, p.market_cap, p.volume_24h, p.price_usd, 
                   p.price_change_24h, p.image_url, p.popularity_score,
                   p.trend_score, p.primary_category AS category, p.chain
            FROM recommendations r
            JOIN projects p ON r.project_id = p.id
            WHERE r.user_id = %s 
            AND r.recommendation_type = %s
            {category_condition}
            {chain_condition}
            AND r.created_at > NOW() - INTERVAL '12 hours'
            ORDER BY r.rank
            """
            
            cursor.execute(query, params)
            recommendations = cursor.fetchall()
            
            if not recommendations:
                return []
            
            # Convert to list of dicts and normalize numeric values
            result = []
            for rec in recommendations:
                rec_dict = dict(rec)
                
                # Convert numeric values to correct types
                for key in ['score', 'market_cap', 'volume_24h', 'price_usd', 
                            'price_change_24h', 'popularity_score', 'trend_score']:
                    if key in rec_dict and rec_dict[key] is not None:
                        rec_dict[key] = float(rec_dict[key])
                
                # Rename score to recommendation_score for consistency
                rec_dict['recommendation_score'] = rec_dict.pop('score')
                
                result.append(rec_dict)
            
            return result
    
    except Exception as e:
        logger.error(f"Error getting recommendations from DB: {e}", exc_info=True)
        return []

def save_recommendations_to_db(
    user_id: str, 
    recommendations: List[Dict[str, Any]], 
    rec_type: str,
    category: Optional[str] = None,
    chain: Optional[str] = None
) -> None:
    """
    Save recommendations to database
    
    Args:
        user_id: User ID
        recommendations: List of recommendations
        rec_type: Recommendation type
        category: Category filter
        chain: Chain filter
    """
    try:
        with DatabaseConnection() as cursor:
            # Delete existing recommendations of this type for this user
            delete_query = """
            DELETE FROM recommendations 
            WHERE user_id = %s AND recommendation_type = %s
            """
            cursor.execute(delete_query, (user_id, rec_type))
            
            # Insert new recommendations
            insert_query = """
            INSERT INTO recommendations 
            (user_id, project_id, score, rank, recommendation_type, category_filter, chain_filter, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """
            
            for i, rec in enumerate(recommendations):
                if isinstance(rec, dict):
                    project_id = rec.get('id')
                    score = rec.get('recommendation_score', 0)
                elif isinstance(rec, tuple) and len(rec) == 2:
                    project_id = rec[0]
                    score = rec[1]
                else:
                    logger.warning(f"Unknown recommendation format: {rec}")
                    continue
                
                cursor.execute(
                    insert_query, 
                    (user_id, project_id, score, i+1, rec_type, category, chain)
                )
    
    except Exception as e:
        logger.error(f"Error saving recommendations to DB: {e}", exc_info=True)

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

def invalidate_user_cache(user_id: str) -> None:
    """
    Invalidate all cache entries for a specific user
    
    Args:
        user_id: User ID
    """
    if not CACHE_ENABLED:
        return
        
    global cache
    keys_to_remove = []
    
    for key in cache:
        if f"user_id={user_id}" in key:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del cache[key]
        
    logger.debug(f"Invalidated {len(keys_to_remove)} cache entries for user {user_id}")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# JSON encoder for datetime objects
@app.after_request
def set_response_headers(response):
    """Set common response headers"""
    response.headers['Content-Type'] = 'application/json'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

@app.before_request
def validate_content_type():
    """Validate content type for POST/PUT requests"""
    if request.method in ['POST', 'PUT'] and request.headers.get('Content-Type') != 'application/json':
        return jsonify({"error": "Content-Type must be application/json"}), 415

if __name__ == '__main__':
    # Use Gunicorn for production
    if os.environ.get('FLASK_ENV') == 'production':
        # Let Gunicorn handle the server
        pass
    else:
        # Development server
        app.run(host='0.0.0.0', port=5000, debug=False)