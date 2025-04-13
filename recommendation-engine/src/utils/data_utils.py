"""
Enhanced utilities for data manipulation and transformation with cryptocurrency-specific features
"""

import os
import pandas as pd
import numpy as np
import json
import csv
import logging
import traceback
import sys
import re
import itertools
import hashlib
from datetime import datetime, date, timezone
from typing import Dict, List, Any, Union, Optional, Tuple, Callable, Set, TypeVar, Iterable, Generator, cast
from functools import lru_cache, partial
import concurrent.futures
from io import StringIO
import warnings
from pathlib import Path
import gzip
import bz2
from joblib import Parallel, delayed
from collections import defaultdict

# Set up pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 6)

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Setup logging
try:
    from central_logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback if central_logging is not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
S = TypeVar('S')

# Constants
DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
COMPRESSION_EXTENSIONS = {'.gz': 'gzip', '.bz2': 'bz2', '.zip': 'zip', '.xz': 'xz'}
CRYPTO_MARKET_FIELDS = [
    'price_usd', 'market_cap', 'volume_24h', 'circulating_supply', 
    'total_supply', 'max_supply', 'price_change_24h', 'price_change_7d'
]

class DateTimeEncoder(json.JSONEncoder):
    """
    Enhanced JSON encoder for handling datetime objects, numpy types, and other non-standard JSON types
    """
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def safe_convert(value: Any, target_type: Callable[[Any], T], default_value: T) -> T:
    """
    Safely converts a value to a specified type with enhanced error handling
    
    Args:
        value: Value to convert
        target_type: Function to convert value (int, float, etc.)
        default_value: Value to return if conversion fails
        
    Returns:
        Converted value or default_value if conversion fails
    """
    if value is None:
        return default_value
        
    try:
        if pd.isna(value):
            return default_value
            
        # Special handling for certain types
        if target_type == bool and isinstance(value, str):
            value = value.lower()
            if value in ('true', 't', 'yes', 'y', '1'):
                return cast(T, True)
            elif value in ('false', 'f', 'no', 'n', '0'):
                return cast(T, False)
                
        # Handle lists and dicts specially when they come as strings
        if target_type == list and isinstance(value, str):
            if value.strip().startswith('[') and value.strip().endswith(']'):
                try:
                    return cast(T, json.loads(value))
                except json.JSONDecodeError:
                    pass
                    
        if target_type == dict and isinstance(value, str):
            if value.strip().startswith('{') and value.strip().endswith('}'):
                try:
                    return cast(T, json.loads(value))
                except json.JSONDecodeError:
                    pass
        
        # Handle numpy types
        if isinstance(value, np.generic):
            if issubclass(target_type, int) and isinstance(value, np.integer):
                return cast(T, int(value))
            elif issubclass(target_type, float) and isinstance(value, np.floating):
                return cast(T, float(value))
            elif issubclass(target_type, bool) and isinstance(value, np.bool_):
                return cast(T, bool(value))
                
        return cast(T, target_type(value))
        
    except (ValueError, TypeError, AttributeError, OverflowError):
        return default_value

def infer_compression(filepath: str) -> Optional[str]:
    """
    Infer compression type from file extension
    
    Args:
        filepath: Path to the file
        
    Returns:
        str or None: Compression type for pandas or None if uncompressed
    """
    extension = os.path.splitext(filepath)[1].lower()
    return COMPRESSION_EXTENSIONS.get(extension)

def detect_file_encoding(filepath: str, n_lines: int = 1000) -> str:
    """
    Attempt to detect file encoding
    
    Args:
        filepath: Path to the file
        n_lines: Number of lines to read for detection
        
    Returns:
        str: Detected encoding
    """
    try:
        import chardet
        
        # Read a sample of the file
        with open(filepath, 'rb') as f:
            sample = b''.join(f.readline() for _ in range(n_lines))
            
        # Detect encoding
        result = chardet.detect(sample)
        encoding = result['encoding'] or 'utf-8'
        confidence = result['confidence']
        
        logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
        return encoding
        
    except ImportError:
        logger.debug("chardet not installed, defaulting to utf-8")
        return 'utf-8'
    except Exception as e:
        logger.warning(f"Error detecting encoding: {e}, defaulting to utf-8")
        return 'utf-8'

def load_csv_data(
    filepath: str, 
    encoding: Optional[str] = None,
    detect_encoding: bool = True,
    parse_dates: Union[bool, List[str]] = True,
    low_memory: bool = False,
    **kwargs
) -> Optional[pd.DataFrame]:
    """
    Enhanced CSV loader with encoding detection and better error handling
    
    Args:
        filepath: Path to the CSV file
        encoding: File encoding or None to auto-detect
        detect_encoding: Whether to attempt encoding detection
        parse_dates: List of columns to parse as dates or True to auto-detect
        low_memory: Set to True for large files with mixed types
        **kwargs: Additional arguments for pd.read_csv()
        
    Returns:
        pd.DataFrame or None: Loaded data or None if file not found/invalid
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
        
        # Detect encoding if needed
        if encoding is None and detect_encoding:
            encoding = detect_file_encoding(filepath)
            
        # If no encoding is specified or detected, default to utf-8
        if encoding is None:
            encoding = 'utf-8'
            
        # Determine compression
        compression = infer_compression(filepath)
        
        # Handle date parsing
        if parse_dates is True:
            # Try to automatically detect date columns by regex patterns
            parse_dates = []
            sample = pd.read_csv(filepath, nrows=5, encoding=encoding, compression=compression)
            
            date_patterns = [
                r'date', r'time', r'datetime', r'timestamp', r'created_at',
                r'updated_at', r'day', r'month', r'year'
            ]
            
            for col in sample.columns:
                if any(re.search(pattern, col, re.IGNORECASE) for pattern in date_patterns):
                    parse_dates.append(col)
        
        # Load data with better error handling
        df = pd.read_csv(
            filepath, 
            encoding=encoding, 
            compression=compression,
            parse_dates=parse_dates,
            low_memory=low_memory,
            **kwargs
        )
        
        # Check if date parsing was successful, and try again for failed columns
        if isinstance(parse_dates, list) and parse_dates:
            for col in parse_dates:
                if col in df.columns and pd.api.types.is_object_dtype(df[col]):
                    # Try to parse dates with different formats
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Failed to parse dates for column {col}: {e}")
        
        logger.debug(f"Loaded CSV data from {filepath}: {len(df)} rows, {len(df.columns)} columns")
        
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {filepath}")
        return pd.DataFrame()
        
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
        # Try again with Python engine which is more forgiving
        try:
            logger.info("Retrying with Python engine...")
            return pd.read_csv(
                filepath, 
                encoding=encoding or 'utf-8',
                engine='python', 
                on_bad_lines='skip',
                **kwargs
            )
        except Exception:
            return None
        
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        return None

def save_csv_data(
    df: pd.DataFrame, 
    filepath: str, 
    compression: Optional[str] = None,
    encoding: str = 'utf-8',
    float_format: str = '%.6f',
    date_format: str = None,
    optimize_size: bool = False,
    **kwargs
) -> bool:
    """
    Enhanced CSV saving with compression and optimization options
    
    Args:
        df: DataFrame to save
        filepath: Path to save the file
        compression: Compression type (None, 'gzip', 'bz2', etc.)
        encoding: File encoding
        float_format: Format string for float values
        date_format: Format string for date values
        optimize_size: Optimize data types to reduce file size
        **kwargs: Additional arguments for df.to_csv()
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Infer compression if not specified
        if compression is None:
            compression = infer_compression(filepath)
        
        # Set default kwargs if not provided
        if 'index' not in kwargs:
            kwargs['index'] = False
            
        # Make a copy for optimization if needed
        save_df = df
        
        if optimize_size:
            save_df = optimize_dataframe_dtypes(df.copy())
        
        # Save with enhanced options
        save_df.to_csv(
            filepath, 
            compression=compression,
            encoding=encoding,
            float_format=float_format,
            date_format=date_format,
            **kwargs
        )
        
        # Log file size
        file_size = os.path.getsize(filepath)
        logger.debug(f"Saved {len(df)} rows to {filepath} ({file_size/1024:.1f} KB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving CSV data: {e}")
        return False

def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        DataFrame: Optimized DataFrame
    """
    result = df.copy()
    
    # Downcast numeric columns
    for col in result.select_dtypes(include=['integer']).columns:
        # Try to convert to the smallest possible integer type
        result[col] = pd.to_numeric(result[col], downcast='integer')
        
    for col in result.select_dtypes(include=['float']).columns:
        # Try to convert to the smallest possible float type
        result[col] = pd.to_numeric(result[col], downcast='float')
    
    # Convert object columns to categorical for repeated values
    for col in result.select_dtypes(include=['object']).columns:
        # Check if values are mostly repeated (good candidates for categorical)
        unique_count = result[col].nunique()
        if unique_count > 0 and unique_count < len(result) * 0.5:  # Less than 50% unique values
            result[col] = result[col].astype('category')
    
    return result

def load_json_data(
    filepath: str,
    encoding: str = 'utf-8',
    default_value: Any = None,
    flatten: bool = False
) -> Optional[Union[Dict, List]]:
    """
    Enhanced JSON loader with flattening option
    
    Args:
        filepath: Path to the JSON file
        encoding: File encoding
        default_value: Value to return if file not found or invalid
        flatten: Whether to flatten nested JSON objects
        
    Returns:
        dict/list or default_value: Loaded data or default_value if file not found/invalid
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return default_value
        
        # Determine compression
        compression = infer_compression(filepath)
        
        # Open file based on compression
        if compression == 'gzip':
            with gzip.open(filepath, 'rt', encoding=encoding) as f:
                data = json.load(f)
        elif compression == 'bz2':
            with bz2.open(filepath, 'rt', encoding=encoding) as f:
                data = json.load(f)
        else:
            with open(filepath, 'r', encoding=encoding) as f:
                data = json.load(f)
        
        # Flatten if requested
        if flatten:
            data = flatten_json(data)
            
        logger.debug(f"Loaded JSON data from {filepath}")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from {filepath}: {e}")
        # Try to handle common JSON issues
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                text = f.read()
                
            # Try to fix common JSON issues
            if text.startswith("'") and text.endswith("'"):
                text = text[1:-1]
                
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1].replace('\\"', '"').replace("\\'", "'")
                
            # Try to parse again
            data = json.loads(text)
            logger.info(f"Successfully fixed JSON parsing issue in {filepath}")
            
            if flatten:
                data = flatten_json(data)
                
            return data
        except Exception:
            return default_value
        
    except Exception as e:
        logger.error(f"Error loading JSON data from {filepath}: {e}")
        return default_value

def save_json_data(
    data: Union[Dict, List], 
    filepath: str, 
    encoding: str = 'utf-8',
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
    compression: Optional[str] = None,
    date_format: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Enhanced JSON saving with compression and datetime handling
    
    Args:
        data: Data to save
        filepath: Path to save the file
        encoding: File encoding
        indent: Number of spaces for indentation or None for compact JSON
        ensure_ascii: Whether to escape non-ASCII characters
        compression: Compression type (None, 'gzip', 'bz2')
        date_format: Format string for date values
        **kwargs: Additional arguments for json.dump()
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Infer compression if not specified
        if compression is None:
            compression = infer_compression(filepath)
            
        # Use custom encoder for handling datetime, numpy types, etc.
        encoder_kwargs = {'cls': DateTimeEncoder}
        if date_format:
            encoder_kwargs = {
                'default': lambda obj: obj.strftime(date_format) if isinstance(obj, (datetime, date)) else DateTimeEncoder().default(obj)
            }
            
        # Merge with user kwargs
        dump_kwargs = {**encoder_kwargs, **kwargs}
            
        # Open file based on compression
        if compression == 'gzip':
            with gzip.open(filepath, 'wt', encoding=encoding) as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, **dump_kwargs)
        elif compression == 'bz2':
            with bz2.open(filepath, 'wt', encoding=encoding) as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, **dump_kwargs)
        else:
            with open(filepath, 'w', encoding=encoding) as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, **dump_kwargs)
                
        # Log file size
        file_size = os.path.getsize(filepath)
        logger.debug(f"Saved JSON data to {filepath} ({file_size/1024:.1f} KB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving JSON data: {e}")
        return False

def flatten_json(data: Union[Dict, List], separator: str = '.') -> Dict:
    """
    Flatten nested JSON objects for easier analysis
    
    Args:
        data: Nested JSON data
        separator: Separator for nested keys
        
    Returns:
        dict: Flattened dictionary
    """
    if isinstance(data, list):
        # Convert list to dict with indices as keys
        return {f'[{i}]': flatten_json(item, separator) if isinstance(item, (dict, list)) else item 
                for i, item in enumerate(data)}
    
    result = {}
    
    def _flatten(x: Union[Dict, List], name: str = ''):
        if isinstance(x, dict):
            for key, value in x.items():
                _flatten(value, f"{name}{separator}{key}" if name else key)
        elif isinstance(x, list):
            for i, item in enumerate(x):
                _flatten(item, f"{name}[{i}]")
        else:
            result[name] = x
            
    _flatten(data)
    return result

def find_latest_file(
    directory: str, 
    prefix: str = '', 
    suffix: str = '', 
    full_path: bool = True,
    recursive: bool = False,
    min_size: int = 0
) -> Optional[str]:
    """
    Enhanced function to find the latest file with specified criteria
    
    Args:
        directory: Directory to search
        prefix: File prefix to match
        suffix: File suffix to match
        full_path: Return full path if True, filename only if False
        recursive: Search subdirectories
        min_size: Minimum file size in bytes
        
    Returns:
        str or None: Latest file path/name or None if not found
    """
    try:
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return None
        
        # Use pathlib for more robust path handling
        search_dir = Path(directory)
        
        # Set up glob pattern
        pattern = f"{prefix}*{suffix}"
        
        # Search files
        if recursive:
            matching_files = list(search_dir.glob(f"**/{pattern}"))
        else:
            matching_files = list(search_dir.glob(pattern))
        
        # Filter by minimum size if specified
        if min_size > 0:
            matching_files = [f for f in matching_files if f.stat().st_size >= min_size]
            
        if not matching_files:
            logger.warning(f"No files matching '{pattern}' found in {directory}")
            return None
            
        # Sort by modification time
        latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
        
        if full_path:
            return str(latest_file)
        else:
            return latest_file.name
            
    except Exception as e:
        logger.error(f"Error finding latest file: {e}")
        return None

def extract_timestamp_from_filename(
    filename: str, 
    pattern: str = r'_(\d{8}_\d{6})\.',
    datetime_format: str = '%Y%m%d_%H%M%S'
) -> Optional[datetime]:
    """
    Enhanced timestamp extraction with custom pattern support
    
    Args:
        filename: Filename to process
        pattern: Regex pattern to extract timestamp
        datetime_format: Format string for datetime parsing
        
    Returns:
        datetime or None: Extracted timestamp or None if not found
    """
    try:
        match = re.search(pattern, filename)
        if match:
            timestamp_str = match.group(1)
            return datetime.strptime(timestamp_str, datetime_format)
            
        # Try common patterns if the specified pattern fails
        common_patterns = [
            (r'(\d{8}_\d{6})', '%Y%m%d_%H%M%S'),
            (r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', '%Y-%m-%d_%H-%M-%S'),
            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d')
        ]
        
        for pat, fmt in common_patterns:
            match = re.search(pat, filename)
            if match:
                timestamp_str = match.group(1)
                return datetime.strptime(timestamp_str, fmt)
                
        return None
        
    except Exception as e:
        logger.error(f"Error extracting timestamp: {e}")
        return None

def filter_dataframe(
    df: pd.DataFrame, 
    filters: Dict[str, Any],
    operators: Optional[Dict[str, str]] = None,
    combine_with: str = 'and'
) -> pd.DataFrame:
    """
    Enhanced DataFrame filtering with more flexible operators
    
    Args:
        df: DataFrame to filter
        filters: Dictionary mapping column names to filter values
        operators: Dictionary mapping column names to operators
            (valid operators: '=', '!=', '>', '<', '>=', '<=', 'in', 'not in', 
             'contains', 'startswith', 'endswith', 'between', 'isnull', 'notnull')
        combine_with: How to combine multiple filters ('and' or 'or')
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if not filters:
        return df
        
    if operators is None:
        operators = {col: '=' for col in filters}
        
    # Copy DataFrame for filtering
    filtered_df = df.copy()
    
    # Store individual filter masks
    masks = []
    
    # Apply filters
    for col, value in filters.items():
        if col not in filtered_df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping filter")
            continue
            
        operator = operators.get(col, '=')
        
        try:
            # Handle different operator types
            if operator == '=' or operator == '==':
                mask = filtered_df[col] == value
            elif operator == '!=':
                mask = filtered_df[col] != value
            elif operator == '>':
                mask = filtered_df[col] > value
            elif operator == '<':
                mask = filtered_df[col] < value
            elif operator == '>=':
                mask = filtered_df[col] >= value
            elif operator == '<=':
                mask = filtered_df[col] <= value
            elif operator.lower() == 'in':
                if not isinstance(value, (list, tuple, set)):
                    value = [value]
                mask = filtered_df[col].isin(value)
            elif operator.lower() == 'not in':
                if not isinstance(value, (list, tuple, set)):
                    value = [value]
                mask = ~filtered_df[col].isin(value)
            elif operator.lower() == 'contains':
                if pd.api.types.is_string_dtype(filtered_df[col]):
                    # Handle list of values to check containment
                    if isinstance(value, (list, tuple, set)):
                        # Check if any value in the list is contained
                        mask = filtered_df[col].str.contains('|'.join(map(re.escape, value)), na=False, regex=True)
                    else:
                        mask = filtered_df[col].str.contains(str(value), na=False, regex=True)
                else:
                    logger.warning(f"Cannot use 'contains' operator on non-string column '{col}'")
                    continue
            elif operator.lower() == 'startswith':
                if pd.api.types.is_string_dtype(filtered_df[col]):
                    mask = filtered_df[col].str.startswith(str(value), na=False)
                else:
                    logger.warning(f"Cannot use 'startswith' operator on non-string column '{col}'")
                    continue
            elif operator.lower() == 'endswith':
                if pd.api.types.is_string_dtype(filtered_df[col]):
                    mask = filtered_df[col].str.endswith(str(value), na=False)
                else:
                    logger.warning(f"Cannot use 'endswith' operator on non-string column '{col}'")
                    continue
            elif operator.lower() == 'between':
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    mask = (filtered_df[col] >= value[0]) & (filtered_df[col] <= value[1])
                else:
                    logger.warning("'between' operator requires a list or tuple of two values")
                    continue
            elif operator.lower() == 'isnull':
                mask = filtered_df[col].isnull()
            elif operator.lower() == 'notnull':
                mask = filtered_df[col].notnull()
            else:
                logger.warning(f"Unknown operator '{operator}' for column '{col}'")
                continue
                
            # Add mask to list
            masks.append(mask)
                
        except Exception as e:
            logger.warning(f"Error applying filter on column '{col}': {e}")
            continue
    
    # Combine masks
    if not masks:
        return filtered_df
        
    if combine_with.lower() == 'and':
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = final_mask & mask
    else:  # 'or'
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = final_mask | mask
    
    return filtered_df[final_mask]

def handle_missing_values(
    df: pd.DataFrame,
    strategy: Optional[Dict[str, str]] = None,
    custom_values: Optional[Dict[str, Any]] = None,
    time_columns: Optional[List[str]] = None,
    categorical_threshold: int = 10
) -> pd.DataFrame:
    """
    Enhanced missing value handler with specialized strategies for different data types
    
    Args:
        df: DataFrame to process
        strategy: Dictionary mapping column names to strategies
            (valid strategies: 'mean', 'median', 'mode', 'min', 'max', 'zero', 
             'empty_string', 'custom', 'drop', 'interpolate', 'ffill', 'bfill')
        custom_values: Dictionary mapping column names to custom values
        time_columns: List of time series columns for specialized handling
        categorical_threshold: Maximum number of unique values to consider as categorical
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    if strategy is None:
        strategy = {}
        
    if custom_values is None:
        custom_values = {}
        
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Get columns with missing values
    missing_columns = result_df.columns[result_df.isna().any()].tolist()
    
    if not missing_columns:
        logger.debug("No missing values found in DataFrame")
        return result_df
        
    # Auto-detect strategies for columns not in strategy dict
    for col in missing_columns:
        if col not in strategy:
            # Auto-detect appropriate strategy based on column data type
            if pd.api.types.is_numeric_dtype(result_df[col]):
                if col in CRYPTO_MARKET_FIELDS or any(market_field in col.lower() for market_field in ['price', 'volume', 'market_cap']):
                    # Financial data often uses zero for missing values
                    strategy[col] = 'zero'
                else:
                    # Use median for other numeric data (more robust than mean)
                    strategy[col] = 'median'
            elif pd.api.types.is_datetime64_dtype(result_df[col]) or (time_columns and col in time_columns):
                # Time series data often benefits from interpolation or forward fill
                strategy[col] = 'ffill'
            elif pd.api.types.is_string_dtype(result_df[col]):
                # For string columns, check if they're categorical-like
                unique_count = result_df[col].nunique()
                if 0 < unique_count <= categorical_threshold:
                    # For categorical-like columns, use mode
                    strategy[col] = 'mode'
                else:
                    # For other string columns, use empty string
                    strategy[col] = 'empty_string'
            elif pd.api.types.is_categorical_dtype(result_df[col]):
                # For categorical columns, use mode
                strategy[col] = 'mode'
            else:
                # Default strategy
                strategy[col] = 'drop'
    
    # Handle each column based on strategy
    for col in missing_columns:
        col_strategy = strategy.get(col)
        missing_count = result_df[col].isna().sum()
        
        logger.debug(f"Handling missing values in column '{col}': {missing_count} missing, strategy: {col_strategy}")
        
        # Handle based on strategy
        if col_strategy == 'mean':
            if pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = result_df[col].fillna(result_df[col].mean())
            else:
                logger.warning(f"Cannot use 'mean' strategy on non-numeric column '{col}'")
                
        elif col_strategy == 'median':
            if pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = result_df[col].fillna(result_df[col].median())
            else:
                logger.warning(f"Cannot use 'median' strategy on non-numeric column '{col}'")
                
        elif col_strategy == 'mode':
            mode_value = result_df[col].mode().iloc[0] if not result_df[col].mode().empty else None
            if mode_value is not None:
                result_df[col] = result_df[col].fillna(mode_value)
                
        elif col_strategy == 'min':
            if pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = result_df[col].fillna(result_df[col].min())
            else:
                logger.warning(f"Cannot use 'min' strategy on non-numeric column '{col}'")
                
        elif col_strategy == 'max':
            if pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = result_df[col].fillna(result_df[col].max())
            else:
                logger.warning(f"Cannot use 'max' strategy on non-numeric column '{col}'")
                
        elif col_strategy == 'zero':
            if pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = result_df[col].fillna(0)
            else:
                logger.warning(f"Cannot use 'zero' strategy on non-numeric column '{col}'")
                
        elif col_strategy == 'empty_string':
            result_df[col] = result_df[col].fillna('')
            
        elif col_strategy == 'custom':
            if col in custom_values:
                result_df[col] = result_df[col].fillna(custom_values[col])
            else:
                logger.warning(f"Custom value not provided for column '{col}'")
                
        elif col_strategy == 'interpolate':
            # Linear interpolation works well for time series and numeric data
            result_df[col] = result_df[col].interpolate(method='linear')
            # Handle edge cases (first/last values)
            result_df[col] = result_df[col].fillna(method='bfill').fillna(method='ffill')
                
        elif col_strategy == 'ffill':
            # Forward fill (carry last observation forward)
            result_df[col] = result_df[col].fillna(method='ffill')
            # Handle case where first values are NaN
            result_df[col] = result_df[col].fillna(method='bfill')
                
        elif col_strategy == 'bfill':
            # Backward fill (use next valid observation)
            result_df[col] = result_df[col].fillna(method='bfill')
            # Handle case where last values are NaN
            result_df[col] = result_df[col].fillna(method='ffill')
                
        elif col_strategy == 'drop':
            # This will be handled after all columns are processed
            pass
            
        else:
            logger.warning(f"Unknown strategy '{col_strategy}' for column '{col}'")
    
    # Drop rows that still have missing values in columns with 'drop' strategy
    drop_columns = [col for col in missing_columns if strategy.get(col) == 'drop']
    if drop_columns:
        before_count = len(result_df)
        result_df = result_df.dropna(subset=drop_columns)
        after_count = len(result_df)
        dropped_count = before_count - after_count
        
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} rows with missing values in columns: {drop_columns}")
    
    return result_df

def normalize_column(
    df: pd.DataFrame,
    column: str,
    method: str = 'minmax',
    custom_range: Tuple[float, float] = (0, 1),
    handle_outliers: bool = False,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Enhanced column normalization with outlier handling
    
    Args:
        df: DataFrame to process
        column: Column to normalize
        method: Normalization method ('minmax', 'zscore', 'maxabs', 'log', 'robust', 'quantile')
        custom_range: Min and max values for minmax or quantile normalization
        handle_outliers: Whether to handle outliers before normalization
        inplace: Whether to modify the original DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return df
        
    if not pd.api.types.is_numeric_dtype(df[column]):
        logger.warning(f"Cannot normalize non-numeric column '{column}'")
        return df
        
    # Make a copy if not inplace
    result_df = df if inplace else df.copy()
    
    # Handle outliers if requested
    if handle_outliers:
        # Calculate IQR for outlier detection
        q1 = result_df[column].quantile(0.25)
        q3 = result_df[column].quantile(0.75)
        iqr = q3 - q1
        
        # Define outlier bounds (1.5 * IQR)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Clip outliers to bounds
        outlier_mask = (result_df[column] < lower_bound) | (result_df[column] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            logger.debug(f"Clipping {outlier_count} outliers in column '{column}'")
            result_df[column] = result_df[column].clip(lower=lower_bound, upper=upper_bound)
    
    # Handle based on method
    try:
        if method == 'minmax':
            min_val = result_df[column].min()
            max_val = result_df[column].max()
            
            if max_val == min_val:
                result_df[column] = custom_range[0]
            else:
                min_range, max_range = custom_range
                result_df[column] = min_range + (result_df[column] - min_val) * (max_range - min_range) / (max_val - min_val)
                
        elif method == 'zscore':
            mean = result_df[column].mean()
            std = result_df[column].std()
            
            if std == 0:
                result_df[column] = 0
            else:
                result_df[column] = (result_df[column] - mean) / std
                
        elif method == 'maxabs':
            max_abs = result_df[column].abs().max()
            
            if max_abs == 0:
                result_df[column] = 0
            else:
                result_df[column] = result_df[column] / max_abs
                
        elif method == 'log':
            # Handle zero and negative values
            min_val = result_df[column].min()
            if min_val <= 0:
                offset = abs(min_val) + 1
                result_df[column] = np.log1p(result_df[column] + offset)
            else:
                result_df[column] = np.log1p(result_df[column])
                
        elif method == 'robust':
            median = result_df[column].median()
            q1 = result_df[column].quantile(0.25)
            q3 = result_df[column].quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                result_df[column] = 0
            else:
                result_df[column] = (result_df[column] - median) / iqr
                
        elif method == 'quantile':
            # Quantile normalization (distribution-preserving)
            from scipy import stats
            result_df[column] = stats.rankdata(result_df[column]) / len(result_df)
            
            # Rescale to custom range if requested
            if custom_range != (0, 1):
                min_range, max_range = custom_range
                result_df[column] = min_range + result_df[column] * (max_range - min_range)
                
        else:
            logger.warning(f"Unknown normalization method '{method}'")
    
    except Exception as e:
        logger.error(f"Error normalizing column '{column}': {e}")
    
    return result_df

def convert_json_columns(
    df: pd.DataFrame, 
    json_columns: List[str], 
    handle_errors: str = 'ignore',
    flatten: bool = False
) -> pd.DataFrame:
    """
    Enhanced JSON column conversion with flattening option
    
    Args:
        df: DataFrame to process
        json_columns: List of columns containing JSON strings
        handle_errors: How to handle errors ('ignore', 'null', 'raise')
        flatten: Whether to flatten nested JSON objects
        
    Returns:
        pd.DataFrame: DataFrame with converted columns
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    for col in json_columns:
        if col not in result_df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
            
        def parse_json(value):
            if pd.isna(value) or value == '' or value == '[]' or value == '{}':
                return [] if col.endswith('_list') or col.endswith('_array') or col == 'categories' else {}
                
            try:
                if isinstance(value, str):
                    # Try multiple parsing approaches
                    try:
                        # Standard JSON parsing
                        parsed = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            # Clean up excessive quotes and try again
                            cleaned = value.replace('\"\"', '\"').replace('\\"', '"')
                            parsed = json.loads(cleaned)
                        except:
                            try:
                                # Last resort - strip quotes and use eval
                                if value.startswith('"[') and value.endswith(']"'):
                                    value = value[1:-1]
                                if (col == 'categories' and value.startswith('[')) or \
                                    (col == 'platforms' and value.startswith('{')):
                                    return eval(value)
                                else:
                                    return [] if col.endswith('_list') or col.endswith('_array') or col == 'categories' else {}
                            except:
                                return [] if col.endswith('_list') or col.endswith('_array') or col == 'categories' else {}
                    
                    # Flatten if requested
                    if flatten and isinstance(parsed, (dict, list)):
                        flat_dict = flatten_json(parsed)
                        
                        # For list columns, attempt to keep as list if all keys are numeric indices
                        if col.endswith('_list') or col.endswith('_array') or col == 'categories':
                            if all(k.startswith('[') and k.endswith(']') for k in flat_dict.keys()):
                                # Convert back to list
                                max_idx = max(int(k[1:-1]) for k in flat_dict.keys())
                                flat_list = [None] * (max_idx + 1)
                                for k, v in flat_dict.items():
                                    idx = int(k[1:-1])
                                    flat_list[idx] = v
                                return flat_list
                        
                        return flat_dict
                    
                    # Ensure consistent type returned
                    if col.endswith('_list') or col.endswith('_array') or col == 'categories':
                        return parsed if isinstance(parsed, list) else []
                    else:
                        return parsed if isinstance(parsed, dict) else {}
                
                elif isinstance(value, (dict, list)):
                    # Already parsed, just validate type and flatten if needed
                    if flatten and isinstance(value, (dict, list)):
                        flat_dict = flatten_json(value)
                        
                        # For list columns, attempt to keep as list if all keys are numeric indices
                        if col.endswith('_list') or col.endswith('_array') or col == 'categories':
                            if all(k.startswith('[') and k.endswith(']') for k in flat_dict.keys()):
                                # Convert back to list
                                max_idx = max(int(k[1:-1]) for k in flat_dict.keys())
                                flat_list = [None] * (max_idx + 1)
                                for k, v in flat_dict.items():
                                    idx = int(k[1:-1])
                                    flat_list[idx] = v
                                return flat_list
                        
                        return flat_dict
                    
                    if col.endswith('_list') or col.endswith('_array') or col == 'categories':
                        return value if isinstance(value, list) else []
                    else:
                        return value if isinstance(value, dict) else {}
                else:
                    return [] if col.endswith('_list') or col.endswith('_array') or col == 'categories' else {}
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing JSON in column {col}: {e}")
                if handle_errors == 'raise':
                    raise
                elif handle_errors == 'null':
                    return [] if col.endswith('_list') or col.endswith('_array') or col == 'categories' else {}
                else:  # ignore
                    return value
            except Exception as e:
                logger.warning(f"Unexpected error processing JSON in column {col}: {e}")
                if handle_errors == 'raise':
                    raise
                return [] if col.endswith('_list') or col.endswith('_array') or col == 'categories' else {}
        
        result_df[col] = result_df[col].apply(parse_json)
        logger.debug(f"Converted JSON column: {col}")
    
    return result_df

def merge_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    on: Union[str, List[str]],
    how: str = 'inner',
    suffixes: Tuple[str, str] = ('_x', '_y'),
    validate: Optional[str] = None,
    indicator: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Enhanced DataFrame merging with validation and conflict resolution
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        on: Column(s) to merge on
        how: Type of merge ('inner', 'outer', 'left', 'right')
        suffixes: Suffixes for overlapping columns
        validate: Validation type (None, '1:1', '1:m', 'm:1', 'm:m')
        indicator: Add a column showing merge source
        **kwargs: Additional arguments for pd.merge()
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    try:
        # Check if merge columns exist
        if isinstance(on, str):
            on_columns = [on]
        else:
            on_columns = on
            
        for col in on_columns:
            if col not in df1.columns:
                raise ValueError(f"Column '{col}' not found in first DataFrame")
                
            if col not in df2.columns:
                raise ValueError(f"Column '{col}' not found in second DataFrame")
        
        # Check datatypes and convert if necessary
        for col in on_columns:
            if df1[col].dtype != df2[col].dtype:
                logger.warning(f"Data type mismatch for column '{col}': {df1[col].dtype} vs {df2[col].dtype}")
                
                # Try to convert to common type
                if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                    # Both numeric, convert to float
                    df1 = df1.copy()
                    df2 = df2.copy()
                    df1[col] = df1[col].astype(float)
                    df2[col] = df2[col].astype(float)
                else:
                    # Convert both to string
                    df1 = df1.copy()
                    df2 = df2.copy()
                    df1[col] = df1[col].astype(str)
                    df2[col] = df2[col].astype(str)
        
        # Perform merge
        merged_df = pd.merge(
            df1,
            df2,
            on=on,
            how=how,
            suffixes=suffixes,
            validate=validate,
            indicator=indicator,
            **kwargs
        )
        
        # Log merge statistics
        logger.debug(f"Merged DataFrames: {len(merged_df)} rows result from {len(df1)} and {len(df2)} rows")
        
        # If indicator is True, analyze the merge result
        if indicator:
            indicator_col = '_merge' if '_merge' in merged_df.columns else kwargs.get('indicator', True)
            if isinstance(indicator_col, str) and indicator_col in merged_df.columns:
                left_only = (merged_df[indicator_col] == 'left_only').sum()
                right_only = (merged_df[indicator_col] == 'right_only').sum()
                both = (merged_df[indicator_col] == 'both').sum()
                
                logger.info(f"Merge analysis: {both} rows from both DataFrames, "
                           f"{left_only} left_only rows, {right_only} right-only rows")
        
        return merged_df
        
    except pd.errors.MergeError as e:
        logger.error(f"Error merging DataFrames: {e}")
        raise
        
    except ValueError as e:
        logger.error(f"Validation error during merge: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Error merging DataFrames: {e}")
        raise

def extract_nested_data(
    df: pd.DataFrame,
    nested_column: str,
    extract_columns: List[str],
    prefix: Optional[str] = None,
    keep_original: bool = False,
    flatten_lists: bool = False,
    errors: str = 'ignore'
) -> pd.DataFrame:
    """
    Enhanced nested data extraction with better error handling
    
    Args:
        df: DataFrame to process
        nested_column: Column containing nested data
        extract_columns: List of keys to extract
        prefix: Prefix for extracted columns
        keep_original: Whether to keep the original nested column
        flatten_lists: Whether to flatten list values into separate rows
        errors: How to handle errors ('ignore', 'raise')
        
    Returns:
        pd.DataFrame: DataFrame with extracted columns
    """
    if nested_column not in df.columns:
        logger.warning(f"Column '{nested_column}' not found in DataFrame")
        return df
        
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Set prefix if not provided
    if prefix is None:
        prefix = f"{nested_column}_"
    
    # Extract each column
    for key in extract_columns:
        column_name = f"{prefix}{key}"
        
        def extract_value(item):
            try:
                if item is None:
                    return None
                    
                if isinstance(item, dict):
                    # Handle nested paths with dots
                    if '.' in key:
                        parts = key.split('.')
                        value = item
                        for part in parts:
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            else:
                                return None
                        return value
                    return item.get(key)
                elif isinstance(item, str):
                    try:
                        json_item = json.loads(item)
                        # Handle nested paths with dots
                        if '.' in key:
                            parts = key.split('.')
                            value = json_item
                            for part in parts:
                                if isinstance(value, dict) and part in value:
                                    value = value[part]
                                else:
                                    return None
                            return value
                        return json_item.get(key)
                    except json.JSONDecodeError:
                        return None
                else:
                    return None
            except Exception as e:
                if errors == 'raise':
                    raise
                logger.debug(f"Error extracting nested data for key '{key}': {e}")
                return None
        
        result_df[column_name] = result_df[nested_column].apply(extract_value)
    
    # Handle list flattening if requested
    if flatten_lists:
        list_columns = []
        
        for col in result_df.columns:
            if col.startswith(prefix) and result_df[col].apply(lambda x: isinstance(x, list)).any():
                list_columns.append(col)
        
        if list_columns:
            # Create an exploded DataFrame for each list column
            all_exploded_dfs = []
            
            for list_col in list_columns:
                # Create a temp DataFrame for this explosion to avoid modifying original
                temp_df = result_df.copy()
                temp_df[list_col] = temp_df[list_col].apply(lambda x: x if isinstance(x, list) else [x] if x is not None else [None])
                
                try:
                    # Explode the column
                    exploded_df = temp_df.explode(list_col)
                    all_exploded_dfs.append(exploded_df)
                except Exception as e:
                    logger.warning(f"Error exploding column {list_col}: {e}")
                    # Skip this column if explosion fails
                    continue
            
            if all_exploded_dfs:
                # Merge all exploded DataFrames if any were successful
                result_df = pd.concat(all_exploded_dfs).drop_duplicates()
    
    # Remove original column if not keeping it
    if not keep_original:
        result_df = result_df.drop(columns=[nested_column])
    
    return result_df

def get_batch_generator(iterable: Iterable[T], batch_size: int) -> Generator[List[T], None, None]:
    """
    Enhanced batch generator with memory optimization
    
    Args:
        iterable: The iterable to batch
        batch_size: Batch size
        
    Returns:
        generator: Batch generator
    """
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def process_in_parallel(
    items: Iterable[T],
    process_func: Callable[[T], S],
    n_jobs: int = -1,
    batch_size: Optional[int] = None,
    desc: Optional[str] = None,
    verbose: int = 0
) -> List[S]:
    """
    Process items in parallel with progress tracking
    
    Args:
        items: Items to process
        process_func: Function to apply to each item
        n_jobs: Number of parallel jobs
        batch_size: Batch size per job
        desc: Description for progress tracking
        verbose: Verbosity level
        
    Returns:
        list: Processed items
    """
    try:
        # Convert to list if needed
        if not isinstance(items, list):
            items = list(items)
            
        # Check if items are empty
        if not items:
            return []
            
        # Progress tracking if tqdm is available
        if desc:
            try:
                from tqdm import tqdm
                logger.info(f"Processing {len(items)} items in parallel: {desc}")
                results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                    delayed(process_func)(item) for item in tqdm(items, desc=desc)
                )
            except ImportError:
                logger.info(f"Processing {len(items)} items in parallel: {desc}")
                results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                    delayed(process_func)(item) for item in items
                )
        else:
            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(process_func)(item) for item in items
            )
            
        return results
            
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        # Fallback to serial processing
        logger.info("Falling back to serial processing")
        return [process_func(item) for item in items]

def anonymize_column(
    df: pd.DataFrame,
    column: str, 
    method: str = 'hash',
    salt: Optional[str] = None,
    keep_original: bool = False,
    suffix: str = '_anonymized'
) -> pd.DataFrame:
    """
    Enhanced data anonymization with original column preservation
    
    Args:
        df: DataFrame to process
        column: Column to anonymize
        method: Anonymization method ('hash', 'mask', 'categorical', 'noise', 'obfuscate')
        salt: Salt for hashing (if method is 'hash')
        keep_original: Whether to keep the original column
        suffix: Suffix for anonymized column if keeping original
        
    Returns:
        pd.DataFrame: DataFrame with anonymized column
    """
    import hashlib
    from datetime import datetime
    import random
    
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return df
        
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Determine output column name
    output_column = column if not keep_original else f"{column}{suffix}"
    
    # Choose anonymization method
    if method == 'hash':
        # Generate salt if not provided
        if salt is None:
            salt = datetime.now().isoformat()
            
        def hash_value(value):
            if pd.isna(value):
                return None
            hash_obj = hashlib.sha256(f"{value}{salt}".encode())
            return hash_obj.hexdigest()
            
        result_df[output_column] = result_df[column].apply(hash_value)
        
    elif method == 'mask':
        def mask_value(value):
            if pd.isna(value):
                return None
                
            value_str = str(value)
            if len(value_str) <= 4:
                return '*' * len(value_str)
            else:
                return value_str[:2] + '*' * (len(value_str) - 4) + value_str[-2:]
                
        result_df[output_column] = result_df[column].apply(mask_value)
        
    elif method == 'categorical':
        # Replace with category numbers
        categories = {val: i for i, val in enumerate(result_df[column].unique())}
        result_df[output_column] = result_df[column].map(categories)
        
    elif method == 'noise':
        # Add random noise to numeric values
        if pd.api.types.is_numeric_dtype(result_df[column]):
            # Calculate std for noise generation
            std = result_df[column].std() * 0.1
            
            # Add noise
            random.seed(0)  # For reproducibility
            noise = [random.gauss(0, std) for _ in range(len(result_df))]
            result_df[output_column] = result_df[column] + noise
        else:
            logger.warning(f"Cannot use 'noise' method on non-numeric column '{column}'")
            result_df[output_column] = result_df[column]
            
    elif method == 'obfuscate':
        # Replace sensitive values with similar but different values
        if pd.api.types.is_numeric_dtype(result_df[column]):
            # For numeric, round to reduce precision
            result_df[output_column] = result_df[column].round(-1)  # Round to nearest 10
        elif pd.api.types.is_datetime64_dtype(result_df[column]):
            # For dates, round to month
            result_df[output_column] = pd.to_datetime(result_df[column]).dt.to_period('M').dt.to_timestamp()
        else:
            # For strings, replace with first character and length
            def obfuscate_string(val):
                if pd.isna(val):
                    return None
                val_str = str(val)
                if len(val_str) <= 1:
                    return val_str
                return val_str[0] + "_" + str(len(val_str) - 1)
                
            result_df[output_column] = result_df[column].apply(obfuscate_string)
    
    else:
        logger.warning(f"Unknown anonymization method '{method}'")
        result_df[output_column] = result_df[column]
    
    # Remove original column if not keeping it
    if not keep_original:
        result_df = result_df.drop(columns=[column])
    
    return result_df

def extract_time_features(
    df: pd.DataFrame,
    date_column: str,
    features: Optional[List[str]] = None,
    drop_original: bool = False,
    prefix: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract time-based features from a datetime column
    
    Args:
        df: DataFrame containing the date column
        date_column: Name of column with datetime data
        features: List of features to extract (default: all)
        drop_original: Whether to drop the original date column
        prefix: Prefix for feature column names
        
    Returns:
        pd.DataFrame: DataFrame with time features added
    """
    if date_column not in df.columns:
        logger.warning(f"Column '{date_column}' not found in DataFrame")
        return df
        
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_dtype(result_df[date_column]):
        try:
            result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting '{date_column}' to datetime: {e}")
            return df
    
    # Set prefix if not provided
    if prefix is None:
        prefix = f"{date_column}_"
        
    # Default features to extract if not specified
    all_features = [
        'year', 'quarter', 'month', 'day', 'day_of_week', 'day_of_year',
        'week_of_year', 'hour', 'minute', 'is_month_start', 'is_month_end',
        'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end',
        'is_weekend'
    ]
    
    # Use specified features or all features
    extract_features = features if features else all_features
    
    # Extract each feature
    for feature in extract_features:
        try:
            feature_name = f"{prefix}{feature}"
            
            if feature == 'year':
                result_df[feature_name] = result_df[date_column].dt.year
            elif feature == 'quarter':
                result_df[feature_name] = result_df[date_column].dt.quarter
            elif feature == 'month':
                result_df[feature_name] = result_df[date_column].dt.month
            elif feature == 'day':
                result_df[feature_name] = result_df[date_column].dt.day
            elif feature == 'day_of_week':
                result_df[feature_name] = result_df[date_column].dt.dayofweek
            elif feature == 'day_of_year':
                result_df[feature_name] = result_df[date_column].dt.dayofyear
            elif feature == 'week_of_year':
                result_df[feature_name] = result_df[date_column].dt.isocalendar().week
            elif feature == 'hour':
                result_df[feature_name] = result_df[date_column].dt.hour
            elif feature == 'minute':
                result_df[feature_name] = result_df[date_column].dt.minute
            elif feature == 'is_month_start':
                result_df[feature_name] = result_df[date_column].dt.is_month_start.astype(int)
            elif feature == 'is_month_end':
                result_df[feature_name] = result_df[date_column].dt.is_month_end.astype(int)
            elif feature == 'is_quarter_start':
                result_df[feature_name] = result_df[date_column].dt.is_quarter_start.astype(int)
            elif feature == 'is_quarter_end':
                result_df[feature_name] = result_df[date_column].dt.is_quarter_end.astype(int)
            elif feature == 'is_year_start':
                result_df[feature_name] = result_df[date_column].dt.is_year_start.astype(int)
            elif feature == 'is_year_end':
                result_df[feature_name] = result_df[date_column].dt.is_year_end.astype(int)
            elif feature == 'is_weekend':
                result_df[feature_name] = result_df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
            else:
                logger.warning(f"Unknown time feature: {feature}")
                
        except Exception as e:
            logger.warning(f"Error extracting time feature '{feature}': {e}")
    
    # Drop original column if requested
    if drop_original:
        result_df = result_df.drop(columns=[date_column])
    
    return result_df

def identify_outliers(
    df: pd.DataFrame, 
    column: str,
    method: str = 'iqr',
    threshold: float = 1.5,
    return_mask: bool = False
) -> Union[pd.DataFrame, pd.Series]:
    """
    Identify outliers in a DataFrame column
    
    Args:
        df: DataFrame to process
        column: Column to check for outliers
        method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier classification
        return_mask: Whether to return a boolean mask instead of filtered DataFrame
        
    Returns:
        pd.DataFrame or pd.Series: Filtered DataFrame with outliers or boolean mask
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found in DataFrame")
        return df if not return_mask else pd.Series(False, index=df.index)
        
    if not pd.api.types.is_numeric_dtype(df[column]):
        logger.warning(f"Cannot identify outliers in non-numeric column '{column}'")
        return df if not return_mask else pd.Series(False, index=df.index)
    
    # Calculate outlier mask based on method
    outlier_mask = pd.Series(False, index=df.index)
    
    try:
        if method == 'iqr':
            # Use IQR method (Tukey's fences)
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            
        elif method == 'zscore':
            # Use Z-score method
            zscore = (df[column] - df[column].mean()) / df[column].std()
            outlier_mask = zscore.abs() > threshold
            
        elif method == 'modified_zscore':
            # Use modified Z-score method (more robust against outliers)
            median = df[column].median()
            mad = (df[column] - median).abs().median() * 1.4826  # Constant for normal distribution
            
            if mad == 0:
                # Handle case where MAD is zero
                logger.warning(f"MAD is zero for column '{column}', using standard Z-score")
                zscore = (df[column] - df[column].mean()) / df[column].std()
                outlier_mask = zscore.abs() > threshold
            else:
                modified_zscore = 0.6745 * (df[column] - median) / mad
                outlier_mask = modified_zscore.abs() > threshold
                
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            
    except Exception as e:
        logger.error(f"Error identifying outliers in column '{column}': {e}")
    
    # Return mask or filtered DataFrame
    if return_mask:
        return outlier_mask
    else:
        return df[outlier_mask]

def calculate_market_metrics(
    df: pd.DataFrame,
    price_col: str = 'price_usd',
    market_cap_col: str = 'market_cap',
    volume_col: str = 'volume_24h',
    prefix: str = 'market_',
    period_days: List[int] = [1, 7, 30]
) -> pd.DataFrame:
    """
    Calculate cryptocurrency market metrics like volatility, volume ratio, etc.
    
    Args:
        df: DataFrame with time series market data
        price_col: Name of price column
        market_cap_col: Name of market cap column
        volume_col: Name of trading volume column
        prefix: Prefix for new metric columns
        period_days: List of periods in days to calculate metrics for
        
    Returns:
        pd.DataFrame: DataFrame with added market metrics
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Verify required columns exist
    required_cols = [col for col in [price_col, market_cap_col, volume_col] 
                   if col in result_df.columns]
    
    if not required_cols:
        logger.warning("No required market data columns found")
        return result_df
    
    try:
        # 1. Calculate price volatility (standard deviation of returns)
        if price_col in result_df.columns:
            # Calculate returns
            result_df[f'{prefix}returns'] = result_df[price_col].pct_change()
            
            # Calculate volatility for different periods
            for days in period_days:
                if len(result_df) >= days:
                    result_df[f'{prefix}volatility_{days}d'] = result_df[f'{prefix}returns'].rolling(days).std() * np.sqrt(days)
        
        # 2. Calculate volume to market cap ratio
        if volume_col in result_df.columns and market_cap_col in result_df.columns:
            result_df[f'{prefix}volume_to_mcap'] = result_df[volume_col] / result_df[market_cap_col]
        
        # 3. Calculate moving averages for price
        if price_col in result_df.columns:
            for days in period_days:
                result_df[f'{prefix}ma_{days}d'] = result_df[price_col].rolling(days).mean()
        
        # 4. Calculate RSI (Relative Strength Index)
        if price_col in result_df.columns:
            for days in period_days:
                if days <= len(result_df):
                    # Calculate price changes
                    delta = result_df[price_col].diff()
                    
                    # Get gains and losses
                    gains = delta.where(delta > 0, 0)
                    losses = -delta.where(delta < 0, 0)
                    
                    # Calculate average gains and losses
                    avg_gain = gains.rolling(window=days).mean()
                    avg_loss = losses.rolling(window=days).mean()
                    
                    # Calculate RS and RSI
                    rs = avg_gain / avg_loss
                    result_df[f'{prefix}rsi_{days}d'] = 100 - (100 / (1 + rs))
        
        # 5. Calculate MACD (Moving Average Convergence Divergence)
        if price_col in result_df.columns and len(result_df) >= 26:
            # Common MACD parameters
            ema12 = result_df[price_col].ewm(span=12, adjust=False).mean()
            ema26 = result_df[price_col].ewm(span=26, adjust=False).mean()
            result_df[f'{prefix}macd'] = ema12 - ema26
            result_df[f'{prefix}macd_signal'] = result_df[f'{prefix}macd'].ewm(span=9, adjust=False).mean()
            result_df[f'{prefix}macd_hist'] = result_df[f'{prefix}macd'] - result_df[f'{prefix}macd_signal']
        
        # 6. Calculate price momentum
        if price_col in result_df.columns:
            for days in period_days:
                if len(result_df) > days:
                    result_df[f'{prefix}momentum_{days}d'] = result_df[price_col] / result_df[price_col].shift(days) - 1
        
        # 7. Calculate Bollinger Bands
        if price_col in result_df.columns:
            for days in period_days:
                if len(result_df) >= days:
                    # Calculate moving average
                    ma = result_df[price_col].rolling(days).mean()
                    # Calculate standard deviation
                    std = result_df[price_col].rolling(days).std()
                    # Calculate Bollinger Bands
                    result_df[f'{prefix}bb_upper_{days}d'] = ma + (std * 2)
                    result_df[f'{prefix}bb_lower_{days}d'] = ma - (std * 2)
                    # Calculate %B (position within Bollinger Bands)
                    result_df[f'{prefix}bb_pct_{days}d'] = (result_df[price_col] - result_df[f'{prefix}bb_lower_{days}d']) / (result_df[f'{prefix}bb_upper_{days}d'] - result_df[f'{prefix}bb_lower_{days}d'])
        
    except Exception as e:
        logger.error(f"Error calculating market metrics: {e}")
    
    return result_df

def calculate_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'pearson',
    min_periods: int = 1,
    numeric_only: bool = True
) -> pd.DataFrame:
    """
    Calculate correlation matrix with enhanced options
    
    Args:
        df: DataFrame containing data
        columns: List of columns to include (None for all)
        method: Correlation method ('pearson', 'kendall', 'spearman')
        min_periods: Minimum number of observations required
        numeric_only: Whether to include only numeric columns
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    # Select columns if specified
    if columns:
        # Check which specified columns exist
        existing_columns = [col for col in columns if col in df.columns]
        if not existing_columns:
            logger.warning("None of the specified columns exist in DataFrame")
            return pd.DataFrame()
            
        data = df[existing_columns]
    else:
        data = df
    
    # Select only numeric columns if required
    if numeric_only:
        numeric_columns = data.select_dtypes(include=['number']).columns
        if not len(numeric_columns):
            logger.warning("No numeric columns found")
            return pd.DataFrame()
            
        data = data[numeric_columns]
    
    try:
        # Calculate correlation matrix
        corr_matrix = data.corr(method=method, min_periods=min_periods)
        
        return corr_matrix
        
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return pd.DataFrame()

def create_pivot_table(
    df: pd.DataFrame,
    index: Union[str, List[str]],
    columns: Optional[Union[str, List[str]]] = None,
    values: Optional[Union[str, List[str]]] = None,
    aggfunc: Union[str, Callable, Dict[str, Union[str, Callable]]] = 'mean',
    fill_value: Optional[Any] = None,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Create a pivot table with enhanced options and error handling
    
    Args:
        df: DataFrame to pivot
        index: Column(s) to use as index
        columns: Column(s) to use as columns
        values: Column(s) to aggregate
        aggfunc: Aggregation function(s)
        fill_value: Value to fill NaNs with
        dropna: Whether to drop columns with all NaN values
        
    Returns:
        pd.DataFrame: Pivot table
    """
    try:
        # Verify columns exist
        index_cols = [index] if isinstance(index, str) else index
        for col in index_cols:
            if col not in df.columns:
                logger.error(f"Index column '{col}' not found in DataFrame")
                return pd.DataFrame()
        
        if columns is not None:
            col_cols = [columns] if isinstance(columns, str) else columns
            for col in col_cols:
                if col not in df.columns:
                    logger.error(f"Column column '{col}' not found in DataFrame")
                    return pd.DataFrame()
        
        if values is not None:
            val_cols = [values] if isinstance(values, str) else values
            for col in val_cols:
                if col not in df.columns:
                    logger.error(f"Values column '{col}' not found in DataFrame")
                    return pd.DataFrame()
        
        # Handle aggregation function
        agg_funcs = {
            'mean': np.mean,
            'median': np.median,
            'sum': np.sum,
            'count': len,
            'min': np.min,
            'max': np.max,
            'std': np.std,
            'var': np.var,
            'first': lambda x: x.iloc[0] if len(x) > 0 else None,
            'last': lambda x: x.iloc[-1] if len(x) > 0 else None
        }
        
        if isinstance(aggfunc, str) and aggfunc in agg_funcs:
            aggfunc = agg_funcs[aggfunc]
        
        # Create pivot table
        pivot = pd.pivot_table(
            df, 
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=fill_value,
            dropna=dropna
        )
        
        return pivot
        
    except Exception as e:
        logger.error(f"Error creating pivot table: {e}")
        return pd.DataFrame()

def create_time_windows(
    df: pd.DataFrame,
    date_column: str,
    window_size: str = '1D',
    date_format: Optional[str] = None,
    groupby_columns: Optional[List[str]] = None,
    agg_dict: Optional[Dict[str, Union[str, Callable]]] = None
) -> pd.DataFrame:
    """
    Create time-based aggregation windows
    
    Args:
        df: DataFrame containing time data
        date_column: Name of the date/time column
        window_size: Size of the window ('1D', '1W', '1M', etc.)
        date_format: Format for parsing dates
        groupby_columns: Additional columns to group by
        agg_dict: Dictionary mapping columns to aggregation functions
        
    Returns:
        pd.DataFrame: Aggregated DataFrame
    """
    if date_column not in df.columns:
        logger.warning(f"Column '{date_column}' not found in DataFrame")
        return df
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    try:
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_dtype(result_df[date_column]):
            if date_format:
                result_df[date_column] = pd.to_datetime(result_df[date_column], format=date_format, errors='coerce')
            else:
                result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
        
        # Handle NaT values
        if result_df[date_column].isna().any():
            logger.warning(f"Found {result_df[date_column].isna().sum()} NaT values in '{date_column}'")
            result_df = result_df.dropna(subset=[date_column])
        
        # Create time window column
        result_df['time_window'] = result_df[date_column].dt.to_period(window_size)
        
        # Define default aggregation if not provided
        if agg_dict is None:
            # Auto-detect appropriate aggregations based on column types
            agg_dict = {}
            for col in result_df.columns:
                if col == date_column or col == 'time_window':
                    continue
                
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    # For numeric columns, use common financial aggregations
                    agg_dict[col] = ['mean', 'min', 'max', 'std']
                elif pd.api.types.is_string_dtype(result_df[col]):
                    # For string columns, use mode and count unique
                    agg_dict[col] = [
                        ('mode', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
                        ('nunique', 'nunique')
                    ]
        
        # Group by time window and additional columns if provided
        groupby_cols = ['time_window']
        if groupby_columns:
            groupby_cols.extend(groupby_columns)
        
        # Perform aggregation
        aggregated = result_df.groupby(groupby_cols).agg(agg_dict)
        
        # Flatten multi-level column names if needed
        if isinstance(aggregated.columns, pd.MultiIndex):
            aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        
        # Reset index for easier use
        aggregated = aggregated.reset_index()
        
        # Convert Period to datetime
        aggregated['time_window'] = aggregated['time_window'].dt.to_timestamp()
        
        return aggregated
        
    except Exception as e:
        logger.error(f"Error creating time windows: {e}")
        return df

def encode_categorical_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'onehot',
    drop_original: bool = True,
    max_unique: int = 20
) -> pd.DataFrame:
    """
    Encode categorical features with various methods
    
    Args:
        df: DataFrame containing categorical features
        columns: List of columns to encode or None for auto-detection
        method: Encoding method ('onehot', 'ordinal', 'binary', 'frequency', 'target')
        drop_original: Whether to drop original columns
        max_unique: Maximum number of unique values for auto-detection
        
    Returns:
        pd.DataFrame: DataFrame with encoded features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    try:
        # Auto-detect categorical columns if not specified
        if columns is None:
            columns = []
            for col in result_df.columns:
                if pd.api.types.is_object_dtype(result_df[col]):
                    # For object columns, check if they're categorical-like
                    unique_count = result_df[col].nunique()
                    if 1 < unique_count <= max_unique:
                        columns.append(col)
                elif pd.api.types.is_categorical_dtype(result_df[col]):
                    columns.append(col)
        
        # Filter to include only existing columns
        columns = [col for col in columns if col in result_df.columns]
        
        if not columns:
            logger.warning("No categorical columns found to encode")
            return result_df
        
        # Apply encoding based on method
        if method == 'onehot':
            # One-hot encoding
            from sklearn.preprocessing import OneHotEncoder
            
            for col in columns:
                try:
                    # Get unique values with handling for NaNs
                    values = result_df[col].dropna().unique()
                    
                    # Skip if too many unique values
                    if len(values) > max_unique:
                        logger.warning(f"Column '{col}' has {len(values)} unique values, skipping one-hot encoding")
                        continue
                    
                    # Initialize encoder
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    
                    # Reshape for fitting
                    encoded = encoder.fit_transform(result_df[[col]].fillna('missing'))
                    
                    # Get feature names
                    feature_names = [f"{col}_{val}" for val in encoder.categories_[0]]
                    
                    # Add encoded columns
                    for i, feat_name in enumerate(feature_names):
                        result_df[feat_name] = encoded[:, i]
                    
                    # Drop original column if requested
                    if drop_original:
                        result_df = result_df.drop(columns=[col])
                except Exception as e:
                    logger.warning(f"Error one-hot encoding column '{col}': {e}")
                
        elif method == 'ordinal':
            # Ordinal encoding
            from sklearn.preprocessing import OrdinalEncoder
            
            for col in columns:
                try:
                    # Initialize encoder
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    
                    # Encode
                    result_df[f"{col}_encoded"] = encoder.fit_transform(result_df[[col]].fillna('missing'))
                    
                    # Drop original column if requested
                    if drop_original:
                        result_df = result_df.drop(columns=[col])
                except Exception as e:
                    logger.warning(f"Error ordinal encoding column '{col}': {e}")
                
        elif method == 'binary':
            # Binary encoding (Base-2 representation)
            for col in columns:
                try:
                    # Get mapping of categories to ordinal values
                    ordinal_map = {val: i for i, val in enumerate(result_df[col].dropna().unique())}
                    
                    # Apply mapping
                    ordinal_encoded = result_df[col].map(ordinal_map)
                    
                    # Convert to binary representation
                    max_val = max(ordinal_map.values())
                    num_binary_cols = max(1, (max_val + 1).bit_length() - 1)
                    
                    for i in range(num_binary_cols):
                        result_df[f"{col}_bin_{i}"] = ((ordinal_encoded.fillna(-1) >> i) & 1).astype(int)
                    
                    # Drop original column if requested
                    if drop_original:
                        result_df = result_df.drop(columns=[col])
                except Exception as e:
                    logger.warning(f"Error binary encoding column '{col}': {e}")
                
        elif method == 'frequency':
            # Frequency encoding (replace with frequency)
            for col in columns:
                try:
                    freq_map = result_df[col].value_counts(normalize=True)
                    result_df[f"{col}_freq"] = result_df[col].map(freq_map)
                    
                    # Drop original column if requested
                    if drop_original:
                        result_df = result_df.drop(columns=[col])
                except Exception as e:
                    logger.warning(f"Error frequency encoding column '{col}': {e}")
                
        elif method == 'target':
            # Target encoding requires a target column
            logger.warning("Target encoding requires specifying a target column. Using frequency encoding instead.")
            
            # Fall back to frequency encoding
            for col in columns:
                try:
                    freq_map = result_df[col].value_counts(normalize=True)
                    result_df[f"{col}_freq"] = result_df[col].map(freq_map)
                    
                    # Drop original column if requested
                    if drop_original:
                        result_df = result_df.drop(columns=[col])
                except Exception as e:
                    logger.warning(f"Error frequency encoding column '{col}': {e}")
                
        else:
            logger.warning(f"Unknown encoding method: {method}")
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error encoding categorical features: {e}")
        return df

def detect_market_events(
    price_df: pd.DataFrame, 
    price_col: str = 'price_usd',
    date_col: str = 'date',
    window_size: int = 7,
    threshold_std: float = 2.0
) -> pd.DataFrame:
    """
    Detect significant market events (pumps, dumps, high volatility)
    
    Args:
        price_df: DataFrame with price history
        price_col: Name of price column
        date_col: Name of date column
        window_size: Rolling window size for volatility calculation
        threshold_std: Number of standard deviations for event detection
        
    Returns:
        pd.DataFrame: DataFrame with detected events
    """
    if price_col not in price_df.columns:
        logger.warning(f"Price column '{price_col}' not found in DataFrame")
        return pd.DataFrame()
        
    if date_col not in price_df.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        return pd.DataFrame()
    
    # Make a copy for processing
    df = price_df.copy()
    
    try:
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Calculate returns
        df['return'] = df[price_col].pct_change()
        
        # Calculate rolling volatility (standard deviation of returns)
        df['volatility'] = df['return'].rolling(window=window_size).std()
        
        # Calculate moving average and standard deviation for event detection
        df['price_ma'] = df[price_col].rolling(window=window_size).mean()
        df['price_std'] = df[price_col].rolling(window=window_size).std()
        
        # Calculate upper and lower bounds
        df['upper_bound'] = df['price_ma'] + (threshold_std * df['price_std'])
        df['lower_bound'] = df['price_ma'] - (threshold_std * df['price_std'])
        
        # Detect pump events (price significantly above moving average)
        df['pump'] = (df[price_col] > df['upper_bound']).astype(int)
        
        # Detect dump events (price significantly below moving average)
        df['dump'] = (df[price_col] < df['lower_bound']).astype(int)
        
        # Detect high volatility events
        vol_mean = df['volatility'].mean()
        vol_std = df['volatility'].std()
        df['high_volatility'] = (df['volatility'] > vol_mean + (threshold_std * vol_std)).astype(int)
        
        # Detect consecutive days of price increase
        df['price_up'] = (df[price_col] > df[price_col].shift(1)).astype(int)
        df['consecutive_up'] = df['price_up'].rolling(window=3).sum()
        df['price_streak_up'] = (df['consecutive_up'] >= 3).astype(int)
        
        # Detect consecutive days of price decrease
        df['price_down'] = (df[price_col] < df[price_col].shift(1)).astype(int)
        df['consecutive_down'] = df['price_down'].rolling(window=3).sum()
        df['price_streak_down'] = (df['consecutive_down'] >= 3).astype(int)
        
        # Detect peak and trough patterns
        df['local_peak'] = ((df[price_col] > df[price_col].shift(1)) & 
                           (df[price_col] > df[price_col].shift(-1))).astype(int)
        df['local_trough'] = ((df[price_col] < df[price_col].shift(1)) & 
                             (df[price_col] < df[price_col].shift(-1))).astype(int)
        
        # Detect breakout (price crossing above moving average)
        df['above_ma'] = (df[price_col] > df['price_ma']).astype(int)
        df['breakout'] = ((df['above_ma'] == 1) & (df['above_ma'].shift(1) == 0)).astype(int)
        
        # Detect breakdown (price crossing below moving average)
        df['below_ma'] = (df[price_col] < df['price_ma']).astype(int)
        df['breakdown'] = ((df['below_ma'] == 1) & (df['below_ma'].shift(1) == 0)).astype(int)
        
        # Detect acceleration in price movements
        df['return_change'] = df['return'].diff()
        df['acceleration'] = df['return_change'].rolling(window=3).mean()
        df['price_acceleration'] = (df['acceleration'].abs() > df['acceleration'].abs().mean() + 
                                  threshold_std * df['acceleration'].abs().std()).astype(int)
        
        # Detect major volume spikes
        if 'volume_24h' in df.columns:
            df['volume_ma'] = df['volume_24h'].rolling(window=window_size).mean()
            df['volume_std'] = df['volume_24h'].rolling(window=window_size).std()
            df['volume_spike'] = (df['volume_24h'] > df['volume_ma'] + threshold_std * df['volume_std']).astype(int)
        
        # Consolidate events into a single event type column
        event_map = {
            0: 'normal',
            1: 'pump',
            2: 'dump',
            3: 'high_volatility',
            4: 'streak_up',
            5: 'streak_down',
            6: 'breakout',
            7: 'breakdown',
            8: 'volume_spike'
        }
        
        # Create an event type column with priority
        df['event_type'] = 0  # Default to normal
        
        # Apply events with priority (higher numbers take precedence)
        for event, value in [('pump', 1), ('dump', 2), ('high_volatility', 3), 
                           ('price_streak_up', 4), ('price_streak_down', 5),
                           ('breakout', 6), ('breakdown', 7), ('volume_spike', 8)]:
            if event in df.columns:
                df.loc[df[event] == 1, 'event_type'] = value
        
        # Map numeric event types to strings
        df['event_name'] = df['event_type'].map(event_map)
        
        # Calculate event duration
        df['event_start'] = 0
        for event in ['pump', 'dump', 'high_volatility']:
            if event in df.columns:
                # Mark start of a new event
                df.loc[(df[event] == 1) & (df[event].shift(1) == 0), 'event_start'] = 1
        
        # Calculate event duration by creating a group ID for each event
        df['event_group'] = df['event_start'].cumsum()
        
        # Calculate event duration for each group
        event_durations = df.groupby('event_group').size()
        df['event_duration'] = df['event_group'].map(event_durations)
        
        # Only keep event duration for rows where an event is happening
        df.loc[df['event_type'] == 0, 'event_duration'] = 0
        
        return df
        
    except Exception as e:
        logger.error(f"Error detecting market events: {e}")
        logger.error(traceback.format_exc())
        return price_df
    
if __name__ == "__main__":
    # Test functionality
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    print("Testing data_utils.py functions...")
    
    # Test 1: detect_market_events
    print("\n1. Testing detect_market_events:")
    # Create sample price data
    dates = pd.date_range(start='2023-01-01', periods=100)
    
    # Generate a synthetic price series with some patterns
    # Menggunakan Generator baru dengan seed 42
    rng = np.random.default_rng(42)
    base_price = 1000
    noise = rng.normal(0, 20, 100)
    trend = np.linspace(0, 200, 100)  # Upward trend
    # Add two pump events
    noise[30:35] += 100
    noise[70:75] += 150
    # Add one dump event
    noise[50:55] -= 130
    
    prices = base_price + trend + noise.cumsum()
    
    sample_df = pd.DataFrame({
        'date': dates,
        'price_usd': prices,
        'volume_24h': rng.normal(10000, 5000, 100).clip(min=0)
    })
    
    # Detect market events
    events_df = detect_market_events(sample_df)
    
    # Print summary
    print(f"Detected {events_df['pump'].sum()} pump events")
    print(f"Detected {events_df['dump'].sum()} dump events")
    print(f"Detected {events_df['high_volatility'].sum()} high volatility events")
    print(f"Detected {events_df['price_streak_up'].sum()} price streak up events")
    print(f"Detected {events_df['price_streak_down'].sum()} price streak down events")
    
    # Test 2: handle_missing_values
    print("\n2. Testing handle_missing_values:")
    # Create sample data with missing values
    df_missing = pd.DataFrame({
        'numeric': [1, 2, None, 4, 5],
        'categorical': ['A', 'B', None, 'B', 'C'],
        'date': pd.to_datetime(['2023-01-01', None, '2023-01-03', '2023-01-04', '2023-01-05'])
    })
    print("Before handling missing values:")
    print(df_missing.isna().sum())
    
    # Apply missing value handling
    df_clean = handle_missing_values(df_missing)
    print("After handling missing values:")
    print(df_clean.isna().sum())
    
    # Test 3: normalize_column
    print("\n3. Testing normalize_column:")
    # Create sample data
    df_normalize = pd.DataFrame({
        'value': [100, 200, 300, 400, 500],
        'outlier': [10, 20, 1000, 30, 40]
    })
    print("Original data:")
    print(df_normalize.describe())
    
    # Normalize columns
    df_norm = normalize_column(df_normalize, 'value', method='minmax')
    df_norm = normalize_column(df_norm, 'outlier', method='robust', handle_outliers=True)
    print("After normalization:")
    print(df_norm.describe())
    
    # Test 4: calculate_market_metrics
    print("\n4. Testing calculate_market_metrics:")

    # Menggunakan Generator baru dengan seed 44
    rng = np.random.default_rng(44)

    # Create sample market data
    market_dates = pd.date_range(start='2023-01-01', periods=30)
    market_prices = rng.normal(0, 1, 30).cumsum() + 100
    
    market_df = pd.DataFrame({
        'date': market_dates,
        'price_usd': market_prices,
        'market_cap': market_prices * 1000000,  # Market cap correlates with price
        'volume_24h': rng.normal(500000, 100000, 30).clip(min=0)
    })
    
    # Calculate metrics
    metrics_df = calculate_market_metrics(market_df)
    print(f"Added {len(metrics_df.columns) - len(market_df.columns)} new market metrics")
    print("New columns:", [col for col in metrics_df.columns if col not in market_df.columns])
    
    # Test 5: extract_time_features
    print("\n5. Testing extract_time_features:")

    # Menggunakan Generator baru dengan seed 42
    rng = np.random.default_rng(42)

    # Create sample datetime data
    time_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'value': rng.normal(0, 1, 100).cumsum()
    })
    
    # Extract time features
    time_features_df = extract_time_features(time_df, 'timestamp', features=['year', 'month', 'day_of_week', 'is_weekend'])
    print(f"Extracted {len(time_features_df.columns) - len(time_df.columns)} time features")
    print("Time feature columns:", [col for col in time_features_df.columns if col.startswith('timestamp_')])
    
    # Test 6: identify_outliers
    print("\n6. Testing identify_outliers:")

    rng_normal = np.random.default_rng(42)
    rng_outliers = np.random.default_rng(43)

    # Create sample data with outliers
    outlier_df = pd.DataFrame({
        'normal': rng_normal.normal(0, 1, 100),
        'with_outliers': rng_outliers.normal(0, 1, 100)
    })
    # Add outliers
    outlier_df.loc[10:15, 'with_outliers'] += 5
    outlier_df.loc[50:55, 'with_outliers'] -= 5
    
    # Identify outliers
    outlier_mask = identify_outliers(outlier_df, 'with_outliers', method='zscore', return_mask=True)
    print(f"Identified {outlier_mask.sum()} outliers with z-score method")
    
    # Test 7: convert_json_columns
    print("\n7. Testing convert_json_columns:")
    # Create sample data with JSON strings
    json_df = pd.DataFrame({
        'platforms': ['{"ethereum":"0x1234", "binance":"0x5678"}', '{"solana":"0xabcd"}', None],
        'categories': ['["defi", "lending"]', '["nft", "gaming"]', '[]']
    })
    
    # Convert JSON columns
    converted_df = convert_json_columns(json_df, ['platforms', 'categories'])
    print("JSON columns converted successfully:")
    for i, row in converted_df.iterrows():
        print(f"Row {i}:")
        print(f"  Platforms: {type(row['platforms']).__name__} with {len(row['platforms'])} items")
        print(f"  Categories: {type(row['categories']).__name__} with {len(row['categories'])} items")
    
    print("\nData utils testing completed successfully!")