"""
Central logging configuration for recommendation engine
"""
import os
import logging
import shutil
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
import glob

# Global variable to track if logging has been initialized
_logging_initialized = False
_current_log_file = None

def setup_logging(log_dir=None, default_level=logging.INFO, service_name=None, 
                  max_size_mb=50, backup_count=5, clean_old_logs=True):
    """
    Setup logging configuration with enhanced features
    
    Args:
        log_dir (str): Directory to store logs
        default_level: Default logging level
        service_name (str): Nama service untuk file log terpisah
        max_size_mb (int): Maximum size of log file in MB before rotation
        backup_count (int): Number of backup files to keep
        clean_old_logs (bool): Whether to clean logs older than 30 days
        
    Returns:
        logging.Logger: Configured root logger
    """
    global _logging_initialized, _current_log_file
    
    # Skip jika sudah diinisialisasi
    if _logging_initialized:
        return logging.getLogger()
    
    # Tentukan path logs
    if log_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(current_dir, 'logs')
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Log format - extended with process ID and thread name
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(process)d:%(threadName)s] - %(message)s'
    
    # Timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Log filename - custom nama berdasarkan service
    if service_name:
        log_file = os.path.join(log_dir, f'{service_name}_{timestamp}.log')
    else:
        log_file = os.path.join(log_dir, f'recommendation_engine_{timestamp}.log')
    
    _current_log_file = log_file
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(default_level)
    
    # Remove existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handlers with rotation
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_size_mb * 1024 * 1024,  # Convert MB to bytes
        backupCount=backup_count
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # Clean old logs if requested
    if clean_old_logs:
        clean_old_log_files(log_dir, days=30)
    
    # Tandai logging sudah diinisialisasi
    _logging_initialized = True
    
    # Buat symlink ke log terbaru untuk akses mudah
    latest_link = os.path.join(log_dir, "latest.log")
    try:
        if os.path.exists(latest_link):
            os.remove(latest_link)
        if hasattr(os, 'symlink'):  # Windows might not have symlink capability
            os.symlink(os.path.basename(log_file), latest_link)
    except (OSError, AttributeError):
        # Skip symlink creation if not supported
        pass
    
    # Test logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    # Get level name correctly without using getLevelName
    level_name = _get_level_name(default_level)
    logger.info(f"Log level: {level_name}")
    
    return root_logger

def get_logger(name):
    """
    Get a logger with the specified name, and setup logging if not already done
    
    Args:
        name (str): Name for the logger
        
    Returns:
        logging.Logger: Logger object
    """
    # Pastikan logging sudah disetup
    if not _logging_initialized:
        setup_logging()
        
    return logging.getLogger(name)

def clean_old_log_files(log_dir, days=30):
    """
    Clean log files older than specified days
    
    Args:
        log_dir (str): Directory containing log files
        days (int): Number of days to keep logs
    """
    try:
        now = datetime.now()
        cutoff = now - timedelta(days=days)
        
        # Get all log files
        log_files = glob.glob(os.path.join(log_dir, "*.log"))
        
        # Add backup log files (rotated logs)
        log_files.extend(glob.glob(os.path.join(log_dir, "*.log.*")))
        
        # Check each file's modification time
        for log_file in log_files:
            # Skip latest.log symlink
            if os.path.basename(log_file) == "latest.log":
                continue
                
            # Get file's last modification time
            if os.path.isfile(log_file):
                mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
                
                # Remove if older than cutoff
                if mtime < cutoff:
                    os.remove(log_file)
                    
        # Log cleanup info
        logger = logging.getLogger(__name__)
        logger.info(f"Cleaned log files older than {days} days from {log_dir}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error cleaning old log files: {e}")

def get_current_log_file():
    """
    Get the path to the current log file
    
    Returns:
        str: Path to current log file or None if logging not initialized
    """
    return _current_log_file

def set_log_level(level):
    """
    Change the logging level dynamically
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    if _logging_initialized:
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(level)
            
        logger = logging.getLogger(__name__)
        level_name = _get_level_name(level)
        logger.info(f"Log level changed to: {level_name}")

def _get_level_name(level):
    """
    Convert a logging level to its name safely
    
    Args:
        level: Logging level integer
        
    Returns:
        str: Name of the logging level
    """
    # Map logging levels to their names using hardcoded mapping
    # This avoids using the deprecated getLevelName function
    level_names = {
        logging.CRITICAL: "CRITICAL",
        logging.ERROR: "ERROR",
        logging.WARNING: "WARNING",
        logging.INFO: "INFO",
        logging.DEBUG: "DEBUG",
        logging.NOTSET: "NOTSET"
    }
    
    return level_names.get(level, f"Level {level}")