"""
Enhanced utilities for handling API requests with advanced features:
- Async request support
- Improved caching
- Adaptive rate limiting
- Session management
- Enhanced error handling
"""

import os
import sys
import requests
import aiohttp
import asyncio
import time
import json
import logging
import hashlib
import datetime
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, TypeVar, Generic
from dataclasses import dataclass
from functools import wraps
import backoff
from urllib.parse import urljoin, urlencode
import random
import inspect
from contextlib import contextmanager
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Setup logging
from central_logging import get_logger
logger = get_logger(__name__)

# Type variables for improved type hints
T = TypeVar('T')
R = TypeVar('R')

# Custom exceptions for better error handling
class APIRateLimitError(Exception):
    """Custom exception for API rate limiting situations"""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(message)

class APIResponseError(Exception):
    """Custom exception for API response errors with status code details"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(message)

class APITimeoutError(Exception):
    """Custom exception for API timeout situations"""
    pass

class APIConnectionError(Exception):
    """Custom exception for API connection problems"""
    pass

class APIAuthError(Exception):
    """Custom exception for API authentication issues"""
    pass

@dataclass
class APIResponse(Generic[T]):
    """Container for API responses with metadata"""
    data: T
    status_code: int
    headers: Dict[str, str]
    request_time: float
    success: bool
    cached: bool = False
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[int] = None
    
    @property
    def is_rate_limited(self) -> bool:
        """Check if response indicates rate limiting"""
        return self.status_code == 429 or (self.rate_limit_remaining is not None and self.rate_limit_remaining <= 0)

def create_requests_session(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: List[int] = [500, 502, 503, 504],
    pool_connections: int = 10,
    pool_maxsize: int = 10
) -> requests.Session:
    """
    Create a requests Session with retry capabilities and connection pooling
    
    Args:
        retries: Number of retries for failed requests
        backoff_factor: Backoff factor for retry delay
        status_forcelist: HTTP status codes that should trigger a retry
        pool_connections: Number of connection pools to cache
        pool_maxsize: Maximum number of connections to save in the pool
        
    Returns:
        requests.Session: Configured session object
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
    )
    
    # Add retry adapter to session
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize
    )
    
    # Mount adapter for both HTTP and HTTPS
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Global session for reuse
_GLOBAL_SESSION = create_requests_session()

def adaptive_rate_limit_delay(
    response: requests.Response,
    default_delay: float = 1.0,
    min_remaining_threshold: int = 10
) -> float:
    """
    Calculate adaptive delay based on rate limit headers
    
    Args:
        response: API response
        default_delay: Default delay if no rate limit info
        min_remaining_threshold: Threshold for remaining requests
        
    Returns:
        float: Delay in seconds
    """
    # Check for standard rate limit headers
    remaining = None
    reset = None
    
    # Try different common header formats
    for header in ['X-RateLimit-Remaining', 'RateLimit-Remaining', 'X-Rate-Limit-Remaining', 'rate-limit-remaining']:
        if header in response.headers:
            try:
                remaining = int(response.headers[header])
                break
            except (ValueError, TypeError):
                pass
    
    for header in ['X-RateLimit-Reset', 'RateLimit-Reset', 'X-Rate-Limit-Reset', 'rate-limit-reset']:
        if header in response.headers:
            try:
                reset = int(response.headers[header])
                break
            except (ValueError, TypeError):
                pass
    
    # If Retry-After header exists, use it
    if 'Retry-After' in response.headers:
        try:
            return float(response.headers['Retry-After'])
        except (ValueError, TypeError):
            pass
    
    # If we have both remaining and reset, calculate adaptive delay
    if remaining is not None and reset is not None:
        now = time.time()
        time_to_reset = max(0, reset - now)
        
        if remaining <= 0:
            # No requests left, wait until reset plus small buffer
            return time_to_reset + 0.5
        elif remaining < min_remaining_threshold:
            # Few requests left, use proportional delay
            return time_to_reset / remaining
            
    # Use default delay with small random jitter for load distribution
    return default_delay * (0.8 + 0.4 * random.random())  # 0.8x to 1.2x of default

def cache_key(url: str, params: Dict[str, Any], method: str, body: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate consistent cache key for request
    
    Args:
        url: Request URL
        params: URL parameters
        method: HTTP method
        body: Request body for POST/PUT
        
    Returns:
        str: Cache key
    """
    # Create canonical representation of request
    canonical = {
        'url': url,
        'params': params,
        'method': method
    }
    
    if body:
        canonical['body'] = body
    
    # Sort keys for consistency
    canonical_str = json.dumps(canonical, sort_keys=True)
    
    # Generate hash for the key
    return hashlib.md5(canonical_str.encode('utf-8')).hexdigest()

def is_cache_valid(cache_path: str, ttl: int) -> bool:
    """
    Check if cached response is still valid
    
    Args:
        cache_path: Path to cache file
        ttl: Time-to-live in seconds
        
    Returns:
        bool: True if cache is valid
    """
    if not os.path.exists(cache_path):
        return False
        
    # Check file modification time
    file_age = time.time() - os.path.getmtime(cache_path)
    return file_age < ttl

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, APIRateLimitError),
    max_tries=5,
    max_time=30
)
def make_api_request(
    url: str, 
    headers: Optional[Dict[str, str]] = None, 
    params: Optional[Dict[str, Any]] = None, 
    method: str = "GET",
    json_body: Optional[Dict[str, Any]] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
    verify: bool = True,
    stream: bool = False,
    parse_json: bool = True
) -> APIResponse:
    """
    Makes an API request with advanced error handling, retry logic, and session reuse
    
    Args:
        url: The API endpoint URL
        headers: Request headers
        params: URL parameters
        method: HTTP method (GET, POST, etc.)
        json_body: JSON body for POST/PUT requests
        session: Requests session (uses global session if None)
        timeout: Request timeout in seconds
        verify: Verify SSL certificates
        stream: Stream response content
        parse_json: Parse response as JSON
        
    Returns:
        APIResponse: Enhanced response object with metadata
        
    Raises:
        APIRateLimitError: When rate limit is hit
        APIResponseError: When API returns an error
        APITimeoutError: When request times out
        APIConnectionError: When connection fails
        APIAuthError: When authentication fails
    """
    if headers is None:
        headers = {"Accept": "application/json"}
    
    if params is None:
        params = {}
    
    method = method.upper()
    
    # Start timing
    start_time = time.time()
    
    # Use provided session or global session
    request_session = session or _GLOBAL_SESSION
    
    try:
        # Prepare request arguments
        request_kwargs = {
            'headers': headers,
            'params': params,
            'timeout': timeout,
            'verify': verify,
            'stream': stream
        }
        
        # Add JSON body for appropriate methods
        if json_body is not None and method in ["POST", "PUT", "PATCH"]:
            request_kwargs['json'] = json_body
        
        # Make request with appropriate method
        response = request_session.request(method, url, **request_kwargs)
        
        # Calculate request time
        request_time = time.time() - start_time
        
        # Log request details
        log_level = logging.DEBUG if response.status_code < 400 else logging.WARNING
        logger.log(log_level, f"{method} {url} - Status: {response.status_code} - Time: {request_time:.2f}s")
        
        # Extract rate limit information
        rate_limit_remaining = None
        rate_limit_reset = None
        
        for header in ['X-RateLimit-Remaining', 'RateLimit-Remaining']:
            if header in response.headers:
                try:
                    rate_limit_remaining = int(response.headers[header])
                    break
                except (ValueError, TypeError):
                    pass
        
        for header in ['X-RateLimit-Reset', 'RateLimit-Reset']:
            if header in response.headers:
                try:
                    rate_limit_reset = int(response.headers[header])
                    break
                except (ValueError, TypeError):
                    pass
        
        # Handle various status codes
        if response.status_code == 429:
            retry_after = None
            if 'Retry-After' in response.headers:
                try:
                    retry_after = int(response.headers['Retry-After'])
                except (ValueError, TypeError):
                    pass
            
            raise APIRateLimitError(f"Rate limit exceeded. URL: {url}", retry_after=retry_after)
        
        # Check for error status codes
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if 400 <= response.status_code < 500:
                if response.status_code == 401:
                    raise APIAuthError(f"Authentication failed. URL: {url}")
                else:
                    raise APIResponseError(
                        f"Client error: {e}. URL: {url}",
                        status_code=response.status_code,
                        response_text=response.text
                    )
            else:
                raise APIResponseError(
                    f"Server error: {e}. URL: {url}",
                    status_code=response.status_code,
                    response_text=response.text
                )
        
        # Return data based on content type
        if parse_json:
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise APIResponseError(
                    f"Invalid JSON response: {e}. URL: {url}",
                    status_code=response.status_code,
                    response_text=response.text
                )
        else:
            data = response.text
        
        return APIResponse(
            data=data,
            status_code=response.status_code,
            headers=dict(response.headers),
            request_time=request_time,
            success=True,
            rate_limit_remaining=rate_limit_remaining,
            rate_limit_reset=rate_limit_reset
        )
        
    except requests.exceptions.Timeout:
        request_time = time.time() - start_time
        logger.error(f"Request timeout after {request_time:.2f}s: {url}")
        raise APITimeoutError(f"Request timed out after {timeout}s. URL: {url}")
        
    except requests.exceptions.ConnectionError as e:
        request_time = time.time() - start_time
        logger.error(f"Connection error after {request_time:.2f}s: {url} - {str(e)}")
        raise APIConnectionError(f"Connection error: {str(e)}. URL: {url}")
        
    except (APIRateLimitError, APIResponseError, APIAuthError):
        # Re-raise custom exceptions
        raise
        
    except Exception as e:
        request_time = time.time() - start_time
        logger.error(f"Unexpected error after {request_time:.2f}s: {url} - {str(e)}")
        raise

async def async_make_api_request(
    url: str, 
    headers: Optional[Dict[str, str]] = None, 
    params: Optional[Dict[str, Any]] = None, 
    method: str = "GET",
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    verify: bool = True,
    parse_json: bool = True
) -> APIResponse:
    """
    Makes an asynchronous API request using aiohttp
    
    Args:
        url: The API endpoint URL
        headers: Request headers
        params: URL parameters
        method: HTTP method (GET, POST, etc.)
        json_body: JSON body for POST/PUT requests
        timeout: Request timeout in seconds
        verify: Verify SSL certificates
        parse_json: Parse response as JSON
        
    Returns:
        APIResponse: Enhanced response object with metadata
    """
    if headers is None:
        headers = {"Accept": "application/json"}
    
    if params is None:
        params = {}
    
    method = method.upper()
    
    # Start timing
    start_time = time.time()
    
    # Set up timeout
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            # Prepare request arguments
            request_kwargs = {
                'headers': headers,
                'params': params,
                'ssl': verify
            }
            
            # Add JSON body for appropriate methods
            if json_body is not None and method in ["POST", "PUT", "PATCH"]:
                request_kwargs['json'] = json_body
            
            # Make request with appropriate method
            async with session.request(method, url, **request_kwargs) as response:
                # Calculate request time
                request_time = time.time() - start_time
                
                # Extract headers
                response_headers = dict(response.headers)
                
                # Extract rate limit information
                rate_limit_remaining = None
                rate_limit_reset = None
                
                for header in ['X-RateLimit-Remaining', 'RateLimit-Remaining']:
                    if header in response_headers:
                        try:
                            rate_limit_remaining = int(response_headers[header])
                            break
                        except (ValueError, TypeError):
                            pass
                
                for header in ['X-RateLimit-Reset', 'RateLimit-Reset']:
                    if header in response_headers:
                        try:
                            rate_limit_reset = int(response_headers[header])
                            break
                        except (ValueError, TypeError):
                            pass
                
                # Handle rate limiting
                if response.status == 429:
                    retry_after = None
                    if 'Retry-After' in response_headers:
                        try:
                            retry_after = int(response_headers['Retry-After'])
                        except (ValueError, TypeError):
                            pass
                    
                    raise APIRateLimitError(f"Rate limit exceeded. URL: {url}", retry_after=retry_after)
                
                # Check for error status
                if response.status >= 400:
                    response_text = await response.text()
                    
                    if 400 <= response.status < 500:
                        if response.status == 401:
                            raise APIAuthError(f"Authentication failed. URL: {url}")
                        else:
                            raise APIResponseError(
                                f"Client error: {response.status}. URL: {url}",
                                status_code=response.status,
                                response_text=response_text
                            )
                    else:
                        raise APIResponseError(
                            f"Server error: {response.status}. URL: {url}",
                            status_code=response.status,
                            response_text=response_text
                        )
                
                # Parse response
                if parse_json:
                    try:
                        data = await response.json()
                    except json.JSONDecodeError as e:
                        response_text = await response.text()
                        raise APIResponseError(
                            f"Invalid JSON response: {e}. URL: {url}",
                            status_code=response.status,
                            response_text=response_text
                        )
                else:
                    data = await response.text()
                
                return APIResponse(
                    data=data,
                    status_code=response.status,
                    headers=response_headers,
                    request_time=request_time,
                    success=True,
                    rate_limit_remaining=rate_limit_remaining,
                    rate_limit_reset=rate_limit_reset
                )
                
    except asyncio.TimeoutError:
        request_time = time.time() - start_time
        logger.error(f"Async request timeout after {request_time:.2f}s: {url}")
        raise APITimeoutError(f"Request timed out after {timeout}s. URL: {url}")
        
    except aiohttp.ClientConnectorError as e:
        request_time = time.time() - start_time
        logger.error(f"Async connection error after {request_time:.2f}s: {url} - {str(e)}")
        raise APIConnectionError(f"Connection error: {str(e)}. URL: {url}")
        
    except (APIRateLimitError, APIResponseError, APIAuthError):
        # Re-raise custom exceptions
        raise
        
    except Exception as e:
        request_time = time.time() - start_time
        logger.error(f"Unexpected async error after {request_time:.2f}s: {url} - {str(e)}")
        raise

def save_api_response(
    response: Union[APIResponse, Dict[str, Any]],
    filepath: str,
    prettify: bool = True,
    include_metadata: bool = False
) -> None:
    """
    Saves API response to a JSON file with enhanced options
    
    Args:
        response: API response or data dictionary
        filepath: Path to save the file
        prettify: Whether to format JSON with indentation
        include_metadata: Include response metadata with the data
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data to save
        if isinstance(response, APIResponse):
            if include_metadata:
                data_to_save = {
                    'data': response.data,
                    'metadata': {
                        'status_code': response.status_code,
                        'request_time': response.request_time,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'success': response.success,
                        'cached': response.cached
                    }
                }
            else:
                data_to_save = response.data
        else:
            data_to_save = response
        
        # Save JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            if prettify:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            else:
                json.dump(data_to_save, f, ensure_ascii=False)
                
        logger.debug(f"Saved API response to {filepath}")
    
    except Exception as e:
        logger.error(f"Error saving API response: {e}")

def load_api_response(
    filepath: str,
    default_value: Optional[T] = None,
    include_metadata: bool = False
) -> Optional[Union[Dict[str, Any], T]]:
    """
    Loads API response from a JSON file with enhanced error handling
    
    Args:
        filepath: Path to the file
        default_value: Value to return if file not found or invalid
        include_metadata: Whether the file includes metadata
        
    Returns:
        dict/list/T: API response or default_value if not found
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return default_value
        
        with open(filepath, 'r', encoding='utf-8') as f:
            response = json.load(f)
            
        # Extract data if metadata was included
        if include_metadata and isinstance(response, dict) and 'data' in response and 'metadata' in response:
            return response['data']
            
        return response
    
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from {filepath}: {e}")
        return default_value
    
    except Exception as e:
        logger.error(f"Error loading API response: {e}")
        return default_value

def build_url(base_url: str, endpoint: str, api_version: Optional[str] = None) -> str:
    """
    Builds a full URL from base URL and endpoint with improved handling
    
    Args:
        base_url: Base API URL
        endpoint: API endpoint
        api_version: API version
        
    Returns:
        str: Full URL
    """
    # Normalize URL components
    base_url = base_url.rstrip('/')
    
    if api_version:
        # Handle both formats: 'v1' and '/v1/'
        api_version = api_version.strip('/') 
        middle_path = f"{api_version}/"
    else:
        middle_path = ""
    
    # Normalize endpoint
    endpoint = endpoint.lstrip('/')
    
    # Combine parts using urljoin for proper URL handling
    if middle_path:
        full_url = urljoin(f"{base_url}/", urljoin(f"{middle_path}/", endpoint))
    else:
        full_url = urljoin(f"{base_url}/", endpoint)
    
    return full_url

async def paginated_fetch_all(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    page_param: str = "page",
    per_page_param: str = "per_page",
    per_page: int = 100,
    max_pages: int = 10,
    page_starts_at: int = 1,
    result_key: Optional[str] = None,
    total_key: Optional[str] = None,
    concurrent_requests: int = 3
) -> List[Dict[str, Any]]:
    """
    Fetches all pages from a paginated API endpoint asynchronously
    
    Args:
        url: The API endpoint URL
        headers: Request headers
        params: URL parameters
        page_param: Name of the page parameter
        per_page_param: Name of the per_page parameter
        per_page: Number of items per page
        max_pages: Maximum number of pages to fetch
        page_starts_at: Starting page number (0 or 1)
        result_key: Key containing results in the response
        total_key: Key containing total count in the response
        concurrent_requests: Number of concurrent requests
        
    Returns:
        list: Combined results from all pages
    """
    if headers is None:
        headers = {}
    
    if params is None:
        params = {}
    
    # Set per_page parameter
    params[per_page_param] = per_page
    
    all_results = []
    
    # Make initial request to get first page and total count
    first_page_params = params.copy()
    first_page_params[page_param] = page_starts_at
    
    first_response = await async_make_api_request(url, headers=headers, params=first_page_params)
    
    # Extract results from first page
    if result_key:
        if isinstance(first_response.data, dict) and result_key in first_response.data:
            first_page_results = first_response.data[result_key]
        else:
            logger.warning(f"Result key '{result_key}' not found in response")
            return []
    else:
        first_page_results = first_response.data
    
    if not isinstance(first_page_results, list):
        logger.warning(f"Expected list of results, got {type(first_page_results)}")
        return []
    
    all_results.extend(first_page_results)
    
    # Determine total pages
    total_items = None
    if total_key and isinstance(first_response.data, dict):
        total_items = first_response.data.get(total_key)
    
    if total_items is not None:
        total_pages = (total_items + per_page - 1) // per_page
    else:
        # If no total count, check if we got a full page
        total_pages = 1
        if len(first_page_results) >= per_page:
            # Assume there are more pages
            total_pages = max_pages
    
    # Limit to max_pages
    total_pages = min(total_pages, max_pages)
    
    if total_pages <= 1:
        # Only one page, already fetched
        return all_results
    
    # Fetch remaining pages in parallel
    remaining_pages = list(range(page_starts_at + 1, page_starts_at + total_pages))
    
    # Process pages in batches to control concurrency
    for i in range(0, len(remaining_pages), concurrent_requests):
        batch = remaining_pages[i:i + concurrent_requests]
        
        # Create tasks for each page
        tasks = []
        for page in batch:
            page_params = params.copy()
            page_params[page_param] = page
            tasks.append(async_make_api_request(url, headers=headers, params=page_params))
        
        # Run tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        for response in responses:
            if isinstance(response, Exception):
                logger.error(f"Error fetching page: {str(response)}")
                continue
                
            # Extract results
            if result_key:
                if isinstance(response.data, dict) and result_key in response.data:
                    page_results = response.data[result_key]
                else:
                    continue
            else:
                page_results = response.data
            
            if isinstance(page_results, list):
                all_results.extend(page_results)
                
                # Check if we got a partial page, indicating the end
                if len(page_results) < per_page:
                    break
    
    return all_results

def pagination_handler(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    page_param: str = "page",
    per_page_param: str = "per_page",
    per_page: int = 100,
    max_pages: int = 10,
    page_starts_at: int = 1,
    result_key: Optional[str] = None,
    session: Optional[requests.Session] = None
) -> List[Dict[str, Any]]:
    """
    Enhanced synchronous handler for paginated API endpoints
    
    Args:
        url: The API endpoint URL
        headers: Request headers
        params: URL parameters
        page_param: Name of the page parameter
        per_page_param: Name of the per_page parameter
        per_page: Number of items per page
        max_pages: Maximum number of pages to fetch
        page_starts_at: Starting page number (0 or 1)
        result_key: Key containing results in the response
        session: Requests session
        
    Returns:
        list: Combined results from all pages
    """
    if params is None:
        params = {}
    
    # Set per_page parameter
    params[per_page_param] = per_page
    
    all_results = []
    current_page = page_starts_at
    
    request_session = session or _GLOBAL_SESSION
    
    while current_page < page_starts_at + max_pages:
        # Set page parameter
        page_params = params.copy()
        page_params[page_param] = current_page
        
        # Make request
        response = make_api_request(
            url, 
            headers=headers, 
            params=page_params,
            session=request_session
        )
        
        # Extract results
        if result_key:
            if isinstance(response.data, dict) and result_key in response.data:
                results = response.data[result_key]
            else:
                logger.warning(f"Result key '{result_key}' not found in response")
                break
        else:
            results = response.data
        
        # If results is not a list, handle accordingly
        if not isinstance(results, list):
            logger.warning(f"Expected list, got {type(results)}")
            break
        
        # Add results to full list
        all_results.extend(results)
        
        # Check if we've reached the last page
        if len(results) < per_page:
            break
        
        # Apply adaptive rate limiting if needed
        if response.rate_limit_remaining is not None and response.rate_limit_remaining < 10:
            delay = adaptive_rate_limit_delay(requests.Response(), default_delay=1.0)
            logger.debug(f"Rate limit getting low ({response.rate_limit_remaining}), sleeping for {delay:.2f}s")
            time.sleep(delay)
        
        # Increment page
        current_page += 1
    
    return all_results

def cache_api_response(
    url: str,
    cache_dir: str,
    cache_filename: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    method: str = "GET",
    json_body: Optional[Dict[str, Any]] = None,
    cache_ttl: int = 3600,  # 1 hour
    force_refresh: bool = False,
    session: Optional[requests.Session] = None
) -> APIResponse:
    """
    Enhanced API request caching with better key generation and metadata
    
    Args:
        url: The API endpoint URL
        cache_dir: Directory to store cache files
        cache_filename: Custom filename for cache
        headers: Request headers
        params: URL parameters
        method: HTTP method (GET, POST, etc.)
        json_body: JSON body for POST requests
        cache_ttl: Cache TTL in seconds
        force_refresh: Force refresh the cache
        session: Requests session
        
    Returns:
        APIResponse: API response with cache metadata
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename if not provided
    if cache_filename is None:
        cache_key_str = cache_key(url, params or {}, method, json_body)
        cache_filename = f"cache_{cache_key_str}.json"
    
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if cache exists and is valid
    if not force_refresh and is_cache_valid(cache_path, cache_ttl):
        try:
            # Load cached response
            logger.debug(f"Loading from cache: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Extract metadata if it exists
            if isinstance(cached_data, dict) and 'data' in cached_data and 'metadata' in cached_data:
                data = cached_data['data']
                metadata = cached_data['metadata']
                
                # Create APIResponse from cache
                return APIResponse(
                    data=data,
                    status_code=metadata.get('status_code', 200),
                    headers={},
                    request_time=0.0,
                    success=True,
                    cached=True
                )
            else:
                # Just the data was cached
                return APIResponse(
                    data=cached_data,
                    status_code=200,
                    headers={},
                    request_time=0.0,
                    success=True,
                    cached=True
                )
        except Exception as e:
            logger.warning(f"Error loading cache, making fresh request: {e}")
    
    # Cache doesn't exist, is invalid, or force_refresh is True
    logger.debug(f"Making fresh API request to {url}")
    response = make_api_request(
        url,
        headers=headers,
        params=params,
        method=method,
        json_body=json_body,
        session=session
    )
    
    # Save to cache (only successful responses)
    if response.success:
        try:
            # Include metadata with the cached data
            cache_data = {
                'data': response.data,
                'metadata': {
                    'status_code': response.status_code,
                    'request_time': response.request_time,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'url': url,
                    'method': method
                }
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"Saved response to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    return response

def monitor_api_usage(func: Callable) -> Callable:
    """
    Enhanced decorator to monitor API usage with detailed metrics
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        
        # Extract URL from args or kwargs
        url = None
        if len(args) > 0 and isinstance(args[0], str):
            url = args[0]
        elif 'url' in kwargs:
            url = kwargs['url']
        
        # Prepare metrics
        metrics = {
            'function': function_name,
            'url': url,
            'start_time': start_time,
            'status': None,
            'response_size': None,
            'error': None
        }
        
        try:
            result = func(*args, **kwargs)
            
            # Capture metrics from result if it's an APIResponse
            if isinstance(result, APIResponse):
                metrics['status'] = result.status_code
                metrics['response_size'] = len(json.dumps(result.data)) if result.data else 0
                metrics['rate_limit_remaining'] = result.rate_limit_remaining
                
                status = "success"
            else:
                status = "success"
                
            metrics['status'] = status
        except Exception as e:
            metrics['status'] = "error"
            metrics['error'] = str(e)
            
            # Include more context for certain error types
            if isinstance(e, APIRateLimitError):
                metrics['error_type'] = "rate_limit"
                metrics['retry_after'] = getattr(e, 'retry_after', None)
            elif isinstance(e, APIResponseError):
                metrics['error_type'] = "response_error"
                metrics['status_code'] = getattr(e, 'status_code', None)
            elif isinstance(e, APITimeoutError):
                metrics['error_type'] = "timeout"
            elif isinstance(e, APIConnectionError):
                metrics['error_type'] = "connection"
            
            raise
        finally:
            # Calculate elapsed time
            end_time = time.time()
            elapsed = end_time - start_time
            metrics['elapsed'] = elapsed
            
            # Log metrics
            log_level = logging.INFO if metrics['status'] == "success" else logging.WARNING
            
            # Create detailed log message
            if url:
                log_message = f"API call to {function_name} ({url}): status={metrics['status']}, time={elapsed:.2f}s"
            else:
                log_message = f"API call to {function_name}: status={metrics['status']}, time={elapsed:.2f}s"
                
            if metrics.get('rate_limit_remaining') is not None:
                log_message += f", rate_limit_remaining={metrics['rate_limit_remaining']}"
                
            if metrics.get('error'):
                log_message += f", error={metrics['error']}"
            
            logger.log(log_level, log_message)
            
            # Could add metrics collection/reporting here
        
        return result
    
    return wrapper

@contextmanager
def api_retry_context(
    max_retries: int = 3, 
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_errors: List[type] = None
):
    """
    Context manager for retrying API operations with exponential backoff
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries
        backoff_factor: Factor to increase delay between retries
        retry_errors: List of error types to retry on
    """
    if retry_errors is None:
        retry_errors = [APIRateLimitError, APIConnectionError, APITimeoutError]
    
    retries = 0
    delay = retry_delay
    
    while True:
        try:
            yield
            break  # Success, exit the loop
        except tuple(retry_errors) as e:
            retries += 1
            if retries > max_retries:
                logger.error(f"Maximum retries ({max_retries}) exceeded")
                raise
            
            # Adjust delay for rate limit errors if retry-after is provided
            if isinstance(e, APIRateLimitError) and e.retry_after is not None:
                delay = e.retry_after
            
            logger.warning(f"Retry {retries}/{max_retries} after error: {str(e)}. Waiting {delay:.2f}s")
            time.sleep(delay)
            
            # Increase delay for next retry
            delay *= backoff_factor

async def async_batch_requests(
    urls: List[str],
    headers: Optional[Dict[str, str]] = None,
    concurrent_limit: int = 5,
    timeout: int = 30
) -> List[APIResponse]:
    """
    Execute multiple API requests in parallel with controlled concurrency
    
    Args:
        urls: List of URLs to request
        headers: Common headers for all requests
        concurrent_limit: Maximum number of concurrent requests
        timeout: Request timeout in seconds
        
    Returns:
        list: List of APIResponse objects
    """
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    async def fetch_with_semaphore(url):
        async with semaphore:
            return await async_make_api_request(url, headers=headers, timeout=timeout)
    
    # Create tasks for all URLs
    tasks = [fetch_with_semaphore(url) for url in urls]
    
    # Execute all tasks and gather results
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process responses, converting exceptions to APIResponse objects with error status
    processed_responses = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            logger.error(f"Error fetching {urls[i]}: {str(response)}")
            # Create error response
            processed_responses.append(APIResponse(
                data=None,
                status_code=0,
                headers={},
                request_time=0.0,
                success=False,
                cached=False
            ))
        else:
            processed_responses.append(response)
    
    return processed_responses


if __name__ == "__main__":
    # Example usage
    async def test_async_api():
        print("Testing async API calls...")
        
        try:
            # Test async single request
            response = await async_make_api_request("https://api.github.com/users/feyfry")
            print(f"Async API call status: {response.status_code}")
            print(f"Response time: {response.request_time:.2f}s")
            
            if response.success:
                print(f"Username: {response.data.get('login')}")
                print(f"Name: {response.data.get('name')}")
            
            # Test async pagination
            print("\nFetching paginated data...")
            results = await paginated_fetch_all(
                "https://api.github.com/users/feyfry/repos",
                per_page=5,
                max_pages=3
            )
            print(f"Fetched {len(results)} repositories")
            
            # Test parallel requests
            print("\nExecuting parallel requests...")
            urls = [
                "https://api.github.com/users/feyfry",
                "https://api.github.com/users/RanelaRoe",
                "https://api.github.com/users/0xBitLoc"
            ]
            
            parallel_responses = await async_batch_requests(urls, concurrent_limit=2)
            print(f"Parallel responses: {len(parallel_responses)}")
            
            for i, resp in enumerate(parallel_responses):
                if resp.success:
                    print(f"Response {i+1}: {resp.data.get('name')}")
                else:
                    print(f"Response {i+1}: Error")
            
        except Exception as e:
            print(f"Error in async test: {e}")
    
    # Run synchronous tests
    print("Testing API utils...")
    
    # Test URL building
    url = build_url("https://api.vespertine.my.id", "users", "v2")
    print(f"Built URL: {url}")
    
    # Test caching
    cache_dir = "./cache"
    
    response = cache_api_response(
        "https://api.github.com/users/feyfry",
        cache_dir=cache_dir,
        cache_ttl=600
    )
    
    print(f"Cached API call status: {response.status_code}")
    print(f"From cache: {response.cached}")
    
    # Run async tests if possible
    if hasattr(asyncio, 'run'):
        asyncio.run(test_async_api())
    else:
        print("Skipping async tests (Python < 3.7)")