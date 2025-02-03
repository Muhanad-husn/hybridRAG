import os
import pickle
import hashlib
from typing import Optional, List, Dict, Any, TypeVar, Generic
from datetime import datetime, timedelta
import json
from src.utils.logger import get_logger
from src.utils.config_handler import config

logger = get_logger(__name__)

T = TypeVar('T')  # Generic type for cached data

class CacheMetadata:
    """Metadata for cached items."""
    def __init__(self, 
                 content_hash: str,
                 created_at: datetime,
                 expires_at: datetime,
                 version: str = "1.0.0",
                 extra: Dict[str, Any] = None):
        self.content_hash = content_hash
        self.created_at = created_at
        self.expires_at = expires_at
        self.version = version
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'content_hash': self.content_hash,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'version': self.version,
            'extra': self.extra
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheMetadata':
        """Create metadata from dictionary."""
        return cls(
            content_hash=data['content_hash'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            version=data['version'],
            extra=data['extra']
        )

class DocumentCache(Generic[T]):
    """Enhanced caching system with versioning and content validation."""
    
    def __init__(self, cache_dir: str = None, ttl_hours: int = 24):
        """Initialize the cache handler."""
        self.cache_dir = cache_dir or config.get("cache.directory", "data/cache")
        self.ttl_hours = ttl_hours or config.get("cache.ttl_hours", 24)
        self.version = "1.0.0"  # Cache version for compatibility checking
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create metadata directory
        self.metadata_dir = os.path.join(self.cache_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
    
    def _compute_content_hash(self, content: Any) -> str:
        """Compute hash of content for validation."""
        try:
            # Convert content to bytes for consistent hashing
            if isinstance(content, (str, bytes)):
                content_bytes = content.encode() if isinstance(content, str) else content
            else:
                content_bytes = pickle.dumps(content)
            return hashlib.sha256(content_bytes).hexdigest()
        except Exception as e:
            logger.error(f"Error computing content hash: {str(e)}")
            return ""
    
    def _get_cache_path(self, key: str) -> tuple[str, str]:
        """Get paths for cache and metadata files."""
        cache_key = hashlib.md5(key.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pickle")
        meta_file = os.path.join(self.metadata_dir, f"{cache_key}.json")
        return cache_file, meta_file
    
    def _save_metadata(self, meta_file: str, metadata: CacheMetadata) -> None:
        """Save metadata to file."""
        try:
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f)
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise
    
    def _load_metadata(self, meta_file: str) -> Optional[CacheMetadata]:
        """Load metadata from file."""
        try:
            if not os.path.exists(meta_file):
                return None
            with open(meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return CacheMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return None
    
    def get(self, key: str) -> Optional[T]:
        """
        Retrieve item from cache with validation.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found/invalid
        """
        try:
            cache_file, meta_file = self._get_cache_path(key)
            
            # Check if cache exists
            if not os.path.exists(cache_file) or not os.path.exists(meta_file):
                return None
            
            # Load metadata
            metadata = self._load_metadata(meta_file)
            if metadata is None:
                return None
            
            # Check expiration
            if datetime.now() > metadata.expires_at:
                self.invalidate(key)
                return None
            
            # Check version compatibility
            if metadata.version != self.version:
                logger.warning(f"Cache version mismatch: {metadata.version} != {self.version}")
                self.invalidate(key)
                return None
            
            # Load cached data
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Validate content hash
            current_hash = self._compute_content_hash(cached_data)
            if current_hash != metadata.content_hash:
                logger.warning("Cache content hash mismatch")
                self.invalidate(key)
                return None
            
            return cached_data
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None
    
    def set(self, key: str, value: T, extra_metadata: Dict[str, Any] = None) -> None:
        """
        Store item in cache with metadata.
        
        Args:
            key: Cache key
            value: Item to cache
            extra_metadata: Additional metadata to store
        """
        try:
            cache_file, meta_file = self._get_cache_path(key)
            
            # Compute content hash
            content_hash = self._compute_content_hash(value)
            
            # Create metadata
            metadata = CacheMetadata(
                content_hash=content_hash,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=self.ttl_hours),
                version=self.version,
                extra=extra_metadata
            )
            
            # Save data and metadata
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            self._save_metadata(meta_file, metadata)
            
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            raise
    
    def invalidate(self, key: str) -> None:
        """
        Remove item from cache.
        
        Args:
            key: Cache key to invalidate
        """
        try:
            cache_file, meta_file = self._get_cache_path(key)
            
            # Remove cache and metadata files
            if os.path.exists(cache_file):
                os.remove(cache_file)
            if os.path.exists(meta_file):
                os.remove(meta_file)
                
        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")
            raise
    
    def clear(self) -> None:
        """Clear all cached items."""
        try:
            # Remove all files in cache and metadata directories
            for file in os.listdir(self.cache_dir):
                if file != "metadata":  # Skip metadata directory
                    os.remove(os.path.join(self.cache_dir, file))
            
            for file in os.listdir(self.metadata_dir):
                os.remove(os.path.join(self.metadata_dir, file))
                
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for cached item.
        
        Args:
            key: Cache key
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            _, meta_file = self._get_cache_path(key)
            metadata = self._load_metadata(meta_file)
            return metadata.to_dict() if metadata else None
        except Exception as e:
            logger.error(f"Error getting metadata: {str(e)}")
            return None