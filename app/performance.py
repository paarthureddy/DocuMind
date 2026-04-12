"""
Performance optimizations for the talent search engine.
Includes caching, batch processing, and memory management.
"""

import os
import pickle
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import functools
import threading

from config import FAISS_INDEX_DIR, DATA_DIR

class PerformanceOptimizer:
    """Manages performance optimizations for the talent search engine."""
    
    def __init__(self):
        self.cache_dir = DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Thread-safe cache
        self._embedding_cache = {}
        self._cache_lock = threading.RLock()
        
        # Performance settings
        self.batch_size = 32
        self.max_cache_size = 1000
        
    def cache_result(self, key: str, data: Any, ttl_seconds: int = 3600):
        """Cache a result with TTL."""
        with self._cache_lock:
            cache_file = self.cache_dir / f"{key}.pkl"
            cache_entry = {
                "data": data,
                "timestamp": time.time(),
                "ttl": ttl_seconds
            }
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_entry, f)
            except Exception as e:
                print(f"Failed to cache result: {e}")
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if not expired."""
        with self._cache_lock:
            cache_file = self.cache_dir / f"{key}.pkl"
            
            if not cache_file.exists():
                return None
            
            try:
                with open(cache_file, 'rb') as f:
                    cache_entry = pickle.load(f)
                
                # Check if expired
                if time.time() - cache_entry["timestamp"] > cache_entry["ttl"]:
                    cache_file.unlink()  # Remove expired cache
                    return None
                
                return cache_entry["data"]
                
            except Exception as e:
                print(f"Failed to load cached result: {e}")
                return None
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cache files."""
        with self._cache_lock:
            for cache_file in self.cache_dir.glob("*.pkl"):
                if pattern is None or pattern in cache_file.name:
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        print(f"Failed to delete cache file {cache_file}: {e}")
    
    def memoize_with_cache(self, ttl_seconds: int = 3600):
        """Decorator for memoizing functions with disk cache."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = "_".join(key_parts).replace("/", "_").replace("\\", "_")
                
                # Try to get from cache
                cached_result = self.get_cached_result(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Compute and cache result
                result = func(*args, **kwargs)
                self.cache_result(cache_key, result, ttl_seconds)
                
                return result
            return wrapper
        return decorator
    
    def batch_process_embeddings(self, texts: List[str], embeddings_model) -> List[List[float]]:
        """Process embeddings in batches for better performance."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                batch_embeddings = embeddings_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to prevent overwhelming the model
                if i + self.batch_size < len(texts):
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Error processing batch {i//self.batch_size}: {e}")
                # Add dummy embeddings for failed batch
                dummy_embedding = [0.0] * 384  # Default embedding dimension
                all_embeddings.extend([dummy_embedding] * len(batch))
        
        return all_embeddings
    
    def optimize_faiss_index(self, vector_store):
        """Optimize FAISS index for better performance."""
        try:
            # Convert to IVF index for faster search on large datasets
            import faiss
            
            # Get current index
            index = vector_store.index
            
            # Get index dimension
            d = index.d
            
            # Get number of vectors
            ntotal = index.ntotal
            
            if ntotal > 10000:  # Only optimize for larger datasets
                print(f"Optimizing FAISS index for {ntotal} vectors...")
                
                # Create IVF index
                nlist = min(int(np.sqrt(ntotal)), 1000)  # Number of clusters
                quantizer = faiss.IndexFlatL2(d)
                ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist)
                
                # Train the index
                print("Training IVF index...")
                ivf_index.train(index.reconstruct_n(0, ntotal))
                
                # Add vectors
                ivf_index.add(index.reconstruct_n(0, ntotal))
                
                # Replace the index
                vector_store.index = ivf_index
                
                # Save optimized index
                vector_store.save_local(str(FAISS_INDEX_DIR))
                
                print("FAISS index optimized successfully")
                
        except Exception as e:
            print(f"Failed to optimize FAISS index: {e}")
    
    def monitor_memory_usage(self):
        """Monitor and report memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available for memory monitoring"}
        except Exception as e:
            return {"error": f"Memory monitoring failed: {e}"}
    def preload_embeddings(self, profile_ids: List[str], embeddings_model):
        """Preload embeddings for frequently accessed profiles."""
        preload_cache_file = self.cache_dir / "preloaded_embeddings.pkl"
        
        try:
            # Check if preloaded embeddings exist
            if preload_cache_file.exists():
                with open(preload_cache_file, 'rb') as f:
                    preloaded = pickle.load(f)
                
                # Update cache with new profiles if needed
                existing_ids = set(preloaded.keys())
                new_ids = set(profile_ids) - existing_ids
                
                if new_ids:
                    print(f"Preloading embeddings for {len(new_ids)} new profiles...")
                    # Load new embeddings and add to cache
                    # This would need to be implemented based on your data structure
                    
                return preloaded
            else:
                print("No preloaded embeddings found")
                return {}
                
        except Exception as e:
            print(f"Failed to preload embeddings: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "cache_files": len(list(self.cache_dir.glob("*.pkl"))),
            "cache_size_mb": sum(
                f.stat().st_size for f in self.cache_dir.glob("*.pkl")
            ) / 1024 / 1024,
            "memory_usage": self.monitor_memory_usage()
        }
        
        return stats

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    return wrapper

def async_timing_decorator(func):
    """Decorator to measure async function execution time."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    return wrapper

class BatchProcessor:
    """Utility for batch processing large datasets."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    def process_in_batches(self, items: List[Any], process_func, progress_callback=None):
        """Process items in batches with progress tracking."""
        results = []
        total_items = len(items)
        
        for i in range(0, total_items, self.batch_size):
            batch = items[i:i + self.batch_size]
            
            try:
                batch_results = process_func(batch)
                results.extend(batch_results)
                
                if progress_callback:
                    progress = min((i + self.batch_size) / total_items, 1.0)
                    progress_callback(progress, i + self.batch_size, total_items)
                    
            except Exception as e:
                print(f"Error processing batch {i//self.batch_size}: {e}")
                # Add empty results for failed batch
                results.extend([None] * len(batch))
        
        return results
    
    def process_with_retry(self, items: List[Any], process_func, max_retries=3):
        """Process items with automatic retry on failure."""
        results = []
        
        for i, item in enumerate(items):
            retries = 0
            while retries < max_retries:
                try:
                    result = process_func(item)
                    results.append(result)
                    break
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        print(f"Failed to process item {i} after {max_retries} retries: {e}")
                        results.append(None)
                    else:
                        time.sleep(0.1 * retries)  # Exponential backoff
        
        return results
