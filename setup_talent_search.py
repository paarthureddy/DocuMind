"""
Setup script for the Talent Search Engine.
Handles data processing, index building, and system initialization.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.data_pipeline import DataPipeline
from app.embeddings_pipeline import EmbeddingsPipeline
from app.retrieval import RetrievalEngine
from app.config import PROFILES_JSON_PATH, PROCESSED_PROFILES_PATH

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check if source data exists
    if not os.path.exists(PROFILES_JSON_PATH):
        print(f"❌ Source data file not found: {PROFILES_JSON_PATH}")
        print("Please ensure the profiles JSON file exists at the specified path.")
        return False
    
    print(f"✅ Source data found: {PROFILES_JSON_PATH}")
    
    # Check if Ollama is running
    try:
        from app.llm import get_llm
        llm = get_llm()
        # Test connection
        response = llm.invoke("test")
        print("✅ Ollama connection successful")
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("Please ensure Ollama is running and the model is available.")
        return False
    
    return True

def process_data(force_reprocess=False, resume=False):
    """Process the raw talent data."""
    print("\n📊 Processing talent data...")
    
    # Check if already processed (skip check if resuming)
    if PROCESSED_PROFILES_PATH.exists() and not force_reprocess and not resume:
        print("📋 Data already processed. Use --force to reprocess or --resume to continue.")
        return True
    
    try:
        pipeline = DataPipeline()
        processed_profiles = pipeline.process_dataset(PROFILES_JSON_PATH, resume=resume)
        
        print(f"✅ Successfully processed {len(processed_profiles)} profiles")
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def build_index(force_rebuild=False):
    """Build the search index."""
    print("\n🔍 Building search index...")
    
    # Check if index already exists
    from app.config import FAISS_INDEX_DIR
    if FAISS_INDEX_DIR.exists() and not force_rebuild:
        print("📋 Index already exists. Use --force to rebuild.")
        return True
    
    try:
        retrieval_engine = RetrievalEngine()
        result = retrieval_engine.build_index(force_rebuild=force_rebuild)
        
        if result["success"]:
            print("✅ Search index built successfully")
            stats = result.get("stats", {})
            print(f"   - Total profiles: {stats.get('total_profiles', 'Unknown')}")
            print(f"   - Embedding dimension: {stats.get('embedding_dimension', 'Unknown')}")
            return True
        else:
            print(f"❌ Index building failed: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Index building failed: {e}")
        return False

def test_system():
    """Test the system with sample queries."""
    print("\n🧪 Testing system with sample queries...")
    
    try:
        retrieval_engine = RetrievalEngine()
        
        # Sample queries
        test_queries = [
            "male actor",
            "female model",
            "villain role",
            "tall actor"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            result = retrieval_engine.search_profiles(query, max_results=3)
            
            if result["success"] and result["results"]:
                print(f"✅ Found {len(result['results'])} results")
                for i, res in enumerate(result["results"][:2], 1):
                    print(f"   {i}. {res['name']} (Score: {res['score']:.2f})")
            else:
                print(f"❌ No results found")
        
        print("\n✅ System test completed")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def start_api():
    """Start the API server."""
    print("\n🚀 Starting API server...")
    
    try:
        import uvicorn
        from app.talent_api import app
        
        print("🌐 API server starting on http://localhost:8000")
        print("📖 API documentation available at http://localhost:8000/docs")
        print("🔍 Try the health endpoint: http://localhost:8000/health")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Talent Search Engine")
    parser.add_argument("--check", action="store_true", help="Check prerequisites only")
    parser.add_argument("--process", action="store_true", help="Process data only")
    parser.add_argument("--index", action="store_true", help="Build index only")
    parser.add_argument("--test", action="store_true", help="Test system only")
    parser.add_argument("--api", action="store_true", help="Start API server only")
    parser.add_argument("--force", action="store_true", help="Force reprocess/rebuild")
    parser.add_argument("--all", action="store_true", help="Run full setup (default)")
    
    args = parser.parse_args()
    
    # If no arguments provided, run full setup
    if not any([args.check, args.process, args.index, args.test, args.api]):
        args.all = True
    
    print("🎭 Talent Search Engine Setup")
    print("=" * 50)
    
    success = True
    
    # Step 1: Check prerequisites
    if args.check or args.all:
        if not check_prerequisites():
            success = False
            if not args.all:  # Exit if only checking
                return
    
    if not success:
        print("\n❌ Setup failed due to missing prerequisites")
        return
    
    # Step 2: Process data
    if (args.process or args.all) and success:
        if not process_data(args.force):
            success = False
    
    # Step 3: Build index
    if (args.index or args.all) and success:
        if not build_index(args.force):
            success = False
    
    # Step 4: Test system
    if (args.test or args.all) and success:
        if not test_system():
            success = False
    
    # Step 5: Start API
    if args.api and success:
        start_api()
    elif args.all and success:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the API server: python setup_talent_search.py --api")
        print("2. Open http://localhost:8000/docs for API documentation")
        print("3. Try a sample query: curl -X POST http://localhost:8000/search_profiles -H 'Content-Type: application/json' -d '{\"query\": \"male actor\"}'")
    
    if not success:
        print("\n❌ Setup encountered errors. Please check the messages above.")

if __name__ == "__main__":
    main()
