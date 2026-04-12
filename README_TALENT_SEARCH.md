# 🎭 Talent Search Engine - AI-Powered Casting Assistant

A production-ready semantic talent search system that transforms natural language queries like "male villain brown 6 feet intense actor" into intelligent profile matches using hybrid retrieval + advanced ranking.

## ✨ Features

- **🧠 Semantic Understanding**: Natural language queries with intelligent parsing
- **🎯 Hybrid Search**: Combines vector similarity with structured filtering
- **⚡ Advanced Ranking**: Multi-factor scoring system with explainable results
- **🤖 Agentic AI**: LangGraph-powered casting assistant
- **🔍 Real-time Search**: Sub-second response times with caching
- **📊 Rich Profiles**: Semantic enrichment using LLM analysis
- **🚀 Production Ready**: Scalable architecture with performance optimizations

## 🏗️ Architecture

```
Natural Language Query
         │
         ▼
   Query Parser
    │           │
    ▼           ▼
Filters   Semantic Terms
    │           │
    ▼           ▼
Vector Search + Filtering
         │
         ▼
   Advanced Ranking
         │
         ▼
  Explained Results
```

## 📋 System Requirements

- Python 3.10+
- Ollama (with llama3 model)
- 8GB+ RAM (for large datasets)
- Storage space for embeddings and index

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
# Install Ollama if not already installed
# https://ollama.com/

# Pull the required model
ollama pull llama3

# Start Ollama server
ollama serve
```

### 3. Prepare Your Data

Place your talent profiles JSON file at:
```
C:\Users\Lenovo\OneDrive\projectss\1500_profiles.json
```

### 4. Setup the System

```bash
# Run complete setup
python setup_talent_search.py --all

# Or run steps individually
python setup_talent_search.py --check    # Check prerequisites
python setup_talent_search.py --process  # Process data
python setup_talent_search.py --index    # Build search index
python setup_talent_search.py --test     # Test system
```

### 5. Start the API Server

```bash
python setup_talent_search.py --api
```

The API will be available at `http://localhost:8000`

## 📖 API Usage

### Search Profiles

```bash
curl -X POST http://localhost:8000/search_profiles \
  -H "Content-Type: application/json" \
  -d '{
    "query": "male villain brown 6 feet intense actor",
    "max_results": 10
  }'
```

### Chat with AI Assistant

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find me a tall male actor for a villain role"
  }'
```

### Get Similar Profiles

```bash
curl -X POST http://localhost:8000/similar_profiles \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": "actor_123",
    "max_results": 5
  }'
```

## 🔍 Query Examples

The system understands natural language queries like:

- **Physical Attributes**: "male 6 feet tall brown complexion"
- **Role Types**: "villain actor intense personality", "romantic lead charming"
- **Experience Level**: "experienced female actor", "new male model"
- **Combinations**: "tall dark villain with good acting skills"
- **Specific Crafts**: "male dancer with comic timing", "female model for fashion"

## 📊 System Components

### Data Pipeline (`data_pipeline.py`)
- Profile normalization and cleaning
- LLM-powered semantic enrichment
- Structured field extraction

### Embeddings Pipeline (`embeddings_pipeline.py`)
- Multi-embedding generation
- FAISS vector storage
- Metadata management

### Query Parser (`query_parser.py`)
- Natural language understanding
- Structured filter extraction
- Query expansion with synonyms

### Retrieval Engine (`retrieval.py`)
- Hybrid search implementation
- Vector similarity + filtering
- Candidate selection

### Ranking System (`ranking.py`)
- Multi-factor scoring
- Semantic similarity weighting
- Result explanation generation

### Talent Agent (`talent_agent.py`)
- LangGraph integration
- Intelligent tool selection
- Conversational interface

### API Layer (`talent_api.py`)
- RESTful endpoints
- Request validation
- Error handling

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Scoring weights
SCORING_WEIGHTS = {
    "semantic_similarity": 0.4,
    "casting_tag_match": 0.2,
    "personality_match": 0.1,
    "experience_score": 0.1,
    "rating_score": 0.1,
    "engagement_score": 0.1
}
```

## 🔧 Advanced Usage

### Custom Scoring Weights

```python
from app.ranking import RankingEngine

engine = RankingEngine()
engine.update_weights({
    "semantic_similarity": 0.5,
    "casting_tag_match": 0.3,
    "personality_match": 0.1,
    "experience_score": 0.05,
    "rating_score": 0.03,
    "engagement_score": 0.02
})
```

### Direct Python Usage

```python
from app.retrieval import RetrievalEngine

# Initialize search engine
engine = RetrievalEngine()

# Search profiles
results = engine.search_profiles(
    query="male villain actor",
    max_results=10
)

# Process results
for result in results["results"]:
    print(f"{result['name']}: {result['score']:.2f}")
    print(f"Reason: {result['explanation']}")
```

### Performance Optimization

```python
from app.performance import performance_optimizer

# Clear cache
performance_optimizer.clear_cache()

# Get performance stats
stats = performance_optimizer.get_performance_stats()
print(stats)
```

## 📈 Performance Features

- **Caching**: Intelligent result caching with TTL
- **Batch Processing**: Efficient embedding generation
- **Memory Management**: Optimized FAISS indexing
- **Async Support**: Non-blocking operations

## 🧪 Testing

```bash
# Test individual components
python -c "from app.query_parser import QueryParser; print(QueryParser().parse_query('male actor'))"

# Test search functionality
python -c "from app.retrieval import RetrievalEngine; print(RetrievalEngine().search_profiles('female model'))"

# Run system tests
python setup_talent_search.py --test
```

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | System health check |
| `POST` | `/search_profiles` | Search talent profiles |
| `POST` | `/similar_profiles` | Find similar profiles |
| `POST` | `/filter_profiles` | Filter by criteria |
| `POST` | `/chat` | Chat with AI assistant |
| `GET` | `/profile/{id}` | Get profile details |
| `POST` | `/process_data` | Process raw data |
| `POST` | `/build_index` | Build search index |
| `GET` | `/stats` | System statistics |
| `GET` | `/query_explain` | Explain query parsing |

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Restart Ollama
   ollama serve
   ```

2. **Memory Issues**
   ```bash
   # Clear cache and rebuild
   python setup_talent_search.py --force
   ```

3. **No Search Results**
   ```bash
   # Check if index is built
   curl http://localhost:8000/health
   
   # Rebuild index
   python setup_talent_search.py --index --force
   ```

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -uvicorn app.talent_api:app --reload --log-level debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - feel free to use and modify for your projects.

## 👥 Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation at `/docs`
- Examine system logs for detailed error information

---

**Built with ❤️ for the entertainment industry**
