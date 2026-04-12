# Document Assistant (Talent Search Engine) - Technical Documentation

This report provides a comprehensive overview of the technical architecture, workflow, and implementation details of the Talent Search Engine project.

## 1. Project Overview
The Talent Search Engine is an AI-powered platform designed for casting directors and recruiters to find talent (actors, models, performers) using natural language queries. It combines semantic vector search with strict metadata filtering to provide highly relevant casting suggestions.

## 2. Technology Stack & Libraries

The project is built using a modern AI/ML stack:

| Category | Libraries Used | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | `FastAPI`, `Uvicorn` | High-performance REST API and server management. |
| **AI Orchestration** | `LangChain`, `LangGraph` | Managing LLM workflows, tools, and agent logic. |
| **LLM Interface** | `langchain-ollama` | Interfacing with local LLMs (like Llama 3) for processing. |
| **Embeddings** | `sentence-transformers` | Generating semantic vector representations of profiles. |
| **Vector Database** | `FAISS (faiss-cpu)` | Efficient similarity search for high-dimensional vectors. |
| **Data Processing** | `pypdf`, `docx2txt` | Parsing talent resumes and documents. |
| **Search Tools** | `duckduckgo-search` | Web search capabilities for external research. |
| **Utilities** | `pydantic`, `python-dotenv` | Data validation and environment management. |

---

## 3. Full Workflow: Step-by-Step

### Step 1: Data Ingestion & Normalization (`data_pipeline.py`)
1.  **Input**: Raw talent data (usually in JSON format) containing personal and professional info.
2.  **Cleaning**: Values are standardized (e.g., "M" becomes "male", "5'11" becomes "180 cm").
3.  **Semantic Enrichment**: 
    *   The system uses an LLM to generate a **Rich Semantic Profile**—a descriptive paragraph highlighting the talent's suitability for specific roles (e.g., "ideal for intense villain roles").
    *   **Casting Tags** are extracted (e.g., `["villain", "antagonist", "action"]`).
4.  **Storage**: Processed data is saved to `processed_profiles.json`.

### Step 2: Indexing (`embeddings_pipeline.py`)
1.  **Vectorization**: The system takes the "Rich Semantic Profile" of every talent and converts it into a numerical vector (embedding) using `sentence-transformers`.
2.  **Indexing**: These vectors are stored in a **FAISS Index**. 
3.  **Persistence**: The index is saved to the `faiss_index` directory, allowing fast lookups without re-calculating embeddings.

### Step 3: Query Handing (`intent_classifier.py` & `query_parser.py`)
When a user types a query (e.g., *"Find a tall male villain with brown skin"*):
1.  **Intent Classification**: The system determines if the query is a **Talent Search**, a **Calculation**, or **General Chat**.
2.  **Natural Language Parsing**: If it's a search, the `QueryParser` extracts:
    *   **Hard Filters**: Gender (Male), Height (Tall), Complexion (Brown).
    *   **Semantic Keywords**: "Villain".

### Step 4: Hybrid Retrieval (`retrieval.py`)
The system performs a two-stage retrieval:
1.  **Vector Search**: Finds profiles that are semantically similar to "Villain" or "Antagonist".
2.  **Metadata Filtering**: Filters the vector results using strict criteria (must be Male, must be Tall).
3.  **Hybrid Merge**: Combines both to ensure the results match the description *and* the specific physical requirements.

### Step 5: Ranking & Scoring (`ranking.py`)
1.  **Score Normalization**: Raw similarity scores (distances) are converted into a 0-100% percentage.
2.  **Multi-Factor Ranking**: Final scores are calculated based on:
    *   **Semantic Match**: How well they fit the description.
    *   **Experience Factor**: Higher weight for more years of experience.
    *   **Rating Factor**: Influence from user ratings.
3.  **Final Response**: The top sorted results are returned to the user with an explanation of *why* they were selected.

---

## 4. Specific Query Type Handling
 
| Query Type | Handling Logic | Core Tool |
| :--- | :--- | :--- |
| **Casting Search** | Hybrid Vector search + Metadata filters + Experience ranking. | `RetrievalEngine` |
| **Similarity Search** | "Find someone like Profile X" - uses vector distance between profiles. | `FAISS index` |
| **Mathematical** | Extracted via regex/intent; solved using a calculator tool. | `LangChain Calculator` |
| **General Info** | Answered via the AI Agent's internal knowledge or Web Search. | `Ollama / DDG Search` |

---

## 5. Directory Structure
*   `/app`: Core logic (API, Agent, Pipelines).
*   `/data`: Source raw data files.
*   `/faiss_index`: Local vector database storage.
*   `/static`: Frontend UI files (HTML, CSS, JS).
*   `/uploads`: Directory for newly uploaded documents.

---

## 6. How to Run the Project
1.  **Install Dependencies**: `pip install -r requirements.txt`
2.  **Environment Setup**: Configure `.env` with model names and API keys.
3.  **Data Initialization**: Run `python app/data_pipeline.py` to process profiles.
4.  **Index Building**: Run `python app/embeddings_pipeline.py` to build the search index.
5.  **Start Server**: `uvicorn app.main:app --reload`
6.  **Access UI**: Open `http://localhost:8000/ui` in your browser.
