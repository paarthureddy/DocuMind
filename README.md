# DocuMind – AI Research Assistant 🤖

An **Agentic AI Research Assistant** that can answer questions from uploaded documents, perform web searches, and do calculations — all automatically, using local LLMs via Ollama.

![DocuMind UI](static/preview.png)

---

## ✨ Features

- 📄 **Document Upload** – Upload PDF, DOCX, or TXT files
- 🔍 **Document Q&A** – Ask questions, get answers from your documents (RAG)
- 🌐 **Web Search** – Auto-searches the internet when documents don't have the answer
- 🧮 **Calculator** – Solves math expressions on the fly
- 🤖 **Agentic AI** – Automatically decides which tool to use for each query
- 🖥️ **Beautiful UI** – Dark glassmorphism design with real-time chat
- 🏠 **100% Local** – Runs entirely on your machine using Ollama (no API keys needed)

---

## 🏗️ Architecture

```
User Question
      │
      ▼
  LangGraph ReAct Agent (llama3 via Ollama)
      │
      ├── 📄 document_search → FAISS Vector DB → Uploaded Docs
      ├── 🧮 calculator      → Math Expression Evaluator
      └── 🌐 web_search      → DuckDuckGo Search
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Pull the model
```bash
ollama pull llama3
```

### 2. Clone & install dependencies
```bash
git clone https://github.com/paarthureddy/DocuMind.git
cd DocuMind
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the UI
Visit [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📁 Project Structure

```
DocuMind/
├── app/
│   ├── main.py              # FastAPI backend (REST API)
│   ├── agent.py             # LangGraph ReAct agent + tools
│   ├── llm.py               # Ollama LLM config
│   ├── document_loader.py   # PDF/DOCX/TXT loader & chunker
│   ├── embeddings.py        # HuggingFace sentence embeddings
│   ├── vector_store.py      # FAISS vector store management
│   └── rag_pipeline.py      # Standalone RAG pipeline (dev/test)
├── static/
│   ├── index.html           # Frontend UI
│   ├── style.css            # Glassmorphism CSS design
│   └── app.js               # Chat & upload JavaScript logic
├── uploads/                 # Uploaded documents (gitignored)
├── faiss_index/             # Vector DB index (gitignored)
├── requirements.txt
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the UI |
| `GET` | `/health` | Server health check |
| `POST` | `/upload` | Upload & index a document |
| `POST` | `/chat` | Send a message to the agent |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/documents` | Clear all documents |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | [Ollama](https://ollama.com) + llama3 |
| Agent | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | [FAISS](https://github.com/facebookresearch/faiss) |
| Backend | [FastAPI](https://fastapi.tiangolo.com) |
| Frontend | Vanilla HTML + CSS + JS |
| Web Search | [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) |

---

## 📝 License

MIT License – feel free to use and modify for your projects.

---

## 👨‍💻 Author

Built by **Paarthu Reddy** for the Agentic AI assignment.
