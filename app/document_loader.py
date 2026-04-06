import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_document(file_path: str):
    """Load and split a document (PDF, DOCX, TXT, or JSON) into chunks."""
    import json
    from langchain_core.documents import Document

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".json":
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    name = item.get('name', 'Unknown Actor')
                    pid = item.get('id', 'Unknown ID')
                    
                    personal = item.get('personal_info', {})
                    prof = item.get('professional_info', {})
                    crafts = prof.get('crafts', {}).get('primary', {})
                    tags = item.get('appearance_tags', [])
                    skills = prof.get('skills', [])
                    loc = item.get('location', {})
                    rating = item.get('rating', {})
                    avg_rating = rating.get('average', 'N/A')
                    
                    content = f"Actor Name: {name} (ID: {pid})\n"
                    content += f"Age: {personal.get('age')}\n"
                    content += f"Gender: {personal.get('gender')}\n"
                    content += f"Height: {personal.get('height_cm')} cm\n"
                    content += f"Build: {personal.get('build')}\n"
                    content += f"Eye color: {personal.get('eye_color')}\n"
                    content += f"Hair color: {personal.get('hair_color')}\n"
                    content += f"Complexion: {personal.get('complexion')}\n"
                    content += f"Role/Craft: {crafts.get('craft', '')} ({crafts.get('subcraft', '')})\n"
                    content += f"Skills: {', '.join(skills) if skills else 'None'}\n"
                    content += f"Appearance Tags/Features: {', '.join(tags) if tags else 'None'}\n"
                    content += f"City/Location: {loc.get('current_city', '')}, {loc.get('country', '')}\n"
                    content += f"Experience: {prof.get('experience_years')} years\n"
                    content += f"Rating: {avg_rating}\n"
                    content += f"Original Search String: {item.get('search_text', '')}\n"
                    
                    doc = Document(page_content=content, metadata={"source": file_path, "page": name})
                    documents.append(doc)
            else:
                doc = Document(page_content=json.dumps(data), metadata={"source": file_path})
                documents.append(doc)
        
        # Because we already structured the actor data perfectly into logical 1-profile chunks,
        # we do not need to recursive split these, they are already ideal sizes for RAG semantic search.
        return documents

    # For other standard parsing text docs:
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    return chunks