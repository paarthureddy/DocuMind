import sys
sys.path.append('app')
from app.retrieval import RetrievalEngine

engine = RetrievalEngine()
docs = engine.embeddings_pipeline.similarity_search('villain actor')
with open('debug_out.txt', 'w') as f:
    f.write(f'Type of docs: {type(docs)}\n')
    f.write(f'Length: {len(docs)}\n')
    for i, item in enumerate(docs):
        f.write(f'Item {i} type: {type(item)}\n')
        if isinstance(item, tuple):
            f.write(f'Item {i} length: {len(item)}\n')
            f.write(f'Item {i}[0] type: {type(item[0])}\n')
            if hasattr(item[0], 'metadata'):
                f.write(f'Item {i}[0] has metadata: {item[0].metadata}\n')
            else:
                f.write(f'Item {i}[0] does NOT have metadata attribute\n')
                f.write(f'Item {i}[0] value: {item[0]}\n')
