import sys
sys.path.append('app')
from app.retrieval import RetrievalEngine

try:
    engine = RetrievalEngine()
    print("Engine instantiated.")
    result = engine.search_profiles("villain actor")
    import json
    with open('out.json', 'w') as f:
        json.dump(result, f, indent=2)
except Exception as e:
    import traceback
    traceback.print_exc()
