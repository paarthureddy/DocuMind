"""Quick debug script to trace the full gender filtering pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from query_parser import QueryParser
from retrieval import RetrievalEngine

parser = QueryParser()
parsed = parser.parse_query("female villain")

print("=" * 60)
print("PARSED QUERY:")
print(f"  target_gender  = {parsed.get('target_gender')!r}")
print(f"  target_role    = {parsed.get('target_role')!r}")
print(f"  target_roles   = {parsed.get('target_roles')!r}")
print(f"  filters        = {parsed.get('filters')!r}")
print(f"  semantic_query = {parsed.get('semantic_query')!r}")
print("=" * 60)

# Now do a real search
engine = RetrievalEngine()
result = engine.search_profiles("female villain", max_results=10, enable_reranking=True)

if result["success"]:
    print(f"\nReturned {len(result['results'])} profiles:")
    for i, p in enumerate(result["results"][:10]):
        gender = p["metadata"].get("gender", "???")
        name = p.get("name", "???")
        score = p.get("score", 0)
        print(f"  {i+1}. {name} | Gender: {gender} | Score: {score}")
else:
    print(f"ERROR: {result.get('error')}")
