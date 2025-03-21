from duckduckgo_search import DDGS
import json

def web_search(query, max_results=5):
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
    
    return json.dumps({
        "results": [{
            "title": r["title"],
            "snippet": r["body"],
            "url": r["href"]
        } for r in results]
    })