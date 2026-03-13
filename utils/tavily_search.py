from tavily import TavilyClient
from config.config import TAVILY_API_KEY

client = TavilyClient(api_key=TAVILY_API_KEY)

def search_web(query):
    response = client.search(query=query)
    results_full = response.get("results", [])

    # top 5 retrievals
    top_contents = [r.get("content", "") for r in results_full]
    web_context = "\n".join(top_contents[:5])

    # results here
    web_results = []
    for r in results_full[:6]:
        web_results.append({
            "title": r.get("title", "") or r.get("url", "Result"),
            "url": r.get("url", ""),
            "snippet": (r.get("content", "") or "")[:300]
        })

    return web_context, web_results