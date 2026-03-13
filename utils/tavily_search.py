from tavily import TavilyClient
from config.config import TAVILY_API_KEY

client = TavilyClient(api_key=TAVILY_API_KEY)

def search_web(query):
    response = client.search(query=query)
    results_full = response.get("results", [])

    # context = top 3 contents joined for the LLM context
    top_contents = [r.get("content", "") for r in results_full]
    web_context = "\n".join(top_contents[:3])

    # web_results = structured list for UI (title, url, snippet)
    web_results = []
    for r in results_full[:6]:
        web_results.append({
            "title": r.get("title", "") or r.get("url", "Result"),
            "url": r.get("url", ""),
            "snippet": (r.get("content", "") or "")[:300]
        })

    return web_context, web_results