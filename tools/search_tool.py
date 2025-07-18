from .base_tool import BaseTool
from ddgs import DDGS
from bs4 import BeautifulSoup
import requests

class SearchTool(BaseTool):
    def __init__(self, config: dict):
        self.config = config
    
    @property
    def name(self) -> str:
        return "search_web"
    
    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo for current information"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find information on the web"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str, max_results: int = 5) -> list:
        """Search the web using DuckDuckGo and fetch page content"""
        try:
            # Use ddgs library
            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)
            
            simplified_results = []
            
            for result in results:
                try:
                    # Fetch content with requests
                    response = requests.get(
                        result['href'], 
                        headers={'User-Agent': self.config.get('search', {}).get('user_agent', 'Mozilla/5.0')},
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    # Parse HTML with BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    # Clean up whitespace
                    text = ' '.join(text.split())
                    
                    # Limit content length
                    content_snippet = text[:1000] + "..." if len(text) > 1000 else text
                    
                    simplified_results.append({
                        "title": result['title'],
                        "url": result['href'],
                        "snippet": result['body'],
                        "content": content_snippet
                    })
                
                except Exception as e:
                    # If we can't fetch the page, still include the search result
                    simplified_results.append({
                        "title": result['title'],
                        "url": result['href'],
                        "snippet": result['body'],
                        "content": f"Could not fetch content: {str(e)}"
                    })
            
            return simplified_results
        
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]