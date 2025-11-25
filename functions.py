from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv("TAVILY_SEARCH_API_KEY")


def google_search(query) : 
    s = TavilySearch()

    response = s.invoke(query)
    return response