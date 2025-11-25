from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
load_dotenv()


def google_search(query: str):
    tavily = TavilySearch(tavily_api_key=os.getenv("TAVILY_API_KEY"))
    return tavily.invoke({"query": query})