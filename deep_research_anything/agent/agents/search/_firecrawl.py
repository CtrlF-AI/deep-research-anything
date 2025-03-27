import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp


load_dotenv()


firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_KEY", ""))


def firecrawl_search(query: str):
    result = firecrawl.search(
        query,
        {"timeout": 15000, "limit": 5, "scrapeOptions": {"formats": ["markdown"]}},
    )
    return result


if __name__ == "__main__":
    result = firecrawl_search("Sam Altman")
