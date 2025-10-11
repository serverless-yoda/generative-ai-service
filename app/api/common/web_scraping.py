# app/api/common/web_scraping.py

import re, asyncio, aiohttp
from bs4 import BeautifulSoup
from loguru import logger


def extract_url(text: str) -> list[str]:
    url_pattern = r"(?P<url>https?:\/\/[^\s]+)"
    return re.findall(url_pattern,text)

def parse_inner_text(html_string: str) -> str:
    soup = BeautifulSoup(html_string, "lxml")
    if content := soup.find("div",id="bodyContent"):
        return content.get_text()
    logger.warning("Could not parse HTML Content")
    return ""

async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:
        html_string = await response.text()
        return parse_inner_text(html_string)

async def fetch_all_urls(urls: list[str]) -> str:
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[fetch(session, url) for url in urls], return_exceptions=True
        )
    success_results = [results for result in results if isinstance(result, str)]
    if len(results) != len(success_results):
        logger.warning('Some URL could not be fetch')
    return " ".join(success_results)