import asyncio
import aiohttp #type: ignore
from bs4 import BeautifulSoup
import json
import logging
import random
import argparse
from aiohttp import ClientError #type: ignore
from tqdm.asyncio import tqdm_asyncio #type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RateLimiter:
    def __init__(self, rate_limit=5, period=60):
        self.rate_limit = rate_limit
        self.period = period
        self.request_times = asyncio.Queue(maxsize=rate_limit)
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            if self.request_times.full():
                oldest_request_time = await self.request_times.get()
                sleep_time = max(0, self.period - (now - oldest_request_time))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            await self.request_times.put(now)

async def backoff_request(session, url, rate_limiter, max_retries=5, initial_delay=1):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            await rate_limiter.acquire()
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.text()
        except ClientError as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to fetch {url} after {max_retries} attempts: {str(e)}")
                raise
            delay = (2 ** attempt) * initial_delay + random.uniform(0, 1)
            logging.warning(f"Attempt {attempt + 1} failed. Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)

def json_items(question, answers):
    return {
        "question": question,
        "answers": answers
    }

def contains_code_tag(html):
    return '<pre' in html

async def fetch_question(session, question_link, rate_limiter):
    q_link = "https://stats.stackexchange.com" + question_link
    q_page = await backoff_request(session, q_link, rate_limiter)
    document = BeautifulSoup(q_page, "html.parser")
    
    q = document.find(id="question").find(class_="js-post-body")
    q_body = str(q)
    
    if contains_code_tag(q_body):
        return None
    
    a_body = None
    
    answer = document.find_all(class_="js-answer")
    if answer:
        answer_score = answer[0].find(class_="fs-subheading")
        if answer_score and int(answer_score.get_text(strip=True)) > 0:
            a = answer[0].find(class_="js-post-body")
            a_body = str(a)
            
            if contains_code_tag(a_body):
                return None

    if a_body:
        return json_items(q_body, a_body)
    return None

async def fetch_page(session, page_number, rate_limiter):
    url = f"https://stats.stackexchange.com/questions?tab=votes&pagesize=50&page={page_number}"
    page = await backoff_request(session, url, rate_limiter)
    soup = BeautifulSoup(page, "html.parser")
    
    results = soup.find(id="questions")
    questions = results.find_all(class_="s-link")
    
    tasks = [fetch_question(session, question.get('href'), rate_limiter) for question in questions]
    return [item for item in await tqdm_asyncio.gather(*tasks) if item]

async def scrape_page_range(session, start_page, end_page, rate_limiter):
    tasks = [fetch_page(session, page_number, rate_limiter) 
             for page_number in range(start_page, end_page)]
    results = await tqdm_asyncio.gather(*tasks, desc=f"Scraping pages {start_page}-{end_page}")
    
    items = [item for page_items in results for item in page_items]
    
    with open(f"data/page{start_page}-{end_page}.json", "w", encoding="utf-8") as file:
        json.dump(items, file, indent=4, ensure_ascii=False)

async def main():
    pages = [1, 102, 203, 304, 405, 506, 607, 708, 809, 910]
    
    rate_limiter = RateLimiter(rate_limit=6, period=8)  # 6 requests per 8 seconds

    async with aiohttp.ClientSession() as session:
        for page in pages:
            await scrape_page_range(session, page, page + 100, rate_limiter)
            await asyncio.sleep(10)  # Add a delay between page ranges to avoid overloading

if __name__ == "__main__":
    asyncio.run(main())