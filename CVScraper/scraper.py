# Required imports
import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import json
from tqdm import tqdm
import time
from requests.exceptions import RequestException
import logging
import random
import argparse

def backoff_request(url, max_retries=5, initial_delay=1):
    """
    Make HTTP requests with exponential backoff retry logic
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Add a small random delay between requests
            time.sleep(random.uniform(1, 3))
            
            return response
            
        except RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                logging.error(f"Failed to fetch {url} after {max_retries} attempts: {str(e)}")
                raise
            
            # Calculate delay with exponential backoff and jitter
            delay = (2 ** attempt) * initial_delay + random.uniform(0, 1)
            logging.warning(f"Attempt {attempt + 1} failed. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

def json_items(question, answers):
    return {
        "question": question,
        "answers": answers
    }

def main():
    parser = argparse.ArgumentParser(description="Scrape Cross Validated questions and answers")
    parser.add_argument('--page', type=int, required=True, help="Which page should be scraped")
    args = parser.parse_args()
    
    items = []
    for page_number in tqdm(range(args.page, args.page+10)):
        url = f"https://stats.stackexchange.com/questions?tab=votes&pagesize=50&page={page_number}"
        print(url)
        page = backoff_request(url)
        soup = BeautifulSoup(page.text, "html.parser")
        
        results = soup.find(id="questions")
        questions = results.find_all(class_="s-link")
        
        for question in tqdm(questions):
            question_link = question.get('href')
            q_link = "https://stackoverflow.com" + question_link
            
            q_page = backoff_request(q_link)
            document = BeautifulSoup(q_page.text, "html.parser")
            
            q = document.find(id="question").find(class_="js-post-body")
            q_body = str(q)
            a_body = None
            
            answer = document.find_all(class_="js-answer")
            answer_score = answer[0].find(class_="fs-subheading")
            if answer_score:
                if int(answer_score.get_text(strip=True)) > 0:
                    a = answer[0].find(class_="js-post-body")
                    a_body = str(a)

            if a_body:
                item = json_items(q_body, a_body)
                items.append(item)
    
    with open(f"data/page{args.page}-{args.page+10}.json", "w", encoding="utf-8") as file:
        json.dump(items, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()