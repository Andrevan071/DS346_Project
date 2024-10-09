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


def scrape_ds():
    items = []
    
    for page_number in tqdm(range(0, 5)):
        url = f"https://stackoverflow.com/questions/tagged/data-science?tab=votes&pagesize=50&page={page_number+1}"
        
        page = backoff_request(url)
        soup = BeautifulSoup(page.text, "html.parser")
        
        results = soup.find(id="questions")
        questions = results.find_all(class_="s-link")
        
        for question in tqdm(questions):
            answers_list = []
            question_link = question.get('href')
            q_link = "https://stackoverflow.com" + question_link
            
            q_page = backoff_request(q_link)
            document = BeautifulSoup(q_page.text, "html.parser")
            
            q = document.find(id="question").find(class_="js-post-body")
            q_body = str(q)
            
            answers = document.find_all(class_="js-answer")
            answer_count = 0
            for answer in answers:
                answer_score = answer.find(class_="fs-subheading")
                if answer_score:
                    if int(answer_score.get_text(strip=True)) <= 0 or answer_count >= 3:
                        break
                    
                    a = answer.find(class_="js-post-body")
                    a_body = str(a)
                    answers_list.append(a_body)
                    
                    answer_count += 1
                else:
                    break
                
            if answers_list:
                item = json_items(q_body, answers_list)
                items.append(item)
    
    data = {
        "items": items
    }
    with open("data/data_science.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def scrape_ml():
    items = []
    
    for page_number in tqdm(range(0, 8)):
        url = f"https://stackoverflow.com/questions/tagged/machine-learning?tab=votes&pagesize=50&page={page_number+1}"
        
        page = backoff_request(url)
        soup = BeautifulSoup(page.text, "html.parser")
        
        results = soup.find(id="questions")
        questions = results.find_all(class_="s-link")
        
        for question in tqdm(questions):
            answers_list = []
            question_link = question.get('href')
            q_link = "https://stackoverflow.com" + question_link
            
            q_page = backoff_request(q_link)
            document = BeautifulSoup(q_page.text, "html.parser")
            
            q = document.find(id="question").find(class_="js-post-body")
            q_body = str(q)
            
            answers = document.find_all(class_="js-answer")
            answer_count = 0
            for answer in answers:
                answer_score = answer.find(class_="fs-subheading")
                if answer_score:
                    if int(answer_score.get_text(strip=True)) <= 0 or answer_count >= 3:
                        break
                    
                    a = answer.find(class_="js-post-body")
                    a_body = str(a)
                    answers_list.append(a_body)
                                        
                    answer_count += 1
                else:
                    break
                
            if answers_list:
                item = json_items(q_body, answers_list)
                items.append(item)
    
    data = {
        "items": items
    }
    with open("data/machine_learning.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        
def scrape_ai():
    items = []
    
    for page_number in tqdm(range(0, 8)):
        url = f"https://stackoverflow.com/questions/tagged/artificial-intelligence?tab=votes&pagesize=50&page={page_number+1}"
        
        page = backoff_request(url)
        soup = BeautifulSoup(page.text, "html.parser")
        
        results = soup.find(id="questions")
        questions = results.find_all(class_="s-link")
        
        for question in tqdm(questions):
            answers_list = []
            question_link = question.get('href')
            q_link = "https://stackoverflow.com" + question_link
            
            q_page = backoff_request(q_link)
            document = BeautifulSoup(q_page.text, "html.parser")
            
            q = document.find(id="question").find(class_="js-post-body")
            q_body = str(q)
            
            answers = document.find_all(class_="js-answer")
            answer_count = 0
            for answer in answers:
                answer_score = answer.find(class_="fs-subheading")
                if answer_score:
                    if int(answer_score.get_text(strip=True)) <= 0 or answer_count >= 3:
                        break
                    
                    a = answer.find(class_="js-post-body")
                    a_body = str(a)
                    answers_list.append(a_body)
                    
                    answer_count += 1
                else:
                    break
            if answers_list:
                item = json_items(q_body, answers_list)
                # item = json_items(q, answers_list)
                items.append(item)
    
    data = {
        "items": items
    }
    with open("data/artificial_intelligence.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Scrape StackOverflow questions and answers")
    parser.add_argument('--tag', type=str, required=True, help="Which tag should be scraped")
    args = parser.parse_args()
    if args.tag == "ds":
        scrape_ds()
    elif args.tag == "ml":
        scrape_ml()
    elif args.tag == "ai":
        scrape_ai()


if __name__ == "__main__":
    main()