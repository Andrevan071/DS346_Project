import json
from bs4 import BeautifulSoup
import re
import os
from tqdm import tqdm #type: ignore
import html
from unidecode import unidecode #type: ignore

def unicode_to_html_to_ascii(text):
    text = html.unescape(text)

    text = unidecode(text)

    text = re.sub(r'&#\d+;', '?', text)

    return text


def wrangle_content(content):
    soup = BeautifulSoup(content, 'html.parser')

    # Remove the post notice board, which contains the closed or locked reason and information.
    for board in soup.find_all(class_='js-post-notice'):
        board.decompose()

    allowed_tags = {'p', 'code', 'a', 'ol', 'li', 'ul', 'strong', 'b', 'i', 'u', 'mark', 'small', 'sub', 'sup', 'span', 'table'}
    for tag in soup.find_all():
        if tag.name not in allowed_tags:
            tag.unwrap()

    for a_tag in soup.find_all('a'):
        href = a_tag.get('href', '')
        a_tag.attrs.clear()
        a_tag['href'] = href

    text = soup.get_text().strip()
    text = unicode_to_html_to_ascii(text)
    
    return text

def wrangle_scraped_data(dir):
    combined_data = []

    for filename in os.listdir(dir):
        if filename.endswith('.json'):
            with open(os.path.join(dir, filename), 'r') as file:
                data = json.load(file)
                combined_data.extend(data)

    wrangled_data = []
    for item in tqdm(combined_data):
        wrangled_item = [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": wrangle_content(item['question'])},
            {"role": "assistant", "content": wrangle_content(item['answers'])}
        ]
        wrangled_data.append(wrangled_item)

    return wrangled_data


wrangled_data = wrangle_scraped_data('data')

data = {
    "conversations": wrangled_data
}

with open('wrangled_data/processed_data.json', 'w') as file:
    json.dump(data, file, indent=2)