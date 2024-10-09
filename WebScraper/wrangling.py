import json
from typing import List, Dict, Set, Tuple
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup
import re

class JSONCombiner:
    def __init__(self):
        self.duplicate_indices: Dict[str, List[Tuple[int, str]]] = {}
        self.unique_entries: List[Dict] = []
        self.hash_set: Set[str] = set()
        
    def clean_html(self, html_content: str) -> str:
        """
        Clean HTML content by:
        1. Removing js-post-notice elements
        2. Removing unnecessary HTML attributes
        3. Preserving only essential HTML tags for content structure
        """
        if not isinstance(html_content, str):
            return ""
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove all elements with class 'js-post-notice'
        for notice in soup.find_all(class_='js-post-notice'):
            notice.decompose()
            
        # Remove unnecessary attributes from all tags
        for tag in soup.find_all(True):
            allowed_attrs = {'href': True} if tag.name == 'a' else {}
            attrs = dict(tag.attrs)
            for attr in attrs:
                if attr not in allowed_attrs:
                    del tag[attr]
                    
        # Only keep essential HTML tags
        allowed_tags = {'p', 'code', 'a'}
        for tag in soup.find_all():
            if tag.name not in allowed_tags:
                tag.unwrap()
                
        # Convert back to string and clean up any extra whitespace
        cleaned_html = str(soup)
        cleaned_html = re.sub(r'\s+', ' ', cleaned_html).strip()
        
        return cleaned_html
    
    def clean_question_content(self, question_content: str) -> str:
        """
        Clean the question content for comparison:
        1. Remove js-post-notice elements
        2. Remove HTML tags
        3. Normalize whitespace
        4. Convert to lowercase
        """
        # First remove js-post-notice elements
        cleaned_html = self.clean_html(question_content)
        
        # Remove all HTML tags
        soup = BeautifulSoup(cleaned_html, 'html.parser')
        text = soup.get_text()
        
        # Normalize whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        return text
    
    def extract_question_content(self, entry: Dict) -> str:
        """
        Extract and clean the question content from an entry.
        """
        question = entry.get('question', '')
        if isinstance(question, dict):
            # If question is a dictionary, assume it contains the actual content
            question = question.get('content', '')
        
        return self.clean_question_content(question)
    
    def hash_entry(self, entry: Dict) -> str:
        """
        Create a hash of just the question content for duplicate detection.
        Ignores answers completely for duplicate checking.
        """
        question_content = self.extract_question_content(entry)
        return hashlib.md5(question_content.encode()).hexdigest()
    
    def process_file(self, file_path: str, file_index: int) -> List[Dict]:
        """
        Process a single JSON file:
        1. Clean HTML from both questions and answers
        2. Track duplicates based on questions only
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        processed_entries = []
        for idx, entry in enumerate(data):
            processed_entry = {}
            
            # Clean question
            if 'question' in entry:
                if isinstance(entry['question'], str):
                    processed_entry['question'] = self.clean_html(entry['question'])
                elif isinstance(entry['question'], dict):
                    processed_entry['question'] = {
                        'content': self.clean_html(entry['question'].get('content', ''))
                    }
            
            # Clean answers
            if 'answers' in entry:
                processed_entry['answers'] = [
                    self.clean_html(answer) for answer in entry['answers']
                    if isinstance(answer, str)
                ]
            
            # Check for duplicates based on question content
            entry_hash = self.hash_entry(processed_entry)
            
            if entry_hash in self.hash_set:
                # Track duplicate
                if entry_hash not in self.duplicate_indices:
                    self.duplicate_indices[entry_hash] = []
                self.duplicate_indices[entry_hash].append((idx, file_path))
            else:
                # Add new unique entry
                self.hash_set.add(entry_hash)
                processed_entries.append(processed_entry)
                
        return processed_entries
    
    def combine_files(self, file_paths: List[str], output_path: str) -> Dict:
        """
        Combine multiple JSON files and remove duplicates based on questions.
        Returns statistics about the operation.
        """
        total_entries = 0
        original_counts = {}
        
        # Process each file
        for i, file_path in enumerate(file_paths):
            entries = self.process_file(file_path, i)
            original_counts[file_path] = len(entries)
            total_entries += len(entries)
            self.unique_entries.extend(entries)
            
        # Write combined data to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.unique_entries, f, indent=2)
            
        # Calculate statistics
        stats = {
            'original_file_counts': original_counts,
            'total_original_entries': total_entries,
            'unique_entries': len(self.unique_entries),
            'duplicates_removed': total_entries - len(self.unique_entries),
            'duplicate_indices': self.duplicate_indices
        }
        
        return stats

def print_duplicate_report(stats: Dict) -> None:
    """
    Print a detailed report about the duplicate removal process.
    """
    print("\n=== Duplicate Removal Report ===")
    print("\nOriginal File Counts:")
    for file_path, count in stats['original_file_counts'].items():
        print(f"  {Path(file_path).name}: {count} entries")
        
    print(f"\nTotal original entries: {stats['total_original_entries']}")
    print(f"Unique entries: {stats['unique_entries']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    
    if stats['duplicates_removed'] > 0:
        print("\nDuplicate Question Groups:")
        for hash_val, locations in stats['duplicate_indices'].items():
            print(f"\nDuplicate group:")
            for idx, file_path in locations:
                print(f"  - Index {idx} in {Path(file_path).name}")

def main():
    # File paths
    file_paths = [
        'data/data_science.json',
        'data/machine_learning.json',
        'data/artificial_intelligence.json'
    ]
    output_path = 'data/combined_data.json'
    
    # Create and run the combiner
    combiner = JSONCombiner()
    
    try:
        stats = combiner.combine_files(file_paths, output_path)
        print_duplicate_report(stats)
    except Exception as e:
        print(f"Error processing files: {str(e)}")

if __name__ == "__main__":
    main()