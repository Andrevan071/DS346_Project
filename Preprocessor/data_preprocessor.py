import json
import re
from bs4 import BeautifulSoup
import html
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pathlib import Path
import pickle
import unicodedata

class DataPreprocessor:
    def __init__(self, min_word_freq = 2, max_vocab_size=50000):
        self.min_word_freq = min_word_freq
        self.max_vocab_size = max_vocab_size
        self.code_vectoriser = CountVectorizer(token_pattern=r'[A-Za-z_][A-Za-z0-9_]*|\S+', max_features=max_vocab_size)
        self.text_vectoriser = CountVectorizer(max_features=max_vocab_size)
    
    def unescape_json(self, text):
        """
        Unescape JSON escaped characters.
        Example: \" -> ", \n -> newline, \t -> tab
        """
        # Define mapping of escaped characters
        escape_mapping = {
            '\\"': '"',    # Escaped quotation mark
            '\\n': ' ',    # Newline to space
            '\\t': ' ',    # Tab to space
            '\\r': ' ',    # Carriage return to space
            '\\\\': '\\',  # Escaped backslash
            '\\/': '/'     # Escaped forward slash
        }
        
        # Replace each escaped character
        for escaped, unescaped in escape_mapping.items():
            text = text.replace(escaped, unescaped)
            
        return text
    
    def decode_html(self, text):
        """
        Decode HTML entities and convert them to ASCII characters. An example would be &gt; is set to >
        """
        return html.unescape(text)
    
    def clean_text(self, text):
        """
        Clean text content by:
        1. Unescaping JSON characters
        2. Decoding HTML entities
        3. Converting to ASCII
        4. Normalizing whitespace
        """
        if not isinstance(text, str):
            return ""
        
        # Unescape JSON characters first
        text = self.unescape_json(text)
        # Decode HTML entities
        text = self.decode_html(text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_code(self, code):
        """
        Preprocesses code by:
        1. Unescaping JSON characters
        2. Removing comments
        3. Normalizing whitespace
        4. Tokenizing special characters
        """
        # Unescape JSON characters first
        code = self.unescape_json(code)
        
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # Single line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Multi-line comments
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        
        # Tokenize special characters
        code = re.sub(r'([^\w\s])', r' \1 ', code)
        
        return code.strip()
    
    def preprocess_text(self, text):
        """
        Preprocesses text by:
        1. Unescaping JSON characters
        2. Converting to lowercase
        3. Removing special characters
        4. Normalizing whitespace
        """
        # Unescape JSON characters first
        text = self.unescape_json(text)
        # Convert to lowercase
        text = text.lower()
        # Remove special characters except periods and spaces
        text = re.sub(r'[^a-z0-9\s\.]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_code_and_text(self, html_content):
        """
        This method separtes the code blocks from the regular text.
        It returns a list of code_blocks and a list of text_blocks
        """
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Extract the code blocks
        code_blocks = []
        code_tags = soup.find_all('code')
        for code in code_tags:
            code_text = code.get_text(strip=True)
            if code_text:
                code_blocks.append(code_text)
            code.decompose()
        
        # Extract the text and links
        text_blocks = []
        text_links = soup.find_all(['p','a'])
        for text in text_links:
            plain_text = text.get_text(strip=True)
            if plain_text:
                text_blocks.append(plain_text)
                
        return code_blocks, text_blocks
    
    def create_vocabulary(self, processed_data):
        """
        Creates separate vocabularies for code and text based on processed data.
        """
        # Combine all code and text separately
        all_code = []
        all_text = []
        
        for entry in processed_data:
            all_code.extend(entry['question_code'])
            all_code.extend(entry['answer_code'])
            all_text.extend(entry['question_text'])
            all_text.extend(entry['answer_text'])
        
        # Fit vectorizers
        self.code_vectoriser.fit(all_code)
        self.text_vectoriser.fit(all_text)
    
    def transform_to_bag_of_words_data(self, processed_entry):
        """
        Transforms processed text and code into bag-of-words representations.
        """
        bag_of_words_data = {}
        
        # Transform code
        if processed_entry['question_code']:
            bag_of_words_data['question_code'] = self.code_vectoriser.transform(processed_entry['question_code']).toarray().sum(axis=0)
        else:
            bag_of_words_data['question_code'] = np.zeros(len(self.code_vectoriser.vocabulary_))
            
        if processed_entry['answer_code']:
            bag_of_words_data['answer_code'] = self.code_vectoriser.transform(
                processed_entry['answer_code']
            ).toarray().sum(axis=0)
        else:
            bag_of_words_data['answer_code'] = np.zeros(len(self.code_vectoriser.vocabulary_))
        
        # Transform text
        if processed_entry['question_text']:
            bag_of_words_data['question_text'] = self.text_vectoriser.transform(processed_entry['question_text']).toarray().sum(axis=0)
        else:
            bag_of_words_data['question_text'] = np.zeros(len(self.text_vectoriser.vocabulary_))
            
        if processed_entry['answer_text']:
            bag_of_words_data['answer_text'] = self.text_vectoriser.transform(processed_entry['answer_text']).toarray().sum(axis=0)
        else:
            bag_of_words_data['answer_text'] = np.zeros(len(self.text_vectoriser.vocabulary_))
        
        return bag_of_words_data
    
    def sanitize_code_block(self, code: str) -> str:
        """
        Sanitize a code block to make it JSON-safe
        """
        # Replace all backslashes first to prevent double escaping
        code = code.replace('\\', '\\\\')
        
        # Escape quotes
        code = code.replace('"', '\\"')
        
        # Replace newlines and other whitespace
        code = code.replace('\n', '\\n')
        code = code.replace('\r', '\\r')
        code = code.replace('\t', '\\t')
        
        # Escape forward slashes in file paths
        code = re.sub(r'(?<!\\)/', '\\/', code)
        
        return code

    def process_code_blocks(self, content: str) -> str:
        """
        Process all code blocks in the content to make them JSON-safe
        """
        def replace_code_block(match):
            code = match.group(1)
            sanitized_code = self.sanitize_code_block(code)
            return f'<code>{sanitized_code}</code>'
        
        # Process each code block
        content = re.sub(r'<code>(.*?)</code>', replace_code_block, content, flags=re.DOTALL)
        return content

    def fix_json_structure(self, content: str) -> str:
        """
        Fix structural issues in the JSON content
        """
        # Remove any BOM and whitespace
        content = content.strip().lstrip('\ufeff')
        
        # Ensure content is wrapped in square brackets
        if not content.startswith('['):
            content = '[' + content
        if not content.endswith(']'):
            content = content + ']'
        
        # Fix missing commas between objects
        content = re.sub(r'}\s*{', '},{', content)
        
        # Remove trailing commas
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        return content

    def validate_json_entry(self, entry: str) -> bool:
        """
        Validate a single JSON entry
        """
        try:
            # Check if the entry has required fields
            if not isinstance(entry, dict):
                return False
            if 'question' not in entry:
                return False
            if 'answers' not in entry or not isinstance(entry['answers'], list):
                return False
            return True
        except Exception:
            return False

    def preprocess_file(self, file_path: str) -> str:
        """
        Preprocess the JSON file with detailed error checking
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create a debug directory
            debug_dir = Path(file_path).parent / 'debug'
            debug_dir.mkdir(exist_ok=True)
            
            # Save original content for comparison
            with open(debug_dir / 'original.txt', 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Process code blocks
            print("Processing code blocks...")
            content = self.process_code_blocks(content)
            
            # Save intermediate result
            with open(debug_dir / 'after_code_processing.txt', 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Fix JSON structure
            print("Fixing JSON structure...")
            content = self.fix_json_structure(content)
            
            # Save preprocessed content
            with open(debug_dir / 'preprocessed.json', 'w', encoding='utf-8') as f:
                f.write(content)
            
            return content
            
        except Exception as e:
            print(f"Error preprocessing file {file_path}: {str(e)}")
            raise

    def load_json_safely(self, file_path: str) -> List[Dict]:
        """
        Load and parse JSON content with extensive error handling
        """
        try:
            # Preprocess the file
            content = self.preprocess_file(file_path)
            
            try:
                # Try to parse the JSON
                data = json.loads(content)
                
                # Validate each entry
                validated_data = []
                for i, entry in enumerate(data):
                    if self.validate_json_entry(entry):
                        validated_data.append(entry)
                    else:
                        print(f"Warning: Skipping invalid entry at index {i}")
                
                if not validated_data:
                    raise ValueError("No valid entries found in the JSON file")
                
                return validated_data
                
            except json.JSONDecodeError as e:
                print(f"\nJSON decode error: {str(e)}")
                
                # Show extended context around the error
                start = max(0, e.pos - 200)
                end = min(len(content), e.pos + 200)
                context = content[start:end]
                
                print("\nExtended context around error:")
                print(context)
                print(" " * (min(200, e.pos - start)) + "^")
                
                # Save error details
                debug_dir = Path(file_path).parent / 'debug'
                with open(debug_dir / 'error_details.txt', 'w', encoding='utf-8') as f:
                    f.write(f"Error position: {e.pos}\n")
                    f.write(f"Error message: {str(e)}\n")
                    f.write("\nContext:\n")
                    f.write(context)
                
                raise
                
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            raise

    
    def process_file(self, input_path, output_path):
        """
        Processes the entire JSON file and creates bag-of-words representations.
        """
        # Load and parse JSON data with better error handling
        data = self.load_json_safely(input_path)
        
        # Process all entries
        processed_data = []
        for entry in data:
            processed_entry = {}
            
            # Process question
            if 'question' in entry:
                question = self.clean_text(entry['question'])
                code_blocks, text_blocks = self.extract_code_and_text(question)
                processed_entry['question_code'] = [self.preprocess_code(code) for code in code_blocks]
                processed_entry['question_text'] = [self.preprocess_text(text) for text in text_blocks]
            
            # Process answers
            if 'answers' in entry:
                processed_entry['answer_code'] = []
                processed_entry['answer_text'] = []
                for answer in entry['answers']:
                    answer = self.clean_text(answer)
                    code_blocks, text_blocks = self.extract_code_and_text(answer)
                    processed_entry['answer_code'].extend(self.preprocess_code(code) for code in code_blocks)
                    processed_entry['answer_text'].extend(self.preprocess_text(text) for text in text_blocks)
            
            processed_data.append(processed_entry)
        
        # Create vocabularies
        self.create_vocabulary(processed_data)
        
        # Transform to bag-of-words
        bag_of_words_dataset = [self.transform_to_bag_of_words_data(entry) for entry in processed_data]
        
        # Save processed data if output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save bag-of-words data
            with open('PreprocessedData/bag_of_words_data.pkl', 'wb') as f:
                pickle.dump(bag_of_words_dataset, f)
            
            # Save vocabularies
            with open('PreprocessedData/code_vocabulary.json', 'w') as f:
                json.dump(str(self.code_vectoriser.vocabulary_), f, indent=2)
                
            with open('PreprocessedData/text_vocabulary.json', 'w') as f:
                json.dump(str(self.text_vectoriser.vocabulary_), f, indent=2)
            
            # Save a sample of processed entries for verification
            with open('PreprocessedData/processed_samples.json', 'w', encoding='utf-8') as f:
                json.dump(processed_data[:5], f, indent=2)
        
        return bag_of_words_dataset

def main():
    try:
        print("Initializing preprocessor...")
        preprocessor = DataPreprocessor(min_word_freq=2, max_vocab_size=50000)
        
        input_path = '../WebScraper/data/combined_data.json'
        output_path = 'PreprocessedData/processed_data'
        
        print(f"\nProcessing file: {input_path}")
        print("A debug directory will be created with intermediate results")
        
        bag_of_words_dataset = preprocessor.process_file(input_path, output_path)
        print("\nProcessing completed successfully")
        
    except Exception as e:
        print(f"\nProcessing failed: {str(e)}")
        print("\nDebug files have been created in the 'debug' directory next to your input file:")
        print("- original.txt: Original file content")
        print("- after_code_processing.txt: Content after processing code blocks")
        print("- preprocessed.json: Final preprocessed JSON")
        print("- error_details.txt: Detailed error information if JSON parsing failed")

if __name__ == "__main__":
    main()