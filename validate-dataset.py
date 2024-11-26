import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from openai import OpenAI
import os
from dotenv import load_dotenv

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('json_validation.log'),
            logging.StreamHandler()
        ]
    )

def initialize_openai() -> Optional[OpenAI]:
    """Initialize OpenAI client with error handling"""
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        logging.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return None
    
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {str(e)}")
        return None

def fix_line_with_openai(client: OpenAI, line: str, max_retries: int = 3) -> Optional[str]:
    """Attempt to fix JSON line using OpenAI API with retry logic"""
    # List of common example patterns to filter out
    example_patterns = [
        '"firstName": "John"',
        '"lastName": "Smith"',
        '"streetAddress": "123',
        'Anytown',
        'example',
        'foo',
        'bar'
    ]
    
    prompt = """Fix this JSON line to make it valid JSON. 
    Important: Do not create example data or placeholder values.
    Only fix the syntax of the existing data.
    If the line cannot be fixed, return null.
    
    Input line to fix:
    {line}
    """
    
    for attempt in range(max_retries):
        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt.format(line=line),
                max_tokens=500,
                temperature=0.1,  # Reduced temperature for more deterministic output
                n=1
            )
            fixed_line = response.choices[0].text.strip()
            
            # Skip if the response contains any example patterns
            if any(pattern in fixed_line for pattern in example_patterns):
                logging.warning("OpenAI returned example data - skipping")
                continue
                
            # Skip if the response is too different from the original
            if len(fixed_line) < len(line) * 0.5 or len(fixed_line) > len(line) * 1.5:
                logging.warning("OpenAI returned significantly different data - skipping")
                continue
            
            # Verify the fixed line is valid JSON
            try:
                json_obj = json.loads(fixed_line)
                # Additional validation to ensure it's not an example
                if isinstance(json_obj, dict) and len(json_obj) <= 2:
                    logging.warning("Response too simple - likely an example")
                    continue
                return fixed_line
            except json.JSONDecodeError:
                logging.warning(f"OpenAI returned invalid JSON on attempt {attempt + 1}")
                continue
                
        except Exception as e:
            logging.error(f"OpenAI API error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            continue
    
    return None

def validate_and_fix_jsonl(input_file: str, output_file: str, use_openai: bool = False) -> Dict[str, int]:
    """Validate and fix JSONL file with detailed statistics"""
    stats = {
        "total_lines": 0,
        "valid_lines": 0,
        "fixed_lines": 0,
        "failed_lines": 0
    }
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{input_file}' does not exist")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI client if needed
    client = initialize_openai() if use_openai else None
    if use_openai and not client:
        logging.error("OpenAI initialization failed. Proceeding without OpenAI fixes.")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                stats["total_lines"] += 1
                line = line.strip()
                
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # First try to parse as-is
                    json_obj = json.loads(line)
                    outfile.write(json.dumps(json_obj) + '\n')
                    stats["valid_lines"] += 1
                    
                except json.JSONDecodeError:
                    # Try basic fixes first
                    fixed_line = line.replace("'", '"')  # Replace single quotes
                    
                    try:
                        json_obj = json.loads(fixed_line)
                        outfile.write(json.dumps(json_obj) + '\n')
                        stats["fixed_lines"] += 1
                        
                    except json.JSONDecodeError:
                        if use_openai and client:
                            # Try OpenAI fix as last resort
                            fixed_json = fix_line_with_openai(client, line)
                            if fixed_json:
                                outfile.write(fixed_json + '\n')
                                stats["fixed_lines"] += 1
                                continue
                        
                        logging.warning(f"Failed to fix line {line_num}: {line[:100]}...")
                        stats["failed_lines"] += 1
                
                if line_num % 1000 == 0:
                    logging.info(f"Processed {line_num} lines...")
        
        logging.info("Validation complete. Statistics:")
        for key, value in stats.items():
            logging.info(f"{key}: {value}")
        
        return stats
    
    except Exception as e:
        logging.error(f"Critical error during validation: {str(e)}")
        raise

if __name__ == "__main__":
    setup_logging()
    
    try:
        input_file = "dataset.jsonl"
        output_file = "validated_dataset.jsonl"
        use_openai = True  # Set to False to disable OpenAI fixes
        
        stats = validate_and_fix_jsonl(input_file, output_file, use_openai)
        logging.info("Process completed successfully")
        
    except Exception as e:
        logging.error(f"Program terminated with error: {str(e)}")
