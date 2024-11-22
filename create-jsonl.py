import os
import json


# Function to process all text files in the /sources directory and create JSONL entries
def process_all_text_files(directory="sources/"):
    entries = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            entry = process_text_file(file_path)  # Process each text file to create an entry
            entries.append(entry)
    return entries

# Main execution block
if __name__ == "__main__":
    entries = process_all_text_files()  # Process all text files in the /sources directory
    save_to_jsonl(entries)  # Save all entries to a JSONL file

# Function to read the content of a text file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to create a JSONL entry with the specified format
def create_jsonl_entry(query, context, answer):
    return {
        "query": query,
        "context": context,
        "answer": answer
    }

# Function to save a list of entries to a JSONL file
def save_to_jsonl(entries, output_file="dataset.jsonl"):
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in entries:
            json.dump(entry, file)
            file.write('\n')
    print(f"JSONL file has been saved to {output_file}")

# Function to process a text file and create a JSONL entry
def process_text_file(file_path):
    text_content = read_text_file(file_path)  # Read the text file content
    # Define a fixed query and answer for demonstration purposes
    query = "What is the main topic of the document?"
    context = text_content  # Use the entire text as the context
    answer = "The document discusses emergency alert systems and related issues."
    return create_jsonl_entry(query, context, answer)  # Create a JSONL entry

# Main execution block
if __name__ == "__main__":
    input_file = "sources/gao-agencies.txt"  # Specify the input text file
    entry = process_text_file(input_file)  # Process the text file to create an entry
    save_to_jsonl([entry])  # Save the entry to a JSONL file

