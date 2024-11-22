import os

# Directory with text files
directory = "sources"

# Load and preprocess text
documents = []
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            text = file.read().strip()
            documents.append({"filename": filename, "content": text})
