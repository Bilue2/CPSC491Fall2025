import os
import json

def merge_jsonl_files(input_directory, output_file):
    merged_entries = []
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(input_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    entry = json.loads(line.strip())
                    merged_entries.append(entry)
    
    # Write merged entries to the output JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in merged_entries:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    print(f"Merged {len(merged_entries)} entries into {output_file}")

# Main execution block
if __name__ == "__main__":
    input_directory = "jsonl"  # Directory containing JSONL files
    output_file = "merged_dataset.jsonl"  # Output JSONL file
    merge_jsonl_files(input_directory, output_file)
