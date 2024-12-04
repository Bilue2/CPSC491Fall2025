from openai import OpenAI
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Create fine-tuning job
response = client.fine_tuning.jobs.create(
    training_file="file-7oU5x8944i5tvnMnmhwNxQ",
    model="gpt-3.5-turbo"
)

print(f"Fine-tuning job created: {response}")

# Track the job's progress
job_id = response.id

while True:
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"Status: {job.status}")
    
    if job.status in ['succeeded', 'failed']:
        print(f"Final status: {job.status}")
        if job.fine_tuned_model:
            print(f"Fine-tuned model ID: {job.fine_tuned_model}")
        break
        
    # Check status every 60 seconds
    time.sleep(60)