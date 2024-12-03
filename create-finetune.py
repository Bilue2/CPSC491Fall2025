import os
from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


client.fine_tuning.jobs.create(
  training_file="file-7oU5x8944i5tvnMnmhwNxQ",
  model="gpt-4o-mini-2024-07-18"
)