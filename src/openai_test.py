import openai
import os

# Directly set your API key here
openai.api_key = ''

#set file path to .txt file with the question
file_path = 'data/prompts/question.txt'

#extract the question from question.txt
with open(file_path, 'r') as file:
    question = file.read().strip()

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": question} #ask question from the file 
    ],
    max_tokens=50
)

print(response.choices[0].message['content'].strip())




