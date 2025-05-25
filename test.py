# import json
# from collections import defaultdict

# # Load all chunks from the .jsonl file
# doc_chunks = defaultdict(list)

# with open("chunked_output.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         entry = json.loads(line)
#         doc_chunks[entry["doc_id"]].append(entry["raw_text"])

# # Now iterate over each document and its chunks
# for doc_id, chunks in doc_chunks.items():
#     print(f"\nProcessing document: {doc_id}")
#     for i, chunk in enumerate(chunks):
#         print(f"  Chunk {i+1}: {chunk[-100:]}")  # printing only first 60 characters


import os
import re
import json
import time
from collections import defaultdict
from tqdm import tqdm
import textwrap
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("hi how are you?")
print(response._result.candidates[0].content.parts[0].text)