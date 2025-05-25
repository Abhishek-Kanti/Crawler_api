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

MAX_CHARS = 1500
OVERLAP = 200

def split_by_documents(raw_text):
    return [doc.strip() for doc in re.split(r'\n{2,}', raw_text) if doc.strip()]

def chunk_with_overlap(text, doc_id="unknown", max_chars=MAX_CHARS, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "doc_id": doc_id,
                "raw_text": chunk_text
            })
        start += max_chars - overlap
    return chunks

def process_text_file_for_training(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    documents = split_by_documents(raw_text)
    all_chunks = []

    for idx, doc in enumerate(documents):
        doc_id = f"doc_{idx+1}"
        doc_chunks = chunk_with_overlap(doc, doc_id)
        all_chunks.extend(doc_chunks)

    return all_chunks

def save_chunks_to_jsonl(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f)
            f.write("\n")

def gemi(final_dataset_path):
    all_response_chunks = []

    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    doc_chunks = defaultdict(list)

    with open("chunked_output.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            doc_chunks[entry["doc_id"]].append(entry["raw_text"])

    for doc_id, chunks in doc_chunks.items():
        context = ""
        response_chunks = []
        for i, chunk in enumerate(chunks):
            prompt = system_prompt + "\n" + f"raw_text: <doc_id = {doc_id}><chunk {i+1}>{chunk}</chunk {i+1}>\n previous_chunk_context: {context}"
            response_obj = model.generate_content(prompt)
            response = response_obj._result.candidates[0].content.parts[0].text
            response_chunks.append({
                f"chunk_{i}": response
            })
            # Save last 150 characters as context (fix from your original chunk[-150] which gives a char)
            context = chunk[-150:]
            time.sleep(4.5)

        all_response_chunks.append({
            "doc_id": doc_id,
            "responses": response_chunks
        })
        print(f"{doc_id} completed.")

    # Write JSONL file: one JSON object per line
    with open(final_dataset_path, "w", encoding="utf-8") as f:
        for item in all_response_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        
    

if __name__ == "__main__":
    # Run this
    input_txt_path = "cleaned_dataset.txt"  # Replace with your actual .txt file path
    output_jsonl_path = "chunked_output.jsonl"
    final_dataset_path = "final_dataset.jsonl"

    chunks = process_text_file_for_training(input_txt_path)
    save_chunks_to_jsonl(chunks, output_jsonl_path)
    gemi(final_dataset_path)
