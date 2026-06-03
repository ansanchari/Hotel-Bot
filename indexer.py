import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env")

client = Mistral(api_key=API_KEY)
EMBEDDING_MODEL = 'mistral-embed'

def create_chunks(data_list, chunk_size=100, overlap=20):
    chunks = []
    for item in data_list:
        category = item['category_name']
        text = item['content']
        
        words = text.split()
        
        if len(words) < chunk_size:
            chunk_text = f"Category: {category} | Details: {' '.join(words)}"
            chunks.append(chunk_text)
            continue
            
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = f"Category: {category} | Details: {' '.join(chunk_words)}"
            chunks.append(chunk_text)
            
    return chunks

def embed_text(text_chunks):
    print(f"Generating embeddings for {len(text_chunks)} chunks via Mistral API...")
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=text_chunks
    )
    
    embeddings = [res.embedding for res in response.data]
    
    return np.array(embeddings).astype('float32')

def main():
    input_path = os.path.join("data", "messy_hotel_kb.json")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find {input_path}. Run scraper.py first.")
        
    with open(input_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    print("Chunking knowledge base...")
    chunks = create_chunks(kb_data, chunk_size=100, overlap=20)
    print(f"Created {len(chunks)} overlapping chunks.")

    embeddings = embed_text(chunks)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, os.path.join("data", "faiss_index.bin"))
    
    with open(os.path.join("data", "chunk_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)
        
    print("Success! Index and chunk mapping saved to the /data folder.")

if __name__ == "__main__":
    main()