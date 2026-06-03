import os
import json
import faiss
import numpy as np
import re
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env")

client = Mistral(api_key=API_KEY)
EMBEDDING_MODEL = 'mistral-embed'
LLM_MODEL = 'mistral-small-latest'

index = faiss.read_index(os.path.join("data", "faiss_index.bin"))
with open(os.path.join("data", "chunk_mapping.json"), "r", encoding="utf-8") as f:
    chunk_mapping = json.load(f)

chat_history = []

def classify_intent(query):
    """
    Categorizes the user's input across English, Hindi, and Hinglish.
    Implements a Hybrid Router: Tries LLM first, falls back to local NLP if rate-limited.
    """
    if query.startswith("#") or any(word in query.lower() for word in ["human", "agent", "staff"]):
        return "staff_command"

    prompt = f"""
    Analyze the following user query (which may be in English, Hindi, or Hinglish).
    Classify it into EXACTLY one of these categories:
    - booking_inquiry (asking about rooms, availability, checking in/out, fees)
    - amenity_question (asking about parking, pool, gym, wifi, food, location)
    - complaint (expressing dissatisfaction, broken items, noise)
    - other (greetings, unrelated topics)
    
    Query: "{query}"
    
    Output ONLY the category name. No punctuation or extra text.
    """
    
    try:
        response = client.chat.complete(
            model=LLM_MODEL, 
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().lower()
        
    except Exception as e:
        q_lower = query.lower()
        
        if any(word in q_lower for word in ["book", "check in", "check-in", "check out", "price", "fee", "cost", "how much", "kitna", "paisa", "allowed", "suite"]):
            return "booking_inquiry"
            
        elif any(word in q_lower for word in ["park", "parking", "where", "location", "located", "airport", "shuttle", "breakfast", "buffet", "wifi", "kaha", "kidhar", "food"]):
            return "amenity_question"
            
        elif any(word in q_lower for word in ["broken", "noise", "dirty", "bad", "complain", "issue", "problem"]):
            return "complaint"
            
        else:
            return "other"

def retrieve_context(query, top_k=3):
    embed_response = client.embeddings.create(model=EMBEDDING_MODEL, inputs=[query])
    query_vector = np.array([embed_response.data[0].embedding]).astype('float32')    
    distances, indices = index.search(query_vector, top_k)
    
    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1 and idx < len(chunk_mapping):
            retrieved_chunks.append(chunk_mapping[idx])

    print("\n[DEBUG: RETRIEVED FAISS CHUNKS]")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"Chunk {i+1}")
        print(chunk)
            
    return "\n\n".join(retrieved_chunks)

def hallucination_catcher(llm_response, context_text):
    fallback_message = "I cannot find that information, would you like me to connect you with a human?"
    
    prices = re.findall(r'[\$\₹]\s*\d+(?:\.\d+)?|\b\d+\s*(?:dollars|rupees|Rs)\b', llm_response, re.IGNORECASE)
    links = re.findall(r'https?://[^\s]+|www\.[^\s]+', llm_response, re.IGNORECASE)
    
    for price in prices:
        num_only = re.search(r'\d+(?:\.\d+)?', price).group()
        if num_only not in context_text:
            print(f"\n[GUARDRAIL TRIGGERED: Invented Price detected -> {price}]")
            return fallback_message
            
    for link in links:
        if link not in context_text:
            print(f"\n[GUARDRAIL TRIGGERED: Invented Link detected -> {link}]")
            return fallback_message
            
    return llm_response

def generate_answer(query, context, intent):
    sys_instruct = (
        "You are a helpful, multilingual hotel assistant for the Jupiter Hotel. "
        "You must ONLY answer using the 'PROVIDED CONTEXT'. "
        "If the answer is not in the context, do not guess. Say exactly: "
        "'I cannot find that information, would you like me to connect you with a human?' "
        "Never invent prices or links."
    )
    
    history_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in chat_history[-4:]])
    
    full_prompt = f"""
    RECENT CHAT HISTORY:
    {history_text}
    
    PROVIDED CONTEXT:
    {context}
    
    CURRENT INTENT: {intent}
    USER QUERY: {query}
    
    Answer the query strictly based on the PROVIDED CONTEXT.
    """
    
    response = client.chat.complete(
        model=LLM_MODEL, 
        messages=[
            {"role": "system", "content": sys_instruct},
            {"role": "user", "content": full_prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()

def main():
    print("StayChat AI: (Type 'quit' to exit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        intent = classify_intent(user_input)
        print(f"[Debug] Intent Detected: {intent}")
        
        if intent == "staff_command" or user_input.startswith("#"):
            print("Bot: [SYSTEM] Staff command acknowledged. Pausing AI logic.")
            continue
            
        context = retrieve_context(user_input)
        
        raw_response = generate_answer(user_input, context, intent)
        
        safe_response = hallucination_catcher(raw_response, context)
        
        chat_history.append({"role": "User", "text": user_input})
        chat_history.append({"role": "Bot", "text": safe_response})
        
        print(f"Bot: {safe_response}")

if __name__ == "__main__":
    main()