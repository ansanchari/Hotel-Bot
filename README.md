
# StayChat AI: Scalable & Grounded Hotel RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline built for the StayChat Developer Assessment.


## Architecture & Core Features

1. **Deterministic Guardrails (The "Make-or-Break")**
Prompt engineering is probabilistic; production safety requires determinism. This system implements a post-generation Regex Hallucination Catcher that mathematically verifies all prices, fees, and `http` links against the retrieved FAISS chunks before outputting to the user. Unverified data triggers an automatic human-handoff.
2. **Hybrid Intent Router for Fault Tolerance**
To ensure high availability at scale, the intent classifier utilizes a hybrid approach. It attempts primary classification via the LLM, but gracefully falls back to a localized, regex-based NLP matching system (supporting English and Hinglish) if the API experiences congestion or rate-limiting.
3. **Model-Agnostic Vector Retrieval**
Uses a Context-Injected Sliding Window chunking strategy to map unstructured hotel data into an in-memory `faiss-cpu` index. The pipeline is currently optimized for 1024-dimensional embeddings via Mistral AI.
4. **Multi-Turn Context Window**
Maintains a rolling conversational state to accurately resolve anaphoric references (e.g., understanding that "how far is the airport?" refers to "PDX" from the previous turn).

## Tech Stack

* **Language:** Python 3.10+
* **LLM & Embeddings:** Mistral AI SDK v2.0 (`mistral-small-latest`, `mistral-embed`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Environment Management:** `python-dotenv`

## Project Structure

* `indexer.py` - Data ingestion engine. Chunks raw data, calls the embedding API, and builds the 1024-dimensional FAISS index.
* `bot.py` - The core RAG engine containing the retrieval math, intent router, multi-turn memory, and guardrail logic.
* `test_eval.py` - The automated evaluation suite. Runs the bot against 10 strict edge cases (including multi-lingual and trap questions) to prove guardrail efficacy.

## Setup Instructions

**1. Clone the repository:**

```bash
git clone <your-repo-url>
cd staychat-hotel-bot

```

**2. Create a virtual environment and install dependencies:**

```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install faiss-cpu mistralai python-dotenv numpy

```

**3. Environment Variables:**
Create a `.env` file in the root directory and add your API key. *(Note: .env is gitignored for security)*.

```env
MISTRAL_API_KEY=your_mistral_api_key_here

```

## 💻 Run Commands

**1. Generate the FAISS Database (Data Ingestion)**
Run this to embed the knowledge base and generate the `faiss_index.bin` file.

```bash
python indexer.py

```

**2. Run the Automated Evaluation Suite (Quality Assurance)**
Runs the bot against 10 predefined test cases to demonstrate the intent router, memory, and hallucination catchers.

```bash
python test_eval.py

```

**3. Run the Interactive Bot (Live Terminal Chat)**
Boot up the engine to test the conversational interface manually.

```bash
python bot.py

```