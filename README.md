# YouTube Transcript Summarizer & Q&A (Flask)

## Features
- Fetch YouTube transcript from a URL
- Build a local FAISS vector index (SentenceTransformer embeddings)
- Summarize transcript using OpenRouter LLM (Llama model)
- Answer questions grounded in transcript using OpenRouter LLM (Qwen model)

## Setup
1. Python 3.11+
2. Install deps:
```bash
py -3.11 -m venv .IBM
pip install -r requirements.txt
```
3. Set your OpenRouter API key as an environment variable (do NOT hardcode secrets):
   - Windows PowerShell:
```powershell
$env:OPENROUTER_API_KEY="YOUR_KEY_HERE"
```
4. Optionally specify exact model slugs (defaults are reasonable free options, adjust if needed):
```powershell
$env:QWEN_MODEL="qwen/qwen2.5-coder-32b-instruct"
$env:LLAMA_MODEL="meta-llama/llama-3.1-70b-instruct"
```
5. Run the app:
```bash
python app.py
```
6. Open http://localhost:5000

## Notes
- This demo keeps the vector index in memory. Restarting the server clears it.
- If a transcript is unavailable (e.g., disabled), the app will report an error.
- The summarization endpoint performs a simple map-reduce over chunks, which can take multiple API calls; avoid overuse to respect rate limits.
- If you want persistent storage or multi-user support, consider saving the FAISS index and chunks to disk keyed by video ID.


# .IBM/Srcipts/Activate
# python app.py
# Open http://localhost:5000
