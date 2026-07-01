# SHL Assessment Recommender API

FastAPI service for the SHL conversational assessment recommender take-home. The API is stateless: each `/chat` request sends the full conversation history, and the service returns the next assistant reply plus a structured shortlist when it has enough information.

## Endpoints

- `GET /health` returns `{"status": "ok"}`.
- `POST /chat` accepts:

```json
{
  "messages": [
    {"role": "user", "content": "I need tests for a senior Java developer with Spring experience."}
  ]
}
```

Response shape:

```json
{
  "reply": "Got it. Here is a catalog-grounded shortlist.",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Setup

```powershell
cd C:\Users\LENOVO\Desktop\Chatbot
pip install -r requirements.txt
```

Create a `.env` file with:

```text
GEMINI_API_KEY=your_key_here
```

## Run Locally

```powershell
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API.

## Test And Evaluate

Syntax check without calling Gemini:

```powershell
python -m py_compile main.py retriever.py test_agent.py evaluate_traces.py
```

Behavior tests, which call `/chat` and may consume Gemini quota:

```powershell
python -m pytest test_agent.py -v -s
```

Public trace Recall@10 evaluation, also quota-consuming:

```powershell
python evaluate_traces.py
```

## Notes

- `catalog.json` is the local SHL product catalog used for grounding.
- `GenAI_SampleConversations_Traces/` contains the 10 public development traces.
- `retriever.py` builds a FAISS index over sentence-transformer embeddings.
- `main.py` validates every returned recommendation against `catalog.json` before responding.
