# SHL Assessment Recommender API

This project is a FastAPI-based AI service that uses Gemini and FAISS to recommend SHL assessments based on a user's conversational input.

## Getting Started

### 1. Install Dependencies
Open your terminal (PowerShell or Command Prompt) and navigate to the project directory:
```bash
cd C:\Users\LENOVO\Desktop\Chatbot
pip install -r requirements.txt
```

### 2. Start the API Server
To run the server locally, run the following command:
```bash
python -m uvicorn main:app --reload
```
You should see output similar to this:
```text
[main] Warming up retriever...
[retriever] Loaded 377 assessments from catalog.json
[retriever] Ready — 377 assessments indexed with model 'sentence-transformers/all-MiniLM-L6-v2'
[main] Retriever ready — 377 assessments indexed.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 3. Test the API

**Using the Interactive Swagger UI (Easiest)**
1. Open your web browser and go to: **http://127.0.0.1:8000/docs**
2. Click on the **`POST /chat`** endpoint.
3. Click **"Try it out"**.
4. In the Request body, paste this JSON:
   ```json
   {
     "messages": [
       {
         "role": "user",
         "content": "I need tests for a senior Java developer with Spring experience."
       }
     ]
   }
   ```
5. Click **"Execute"** and view the response below.

**Using PowerShell**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"messages": [{"role": "user", "content": "I need tests for a senior Java developer with Spring experience."}]}'
```

### 4. Run the Automated Tests
To verify everything is working, you can run the test suite:
```bash
python -m pytest test_agent.py -v -s
```

### 5. Evaluate Trace Recall
To run the evaluation script against the 10 provided conversation traces:
```bash
python evaluate_traces.py
```
