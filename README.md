# Room Decor Analyzer: "Style It, Shop It"

Upload a photo of a room. Two agents analyze its style and find matching decor to buy.

## How it works (LangGraph)

```
START -> analyze_room (Agent A) --is a room?--> shop (Agent B) -> END
                                 \--no--> END (asks for a wider shot)
```

| Agent | Job | Tech |
|---|---|---|
| **A: analyze_room** | Reads the photo: style, colour palette, existing items, 3-4 suggested additions (structured output) | Gemini vision via `langchain-google-genai` |
| **B: shop** | For each suggested item calls tools to find products | `search_web` (DuckDuckGo, free) + `store_search_links` (Amazon.in, Flipkart, IKEA, Pinterest) |

- Framework: LangChain (model, tools, structured output) + LangGraph (state, nodes, conditional edge)
- Free tools only: Gemini free tier, DuckDuckGo search, Render free + Netlify free
- No vector database / RAG in this agent
- Served with FastAPI + LangServe. Playground at `/agent/playground` (use the file picker), plain upload API at `POST /analyze`

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste your free key from https://aistudio.google.com
uvicorn app:app --reload
```

- Playground: http://localhost:8000/agent/playground
- API docs: http://localhost:8000/docs
- Frontend: open `frontend/index.html` in a browser (it calls `http://localhost:8000`)

## Deploy

1. **GitHub**: push this folder to a new repo.
2. **Render (backend)**: New > Web Service > connect the repo. It reads `render.yaml`. Add env var `GOOGLE_API_KEY`. Your backend URL looks like `https://room-decor-agent.onrender.com` (free tier sleeps, so the first request is slow).
3. **Frontend**: in `frontend/index.html` change `API_URL` to your Render URL, then drag the `frontend` folder onto https://app.netlify.com/drop (or connect the repo with base directory `frontend`).
4. Optional: set `ALLOWED_ORIGINS` on Render to your Netlify URL to lock CORS down.

## Limits

- PNG/JPG/WEBP only, max 8 MB
- Suggestions come from the model's reading of the photo, so check sizes and prices on the store pages
