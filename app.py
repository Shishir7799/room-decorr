"""Room Decor Analyzer ("Style It, Shop It")

Two agents wired together with LangGraph:
  Agent A  analyze_room - Gemini vision reads the room photo: style, colour palette,
                          existing items and what to add
  Agent B  shop         - for every suggested item, searches the web (free DuckDuckGo tool)
                          and builds store-search links
A conditional edge stops the graph early if the photo is not a room.
"""
import base64
import os
from typing import Any, Dict, List, TypedDict
from urllib.parse import quote_plus

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langserve import add_routes
from pydantic import BaseModel, Field

load_dotenv()
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-3.1-flash-lite-preview")
MAX_IMAGE_BYTES = 8 * 1024 * 1024

_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini client lazily so the app can start without a key."""
    global _llm
    if _llm is None:
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set on the server.")
        _llm = ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=key, temperature=0.2)
    return _llm


# ------------------------------------------------------------------ tools
@tool
def search_web(query: str) -> List[Dict[str, str]]:
    """Search the web (DuckDuckGo, free) and return title/url/snippet results."""
    try:
        from ddgs import DDGS

        hits = DDGS().text(query, max_results=3) or []
        return [
            {"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")}
            for h in hits
            if h.get("href")
        ]
    except Exception:
        return []


@tool
def store_search_links(query: str) -> List[Dict[str, str]]:
    """Build direct search links for a decor item on popular stores."""
    q = quote_plus(query)
    return [
        {"source": "Amazon.in", "url": f"https://www.amazon.in/s?k={q}"},
        {"source": "Flipkart", "url": f"https://www.flipkart.com/search?q={q}"},
        {"source": "IKEA", "url": f"https://www.ikea.com/in/en/search/?q={q}"},
        {"source": "Pinterest ideas", "url": f"https://www.pinterest.com/search/pins/?q={q}"},
    ]


# ------------------------------------------------------------------ state
class Suggestion(BaseModel):
    item: str = Field(description="A decor item to add, e.g. 'warm floor lamp'.")
    reason: str = Field(description="One sentence on why it suits this room.")
    search_query: str = Field(description="A short shopping search query for this item, including colour/material.")


class RoomAnalysis(BaseModel):
    is_room: bool = Field(description="True if the image shows an interior room/space.")
    style: str = Field(description="Dominant interior style, e.g. 'minimalist', 'boho', 'industrial'.")
    palette: List[str] = Field(description="3-5 dominant colours as plain words.")
    existing_items: List[str] = Field(description="Notable furniture/decor already visible.")
    suggestions: List[Suggestion] = Field(description="3 to 4 items to add that match the style.")
    summary: str = Field(description="Two-sentence summary of the room and the overall advice.")


class AgentState(TypedDict, total=False):
    image_b64: str
    mime: str
    analysis: Dict[str, Any]
    shopping: List[Dict[str, Any]]


# ------------------------------------------------------------------ agents (graph nodes)
def analyze_room_node(state: AgentState) -> AgentState:
    """Agent A: understand the room."""
    prompt = (
        "You are an interior designer. Analyze the room photo: dominant style, colour palette, "
        "notable existing items, and 3-4 specific decor items that would improve it while "
        "matching its style. Do not suggest things that are already in the room. "
        "If the image is not an interior space, set is_room=false and leave lists empty."
    )
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:{state['mime']};base64,{state['image_b64']}"},
        ]
    )
    result = get_llm().with_structured_output(RoomAnalysis).invoke([msg])
    return {"analysis": result.model_dump()}


def route_after_analysis(state: AgentState) -> str:
    a = state["analysis"]
    return "shop" if a["is_room"] and a["suggestions"] else END


def shop_node(state: AgentState) -> AgentState:
    """Agent B: find matching products for every suggestion."""
    style = state["analysis"]["style"]
    shopping = []
    for s in state["analysis"]["suggestions"][:4]:
        q = s["search_query"]
        shopping.append(
            {
                "item": s["item"],
                "web_results": search_web.invoke({"query": f"buy {q} {style}"}),
                "store_links": store_search_links.invoke({"query": q}),
            }
        )
    return {"shopping": shopping}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyze_room", analyze_room_node)
    g.add_node("shop", shop_node)
    g.add_edge(START, "analyze_room")
    g.add_conditional_edges("analyze_room", route_after_analysis, {"shop": "shop", END: END})
    g.add_edge("shop", END)
    return g.compile()


graph = build_graph()


# ------------------------------------------------------------------ helpers
def sniff_mime(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unsupported image. Please upload a PNG, JPG or WEBP photo.")


def run_agent(raw: bytes) -> Dict[str, Any]:
    if not raw:
        raise ValueError("Empty file.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large (max 8 MB).")
    mime = sniff_mime(raw)
    out = graph.invoke({"image_b64": base64.b64encode(raw).decode(), "mime": mime})
    shopping = out.get("shopping")
    return {
        "is_room": shopping is not None,
        "analysis": out["analysis"],
        "shopping": shopping or [],
        "message": "" if shopping is not None else "I couldn't find a room in this photo. Try a wider shot of the space.",
    }


# ------------------------------------------------------------------ API
app = FastAPI(title="Room Decor Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "agent": "room-decor", "playground": "/agent/playground", "docs": "/docs"}


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    try:
        return run_agent(file.file.read(MAX_IMAGE_BYTES + 1))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=502, detail="The AI service failed. Please try again.")


class ImageInput(BaseModel):
    image: str = Field(..., json_schema_extra={"widget": {"type": "base64file"}})


def _from_playground(x) -> Dict[str, Any]:
    b64 = x["image"] if isinstance(x, dict) else x.image
    if "," in b64[:64]:  # strip a "data:image/png;base64," prefix if present
        b64 = b64.split(",", 1)[1]
    return run_agent(base64.b64decode(b64))


add_routes(
    app,
    RunnableLambda(_from_playground).with_types(input_type=ImageInput, output_type=dict),
    path="/agent",
    playground_type="default",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
