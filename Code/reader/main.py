"""
Notes Reader — FastAPI backend
Run: uvicorn main:app --reload --port 8080
Then open: http://localhost:8080
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import re

app = FastAPI(title="Cortex — AI & ML Knowledge Base")

# HTML notes live two directories up: Code/reader/ -> Code/ -> Genai/
NOTES_DIR: Path = Path(__file__).parent.parent.parent.resolve()

# ── Category detection ────────────────────────────────────────────────────────
CATEGORY_MAP = [
    ("pytorch",          ("PyTorch",        "#ee4c2c")),
    ("deep_learning",    ("Deep Learning",  "#9b59b6")),
    ("deep-learning",    ("Deep Learning",  "#9b59b6")),
    ("machine-learning", ("Machine Learning","#8e44ad")),
    ("machine_learning", ("Machine Learning","#8e44ad")),
    ("nlp",              ("NLP",            "#1abc9c")),
    ("langchain",        ("LangChain",      "#e74c3c")),
    ("langgraph",        ("LangGraph",      "#c0392b")),
    ("rag",              ("RAG",            "#3498db")),
    ("ai-search",        ("Azure AI Search","#0078d4")),
    ("azure",            ("Azure",          "#0078d4")),
    ("mcp",              ("MCP",            "#00b4d8")),
    ("fastapi",          ("FastAPI",        "#009688")),
    ("interview",        ("Interview",      "#f39c12")),
    ("python",           ("Python",         "#f7c948")),
]

def detect_category(stem: str):
    name = stem.lower()
    for key, value in CATEGORY_MAP:
        if key in name:
            return value
    return ("Notes", "#6c757d")


def extract_title(content: str, fallback: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", content, re.I | re.S)
    if m:
        t = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if t:
            return t
    return fallback


def estimate_read_time(content: str) -> int:
    text = re.sub(r"<[^>]+>", " ", content)
    words = len(re.sub(r"\s+", " ", text).split())
    return max(1, round(words / 200))


# ── Build notes list (cached after first call) ────────────────────────────────
_cache: list | None = None


def build_notes() -> list:
    global _cache
    if _cache is not None:
        return _cache
    result = []
    for i, f in enumerate(sorted(NOTES_DIR.glob("*.html"))):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            fallback = f.stem.replace("-", " ").replace("_", " ").title()
            category, color = detect_category(f.stem)
            result.append({
                "id":        str(i),
                "filename":  f.name,
                "title":     extract_title(content, fallback),
                "category":  category,
                "color":     color,
                "read_time": estimate_read_time(content),
            })
        except Exception:
            pass
    _cache = result
    return result


# ── API routes ────────────────────────────────────────────────────────────────
@app.get("/api/notes")
def list_notes():
    return build_notes()


@app.get("/api/notes/{note_id}/content", response_class=HTMLResponse)
def get_note_content(note_id: str):
    if not note_id.isdigit():
        raise HTTPException(status_code=400, detail="Invalid note ID")

    notes = build_notes()
    note = next((n for n in notes if n["id"] == note_id), None)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    path = (NOTES_DIR / note["filename"]).resolve()

    # Security: path must stay within NOTES_DIR
    try:
        path.relative_to(NOTES_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    content = path.read_text(encoding="utf-8", errors="ignore")
    return HTMLResponse(
        content=content,
        headers={"X-Frame-Options": "SAMEORIGIN"},
    )


# ── Serve SPA (must be last) ──────────────────────────────────────────────────
_static = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
