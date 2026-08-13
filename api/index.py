"""
Cortex — AI & ML Knowledge Base
FastAPI backend: JWT auth + httpOnly cookie + SQLite + Notes

Local dev  : uvicorn api.index:app --reload --port 8080  (run from Genai/ root)
Vercel     : auto via vercel.json
"""
from __future__ import annotations
import os, re, shutil, sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response, Query
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
from pydantic import BaseModel
from jose import JWTError, jwt

# ── Config ──────────────────────────────────────────────────────────────────────
SECRET_KEY  = os.getenv("SECRET_KEY", "cortex-dev-key-change-in-production-please")
ALGORITHM   = "HS256"
TOKEN_HOURS = int(os.getenv("TOKEN_EXPIRE_HOURS", "72"))
IS_VERCEL   = bool(os.getenv("VERCEL"))
COOKIE_NAME = "cortex_session"

BASE_DIR   = Path(__file__).parent.parent.resolve()   # Genai/
NOTES_DIR  = Path(os.getenv("NOTES_DIR", str(BASE_DIR)))
PUBLIC_DIR = BASE_DIR / "public"

# SQLite: /tmp on Vercel (ephemeral but seeded from env vars), local otherwise
if IS_VERCEL:
    DB_PATH = Path("/tmp/cortex.db")
    _seed = BASE_DIR / "cortex.db"
    if not DB_PATH.exists() and _seed.exists():
        shutil.copy(str(_seed), str(DB_PATH))
else:
    DB_PATH = BASE_DIR / "cortex.db"

# ── Crypto ──────────────────────────────────────────────────────────────────────
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer  = HTTPBearer(auto_error=False)

# ── DB bootstrap ────────────────────────────────────────────────────────────────
def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c

def init_db() -> None:
    c = _conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    DEFAULT (datetime('now'))
        )
    """)
    c.commit()
    # On Vercel, admin user is seeded from env vars on every cold start
    admin_u = os.getenv("ADMIN_USER", "").strip()
    admin_p = os.getenv("ADMIN_PASSWORD", "").strip()
    if admin_u and admin_p:
        existing = c.execute("SELECT id FROM users WHERE username=?", (admin_u,)).fetchone()
        if not existing:
            c.execute(
                "INSERT INTO users (username, password_hash) VALUES (?,?)",
                (admin_u, pwd_ctx.hash(admin_p)),
            )
            c.commit()
    c.close()

init_db()

# ── JWT helpers ──────────────────────────────────────────────────────────────────
def make_token(username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    return jwt.encode({"sub": username, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def _decode(token: str) -> str:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    sub = payload.get("sub")
    if not sub:
        raise ValueError("no sub")
    return sub

def _require_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> str:
    """Bearer-header only (for JSON API endpoints)."""
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return _decode(creds.credentials)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def _require_user_or_cookie(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    session: Optional[str] = Cookie(None, alias=COOKIE_NAME),
) -> str:
    """Bearer-header OR httpOnly cookie (iframe src uses cookie automatically)."""
    token = (creds.credentials if creds else None) or session
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return _decode(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ── Pydantic schemas ─────────────────────────────────────────────────────────────
class AuthReq(BaseModel):
    username: str
    password: str

# ── App ──────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Cortex", docs_url=None, redoc_url=None)

# ── Auth endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/auth/status")
def auth_status():
    """Registration is always open."""
    return {"can_register": True}

@app.post("/api/auth/login")
def login(req: AuthReq, response: Response):
    if not req.username.strip() or not req.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    c = _conn()
    user = c.execute("SELECT * FROM users WHERE username=?", (req.username.strip(),)).fetchone()
    c.close()
    if not user or not pwd_ctx.verify(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = make_token(req.username.strip())
    response.set_cookie(
        COOKIE_NAME, token,
        httponly=True, samesite="lax",
        secure=IS_VERCEL, max_age=TOKEN_HOURS * 3600, path="/"
    )
    return {"access_token": token, "token_type": "bearer", "username": req.username.strip()}

@app.post("/api/auth/register")
def register(req: AuthReq, response: Response):
    uname = req.username.strip()
    if len(uname) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    c = _conn()
    try:
        h = pwd_ctx.hash(req.password)
        c.execute("INSERT INTO users (username, password_hash) VALUES (?,?)", (uname, h))
        c.commit()
        c.close()
    except sqlite3.IntegrityError:
        c.close()
        raise HTTPException(status_code=409, detail="Username already taken")
    token = make_token(uname)
    response.set_cookie(
        COOKIE_NAME, token,
        httponly=True, samesite="lax",
        secure=IS_VERCEL, max_age=TOKEN_HOURS * 3600, path="/"
    )
    return {"access_token": token, "token_type": "bearer", "username": uname}

@app.get("/api/auth/me")
def me(user: str = Depends(_require_user)):
    return {"username": user}

@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}

# ── Notes endpoints ───────────────────────────────────────────────────────────────
CATEGORY_MAP = [
    ("pytorch",          ("PyTorch",         "#ee4c2c")),
    ("deep_learning",    ("Deep Learning",   "#9b59b6")),
    ("deep-learning",    ("Deep Learning",   "#9b59b6")),
    ("machine-learning", ("Machine Learning","#8e44ad")),
    ("machine_learning", ("Machine Learning","#8e44ad")),
    ("nlp",              ("NLP",             "#1abc9c")),
    ("langchain",        ("LangChain",       "#e74c3c")),
    ("langgraph",        ("LangGraph",       "#c0392b")),
    ("rag",              ("RAG",             "#3498db")),
    ("ai-search",        ("Azure AI Search", "#0078d4")),
    ("azure",            ("Azure",           "#0078d4")),
    ("mcp",              ("MCP",             "#00b4d8")),
    ("fastapi",          ("FastAPI",         "#009688")),
    ("interview",        ("Interview",       "#f39c12")),
    ("python",           ("Python",          "#f7c948")),
]

def _detect_category(stem: str):
    name = stem.lower()
    for key, val in CATEGORY_MAP:
        if key in name:
            return val
    return ("Notes", "#6c757d")

def _extract_title(content: str, fallback: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", content, re.I | re.S)
    if m:
        t = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if t:
            return t
    return fallback

def _read_time(content: str) -> int:
    text  = re.sub(r"<[^>]+>", " ", content)
    words = len(re.sub(r"\s+", " ", text).split())
    return max(1, round(words / 200))

# ── Universal reader CSS & template ────────────────────────────────────────────────
_READER_CSS = """
/* ══════════════════════════════════════════════════════════════
   Cortex Reader — Universal Note Stylesheet v1
   Strips original styles; applies consistent, beautiful rendering.
   ══════════════════════════════════════════════════════════════ */

:root {
  --font-sans: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', Consolas, monospace;
  --max-w: 860px;
  --r: 10px;
  --lh: 1.8;
}

[data-theme="dark"] {
  --bg:          #0d1117;
  --bg-1:        #161b22;
  --bg-2:        #1c2128;
  --bg-code:     #161626;
  --text:        #e6edf3;
  --text-h:      #f0f6fc;
  --text-m:      #8b949e;
  --text-s:      #6e7681;
  --accent:      #4f8ef7;
  --accent-2:    #a371f7;
  --border:      #30363d;
  --border-s:    #21262d;
  --link:        #58a6ff;
  --code-fg:     #cdd6f4;
  --warn-bg:     rgba(212,165,32,0.12);
  --warn-bd:     #d4a520;
  --warn-fg:     #e3b341;
  --info-bg:     rgba(79,142,247,0.1);
  --info-bd:     #4f8ef7;
  --info-fg:     #79c0ff;
  --ok-bg:       rgba(63,185,80,0.1);
  --ok-bd:       #3fb950;
  --ok-fg:       #56d364;
  --err-bg:      rgba(248,81,73,0.1);
  --err-bd:      #f85149;
  --err-fg:      #ff7b72;
  --th-bg:       #1c2128;
  --tr-alt:      rgba(255,255,255,0.025);
  --scroll:      #30363d;
}

[data-theme="light"] {
  --bg:          #f6f8fa;
  --bg-1:        #ffffff;
  --bg-2:        #eef2f5;
  --bg-code:     #1e1e2e;
  --text:        #1f2328;
  --text-h:      #0d1117;
  --text-m:      #636c76;
  --text-s:      #9198a1;
  --accent:      #1f6feb;
  --accent-2:    #8250df;
  --border:      #d0d7de;
  --border-s:    #e8ecef;
  --link:        #0969da;
  --code-fg:     #cdd6f4;
  --warn-bg:     #fff8c5;
  --warn-bd:     #d4a72c;
  --warn-fg:     #633c01;
  --info-bg:     #ddf4ff;
  --info-bd:     #54aeff;
  --info-fg:     #0550ae;
  --ok-bg:       #dafbe1;
  --ok-bd:       #2da44e;
  --ok-fg:       #116329;
  --err-bg:      #ffebe9;
  --err-bd:      #cf222e;
  --err-fg:      #82071e;
  --th-bg:       #f0f3f6;
  --tr-alt:      #f6f8fa;
  --scroll:      #d0d7de;
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body {
  font-family: var(--font-sans) !important;
  font-size: 16px !important;
  line-height: var(--lh) !important;
  background: var(--bg) !important;
  color: var(--text) !important;
  -webkit-font-smoothing: antialiased;
  margin: 0 !important;
  padding: 0 !important;
}
body { padding: 32px 20px 80px !important; }

.cortex-body {
  max-width: var(--max-w);
  margin: 0 auto;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-sans) !important;
  color: var(--text-h) !important;
  font-weight: 700 !important;
  line-height: 1.3 !important;
  margin: 1.75em 0 0.55em !important;
  background: none !important;
  -webkit-text-fill-color: var(--text-h) !important;
  text-shadow: none !important;
  border-color: var(--border) !important;
}
h1 { font-size: 2rem !important; letter-spacing: -0.03em; padding-bottom: 0.4em; border-bottom: 2px solid var(--border) !important; }
h2 { font-size: 1.5rem !important; letter-spacing: -0.02em; padding-bottom: 0.35em; border-bottom: 1px solid var(--border) !important; }
h3 { font-size: 1.25rem !important; }
h4 { font-size: 1.1rem !important; }
h5 { font-size: 1rem !important; }
h6 { font-size: 0.88rem !important; color: var(--text-m) !important; }

/* ── Body text ── */
p   { margin: 0 0 0.9em !important; color: var(--text) !important; background: none !important; }
a   { color: var(--link) !important; text-decoration: none !important; }
a:hover { text-decoration: underline !important; }
strong, b { font-weight: 700 !important; color: var(--text-h) !important; }
em, i { font-style: italic; }
mark { background: var(--warn-bg) !important; color: var(--warn-fg) !important; padding: 0 3px; border-radius: 3px; }

kbd {
  font-family: var(--font-mono) !important;
  font-size: 0.74em !important;
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-bottom-width: 2px !important;
  border-radius: 4px !important;
  padding: 1px 6px !important;
  color: var(--text) !important;
  box-shadow: none !important;
}

/* ── Inline code ── */
:not(pre) > code {
  font-family: var(--font-mono) !important;
  font-size: 0.84em !important;
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  padding: 0.12em 0.42em !important;
  color: var(--accent) !important;
}

/* ── Code blocks ── */
pre {
  background: var(--bg-code) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  padding: 20px 24px !important;
  overflow-x: auto !important;
  margin: 1.2em 0 !important;
  line-height: 1.65 !important;
  font-size: 0.875rem !important;
  color: var(--code-fg) !important;
  white-space: pre !important;
}
pre code {
  background: none !important;
  border: none !important;
  padding: 0 !important;
  font-family: var(--font-mono) !important;
  font-size: inherit !important;
  color: var(--code-fg) !important;
  border-radius: 0 !important;
}

/* ── Blockquote ── */
blockquote {
  border-left: 3px solid var(--accent) !important;
  margin: 1em 0 !important;
  padding: 12px 20px !important;
  background: var(--bg-1) !important;
  border-radius: 0 var(--r) var(--r) 0 !important;
  color: var(--text-m) !important;
  border-top: none !important; border-right: none !important; border-bottom: none !important;
}
blockquote p:last-child { margin-bottom: 0 !important; }

/* ── Lists ── */
ul, ol { padding-left: 1.6em !important; margin: 0.4em 0 0.9em !important; color: var(--text) !important; }
li { margin-bottom: 0.3em !important; }
li > ul, li > ol { margin: 0.25em 0 !important; }
li::marker { color: var(--accent) !important; }

/* ── Tables ── */
table { border-collapse: collapse !important; width: 100% !important; margin: 1.2em 0 !important; font-size: 0.9rem !important; background: transparent !important; }
thead th, th {
  background: var(--th-bg) !important;
  color: var(--text-h) !important;
  font-weight: 700 !important;
  text-align: left !important;
  padding: 10px 14px !important;
  border: 1px solid var(--border) !important;
}
td {
  padding: 9px 14px !important;
  border: 1px solid var(--border) !important;
  vertical-align: top !important;
  background: transparent !important;
  color: var(--text) !important;
}
tbody tr:nth-child(even) td { background: var(--tr-alt) !important; }

/* ── HR ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 2em 0 !important; }

/* ── Images ── */
img { max-width: 100% !important; height: auto !important; border-radius: var(--r) !important; }

/* ── Definition lists ── */
dl { margin: 1em 0 !important; }
dt { font-weight: 700 !important; color: var(--text-h) !important; margin-top: 0.8em !important; }
dd { margin-left: 1.5em !important; color: var(--text) !important; }

/* ── Details / summary ── */
details {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  padding: 12px 16px !important;
  margin: 0.8em 0 !important;
}
summary {
  cursor: pointer !important;
  font-weight: 600 !important;
  color: var(--text-h) !important;
  list-style: none !important;
  padding: 2px 0 !important;
}
summary::-webkit-details-marker { display: none; }
summary::before { content: '\25B6  '; font-size: 0.7em; color: var(--accent); }
details[open] summary::before { content: '\25BC  '; }

/* ══════════════════════════════════════════════════════════
   Layout overrides — normalize note-specific layouts
   ══════════════════════════════════════════════════════════ */

/* Kill built-in sidebars (we have our own) */
.sidebar { display: none !important; }

/* Remove left-margin offsets from note's "main" areas */
.main, .content, .page-content, .main-content {
  margin-left: 0 !important;
  max-width: 100% !important;
  padding: 0 !important;
}

/* Strip wrapper max-widths/padding (cortex-body handles it) */
.page-wrapper, .content-wrapper, .wrapper, .container {
  max-width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
}

/* Kill fixed/absolute positioning used by note navbars */
[class*="navbar"], [class*="nav-bar"], [class*="top-nav"],
[id*="navbar"], [id*="nav-bar"] {
  position: static !important;
}

/* ── Hero / banner ── */
.hero, .banner, .page-hero, [class*="-hero"], [class*="-banner"] {
  background: linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 100%) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  padding: 36px 40px 28px !important;
  margin: 0 0 1.8em !important;
  color: var(--text-h) !important;
  position: relative !important;
  overflow: hidden !important;
  text-shadow: none !important;
}
.hero h1, .hero h2, .banner h1, .banner h2 {
  border-bottom: none !important;
  margin-top: 0.3em !important;
}
.hero::before, .banner::before { display: none !important; }

/* ── Cards ── */
.card, .session-card, .section-card, .topic-card, .concept-card,
.lesson-card, .item-card, [class$="-card"] {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  padding: 20px 24px !important;
  margin: 1em 0 !important;
  color: var(--text) !important;
  box-shadow: none !important;
}

/* ── Badges / tags ── */
.badge, .tag, .chip, [class*="-badge"], [class*="-tag"], [class*="-chip"] {
  display: inline-block !important;
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  padding: 2px 10px !important;
  border-radius: 20px !important;
  letter-spacing: 0.04em !important;
  background: rgba(79,142,247,0.15) !important;
  color: var(--accent) !important;
  border: 1px solid rgba(79,142,247,0.3) !important;
  text-transform: uppercase !important;
  vertical-align: middle !important;
}

/* ── Callout boxes ── */
.warn, .warning, .caution {
  background: var(--warn-bg) !important; border: 1px solid var(--warn-bd) !important;
  border-left: 4px solid var(--warn-bd) !important; border-radius: var(--r) !important;
  padding: 14px 18px !important; margin: 1.2em 0 !important; color: var(--warn-fg) !important;
}
.info, .note-info, .callout-info {
  background: var(--info-bg) !important; border: 1px solid var(--info-bd) !important;
  border-left: 4px solid var(--info-bd) !important; border-radius: var(--r) !important;
  padding: 14px 18px !important; margin: 1.2em 0 !important; color: var(--info-fg) !important;
}
.tip, .success, .correct {
  background: var(--ok-bg) !important; border: 1px solid var(--ok-bd) !important;
  border-left: 4px solid var(--ok-bd) !important; border-radius: var(--r) !important;
  padding: 14px 18px !important; margin: 1.2em 0 !important; color: var(--ok-fg) !important;
}
.danger, .error, .wrong {
  background: var(--err-bg) !important; border: 1px solid var(--err-bd) !important;
  border-left: 4px solid var(--err-bd) !important; border-radius: var(--r) !important;
  padding: 14px 18px !important; margin: 1.2em 0 !important; color: var(--err-fg) !important;
}

/* ── Flex/grid card rows ── */
.card-grid, .note-grid, .topic-grid, .flex-grid {
  display: grid !important;
  gap: 16px !important;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)) !important;
  margin: 1.2em 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--scroll); border-radius: 3px; }
* { scrollbar-width: thin; scrollbar-color: var(--scroll) transparent; }
::selection { background: rgba(79,142,247,0.3); color: var(--text-h); }
"""

def _extract_body(raw: str) -> str:
    """Strip <head> and all <style> blocks; return body inner HTML."""
    # Remove every <style> block (head or inline)
    raw = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.I | re.S)
    # Extract <body> contents
    m = re.search(r"<body[^>]*>(.*?)</body>", raw, re.I | re.S)
    if m:
        return m.group(1).strip()
    # Fallback: strip html/head tags
    raw = re.sub(r"<head[^>]*>.*?</head>", "", raw, flags=re.I | re.S)
    raw = re.sub(r"</?(?:html|body)[^>]*>", "", raw, flags=re.I)
    return raw.strip()

def _wrap_note(raw: str, title: str) -> str:
    """Wrap body content in the Cortex reader template with standardized CSS."""
    safe_title = re.sub(r"<[^>]+>", "", title)
    body = _extract_body(raw)
    return (
        '<!DOCTYPE html>\n<html lang="en" data-theme="dark">\n<head>\n'
        '<meta charset="UTF-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        f'<title>{safe_title}</title>\n'
        '<style>' + _READER_CSS + '</style>\n'
        # Theme sync: read parent\'s localStorage theme before first paint
        '<script>'
        "try{var t=localStorage.getItem('nr_theme')||'dark';"
        "document.documentElement.setAttribute('data-theme',t);}catch(e){}\n"
        '</script>\n'
        '</head>\n<body>\n'
        '<div class="cortex-body">\n'
        + body +
        '\n</div>\n'
        # Live theme sync when parent toggles theme
        '<script>'
        "try{window.addEventListener('storage',function(e){"
        "if(e.key==='nr_theme')document.documentElement.setAttribute('data-theme',e.newValue||'dark');"
        "});}catch(e){}\n"
        '</script>\n'
        '</body>\n</html>'
    )

_cache: list | None = None

def _build_notes() -> list:
    global _cache
    if _cache is not None:
        return _cache
    result = []
    for i, f in enumerate(sorted(NOTES_DIR.glob("*.html"))):
        try:
            content  = f.read_text(encoding="utf-8", errors="ignore")
            fallback = f.stem.replace("-", " ").replace("_", " ").title()
            cat, col = _detect_category(f.stem)
            result.append({
                "id":        str(i),
                "filename":  f.name,
                "title":     _extract_title(content, fallback),
                "category":  cat,
                "color":     col,
                "read_time": _read_time(content),
            })
        except Exception:
            pass
    _cache = result
    return result

@app.get("/api/notes")
def list_notes(user: str = Depends(_require_user)):
    return _build_notes()

@app.get("/api/notes/{note_id}/content", response_class=HTMLResponse)
def get_note(note_id: str, user: str = Depends(_require_user_or_cookie)):
    """Accepts Bearer header OR cookie — enables iframe src auth without token in URL."""
    if not note_id.isdigit():
        raise HTTPException(status_code=400, detail="Invalid note ID")
    notes = _build_notes()
    note  = next((n for n in notes if n["id"] == note_id), None)
    if not note:
        raise HTTPException(status_code=404, detail="Not found")
    path = (NOTES_DIR / note["filename"]).resolve()
    try:
        path.relative_to(NOTES_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing")
    raw = path.read_text(encoding="utf-8", errors="ignore")
    wrapped = _wrap_note(raw, note["title"])
    return HTMLResponse(content=wrapped, headers={"X-Frame-Options": "SAMEORIGIN"})

# ── Serve SPA (must be last — only active when public/ exists) ────────────────────
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")
