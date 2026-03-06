# api.py
# FastAPI backend for YouTube RAG — wraps app.py pipeline

import os
import json
import logging
import warnings
import threading

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import (
    get_transcript, process_transcript, build_vector_store,
    load_vector_store, build_qa_chain, ask, maybe_translate
)

# ─────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="YouTube RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# In-Memory Session State
# ─────────────────────────────────────────────────────────────

session = {
    "video_url"   : None,
    "qa_chain"    : None,
    "status"      : "idle",      # idle | processing | ready | error
    "error_msg"   : None,
    "progress_msg": "Waiting for video URL...",
}

INDEX_PATH = "faiss_index"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def get_saved_url() -> str:
    meta_file = os.path.join(INDEX_PATH, "meta.json")
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            return json.load(f).get("video_url", "")
    return ""


def save_url(url: str):
    os.makedirs(INDEX_PATH, exist_ok=True)
    with open(os.path.join(INDEX_PATH, "meta.json"), "w") as f:
        json.dump({"video_url": url}, f)


def _build_pipeline(video_url: str):
    """
    Runs in a background thread — builds / loads FAISS index and QA chain.
    Updates session state as it goes.
    """
    try:
        saved_url    = get_saved_url()
        url_changed  = saved_url != video_url
        index_exists = os.path.exists(INDEX_PATH)

        if index_exists and not url_changed:
            session["progress_msg"] = "Found existing index — loading..."
            vector_store = load_vector_store(INDEX_PATH)
        else:
            session["progress_msg"] = "Fetching transcript..."
            entries, lang_code = get_transcript(video_url)

            session["progress_msg"] = "Translating if needed..."
            entries = maybe_translate(entries, lang_code)

            session["progress_msg"] = "Chunking transcript..."
            chunks = process_transcript(entries, video_url=video_url)

            session["progress_msg"] = "Building vector index (this may take ~30s)..."
            vector_store = build_vector_store(chunks, save_path=INDEX_PATH)
            save_url(video_url)

        session["progress_msg"] = "Loading Q&A chain..."
        qa_chain = build_qa_chain(vector_store)

        session["qa_chain"]     = qa_chain
        session["video_url"]    = video_url
        session["status"]       = "ready"
        session["progress_msg"] = "Ready! Ask your first question."

    except Exception as e:
        session["status"]    = "error"
        session["error_msg"] = str(e)
        session["progress_msg"] = f"Error: {e}"


# ─────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────

class VideoRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    question: str


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Frontend polls this to track processing progress."""
    return {
        "status"      : session["status"],
        "video_url"   : session["video_url"],
        "progress_msg": session["progress_msg"],
        "error_msg"   : session["error_msg"],
    }


@app.post("/api/process")
def process_video(req: VideoRequest):
    """
    Kick off video processing in a background thread.
    Returns immediately — frontend should poll /api/status.
    """
    if session["status"] == "processing":
        raise HTTPException(status_code=409, detail="Already processing a video. Please wait.")

    # Reset state
    session["status"]       = "processing"
    session["qa_chain"]     = None
    session["error_msg"]    = None
    session["video_url"]    = req.url
    session["progress_msg"] = "Starting..."

    thread = threading.Thread(target=_build_pipeline, args=(req.url,), daemon=True)
    thread.start()

    return {"message": "Processing started", "status": "processing"}


@app.post("/api/ask")
def ask_question(req: QuestionRequest):
    """Ask a question about the loaded video."""
    if session["status"] != "ready":
        raise HTTPException(status_code=400, detail="No video loaded yet. Please process a video first.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = ask(session["qa_chain"], req.question)
    return {
        "answer" : result["answer"],
        "sources": result["sources"],
    }


@app.post("/api/reset")
def reset_session():
    """Clear current session entirely — user can load a new video."""
    session["video_url"]    = None
    session["qa_chain"]     = None
    session["status"]       = "idle"
    session["error_msg"]    = None
    session["progress_msg"] = "Session cleared. Paste a new video URL."
    return {"message": "Session reset"}


@app.post("/api/clear-memory")
def clear_memory():
    """
    Clear only the chat memory — keeps the video loaded and index intact.
    VideoRAG.clear_memory() simply wipes the history list. No rebuild needed.
    """
    if session["status"] != "ready" or session["qa_chain"] is None:
        raise HTTPException(status_code=400, detail="No video loaded.")

    session["qa_chain"].clear_memory()
    return {"message": "Chat memory cleared. Same video still loaded."}


@app.post("/api/ask-stream")
def ask_stream(req: QuestionRequest):
    """
    Streaming version of /api/ask.
    Returns a text/event-stream of SSE events:
      data: {"token": "..."}   — one per LLM token
      data: {"done": true, "sources": [...]}  — final event with sources
    """
    if session["status"] != "ready":
        raise HTTPException(status_code=400, detail="No video loaded yet. Please process a video first.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def event_stream():
        try:
            for chunk in session["qa_chain"].stream_ask(req.question):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────
# Serve Frontend
# ─────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")