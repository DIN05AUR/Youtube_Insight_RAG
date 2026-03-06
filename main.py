"""
main.py
-------
CLI runner for the YouTube RAG pipeline.
Run this directly to chat with any YouTube video in your terminal.

Usage:
    python main.py

Commands during chat:
    'new video'  → load a different video (clears memory)
    'clear'      → clear chat history for current video (keeps index)
    'exit'       → quit
"""

import os
import json
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from app import (
    get_transcript,
    process_transcript,
    build_vector_store,
    load_vector_store,
    build_qa_chain,
    ask,
    maybe_translate,
)


# ─────────────────────────────────────────────────────────────
# Index Metadata Helpers
# ─────────────────────────────────────────────────────────────

def _get_saved_url(index_path: str) -> str:
    """Read the video URL that was used to build the saved FAISS index."""
    meta_file = os.path.join(index_path, "meta.json")
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            return json.load(f).get("video_url", "")
    return ""


def _save_url(index_path: str, url: str):
    """Save the current video URL alongside the FAISS index."""
    os.makedirs(index_path, exist_ok=True)
    with open(os.path.join(index_path, "meta.json"), "w") as f:
        json.dump({"video_url": url}, f)


# ─────────────────────────────────────────────────────────────
# Video Processing
# ─────────────────────────────────────────────────────────────

def process_video(video_url: str, index_path: str = "faiss_index"):
    """
    Load an existing FAISS index if the URL matches the last-used one.
    Otherwise rebuild from scratch (fetch transcript → translate → chunk → index).

    This saves you 30-60 seconds on repeat questions about the same video.
    """
    saved_url    = _get_saved_url(index_path)
    url_changed  = saved_url != video_url
    index_exists = os.path.exists(index_path)

    if index_exists and not url_changed:
        print("ℹ️  Same video — loading existing index...")
        return load_vector_store(index_path)

    if url_changed and index_exists:
        print("ℹ️  New video detected — rebuilding index...")
    else:
        print("ℹ️  First run — building index...")

    entries, lang_code = get_transcript(video_url)
    entries            = maybe_translate(entries, lang_code)
    chunks             = process_transcript(entries, video_url=video_url)
    vector_store       = build_vector_store(chunks, save_path=index_path)
    _save_url(index_path, video_url)

    return vector_store


# ─────────────────────────────────────────────────────────────
# Q&A Session
# ─────────────────────────────────────────────────────────────

def run_qa_session(video_url: str) -> str:
    """
    Run a full conversational Q&A session for a video.

    Returns:
        "exit"      → user wants to quit the program
        "new video" → user wants to switch to a different video

    IMPROVED from v1:
    - 'clear' command resets chat memory without rebuilding the index
    - 'new video' rebuilds the qa_chain (and memory) for the new video
    - Memory is in the qa_chain itself — no need to manage it manually
    """
    vector_store = process_video(video_url)
    qa_chain     = build_qa_chain(vector_store)

    print("\n" + "=" * 60)
    print("  YouTube RAG — Ready to chat!")
    print("  Commands: 'new video' | 'clear' | 'exit'")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nInterrupted. Goodbye!")
            return "exit"

        if not question:
            continue

        # ── Special commands ──────────────────────────────
        if question.lower() == "exit":
            print("\nGoodbye!")
            return "exit"

        if question.lower() == "new video":
            return "new video"

        if question.lower() == "clear":
            # Clear conversation memory — rebuild chain (keeps the vector index)
            qa_chain = build_qa_chain(vector_store)
            print("🗑️  Chat history cleared. Starting fresh.\n")
            continue

        # ── Ask the question ──────────────────────────────
        try:
            output = ask(qa_chain, question)
        except Exception as e:
            print(f"\n⚠️  Error getting answer: {e}\n")
            continue

        # ── Print answer ──────────────────────────────────
        print(f"\nAI: {output['answer']}\n")

        # ── Print sources ─────────────────────────────────
        if output["sources"]:
            print("Sources:")
            for s in output["sources"]:
                print(f"  [{s['timestamp']}] {s['yt_link']}")
                print(f"   └─ {s['preview']}")
        print()


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  YouTube RAG Pipeline  (CLI mode)")
    print("=" * 60)

    while True:
        print()
        video_url = input("Paste YouTube URL: ").strip()

        if not video_url:
            print("URL can't be empty. Try again.")
            continue

        action = run_qa_session(video_url)

        if action == "exit":
            break
        elif action == "new video":
            print("\n── Starting new video session ──")
            continue