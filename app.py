"""
app.py
------
Core RAG pipeline — YouTube transcript fetch, clean, chunk, embed, and QA.
"""

import re
import os
import warnings
warnings.filterwarnings("ignore")

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from deep_translator import GoogleTranslator


# ─────────────────────────────────────────────────────────────
# STEP 1: Extract Video ID
# ─────────────────────────────────────────────────────────────

def _extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character video ID from any YouTube URL format."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    if re.match(r"^[0-9A-Za-z_-]{11}$", url_or_id):
        return url_or_id
    raise ValueError(f"Invalid YouTube URL or ID: {url_or_id}")


# ─────────────────────────────────────────────────────────────
# STEP 2: Fetch Transcript
# ─────────────────────────────────────────────────────────────

def get_transcript(url: str) -> tuple:
    """
    Fetch transcript entries from YouTube.
    Tries English first, falls back to any available language.
    Returns (entries, language_code).
    """
    video_id  = _extract_video_id(url)
    api       = YouTubeTranscriptApi()
    lang_code = "en"

    def find_best_transcript(transcript_list):
        nonlocal lang_code
        try:
            t = transcript_list.find_transcript(["en", "en-IN", "en-US", "en-GB"])
            lang_code = t.language_code
            return t.fetch()
        except Exception:
            pass
        for t in transcript_list:
            try:
                fetched   = t.fetch()
                lang_code = t.language_code
                print(f"⚠️  English not found — using '{t.language}' transcript")
                return fetched
            except Exception:
                continue
        raise RuntimeError(f"No transcript available for: {video_id}")

    if hasattr(api, "list"):
        fetched = find_best_transcript(api.list(video_id))
    else:
        fetched = find_best_transcript(YouTubeTranscriptApi.list_transcripts(video_id))

    print(f"✅ Transcript fetched — {len(fetched)} entries | Language: {lang_code}")
    return fetched, lang_code


# ─────────────────────────────────────────────────────────────
# STEP 3: Translation (if needed)
# ─────────────────────────────────────────────────────────────

def _is_roman_script(entries: list) -> bool:
    """
    Returns True if 60%+ of alphabetic characters in the transcript are ASCII (Roman script).
    Used to detect Hinglish (Hindi written in Roman letters) and skip unnecessary translation.
    """
    sample = " ".join([
        (e["text"] if isinstance(e, dict) else e.text)
        for e in entries[:50]
    ])
    roman = sum(1 for c in sample if c.isascii() and c.isalpha())
    total = sum(1 for c in sample if c.isalpha())
    return (roman / total) > 0.6 if total > 0 else True


def maybe_translate(entries: list, language_code: str) -> list:
    """
    Translates the transcript to English if it is in a non-Roman script
    (e.g. Devanagari Hindi, Arabic, Chinese).
    English and Roman-script languages (including Hinglish) are passed through unchanged.
    """
    if language_code.startswith("en"):
        print("✅ English transcript — no translation needed")
        return entries

    if _is_roman_script(entries):
        print("✅ Roman script detected — no translation needed")
        return entries

    print(f"🔄 Non-Roman script (lang={language_code}) — translating to English...")

    translator = GoogleTranslator(source='auto', target='en')
    translated = []
    batch_size = 20

    for i in range(0, len(entries), batch_size):
        batch  = entries[i:i + batch_size]
        texts  = [e["text"]  if isinstance(e, dict) else e.text  for e in batch]
        starts = [e["start"] if isinstance(e, dict) else e.start for e in batch]

        combined = " ||| ".join(texts)

        try:
            translated_combined = translator.translate(combined)
            translated_texts    = translated_combined.split("|||")
            if len(translated_texts) != len(batch):
                translated_texts = texts
        except Exception:
            translated_texts = texts

        for j, start in enumerate(starts):
            translated.append({
                "text" : translated_texts[j].strip() if j < len(translated_texts) else texts[j],
                "start": start,
            })

        print(f"  ✅ {min(i + batch_size, len(entries))}/{len(entries)} entries translated...")

    print("✅ Translation complete!")
    return translated


# ─────────────────────────────────────────────────────────────
# STEP 4: Clean Text
# ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Clean raw transcript text before chunking.
    Removes caption noise tags, filler words, consecutive duplicate words,
    and normalizes whitespace.
    """
    text = re.sub(r'\[.*?\]', '', text)                              # [Music], [Applause], etc.
    text = re.sub(r'\b(uh|um|uh-huh|hmm)\b', '', text, flags=re.IGNORECASE)  # filler words
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)      # duplicate words
    text = re.sub(r'\s+', ' ', text)                                 # normalize whitespace
    return text.strip()


# ─────────────────────────────────────────────────────────────
# STEP 5: Chunk Transcript into Documents
# ─────────────────────────────────────────────────────────────

def process_transcript(
    transcript_entries : list,
    video_url          : str,
    chunk_size         : int = 1000,
    chunk_overlap      : int = 200,
) -> list:
    """
    Cleans and splits transcript entries into overlapping LangChain Document chunks.
    Each chunk carries metadata: source URL, timestamp, and a deep-link to that moment.
    """
    base_url = video_url.split("&t=")[0]

    full_text = _clean_text(" ".join([
        (e["text"] if isinstance(e, dict) else e.text)
        for e in transcript_entries
    ]))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap,
        separators    = ["\n\n", "\n", "  ", " ", ""],
    )
    raw_chunks = splitter.split_text(full_text)

    starts = [
        (e["start"] if isinstance(e, dict) else e.start)
        for e in transcript_entries
    ]

    docs = []
    for i, chunk_text in enumerate(raw_chunks):
        position   = i / max(len(raw_chunks) - 1, 1)
        best_start = starts[min(int(position * len(starts)), len(starts) - 1)]
        minutes, seconds = divmod(int(best_start), 60)

        docs.append(Document(
            page_content = chunk_text,
            metadata     = {
                "source"       : base_url,
                "timestamp"    : f"{minutes:02d}:{seconds:02d}",
                "start_seconds": round(best_start),
                "yt_link"      : f"{base_url}&t={round(best_start)}s",
            }
        ))

    print(f"✅ Chunking done — {len(docs)} chunks created")
    return docs


# ─────────────────────────────────────────────────────────────
# STEP 6: Build / Load Vector Store
# ─────────────────────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFace embedding model.
    Model: all-mpnet-base-v2 — strong English semantic understanding, fast on CPU.
    For non-Roman script languages, switch to: intfloat/multilingual-e5-base
    """
    os.environ["TOKENIZERS_PARALLELISM"]       = "false"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    return HuggingFaceEmbeddings(
        model_name   = "sentence-transformers/all-mpnet-base-v2",
        model_kwargs = {"device": "cpu"},
        encode_kwargs= {"normalize_embeddings": True},
        show_progress= False,
    )


def build_vector_store(chunks: list, save_path: str = "faiss_index") -> FAISS:
    """Embed document chunks and save a FAISS index to disk."""
    print("Loading embedding model...")
    embeddings   = _get_embeddings()
    print("Creating FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    print(f"✅ FAISS index saved to '{save_path}/'")
    return vector_store


def load_vector_store(save_path: str = "faiss_index") -> FAISS:
    """Load an existing FAISS index from disk."""
    embeddings   = _get_embeddings()
    vector_store = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ FAISS index loaded from '{save_path}/'")
    return vector_store


# ─────────────────────────────────────────────────────────────
# STEP 7: Conversational RAG
# ─────────────────────────────────────────────────────────────

class VideoRAG:
    """
    Conversational RAG handler with rolling memory.

    On each ask():
      1. Retrieves relevant chunks via MMR search on the FAISS index
      2. Injects the last `memory_k` exchanges as conversation history
      3. Calls the LLM with the assembled prompt
      4. Stores the Q&A pair for future context
    """

    PROMPT_TEMPLATE = """You are an intelligent assistant helping a user understand a YouTube video.
The transcript may contain informal speech, Hinglish, or filler words — that is fine.

{history_block}Transcript context (relevant parts of the video):
{context}

Instructions:
- Answer ONLY using the transcript context above
- Be specific — mention exact examples, numbers, or names from the transcript if present
- If the question asks for steps or a list, format it clearly with numbers or bullets
- If the answer is not in the context, say exactly: "This wasn't covered in the video."
- Never guess or add outside knowledge not present in the transcript
- Keep answers concise but complete (3–6 sentences for most questions)
- Always answer in clear English regardless of transcript language

Question: {question}

Answer:"""

    def __init__(self, retriever, llm, memory_k: int = 5):
        self.retriever = retriever
        self.llm       = llm
        self.memory_k  = memory_k
        self.history   = []

    def _build_prompt(self, question: str, docs: list) -> str:
        """Assemble the full prompt with history and retrieved context."""
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        history_block = ""
        if self.history:
            recent = self.history[-self.memory_k:]
            lines  = ["Previous conversation:"]
            for turn in recent:
                lines.append(f"  Human: {turn['question']}")
                lines.append(f"  Assistant: {turn['answer']}")
            history_block = "\n".join(lines) + "\n\n"
        return self.PROMPT_TEMPLATE.format(
            history_block = history_block,
            context       = context,
            question      = question,
        )

    def _build_sources(self, docs: list) -> list:
        """Build the sources list from retrieved documents."""
        return [
            {
                "timestamp": doc.metadata["timestamp"],
                "yt_link"  : doc.metadata["yt_link"],
                "preview"  : doc.page_content[:100] + "...",
            }
            for doc in docs
        ]

    def ask(self, question: str) -> dict:
        """Returns {"answer": str, "sources": list}."""
        docs   = self.retriever.invoke(question)
        prompt = self._build_prompt(question, docs)
        answer = self.llm.invoke(prompt)
        self.history.append({"question": question, "answer": answer})
        return {"answer": answer, "sources": self._build_sources(docs)}

    def stream_ask(self, question: str):
        """
        Generator for streaming responses.
        Yields {"token": str} for each chunk, then {"done": True, "sources": list} at the end.
        """
        docs   = self.retriever.invoke(question)
        prompt = self._build_prompt(question, docs)

        full_answer = ""
        for chunk in self.llm.stream(prompt):
            full_answer += chunk
            yield {"token": chunk}

        # Save complete answer to memory once streaming is done
        self.history.append({"question": question, "answer": full_answer})
        yield {"done": True, "sources": self._build_sources(docs)}

    def clear_memory(self):
        """Clear conversation history without reloading the video index."""
        self.history = []
        print("🗑️  Chat memory cleared.")


# ─────────────────────────────────────────────────────────────
# STEP 8: Build QA Chain
# ─────────────────────────────────────────────────────────────

def build_qa_chain(vector_store: FAISS) -> VideoRAG:
    """
    Initialise the LLM and MMR retriever, and return a VideoRAG instance.

    MMR retrieval: fetches 15 candidate chunks, returns the 4 that are
    most relevant and most diverse (lambda_mult=0.7 weights relevance at 70%).
    """
    llm = OllamaLLM(model="llama3.1", temperature=0.2)

    retriever = vector_store.as_retriever(
        search_type  = "mmr",
        search_kwargs= {
            "k"          : 4,
            "fetch_k"    : 15,
            "lambda_mult": 0.7,
        }
    )

    rag = VideoRAG(retriever=retriever, llm=llm, memory_k=5)
    print("✅ Q&A chain ready")
    return rag


def ask(qa_chain: VideoRAG, question: str) -> dict:
    """Thin wrapper around VideoRAG.ask() for compatibility with api.py and main.py."""
    return qa_chain.ask(question)