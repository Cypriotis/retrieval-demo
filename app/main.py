import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core import (CHAT_MODEL, EMBED_MODEL, OLLAMA_URL, QDRANT_URL, COLLECTION,
                      CHUNK_SIZE, CHUNK_OVERLAP, Services, extract_pdf, chunk_text, prompts, friendly_error)
from app.core import RERANK_URL, chunk_spans
from app.focus import sentence_candidates, select_evidence
from app.passages import select_coherent_passages
from app.store import Store
from app.retrieval import focus_events
from app.answer_api import router as answer_router


@asynccontextmanager
async def lifespan(app):
    app.state.store = Store()
    # Serialize ingestion/deletion/comparison so partial indexes never enter a run.
    app.state.lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
        app.state.services = Services(client)
        yield


app = FastAPI(title="retrieval-demo", lifespan=lifespan)
app.include_router(answer_router)
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def event(kind, **data):
    return json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n"


class CompareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20)
    focus: bool = True
    rerank_level: Literal["sentence", "passage"] = "passage"
    top_n: int = Field(default=6, ge=1, le=20)
    neighbors: bool = True
    temperature: float = Field(default=0, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2147483647)
    max_tokens: int = Field(default=768, ge=64, le=2048)
    document_ids: list[str] | None = None


class Feedback(BaseModel):
    winner: str = Field(pattern="^(baseline|rag|focus|tie|neither)$")
    baseline_score: int = Field(ge=1, le=5)
    rag_score: int = Field(ge=1, le=5)
    focus_score: int | None = Field(default=None, ge=1, le=5)
    notes: str = Field(default="", max_length=5000)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health/live")
async def live():
    return {"status": "ok"}


@app.get("/api/status")
async def status():
    async def check(url):
        try:
            response = await app.state.services.client.get(url, timeout=5)
            response.raise_for_status()
            return response.json(), None
        except Exception as exc:
            return None, friendly_error(exc)
    ollama, qdrant, reranker = await asyncio.gather(check(f"{OLLAMA_URL}/api/tags"), check(f"{QDRANT_URL}/collections"), check(f"{RERANK_URL}/health"))
    models = [m["name"] for m in (ollama[0] or {}).get("models", [])]
    return {"ollama": ollama[1] is None, "qdrant": qdrant[1] is None,
            "models_ready": CHAT_MODEL in models and EMBED_MODEL in models,
            "chat_model": CHAT_MODEL, "embedding_model": EMBED_MODEL,
            "collection": COLLECTION, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
            "reranker": reranker[1] is None, "reranker_info": reranker[0],
            "errors": [e for e in (ollama[1], qdrant[1], reranker[1]) if e], "busy": app.state.lock.locked()}


@app.get("/api/documents")
async def documents():
    return app.state.store.documents()


@app.post("/api/documents")
async def upload(file: UploadFile = File(...)):
    filename = (file.filename or "document.pdf").replace("\\", "/").split("/")[-1]
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a PDF file.")
    raw = await file.read(25 * 1024 * 1024 + 1)
    await file.close()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(413, "PDF uploads are limited to 25 MB.")
    if app.state.lock.locked():
        raise HTTPException(409, "Another operation is running. Try again when it completes.")
    document_id = hashlib.sha256(raw).hexdigest()

    async def ingest():
        async with app.state.lock:
            existing = next((d for d in app.state.store.documents() if d["id"] == document_id), None)
            if existing:
                if not existing.get("structure_version"):
                    try:
                        yield event("step", stage="structure", message="Restoring original paragraph metadata without changing vectors or chunks…")
                        pages, _ = await asyncio.to_thread(extract_pdf, raw)
                        await app.state.services.enrich_structure(document_id, pages)
                        existing["structure_version"] = 1
                        app.state.store.save_document(existing)
                    except Exception as exc:
                        yield event("error", message=friendly_error(exc))
                        return
                yield event("complete", document=existing, duplicate=True)
                return
            indexing = False
            try:
                yield event("step", stage="extract", message=f"Reading text from {filename}…")
                pages, empty = await asyncio.to_thread(extract_pdf, raw)
                yield event("extracted", pages=len(pages), empty_pages=empty,
                            preview=pages[0]["text"][:1800])
                chunks = [{"document_id": document_id, "filename": filename, "page": p["page"],
                           "chunk": i, "page_text": " ".join(p["text"].split()), "page_source_text": p["text"], **span}
                          for p in pages for i, span in enumerate(chunk_spans(p["text"]))]
                if len(chunks) > 10000:
                    raise ValueError("PDF exceeds the 10,000 chunk limit. Split it into smaller documents.")
                yield event("chunked", count=len(chunks), size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
                            previews=chunks[:3])
                for start in range(0, len(chunks), 16):
                    batch = chunks[start:start + 16]
                    yield event("step", stage="embed", message=f"Embedding chunks {start + 1}–{start + len(batch)} of {len(chunks)}…")
                    vectors = await app.state.services.embed([c["text"] for c in batch])
                    if start == 0:
                        await app.state.services.ensure_collection(len(vectors[0]))
                        indexing = True
                        yield event("vectors", dimension=len(vectors[0]), sample=vectors[0][:8], model=EMBED_MODEL)
                    points = [{"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{COLLECTION}:{document_id}:{start+i}")),
                               "vector": vector, "payload": chunk}
                              for i, (vector, chunk) in enumerate(zip(vectors, batch))]
                    await app.state.services.upsert(points)
                    yield event("indexed", completed=start + len(batch), total=len(chunks))
                document = {"id": document_id, "filename": filename, "pages": len(pages), "chunks": len(chunks),
                            "empty_pages": empty, "structure_version": 1, "created": datetime.now(timezone.utc).isoformat()}
                app.state.store.save_document(document)
                yield event("complete", document=document, duplicate=False)
            except asyncio.CancelledError:
                if indexing:
                    await asyncio.shield(app.state.services.delete_document(document_id))
                raise
            except Exception as exc:
                if indexing:
                    try:
                        await app.state.services.delete_document(document_id)
                    except Exception:
                        pass  # Orphaned points are excluded by the registered-document filter.
                yield event("error", message=friendly_error(exc))
    return StreamingResponse(ingest(), media_type="application/x-ndjson", headers={"X-Accel-Buffering": "no"})


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    if app.state.lock.locked():
        raise HTTPException(409, "Another operation is running.")
    async with app.state.lock:
        if not any(d["id"] == document_id for d in app.state.store.documents()):
            raise HTTPException(404, "Document not found.")
        try:
            await app.state.services.delete_document(document_id)
            app.state.store.remove_document(document_id)
        except Exception as exc:
            raise HTTPException(502, friendly_error(exc)) from exc
    return {"deleted": document_id}


@app.post("/api/compare")
async def compare(request: CompareRequest):
    if not request.question.strip():
        raise HTTPException(400, "Enter a question.")
    if app.state.lock.locked():
        raise HTTPException(409, "Another operation is running.")

    async def pipeline():
        async with app.state.lock:
            services = app.state.services
            if hasattr(services, "begin_trace"):
                services.begin_trace()
            try:
                ids = [d["id"] for d in app.state.store.documents()]
                selected = request.document_ids if request.document_ids is not None else ids
                if not selected:
                    raise ValueError("Upload and select at least one PDF before comparing.")
                if not set(selected).issubset(ids):
                    raise ValueError("One of the selected documents no longer exists. Refresh the library.")
                started = time.perf_counter()
                yield event("step", stage="query", message="Embedding your question for semantic search…")
                vector = (await app.state.services.embed([request.question], query=True))[0]
                yield event("query_vector", dimension=len(vector), sample=vector[:8])
                yield event("step", stage="retrieve", message=f"Retrieving the closest {request.top_k} passages from Qdrant…")
                sources = await app.state.services.search(vector, request.top_k, selected)
                if not sources:
                    raise ValueError("No indexed passages found. Re-upload the selected PDFs.")
                retrieval_seconds = time.perf_counter() - started
                yield event("sources", sources=sources, seconds=round(retrieval_seconds, 3))
                focus_sources, ranked, rewrite_usage, generalized = [], [], {}, None
                focus_seconds = 0
                focus_chunks = []
                ranked_passages = []
                timings = {"standard_retrieval": retrieval_seconds}
                reranker_info = None
                if request.focus:
                    async for progress in focus_events(app.state.services, request.question, selected,
                                                       request.rerank_level, request.top_n, request.neighbors):
                        if progress["type"] != "focus_result":
                            yield json.dumps(progress, ensure_ascii=False) + "\n"
                        else:
                            prepared = progress["result"]
                            generalized, rewrite_usage = prepared.query, prepared.rewrite_usage
                            focus_chunks, focus_sources = prepared.chunks, prepared.sources
                            if request.rerank_level == "sentence":
                                ranked = prepared.ranked
                            else:
                                ranked_passages = prepared.ranked
                            reranker_info, focus_seconds = prepared.reranker, prepared.seconds
                            timings.update(prepared.timings)
                options = {"temperature": request.temperature, "seed": request.seed,
                           "num_predict": request.max_tokens, "num_ctx": 16384}
                run = {"id": str(uuid.uuid4()), "created": datetime.now(timezone.utc).isoformat(),
                       "question": request.question, "model": CHAT_MODEL, "embedding_model": EMBED_MODEL,
                       "collection": COLLECTION, "options": options, "think": False,
                       "sources": sources, "document_ids": selected, "top_k": request.top_k,
                       "retrieval_seconds": retrieval_seconds, "answers": {}, "prompts": {}}
                run.update(focus_sources=focus_sources, focus_chunks=focus_chunks, ranked_sentences=ranked,
                           ranked_passages=ranked_passages, rerank_level=request.rerank_level,
                           focus_top_k=20, passage_target_tokens=[150, 250] if request.rerank_level == "passage" else None,
                           focus_selection_strategy="ranked_context_budget" if request.rerank_level == "passage" else "top_n_sentences",
                           generalized_query=generalized, rewrite_usage=rewrite_usage, timings=timings,
                           focus_seconds=focus_seconds, top_n=request.top_n if request.rerank_level == "sentence" else None,
                           focus_top_n=request.top_n if request.rerank_level == "sentence" else None,
                           neighbors=request.neighbors if request.rerank_level == "sentence" else False,
                           reranker={k: v for k, v in (reranker_info or {}).items() if k != "scores"})
                run["documents"] = [d for d in app.state.store.documents() if d["id"] in selected]
                run["retrieval_configuration"] = {
                    "backend": "Qdrant", "distance": "Cosine", "chunk_characters": CHUNK_SIZE,
                    "chunk_overlap_characters": CHUNK_OVERLAP, "sentence_language": "en",
                    "focus_context_character_budget": 18000, "minimum_score_threshold": None,
                }
                # Sequential calls use one GPU and prevent competing inference workloads.
                variants = [("baseline", []), ("rag", sources)] + ([("focus", focus_sources)] if request.focus else [])
                for variant, context in variants:
                    messages = prompts(request.question, context)
                    run["prompts"][variant] = messages
                    yield event("prompt", variant=variant, messages=messages, options=options)
                    label = {"baseline": "without RAG", "rag": "standard RAG", "focus": "Focus Mode"}[variant]
                    yield event("step", stage=variant, message=f"Generating {label} using {CHAT_MODEL}…")
                    began = time.perf_counter()
                    answer, final = "", {}
                    async for message in app.state.services.chat(messages, options):
                        token = message.get("message", {}).get("content", "")
                        answer += token
                        if token:
                            yield event("token", variant=variant, text=token)
                        if message.get("done"):
                            final = message
                    if not answer.strip():
                        raise ValueError(f"{variant} returned an empty answer. Increase the output limit or check model compatibility.")
                    result = {"text": answer, "seconds": round(time.perf_counter() - began, 3),
                              "input_tokens": final.get("prompt_eval_count"), "output_tokens": final.get("eval_count"),
                              "load_seconds": round(final.get("load_duration", 0) / 1e9, 3),
                              "done_reason": final.get("done_reason"),
                              "generation_seconds": round(final.get("eval_duration", 0) / 1e9, 3)}
                    result["context_characters"] = sum(len(s["text"]) for s in context)
                    result["pipeline_seconds"] = result["seconds"] + (retrieval_seconds if variant == "rag" else focus_seconds if variant == "focus" else 0)
                    run["answers"][variant] = result
                    yield event("answer_done", variant=variant, result=result)
                run["trace_schema_version"] = 1
                run["service_calls"] = services.end_trace() if hasattr(services, "end_trace") else []
                app.state.store.save_run(run)
                yield event("complete", run=run)
            except Exception as exc:
                yield event("error", message=friendly_error(exc))
            finally:
                if hasattr(services, "end_trace"):
                    services.end_trace()
    return StreamingResponse(pipeline(), media_type="application/x-ndjson", headers={"X-Accel-Buffering": "no"})


@app.get("/api/runs")
async def runs():
    return app.state.store.runs()


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    run = app.state.store.run(run_id)
    if not run:
        raise HTTPException(404, "Comparison not found.")
    return run


@app.put("/api/runs/{run_id}/feedback")
async def save_feedback(run_id: str, feedback: Feedback):
    if not app.state.store.feedback(run_id, feedback.model_dump(exclude_none=True)):
        raise HTTPException(404, "Comparison not found.")
    return {"saved": True}
