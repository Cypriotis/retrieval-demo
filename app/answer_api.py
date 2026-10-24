"""Independent Normal RAG, Focus Mode, and Tree-RAG answer endpoints."""
import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core import CHAT_MODEL, EMBED_MODEL, COLLECTION, prompts, friendly_error
from app.reasoning import run_tree
from app.reasoning_config import ReasoningConfig, default_mode
from app.reasoning_trace import ReasoningTrace, TracedRetrieval
from app.retrieval import retrieve_focus


router = APIRouter()


async def watch_request(request, operation):
    """Disconnects cancel the orchestration and every in-flight dependency task."""
    task = asyncio.create_task(operation)
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=.25)
            if not task.done() and await request.is_disconnected():
                raise HTTPException(499, "Request cancelled by client disconnect")
        return await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["baseline", "focus", "tree"] | None = None
    top_k: int = Field(default=4, ge=1, le=20)
    rerank_level: Literal["sentence", "passage"] = "passage"
    top_n: int = Field(default=6, ge=1, le=20)
    neighbors: bool = True
    temperature: float = Field(default=0, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2147483647)
    max_tokens: int = Field(default=768, ge=64, le=2048)
    document_ids: list[str] | None = None


@router.get("/api/reasoning/config")
async def reasoning_config():
    return {"default_mode": default_mode(), "reasoning": ReasoningConfig.from_env().model_dump(),
            "modes": {"baseline": "Normal RAG (question embedding and top-K retrieval)",
                      "focus": "Focus retrieval and answer generation",
                      "tree": "Dependency-aware orchestration using Focus retrieval per node"}}


async def single_answer(services, request, ids, options, request_id):
    mode = request.mode
    trace = ReasoningTrace(request_id, mode)
    adapter = TracedRetrieval(services, trace, None)
    if mode == "focus":
        prepared = await retrieve_focus(adapter, request.question, ids, level=request.rerank_level,
                                       top_n=request.top_n, neighbors=request.neighbors)
        sources = prepared.sources
        trace.event("focus", query=prepared.query, timings=prepared.timings,
                    selected_ids=[s["id"] for s in sources])
    else:
        vector = (await adapter.embed([request.question], query=True))[0]
        sources = await adapter.search(vector, request.top_k, ids)
    if not sources:
        raise ValueError("No indexed passages found")
    retrieval_seconds = time.perf_counter() - trace.started
    messages = prompts(request.question, sources)
    began, text, final = time.perf_counter(), "", {}
    async for event in services.chat(messages, options):
        text += event.get("message", {}).get("content", "")
        if event.get("done"):
            final = event
    if not text.strip():
        raise ValueError("Model returned an empty answer")
    generation_seconds = time.perf_counter() - began
    trace.call("llm", None, generation_seconds,
               {"input_tokens": final.get("prompt_eval_count"), "output_tokens": final.get("eval_count")})
    snapshot = trace.snapshot()
    return {"status": "answered", "text": text, "sources": sources, "trace": snapshot,
            "seconds": snapshot["total_seconds"], "generation_seconds": generation_seconds,
            "retrieval_seconds": retrieval_seconds, "input_tokens": snapshot["input_tokens"],
            "output_tokens": snapshot["output_tokens"], "prompts": messages}


@router.post("/api/answer")
async def answer(body: AnswerRequest, request: Request):
    app = request.app
    if not body.question.strip():
        raise HTTPException(400, "Enter a question")
    if app.state.lock.locked():
        raise HTTPException(409, "Another operation is running")
    config, mode = ReasoningConfig.from_env(), body.mode or default_mode()
    if mode == "tree" and not config.enabled:
        raise HTTPException(400, "Tree-RAG is disabled by REASONING_ENABLED")
    async with app.state.lock:
        documents = app.state.store.documents()
        registered = {d["id"] for d in documents}
        ids = body.document_ids if body.document_ids is not None else sorted(registered)
        if not ids or not set(ids).issubset(registered):
            raise HTTPException(400, "Select at least one registered PDF")
        run_id = str(uuid.uuid4())
        options = {"temperature": body.temperature, "seed": body.seed,
                   "num_predict": body.max_tokens, "num_ctx": 16384}
        params = body.model_copy(update={"mode": mode})
        requested_mode, fallback_trace = mode, None
        try:
            if mode == "tree":
                result = await watch_request(request, run_tree(app.state.services, body.question, ids, options, config, run_id,
                                        body.rerank_level, body.top_n, body.neighbors))
                remaining = config.timeout_seconds - result["seconds"]
                if result["status"] == "failed" and config.fallback_enabled and remaining > 0:
                    fallback_trace = result
                    fallback_params = params.model_copy(update={"mode": "baseline"})
                    try:
                        fallback_result = await watch_request(request, asyncio.wait_for(
                            single_answer(app.state.services, fallback_params, ids, options, run_id), remaining))
                        mode, result = "baseline", fallback_result
                    except (TimeoutError, ValueError) as exc:
                        result["trace"]["events"].append({"stage": "fallback_failed", "error": str(exc) or "Fallback deadline exceeded"})
                        fallback_trace = None
            else:
                # Normal and Focus runs use a single retrieval path without dependency planning.
                result = await watch_request(request, single_answer(app.state.services, params, ids, options, run_id))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, friendly_error(exc)) from exc
        variant = {"baseline": "rag", "focus": "focus", "tree": "tree"}[mode]
        sources = [{k: v for k, v in s.items() if k not in {"page_text", "page_source_text"}} for s in result["sources"]]
        answer_data = {k: result.get(k) for k in ("text", "seconds", "input_tokens", "output_tokens", "status")}
        answer_data.update(pipeline_seconds=result["seconds"], context_characters=(result["trace"]["answer_context_characters"]
                           if mode == "tree" else sum(len(s["text"]) for s in sources)),
                           generation_seconds=result.get("generation_seconds"))
        run = {"id": run_id, "created": datetime.now(timezone.utc).isoformat(), "question": body.question,
               "execution_mode": mode, "requested_mode": requested_mode, "status": result["status"],
               "model": CHAT_MODEL, "embedding_model": EMBED_MODEL, "collection": COLLECTION,
               "document_ids": ids, "documents": [d for d in documents if d["id"] in ids], "options": options,
               "think": False, "top_k": body.top_k, "rerank_level": body.rerank_level,
               "sources": sources, "answers": {variant: answer_data}, "trace": result["trace"],
               "prompts": {variant: result.get("prompts")}, "retrieval_seconds": result.get("retrieval_seconds", 0)}
        for key in ("analysis", "plan", "node_results", "evidence_ids", "configuration", "error", "failure_stage"):
            if key in result:
                run[key] = result[key]
        if fallback_trace:
            run["fallback"] = {"used": True, "reason": fallback_trace["error"], "tree_attempt": fallback_trace}
        if mode == "focus":
            run.update(focus_sources=sources, focus_seconds=result["retrieval_seconds"])
        app.state.store.save_run(run)
        return run
