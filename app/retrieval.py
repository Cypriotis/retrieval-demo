"""Focus retrieval, reusable without invoking final-answer generation."""
import asyncio
import time
from dataclasses import dataclass, field

from app.focus import sentence_candidates, select_evidence
from app.passages import select_coherent_passages


@dataclass
class FocusEvidence:
    query: str
    rewrite_usage: dict
    chunks: list
    sources: list
    ranked: list
    reranker: dict
    seconds: float
    timings: dict = field(default_factory=dict)


async def focus_events(services, question, document_ids, level="passage", top_n=6, neighbors=True):
    """Yield comparison progress events and a final internal retrieval result."""
    began = time.perf_counter()
    timings = {}
    yield {"type": "step", "stage": "generalize", "message": "Generalizing the question for broad passage retrieval…"}
    query, usage = await services.generalize(question)
    timings["generalization"] = time.perf_counter() - began
    yield {"type": "generalized", "original": question, "query": query, "usage": usage}
    stage = time.perf_counter()
    vector = (await services.embed([query], query=True))[0]
    chunks = await services.search(vector, 20, document_ids)
    if not chunks:
        raise ValueError("The generalized query retrieved no passages.")
    timings["focus_retrieval"] = time.perf_counter() - stage
    yield {"type": "focus_chunks", "chunks": chunks}
    stage = time.perf_counter()
    if level == "sentence":
        yield {"type": "step", "stage": "sentences", "message": "Extracting complete sentences and removing overlapping candidates…"}
        candidates = await asyncio.to_thread(sentence_candidates, chunks)
        if not candidates or len(candidates) > 2000:
            raise ValueError("Sentence candidate count must be between 1 and 2,000. Adjust retrieved passages.")
        timings["sentence_extraction"] = time.perf_counter() - stage
        yield {"type": "sentences", "count": len(candidates), "candidates": candidates}
    else:
        yield {"type": "step", "stage": "sentences", "message": "Building coherent passages with BGE tokenizer counts (target 150–250 tokens)…"}
        candidates = (await services.passages(chunks))["candidates"]
        timings["passage_construction"] = time.perf_counter() - stage
        yield {"type": "passages", "count": len(candidates), "candidates": candidates}
    yield {"type": "step", "stage": "rerank", "message": f"Scoring original question–{level} pairs with the cross-encoder…"}
    stage = time.perf_counter()
    info = await services.rerank(question, [c["text"] for c in candidates], **({"max_length": 4096} if level == "passage" else {}))
    sources, ranked = (select_evidence(candidates, info["scores"], top_n, neighbors) if level == "sentence"
                       else select_coherent_passages(candidates, info["scores"]))
    timings["reranking_and_selection"] = time.perf_counter() - stage
    result = FocusEvidence(query, usage, chunks, sources, ranked, {k: v for k, v in info.items() if k != "scores"},
                           time.perf_counter() - began, timings)
    yield {"type": "focused", "sources": sources, "ranked": ranked, "rerank_level": level,
           "seconds": result.seconds, "reranker": result.reranker}
    yield {"type": "focus_result", "result": result}


async def retrieve_focus(services, question, document_ids, **kwargs):
    async for event in focus_events(services, question, document_ids, **kwargs):
        if event["type"] == "focus_result":
            return event["result"]
