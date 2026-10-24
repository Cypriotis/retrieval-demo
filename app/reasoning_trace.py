"""Request-local measurements; no raw page payloads or embedding vectors."""
import time


class ReasoningTrace:
    def __init__(self, request_id, mode, enabled=True):
        self.started = time.perf_counter()
        self.request_id, self.mode, self.enabled = request_id, mode, enabled
        self.events, self.calls = [], []
        self.contexts = []

    def event(self, stage, **data):
        if self.enabled:
            self.events.append({"stage": stage, "elapsed_seconds": time.perf_counter() - self.started, **data})

    def call(self, kind, node_id, seconds, usage=None, error=None):
        # Counts and consumption remain measurable when detailed event tracing is disabled.
        self.calls.append({"kind": kind, "node_id": node_id, "seconds": seconds,
                           "usage": usage, "error": error})

    def snapshot(self):
        llm = [c for c in self.calls if c["kind"] in {"llm", "generalize"}]
        def tokens(key):
            values = [(c.get("usage") or {}).get(key) for c in llm]
            return sum(values) if all(v is not None for v in values) else None
        return {"request_id": self.request_id, "mode": self.mode, "events": self.events,
                "calls": self.calls, "total_seconds": time.perf_counter() - self.started,
                "llm_calls": len(llm), "retrieval_calls": sum(c["kind"] == "search" for c in self.calls),
                "input_tokens": tokens("input_tokens"), "output_tokens": tokens("output_tokens"),
                "answer_context_characters": sum(c["characters"] for c in self.contexts), "answer_contexts": self.contexts,
                "support_verification_limitation": "Citation IDs and verbatim quotes are checked. They do not prove that every claim follows from the evidence. Optional LLM support checking is fallible."}


class TracedRetrieval:
    def __init__(self, services, trace, node_id):
        self.services, self.trace, self.node_id = services, trace, node_id

    async def _call(self, method, *args, **kwargs):
        began, usage, error = time.perf_counter(), None, None
        try:
            fn = getattr(self.services, method)
            # Serialize node query rewriting and embedding to avoid competing GPU workloads.
            gate = getattr(self.services, "reasoning_llm_lock", None)
            if gate is not None and method in {"generalize", "embed"}:
                async with gate:
                    result = await fn(*args, **kwargs)
            else:
                result = await fn(*args, **kwargs)
            if method == "generalize":
                usage = result[1]
            return result
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self.trace.call(method, self.node_id, time.perf_counter() - began, usage, error)

    async def generalize(self, question):
        return await self._call("generalize", question)

    async def embed(self, texts, query=False):
        return await self._call("embed", texts, query=query)

    async def search(self, vector, top_k, document_ids):
        return await self._call("search", vector, top_k, document_ids)

    async def passages(self, chunks):
        return await self._call("passages", chunks)

    async def rerank(self, question, texts, **kwargs):
        return await self._call("rerank", question, texts, **kwargs)
