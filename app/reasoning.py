"""Tree-RAG orchestration for analysis, dependency planning, and execution."""
import asyncio

from app.reasoning_evidence import EvidenceRegistry
from app.reasoning_executor import ReasoningTreeExecutor
from app.reasoning_models import NodeResult
from app.reasoning_planner import StructuredLLM, StructuralAnalyzer, ReasoningTreePlanner
from app.reasoning_trace import ReasoningTrace


async def run_tree(services, question, document_ids, options, config, request_id,
                   level="passage", top_n=6, neighbors=True):
    trace = ReasoningTrace(request_id, "tree", config.tracing_enabled)
    registry = EvidenceRegistry()
    llm = StructuredLLM(services, trace, options)
    analyzer, planner = StructuralAnalyzer(llm, config), ReasoningTreePlanner(llm, config)
    executor = ReasoningTreeExecutor(services, llm, config, trace, registry, document_ids, level, top_n, neighbors)
    analysis, plan, stage = None, None, "planning"
    root = NodeResult(node_id="N0", status="failed")
    trace.event("request", question=question, document_ids=document_ids, configuration=config.model_dump(),
                extension_points=["planner_selection", "support_verifier", "recovery_strategy"])
    try:
        if not config.enabled:
            raise ValueError("Tree-RAG is disabled by REASONING_ENABLED")
        async with asyncio.timeout(config.timeout_seconds):
            analysis = await analyzer.analyze(question)
            plan = await planner.plan(analysis)
            stage = "execution"
            root = await executor.execute(plan)
    except asyncio.CancelledError:
        trace.event("cancelled")
        raise
    except Exception as exc:
        message = "Total request deadline exceeded" if isinstance(exc, TimeoutError) else str(exc)
        root = executor.results.get(plan.root_node_id) if plan and executor.results else root
        root.status, root.error, root.failure_stage = "failed", message, stage
        trace.event("error", error=message, failure_stage=stage)
    snapshot = trace.snapshot()
    display_answer = root.answer
    if not display_answer and root.status == "failed":
        display_answer = ("Tree-RAG could not complete planning. No document search was performed." if plan is None
                          else "Tree-RAG could not complete execution. Inspect the trace for the failed step.")
    return {"status": root.status, "text": display_answer or "The available evidence could not establish an answer.",
            "error": root.error, "failure_stage": root.failure_stage,
            "sources": list(registry.sources.values()), "evidence_ids": root.evidence_ids,
            "analysis": analysis.model_dump(mode="json") if analysis else None,
            "plan": plan.model_dump(mode="json") if plan else None,
            "node_results": {key: r.model_dump(mode="json") for key, r in executor.results.items()},
            "trace": snapshot, "configuration": config.model_dump(), "seconds": snapshot["total_seconds"],
            "input_tokens": snapshot["input_tokens"], "output_tokens": snapshot["output_tokens"]}
