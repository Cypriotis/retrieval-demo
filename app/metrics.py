import re
import unicodedata
from collections import Counter


def normalize_answer(text):
    text = re.sub(r"\[(?:S|E)\d+\]", "", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def evaluate_run(run, labels):
    # Labels identify relevant source pages, rather than assuming similarity means relevance.
    relevant = {(item["filename"], item["page"]) for item in labels.get("relevant_sources", [])}
    references = labels.get("accepted_answers", [])
    output = {}
    for variant, answer in run["answers"].items():
        retrieval = run.get("focus_chunks", []) if variant == "focus" else run["sources"] if variant == "rag" else []
        retrieved = {(item["filename"], item["page"]) for item in retrieval}
        context = run.get("focus_sources", []) if variant == "focus" else run["sources"] if variant == "rag" else []
        context_pages = {(item["filename"], item["page"]) for item in context}
        output[variant] = {
            "recall_at_k_source_pages": len(relevant & retrieved) / len(relevant) if relevant and variant != "baseline" else None,
            "context_source_page_recall": len(relevant & context_pages) / len(relevant) if relevant and variant != "baseline" else None,
            "answer_exact_match": any(normalize_answer(answer["text"]) == normalize_answer(ref) for ref in references) if references else None,
            "pipeline_seconds": answer["pipeline_seconds"], "generation_seconds": answer["seconds"],
            "input_tokens": answer["input_tokens"], "output_tokens": answer["output_tokens"],
            "rewrite_input_tokens": run.get("rewrite_usage", {}).get("input_tokens", 0) if variant == "focus" else 0,
            "rewrite_output_tokens": run.get("rewrite_usage", {}).get("output_tokens", 0) if variant == "focus" else 0,
            "context_characters": answer["context_characters"],
        }
        tokens = [answer["input_tokens"], answer["output_tokens"]]
        if variant == "focus":
            tokens += [run.get("rewrite_usage", {}).get("input_tokens"), run.get("rewrite_usage", {}).get("output_tokens")]
        output[variant]["total_llm_tokens"] = sum(tokens) if all(t is not None for t in tokens) else None
    return output


def token_f1(answer, reference):
    predicted, expected = normalize_answer(answer).split(), normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    shared = sum((Counter(predicted) & Counter(expected)).values())
    if not shared:
        return 0.0
    precision, recall = shared / len(predicted), shared / len(expected)
    return 2 * precision * recall / (precision + recall)


def source_matches(source, requirement):
    fields = ("filename", "document_id", "page", "chunk", "id")
    if any(source.get(k) != requirement[k] for k in fields if k in requirement):
        return False
    if "chunk_id" in requirement:
        ids = {source.get("chunk_id"), source.get("id")}
        ids.update(s["chunk_id"] for s in source.get("retrieval_sources", []))
        if requirement["chunk_id"] not in ids:
            return False
    return True


def evaluate_mode_run(run, labels):
    """Only annotated targets yield quality scores; unavailable judgments remain null."""
    answer = next(iter(run["answers"].values()))
    mode = run["execution_mode"]
    references = labels.get("accepted_answers", [])
    sources = run["sources"]
    if mode == "tree":
        used = set(run.get("evidence_ids", []))
        sources = [s for s in sources if s["evidence_id"] in used]
    # Page-level requirements count each page once; chunk-level requirements retain chunk occurrences.
    requirements = labels.get("relevant_sources")
    evidence_precision = evidence_recall = None
    if requirements is not None:
        if requirements:
            evidence_precision = sum(any(source_matches(s, r) for r in requirements) for s in sources) / len(sources) if sources else 0.0
            evidence_recall = sum(any(source_matches(s, r) for s in sources) for r in requirements) / len(requirements)
        else:
            evidence_precision = 1.0 if not sources else 0.0
            evidence_recall = None  # No relevant evidence to recall.
    nodes = run.get("node_results", {})
    intermediate_labels = labels.get("required_intermediates", [])
    resolved = []
    for target in intermediate_labels:
        matches = [result for key, result in nodes.items()
                   if ("node_id" not in target or key == target["node_id"])
                   and ("question_contains" not in target or target["question_contains"] in (result.get("resolved_question") or ""))]
        resolved.append(any(result["status"] == "answered" and any(normalize_answer(result["value"]) == normalize_answer(v)
                            for v in target["accepted_values"]) for result in matches))
    retrieval_nodes = [nodes[key] for key, node in (run.get("plan") or {}).get("nodes", {}).items()
                       if node["node_type"] == "retrieval" and key in nodes]
    trace = run.get("trace", {})
    unsupported = labels.get("unsupported_answers", {}).get(mode)
    failures = {"planning": [], "retrieval": [], "interpretation": [], "integration": [], "dependency": [], "execution": []}
    if run.get("failure_stage"):
        failures.setdefault(run["failure_stage"], []).append(run.get("error") or run["status"])
    for key, result in nodes.items():
        if result.get("failure_stage"):
            failures.setdefault(result["failure_stage"], []).append(key)
    return {
        "answer_exact_match": any(normalize_answer(answer["text"]) == normalize_answer(ref) for ref in references) if references else None,
        "answer_token_f1": max((token_f1(answer["text"], ref) for ref in references), default=None),
        "evidence_precision": evidence_precision, "evidence_recall": evidence_recall,
        "dependency_resolution_accuracy": sum(resolved) / len(resolved) if resolved and mode == "tree" else None,
        "multi_hop_completion": (run["status"] == "answered" and bool(retrieval_nodes)
                                 and all(n["status"] == "answered" for n in retrieval_nodes)
                                 and len(retrieval_nodes) >= labels.get("hop_count", 1)
                                 and (all(resolved) if resolved else True)) if mode == "tree" and labels.get("hop_count", 0) > 1 else None,
        # Legacy Normal/Focus status means generation completed, not factual answerability.
        "unanswerable_correct": (run["status"] in {"unanswerable", "blocked"}
                                  if labels.get("expected_unanswerable") and mode == "tree" else None),
        "unsupported_answer": unsupported if isinstance(unsupported, bool) else None,
        "pipeline_seconds": answer["pipeline_seconds"], "llm_calls": trace.get("llm_calls"),
        "retrieval_calls": trace.get("retrieval_calls"), "input_tokens": trace.get("input_tokens"),
        "output_tokens": trace.get("output_tokens"),
        "total_llm_tokens": trace["input_tokens"] + trace["output_tokens"]
                            if trace.get("input_tokens") is not None and trace.get("output_tokens") is not None else None,
        "failures": failures,
    }
