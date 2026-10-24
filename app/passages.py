"""Coherent, page-attributed passage construction; token counts supplied by BGE."""
import hashlib
import math
import re


def is_heading(line):
    text = line.strip()
    if not text or len(text) > 120 or len(text.split()) > 16:
        return False
    numbered = re.match(r"^\d+(?:\.\d+)+(?:\.)?\s+\S", text)
    if numbered and not re.search(r"\b(if|shall|must|should)\b", text, re.I):
        return True
    if text.istitle() and 1 < len(text.split()) <= 10 and not re.search(r"\b(if|when|unless|then|otherwise)\b", text, re.I) and not text.endswith((".", ":", ";")):
        return True
    return text.isupper() and len(text.split()) > 1 and not text.endswith((".", ":", ";"))


def page_blocks(raw):
    """Preserve explicit blank lines; infer paragraphs from PDF line wrapping."""
    words = list(re.finditer(r"\S+", raw))
    normalized = " ".join(m.group() for m in words)
    offsets, cursor = [], 0
    for word in words:
        offsets.append((word.start(), word.end(), cursor, cursor + len(word.group())))
        cursor += len(word.group()) + 1
    lines = list(re.finditer(r"[^\n]+(?:\n|$)", raw))
    width = max((len(m.group().strip()) for m in lines), default=0)
    groups, current = [], []

    def flush():
        if current:
            begin, end = current[0].start(), current[-1].end()
            covered = [w for w in offsets if w[0] < end and w[1] > begin]
            if covered:
                text = normalized[covered[0][2]:covered[-1][3]]
                groups.append({"text": text, "start": covered[0][2], "end": covered[-1][3],
                               "heading": len(current) == 1 and is_heading(text)})
            current.clear()

    for line in lines:
        text = line.group().strip()
        gap = current and "\n\n" in raw[current[-1].start():line.start() + 1]
        heading = is_heading(text)
        short_end = current and current[-1].group().strip().endswith((".", "?", "!")) and len(current[-1].group().strip()) < width * .75
        branch = bool(re.match(r"^(otherwise\b|else\b|then\b|if\b|unless\b|\(?[a-z0-9]+[.)]\s|[-•]\s)", text, re.I))
        if gap or heading or short_end or branch:
            flush()
        current.append(line)
        if heading:
            flush()
    flush()
    return normalized, groups


def coherent_passages(chunks, token_count, min_tokens=150, max_tokens=250):
    pages, candidates = {}, {}
    for chunk in chunks:
        key = (chunk["document_id"], chunk["page"])
        if key not in pages:
            raw = chunk.get("page_source_text")
            if raw is None:
                raise ValueError("Original paragraph structure is missing. Re-import the source PDF to enrich metadata without re-embedding.")
            normalized, blocks = page_blocks(raw)
            if normalized != chunk["page_text"]:
                raise ValueError("Source page text does not match the indexed page; cannot assign accurate passage offsets.")
            units, heading, i = [], None, 0
            while i < len(blocks):
                block = blocks[i]
                if block["heading"]:
                    heading = block; i += 1; continue
                parts = [block]
                conditional = bool(re.search(r"\b(if|unless|provided that|in the event|when)\b", block["text"], re.I))
                i += 1
                if conditional or block["text"].endswith(":"):
                    while i < len(blocks) and not blocks[i]["heading"]:
                        following = blocks[i]["text"]
                        continuation = re.match(r"^(otherwise\b|else\b|then\b|if\b|unless\b|\(?[a-z0-9]+[.)]\s|[-•]\s)", following, re.I)
                        if not continuation and not parts[-1]["text"].endswith(":"):
                            break
                        parts.append(blocks[i]); i += 1
                units.append({"parts": parts, "heading": heading, "conditional": conditional or len(parts) > 1})
            packed, pending = [], []

            def render(items):
                title = items[0]["heading"]
                paragraphs = [part["text"] for item in items for part in item["parts"]]
                return "\n\n".join(([title["text"]] if title else []) + paragraphs)

            def emit():
                if pending:
                    text = render(pending)
                    count = token_count(text)
                    packed.append({"text": text, "char_start": pending[0]["parts"][0]["start"],
                                   "char_end": pending[-1]["parts"][-1]["end"], "token_count": count,
                                   "heading": pending[0]["heading"],
                                   "paragraph_spans": [{"start": p["start"], "end": p["end"]} for u in pending for p in u["parts"]],
                                   "conditional_sequence": any(u["conditional"] for u in pending),
                                   "size_exception": ("coherence_requires_longer_passage" if count > max_tokens else "section_or_page_end" if count < min_tokens else None)})
                    pending.clear()

            for unit in units:
                if pending and (unit["heading"] != pending[0]["heading"] or token_count(render(pending + [unit])) > max_tokens):
                    emit()
                pending.append(unit)
                if token_count(render(pending)) >= min_tokens:
                    emit()
            emit()
            pages[key] = packed
        for passage in pages[key]:
            if passage["char_end"] <= chunk["char_start"] or passage["char_start"] >= chunk["char_end"]:
                continue
            identity = (*key, passage["char_start"], passage["char_end"])
            origin = {"chunk_id": chunk["id"], "chunk": chunk["chunk"], "retrieval_score": chunk["score"],
                      "chunk_char_start": chunk["char_start"], "chunk_char_end": chunk["char_end"]}
            if identity in candidates:
                candidates[identity]["retrieval_sources"].append(origin)
                candidates[identity]["retrieval_score"] = max(candidates[identity]["retrieval_score"], chunk["score"])
            else:
                candidates[identity] = {**passage, "id": hashlib.sha256(str(identity).encode()).hexdigest()[:24],
                    "document_id": chunk["document_id"], "filename": chunk["filename"], "page": chunk["page"],
                    "chunk": chunk["chunk"], "chunk_id": chunk["id"], "retrieval_score": chunk["score"],
                    "retrieval_sources": [origin], "expanded_to_coherent_boundaries": passage["char_start"] < chunk["char_start"] or passage["char_end"] > chunk["char_end"]}
    return list(candidates.values())


def select_coherent_passages(candidates, scores, max_chars=18000):
    if len(scores) != len(candidates) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("Reranker returned invalid or incomplete scores.")
    ranked = sorted(({**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)),
                    key=lambda c: (-c["rerank_score"], c["id"]))
    selected, seen, consumed = [], set(), 0
    for rank, candidate in enumerate(ranked, 1):
        candidate.update(rank=rank, selected=False)
        text_key = (candidate["document_id"], " ".join(candidate["text"].split()).casefold())
        overlap = any(s["document_id"] == candidate["document_id"] and s["page"] == candidate["page"]
                      and s["char_start"] < candidate["char_end"] and candidate["char_start"] < s["char_end"] for s in selected)
        if text_key in seen:
            status = "duplicate_text"
        elif overlap:
            status = "overlaps_selected_passage"
        elif consumed + len(candidate["text"]) > max_chars:
            status = "context_budget"
        else:
            status = "selected"
            candidate["selected"] = True
            selected.append(candidate); seen.add(text_key); consumed += len(candidate["text"])
        candidate["selection_status"] = status
    if not selected:
        raise ValueError("No coherent passage fits the context budget.")
    return selected, ranked
