"""Sentence-level evidence selection with stable positions in normalized page text."""
import hashlib
import math
import re

import pysbd


def sentence_candidates(chunks):
    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    candidates, seen = [], set()
    for chunk in chunks:
        page_text = chunk.get("page_text", chunk["text"])
        start = chunk.get("char_start", 0)
        end = chunk.get("char_end", len(page_text))
        sentences = segmenter.segment(page_text)
        for position, span in enumerate(sentences):
            # Use full-page boundaries to avoid ranking clipped chunk fragments.
            if span.end <= start or span.start >= end or not span.sent.strip():
                continue
            key = (chunk["document_id"], chunk["page"], span.start, span.end)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "id": hashlib.sha256(str(key).encode()).hexdigest()[:24],
                "document_id": chunk["document_id"], "filename": chunk["filename"],
                "page": chunk["page"], "chunk": chunk["chunk"], "chunk_id": chunk["id"],
                "sentence_position": position, "char_start": span.start, "char_end": span.end,
                "text": span.sent.strip(), "retrieval_score": chunk["score"],
                "previous": sentences[position - 1].sent.strip() if position else "",
                "next": sentences[position + 1].sent.strip() if position + 1 < len(sentences) else "",
                "previous_start": sentences[position - 1].start if position else span.start,
                "next_end": sentences[position + 1].end if position + 1 < len(sentences) else span.end,
            })
    return candidates


def select_evidence(candidates, scores, top_n, neighbors, max_chars=18000):
    if len(candidates) != len(scores) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("Reranker returned invalid or incomplete scores.")
    ranked = sorted(({**candidate, "rerank_score": float(score)} for candidate, score in zip(candidates, scores)),
                    key=lambda item: (-item["rerank_score"], item["id"]))
    anchors, seen = [], set()
    for candidate in ranked:
        # Collapse repeated boilerplate within a document, keeping cross-document attribution.
        key = (candidate["document_id"], re.sub(r"\s+", " ", candidate["text"]).casefold())
        if key not in seen:
            anchors.append(candidate)
            seen.add(key)
        if len(anchors) == top_n:
            break
    groups = []
    for anchor in anchors:
        start = anchor["previous_start"] if neighbors and anchor["previous"] else anchor["char_start"]
        end = anchor["next_end"] if neighbors and anchor["next"] else anchor["char_end"]
        parts = ([anchor["previous"]] if neighbors and anchor["previous"] else []) + [anchor["text"]] + ([anchor["next"]] if neighbors and anchor["next"] else [])
        # Merge intersecting windows on the same page; preserve document reading order.
        matching = [g for g in groups if g["document_id"] == anchor["document_id"] and g["page"] == anchor["page"]
                    and not (g["char_end"] < start or g["char_start"] > end)]
        member = {**anchor, "window_start": start, "window_end": end, "window_parts": parts}
        if matching:
            group = matching[0]
            group["members"].append(member)
            for other in matching[1:]:
                group["members"].extend(other["members"]); groups.remove(other)
            group["char_start"] = min(start, *(g["char_start"] for g in matching))
            group["char_end"] = max(end, *(g["char_end"] for g in matching))
        else:
            groups.append({**anchor, "char_start": start, "char_end": end, "members": [member]})
    evidence, consumed = [], 0
    for group in groups:
        ordered = sorted(group["members"], key=lambda item: item["window_start"])
        texts, seen_parts = [], set()
        for member in ordered:
            for part in member["window_parts"]:
                # Windows consist of consecutive sentences; remove their shared sentences.
                if part not in seen_parts:
                    texts.append(part); seen_parts.add(part)
        text = " ".join(texts)
        if consumed + len(text) > max_chars:
            continue  # Never truncate a sentence or misreport an included anchor.
        consumed += len(text)
        evidence.append({k: v for k, v in {**group, "text": text, "score": group["rerank_score"],
                         "anchors": [{k: m[k] for k in ("id", "text", "sentence_position", "char_start", "char_end", "rerank_score")} for m in group["members"]]}.items()
                         if k not in ("members", "previous", "next", "previous_start", "next_end")})
    if not evidence:
        raise ValueError("No complete sentence evidence fits the context budget.")
    return evidence, ranked
