import os
import threading
from typing import Literal

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from passages import coherent_passages

app = FastAPI(title="Local sentence cross-encoder")
MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
REVISION = os.getenv("RERANK_REVISION", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
model = tokenizer = None
lock = threading.Lock()
torch.set_num_threads(int(os.getenv("RERANK_THREADS", "4")))


class Request(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    sentences: list[str] = Field(min_length=1, max_length=2000)
    max_length: Literal[512, 4096] = 512


class PassageRequest(BaseModel):
    chunks: list[dict] = Field(min_length=1, max_length=20)


@app.post("/passages")
def passages(request: PassageRequest):
    global tokenizer
    with lock:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        return {"candidates": coherent_passages(request.chunks, lambda text: len(tokenizer.encode(text, add_special_tokens=False))),
                "tokenizer": MODEL, "target_tokens": [150, 250]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "loaded": model is not None, "device": "cpu"}


@app.post("/rerank")
def rerank(request: Request):
    global model, tokenizer
    with lock:
        if model is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION,
                                                                       trust_remote_code=False).eval()
        scores, truncated = [], 0
        batch_size = 2 if request.max_length == 4096 else 8
        for start in range(0, len(request.sentences), batch_size):
            pairs = [(request.query, sentence) for sentence in request.sentences[start:start + batch_size]]
            lengths = tokenizer(pairs, truncation=False, padding=False)["input_ids"]
            truncated += sum(len(tokens) > request.max_length for tokens in lengths)
            if request.max_length == 4096 and truncated:
                from fastapi import HTTPException
                raise HTTPException(422, "A whole question/passage pair exceeds 4096 tokens. Shorten the question or reduce chunk size; passages are not silently truncated.")
            inputs = tokenizer(pairs, padding=True, truncation=True, max_length=request.max_length, return_tensors="pt")
            with torch.inference_mode():
                scores.extend(model(**inputs).logits.view(-1).float().tolist())
        return {"scores": scores, "model": MODEL, "revision": model.config._commit_hash,
                "device": "cpu", "max_length": request.max_length, "truncated_pairs": truncated}
