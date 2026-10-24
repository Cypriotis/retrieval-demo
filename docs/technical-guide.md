# Technical guide

[Back to README](../README.md)

## Architecture

```text
PDF -> page extraction -> overlapping chunks -> Qwen embeddings -> Qdrant
Question -> retrieval strategy -> selected evidence -> Qwen -> answer
Completed run -> SQLite history -> browser inspection / JSON export
```

- **API:** Python, FastAPI, Uvicorn, and HTTPX.
- **Interface:** HTML, CSS, and JavaScript.
- **Generation:** `qwen3:8b` through Ollama, with thinking disabled.
- **Embeddings:** `qwen3-embedding:0.6b`; query inputs include a retrieval instruction.
- **Vector search:** Qdrant cosine similarity, filtered to selected documents.
- **Reranking:** `BAAI/bge-reranker-v2-m3` through Transformers and CPU PyTorch.
- **Extraction:** pypdf; PySBD supports the optional English sentence-ranking experiment.
- **Persistence:** SQLite metadata, run history, and assessments; Qdrant vectors and passage payloads.

Models are pretrained and are not fine-tuned by this application.

### Normal RAG

Page text is normalized and split into chunks of up to 3,000 characters with approximately 300 characters of overlap. Chunking favors whitespace boundaries. The original question is embedded and used to retrieve top-K chunks (default 4, configurable from 1 to 20).

Evidence includes document, page, chunk, and offset metadata. The answer prompt labels passages with citations such as `[S1]` and asks the model to acknowledge insufficient evidence.

### Focus Mode

Qwen rewrites the question as a broader search query while being instructed to preserve identifiers, constraints, and intent. Qdrant retrieves up to 20 chunks using the rewritten query. The original question is retained for reranking and generation.

The default passage builder reconstructs paragraphs and headings from source-page text. Rules preserve recognizable conditional branches and introductory lists. Passages target 150-250 BGE tokenizer tokens; atomic paragraphs and conditional groups can exceed this target rather than being split.

BGE scores each original-question/passage pair. A greedy selector removes duplicate and overlapping evidence and includes whole passages within an 18,000-character evidence-text budget. There is no fixed passage-count cap or minimum relevance threshold. Question/passage pairs exceeding 4,096 tokens fail explicitly rather than being silently truncated.

The optional sentence mode ranks individual sentences and can add adjacent sentences. Its top-N and neighbor settings do not apply to coherent passage selection.

### Tree-RAG

Tree-RAG adds schema-constrained analysis, dependency planning, validation, and execution around Focus retrieval. Its internal plan is a directed acyclic graph, allowing shared prerequisites to execute once. Independent ready nodes may retrieve concurrently, while reasoning-related Qwen operations are serialized.

Intermediate values used in dependent queries must match supporting evidence. Source IDs and literal quotations are checked before results are integrated. Runs can end with answered, partial, unanswerable, blocked, or failed outcomes. Timeouts, execution limits, retries, and recovery are bounded and recorded.

This is a partial implementation inspired by RT-RAG. It does not implement consensus tree selection, rejection sampling, or general iterative rewriting, and does not claim the research method's benchmark results.

## Configuration

Defaults are listed in `.env.example`. Common settings include:

| Setting | Default | Purpose |
| --- | --- | --- |
| `CHAT_MODEL` | `qwen3:8b` | Generation and planning model |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | Retrieval embedding model |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder model |
| `RAG_MODE` | `baseline` | Default independent API mode |
| `REASONING_ENABLED` | `true` | Enable Tree-RAG |
| `REASONING_TIMEOUT_SECONDS` | `120` | Total reasoning deadline |
| `REASONING_NODE_TIMEOUT_SECONDS` | `60` | Per-node deadline |
| `REASONING_SUPPORT_VERIFIER` | `none` | Optional fallible LLM support check |
| `REASONING_FALLBACK_ENABLED` | `false` | Allow recorded Normal RAG fallback |

The API value `baseline` means Normal RAG. In comparison exports, `answers.baseline` identifies the answer without retrieval.

Changing the embedding model or chunking configuration creates a separate index and requires re-importing documents. Model tags can change upstream; saved runs record settings and outputs, not model-weight snapshots. Use immutable image digests and model revisions when stronger reproducibility is needed.

## API

Interactive documentation: **http://localhost:8000/docs**.

For independent answers, send `POST /api/answer`:

```json
{
  "question": "What requirements apply to this procedure?",
  "mode": "focus"
}
```

Supported modes are `baseline`, `focus`, and `tree`. The API also exposes document management, comparisons, history, assessments, and reasoning configuration. Run `GET /api/reasoning/config` to inspect reasoning settings.

## Operations

```sh
# Stop services while retaining data.
docker compose down

# Inspect startup and service logs.
docker compose logs -f model-init ollama app qdrant reranker

# Inspect loaded models and GPU placement.
docker compose exec ollama ollama ps
```

Four named volumes preserve Ollama weights, Qdrant data, SQLite data, and reranker weights. Original PDF binaries are not retained.

**`docker compose down --volumes` erases the project's persisted data and downloaded container models.**

## Local development

Use Python 3.12 with uv:

```sh
uv sync --python 3.12
```

For execution outside Docker, provide reachable Ollama, Qdrant, and reranker services. Set `OLLAMA_URL`, `QDRANT_URL`, and `RERANK_URL` as needed; defaults use localhost ports 11434, 6333, and 8001. Make the configured models available, then run:

```sh
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For a native process, set environment variables in the shell; the application does not automatically load `.env`.

Tests, evaluation scripts, sample documents, detailed implementation notes, and the host-Ollama Compose variant may be maintained locally but are excluded by this repository's ignore rules. The Docker quick start does not require those materials.

## Limits and interpretation

- This is a local, single-user lab without authentication. Compose publishes the application on `127.0.0.1:8000`; Qdrant, Ollama, and the reranker have no published ports. Public deployment requires additional access controls.
- Upload limits are 25 MB, 1,000 pages, and 10,000 chunks per PDF. Encrypted PDFs are rejected. Scanned PDFs require OCR beforehand; OCR is not included.
- PDF extraction and paragraph heuristics can misinterpret complex layouts, tables, and cross-page conditions.
- Cosine similarity and reranker logits are relevance signals, not confidence probabilities. Retrieved text may be irrelevant when the corpus lacks an answer.
- Citations and exact-quote checks do not prove factual entailment or completeness. Review model answers against the evidence.
- Comparisons use matching generation settings and run sequentially. Timing includes loading effects and is not a controlled latency benchmark. Fixed seeds do not guarantee identical outputs across runtimes.
- Focus and Tree-RAG add processing and model calls. No general accuracy advantage is claimed without a sufficiently reviewed evaluation set.
- Exports and traces can contain document excerpts, prompts, answers, and embedding vectors. Review their contents before sharing.
