function buildComparisonXray(run) {
  const lines = [];
  const json = value => JSON.stringify(value ?? null, null, 2);
  const section = (title, value) => lines.push(`\n## ${title}\n`, typeof value === 'string' ? value : json(value));
  lines.push('# RAG comparison X-ray', `Run: ${run.id}`, `Created: ${run.created}`, `Model: ${run.model}`, `Embedding model: ${run.embedding_model}`);
  section('Original user question', run.question);
  section('Selected documents', run.documents ?? { document_ids: run.document_ids, note: 'Document snapshot was not recorded in this older run.' });
  section('Settings and phase timings', {
    generation: run.options, think: run.think, top_k: run.top_k, top_n: run.top_n,
    focus_reranking_level: run.rerank_level ?? 'sentence',
    focus_retrieval_top_k: run.focus_top_k, passage_target_tokens: run.passage_target_tokens,
    focus_selection_strategy: run.focus_selection_strategy,
    include_neighbor_sentences: run.neighbors, retrieval: run.retrieval_configuration,
    phase_seconds: run.timings, standard_retrieval_seconds: run.retrieval_seconds, focus_preparation_seconds: run.focus_seconds,
  });
  lines.push('\nScore interpretation: Qdrant scores are cosine similarity. Sentence reranker scores are raw cross-encoder logits. Neither is a factual-confidence probability. There is no minimum score cutoff. Citation labels restart at [S1] for each answer.');
  function passages(title, sources, focused = false) {
    section(title, sources?.length ? `${sources.length} evidence blocks, in their supplied order:` : 'None.');
    (sources ?? []).forEach((source, index) => {
      lines.push(`\n[S${index + 1}] ${source.filename} | PDF page ${source.page}`);
      lines.push(json({ document_id: source.document_id, point_id: source.id, chunk_id: source.chunk_id,
        chunk_position: source.chunk, char_start: source.char_start, char_end: source.char_end,
        cosine_similarity: focused ? source.retrieval_score : source.score,
        cross_encoder_score: source.rerank_score, selected_anchor_sentences: source.anchors }));
      if (source.token_count !== undefined) lines.push(json({ token_count: source.token_count,
        rank: source.rank, selection_status: source.selection_status, heading: source.heading,
        paragraph_spans: source.paragraph_spans, size_exception: source.size_exception,
        retrieval_sources: source.retrieval_sources }));
      lines.push('Exact evidence text:', source.text);
    });
  }
  for (const [variant, title] of [['baseline', 'No RAG'], ['rag', 'Standard RAG'], ['focus', 'Focus Mode']]) {
    lines.push(`\n# ${title}`);
    if (!run.answers?.[variant]) { lines.push('Not included in this comparison.'); continue; }
    if (variant === 'baseline') {
      section('Retrieval query and evidence', 'No retrieval query, embeddings, similarity scores, or document passages are used by the No RAG answer. The original user question is supplied with the recorded system prompt.');
    } else if (variant === 'rag') {
      section('Exact retrieval query before embedding instructions', run.question);
      passages('Retrieved chunks and exact passages supplied to Qwen', run.sources);
    } else {
      section('Exact Ollama-generated retrieval query before embedding instructions', run.generalized_query ?? 'Not recorded.');
      section('Query rewrite token usage', run.rewrite_usage);
      passages('Broad chunks retrieved by the generalized query (candidate pool, not final context)', run.focus_chunks);
      const passageMode = run.rerank_level === 'passage';
      section(passageMode ? 'Coherent-passage reranking query' : 'Sentence reranking query', run.question);
      section('Cross-encoder model and truncation information', run.reranker);
      const included = new Set((run.focus_sources ?? []).flatMap(s => passageMode ? [s.id] : (s.anchors ?? []).map(a => a.id)));
      section(passageMode ? 'All scored passage candidates, highest score first' : 'All scored sentence candidates, highest score first', (passageMode ? run.ranked_passages ?? [] : run.ranked_sentences ?? []).map((candidate, index) => ({
        rank: index + 1, selected_in_final_context: included.has(candidate.id),
        included_in_final_context_window: (run.focus_sources ?? []).some(source =>
          source.document_id === candidate.document_id && source.page === candidate.page &&
          source.char_start <= candidate.char_start && source.char_end >= candidate.char_end), ...candidate,
      })));
      lines.push(passageMode ? 'Coherent-passage runs select by rank within the context budget, without a Top-N cap. Candidate records include token counts, retrieval provenance, size exceptions, rank, and selection/rejection status. Historical whole-chunk runs retain their original saved settings and evidence.' : 'Unselected sentences can still appear as neighbors. Top-N applies to anchors; deduplication, window merging, and the context budget determine the final blocks.');
      passages('Exact selected evidence supplied to Qwen', run.focus_sources, true);
    }
    section('Exact generation messages supplied to Qwen', run.prompts?.[variant] ?? 'Not recorded.');
    const { text, ...metrics } = run.answers[variant];
    section('Answer', text);
    section('Generation timing and token consumption', metrics);
  }
  section('Saved manual assessment', run.feedback ?? 'No assessment saved.');
  section('Exact service requests and computed responses, in execution order', run.service_calls?.length ? run.service_calls :
    'Exact network payloads and full query vectors were not recorded for this older comparison. Run a new comparison to capture them. The prompts, evidence, queries, and scores above use the data saved with this run.');
  lines.push('\nThe service-call trace includes actual embedding inputs (including query instructions), complete query vectors, Qdrant filters and results, rewrite requests and responses, reranker input order and scores, and generation requests. Streamed generation responses retain the assembled answer and final event, rather than every individual token event.');
  section('Complete exported run JSON', run);
  return lines.join('\n');
}
