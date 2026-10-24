const $ = id => document.getElementById(id);
let busy = false, currentRun = null, documents = [], statusReady = false;
const selected = new Set();

function fail(error) { $('error').textContent = error.message || String(error); $('error').hidden = false; }
function setBusy(value) {
  busy = value;
  $('compare').disabled = value || !statusReady;
  $('pdf').disabled = value;
  $('execution-mode').disabled = value;
  document.querySelectorAll('.document button,.document input,#history button,#feedback-form button').forEach(b => b.disabled = value);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { const body = await response.json(); message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail); } catch {}
    throw new Error(message);
  }
  return response.json();
}
async function stream(path, options, callback) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || `HTTP ${response.status}`));
  }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '', complete = false;
  function consume(line) {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === 'error') throw new Error(event.message);
    if (event.type === 'complete') complete = true;
    callback(event);
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split('\n'); buffer = lines.pop();
      for (const line of lines) consume(line);
      if (done) break;
    }
    if (buffer.trim()) consume(buffer);
    if (!complete) throw new Error('The connection ended before the operation completed. Check the services and try again.');
  } finally { if (!complete) await reader.cancel().catch(() => {}); reader.releaseLock(); }
}
function log(target, message, preview) {
  const row = document.createElement('div'); row.className = 'trace-row';
  const stamp = document.createElement('time'); stamp.textContent = new Date().toLocaleTimeString();
  row.append(stamp, document.createTextNode(message));
  if (preview !== undefined) {
    const details = document.createElement('details'), summary = document.createElement('summary'), pre = document.createElement('pre');
    summary.textContent = 'Inspect data'; pre.textContent = typeof preview === 'string' ? preview : JSON.stringify(preview, null, 2);
    details.append(summary, pre); row.append(details);
  }
  target.append(row); target.scrollTop = target.scrollHeight;
}
async function refreshStatus() {
  try {
    const s = await api('/api/status');
    statusReady = s.ollama && s.qdrant && s.models_ready && s.reranker;
    $('status').textContent = statusReady ? '● Ollama + Qdrant + reranker connected' : (!s.ollama ? 'Ollama unavailable' : !s.qdrant ? 'Qdrant unavailable' : !s.models_ready ? 'Required models missing' : 'Reranker unavailable');
    $('status').classList.toggle('status-good', statusReady);
    $('status').title = s.errors.join('\n'); $('model-name').textContent = s.chat_model;
    $('compare').disabled = busy || !statusReady;
  } catch { statusReady = false; $('status').textContent = 'App connection unavailable'; $('status').classList.remove('status-good'); $('compare').disabled = true; }
}
async function refreshDocuments() {
  const oldIds = new Set(documents.map(d => d.id));
  documents = await api('/api/documents');
  documents.forEach(d => { if (!oldIds.has(d.id)) selected.add(d.id); });
  for (const id of selected) if (!documents.some(d => d.id === id)) selected.delete(id);
  $('doc-count').textContent = `${documents.length} PDF${documents.length === 1 ? '' : 's'}`;
  $('documents').replaceChildren();
  if (!documents.length) { const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = '> NO DOCUMENTS INDEXED.'; $('documents').append(empty); }
  for (const d of documents) {
    const row = document.createElement('div'); row.className = 'document';
    const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.id = `doc-${d.id}`; checkbox.checked = selected.has(d.id);
    checkbox.addEventListener('change', () => checkbox.checked ? selected.add(d.id) : selected.delete(d.id));
    const label = document.createElement('label'); label.htmlFor = checkbox.id; label.textContent = d.filename;
    const meta = document.createElement('small'); meta.textContent = `${d.pages} text pages · ${d.chunks} chunks`; label.append(meta);
    if (d.empty_pages.length) { const warning = document.createElement('small'); warning.className = 'warning'; warning.textContent = `${d.empty_pages.length} pages skipped (may need OCR)`; label.append(warning); }
    const remove = document.createElement('button'); remove.className = 'text-button'; remove.textContent = '×'; remove.title = `Remove ${d.filename}`; remove.setAttribute('aria-label', remove.title);
    remove.addEventListener('click', async () => {
      if (busy) return;
      if (!confirm(`Remove ${d.filename} and its indexed passages? Saved comparisons will remain.`)) return;
      setBusy(true);
      try { await api(`/api/documents/${d.id}`, { method: 'DELETE' }); await refreshDocuments(); }
      catch (e) { fail(e); } finally { setBusy(false); }
    });
    row.append(checkbox, label, remove); $('documents').append(row);
  }
  setBusy(busy);
}
async function refreshHistory() {
  const history = await api('/api/runs'); $('history').replaceChildren();
  if (!history.length) { const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = '> NO SAVED RUNS.'; $('history').append(empty); }
  for (const run of history) {
    const button = document.createElement('button'); button.textContent = run.question;
    const meta = document.createElement('small'); meta.textContent = `${new Date(run.created).toLocaleString()}${run.feedback ? ' · Rated' : ''}`; button.append(meta);
    button.addEventListener('click', async () => {
      if (busy) return;
      setBusy(true);
      try {
        const saved = await api(`/api/runs/${run.id}`); resetComparison(); renderRun(saved);
        $('question').value = saved.question; $('pipeline-state').textContent = 'Saved comparison';
        log($('trace'), 'Loaded saved prompts, answers, and evidence. No new inference was run.');
      } catch (e) { fail(e); } finally { setBusy(false); }
    }); $('history').append(button);
  }
  setBusy(busy);
}
async function upload(files) {
  if (busy || !files.length) return;
  setBusy(true); $('error').hidden = true;
  $('ingestion').replaceChildren(); $('ingestion-details').open = true;
  try {
    for (const file of files) {
      $('ingestion-state').textContent = '· indexing';
      const data = new FormData(); data.append('file', file);
      log($('ingestion'), `Upload: ${file.name}`);
      await stream('/api/documents', { method: 'POST', body: data }, e => {
        if (e.type === 'step') log($('ingestion'), e.message);
        if (e.type === 'extracted') log($('ingestion'), `${e.pages} text pages extracted. ${e.empty_pages.length} pages skipped.`, { text_preview: e.preview, skipped_pages: e.empty_pages });
        if (e.type === 'chunked') log($('ingestion'), `${e.count} chunks · ${e.size} characters · ${e.overlap} overlap`, e.previews);
        if (e.type === 'vectors') log($('ingestion'), `${e.model} → ${e.dimension}-dimensional vectors`, e.sample);
        if (e.type === 'indexed') log($('ingestion'), `Qdrant stored ${e.completed}/${e.total} chunks.`);
        if (e.type === 'complete') log($('ingestion'), e.duplicate ? 'This PDF is already indexed.' : 'Indexing complete. Ready for retrieval.');
      });
    }
    $('ingestion-state').textContent = '· complete';
  } catch (e) { $('ingestion-state').textContent = '· failed'; log($('ingestion'), e.message); fail(e); }
  finally { $('pdf').value = ''; setBusy(false); await refreshDocuments().catch(fail); }
}
function resetComparison() {
  $('single-result').hidden = true;
  $('copy-status').textContent = '';
  currentRun = null; $('error').hidden = true; $('evaluation').hidden = true; $('sources-section').hidden = true; $('focus-sources-section').hidden = true; $('trace').replaceChildren();
  for (const variant of ['baseline', 'rag', 'focus']) { $(`${variant}-answer`).textContent = ''; $(`${variant}-answer`).classList.remove('placeholder'); $(`${variant}-metrics`).textContent = 'Waiting…'; $(`${variant}-prompt`).textContent = ''; }
  document.querySelectorAll('[data-stage]').forEach(e => e.classList.remove('active', 'done'));
  $('feedback-status').textContent = ''; $('notes').value = ''; $('winner').value = 'rag'; $('baseline-score').value = '3'; $('rag-score').value = '3';
}
function showSources(sources, seconds, variant = 'rag', level = 'sentence') {
  const prefix = variant === 'focus' ? 'focus-' : '';
  $(`${prefix}sources-section`).hidden = false; $(`${prefix}sources`).replaceChildren(); $(`${prefix}retrieval-time`).textContent = `${sources.length} evidence blocks · ${seconds.toFixed(2)}s preparation`;
  sources.forEach((s, i) => {
    const details = document.createElement('details'); details.className = 'source'; details.open = i === 0;
    const summary = document.createElement('summary'), tag = document.createElement('span'); tag.className = 'source-tag'; tag.textContent = `[S${i + 1}]`;
    summary.append(tag, document.createTextNode(`${s.filename} · p. ${s.page}`));
    const meta = document.createElement('small'); meta.textContent = variant === 'focus' ? `Cross-encoder score ${s.rerank_score.toFixed(3)} · ${level === 'passage' ? `${s.token_count ?? '?'} tokens · rank ${s.rank ?? '?'}` : `${s.anchors.length} selected sentences`} · page offsets ${s.char_start}–${s.char_end}` : `Cosine similarity ${s.score.toFixed(3)} · chunk ${s.chunk + 1}`; summary.append(meta);
    const text = document.createElement('p'); text.textContent = s.text; details.append(summary, text); $(`${prefix}sources`).append(details);
  });
}
function showMetrics(variant, result) {
  $(`${variant}-metrics`).textContent = `${result.seconds.toFixed(1)}s · ${result.input_tokens ?? '?'} input / ${result.output_tokens ?? '?'} output tokens${result.done_reason === 'length' ? ' · Output limit reached' : ''}`;
}
function renderRun(run) {
  if (run.execution_mode) { renderSingleRun(run); return; }
  $('execution-mode').value = 'compare';
  updateModeControls();
  $('single-result').hidden = true;
  document.querySelectorAll('.answer-panel,.answers').forEach(node => node.hidden = false);
  $('copy-status').textContent = '';
  currentRun = run; showSources(run.sources, run.retrieval_seconds);
  $('rerank-level').value = run.rerank_level ?? 'sentence';
  updateRankingControls();
  if (run.focus_sources?.length) showSources(run.focus_sources, run.focus_seconds, 'focus', run.rerank_level ?? 'sentence');
  for (const variant of ['baseline', 'rag', 'focus']) {
    if (!run.answers[variant]) { $(`${variant}-answer`).textContent = 'Focus Mode was not included in this saved comparison.'; continue; }
    $(`${variant}-answer`).textContent = run.answers[variant].text; showMetrics(variant, run.answers[variant]); $(`${variant}-prompt`).textContent = JSON.stringify({ messages: run.prompts[variant], options: run.options, think: run.think }, null, 2);
  }
  if (run.generalized_query) log($('trace'), 'Generalized retrieval query', run.generalized_query);
  if (run.ranked_sentences?.length) log($('trace'), `${run.ranked_sentences.length} sentence scores against the original query`, run.ranked_sentences);
  if (run.ranked_passages?.length) log($('trace'), `${run.ranked_passages.length} whole-passage scores against the original query`, run.ranked_passages);
  document.querySelectorAll('[data-stage]').forEach(e => { e.classList.remove('active'); e.classList.add('done'); });
  $('evaluation').hidden = false;
  $('focus-score').value = run.feedback?.focus_score ?? '3';
  if (run.feedback) { $('winner').value = run.feedback.winner; $('baseline-score').value = run.feedback.baseline_score; $('rag-score').value = run.feedback.rag_score; $('notes').value = run.feedback.notes; $('feedback-status').textContent = 'Saved assessment loaded'; }
}
$('compare-form').addEventListener('submit', async e => {
  e.preventDefault(); if (busy) return;
  if (!selected.size) { fail(new Error('Upload and select at least one PDF.')); return; }
  setBusy(true); resetComparison(); $('pipeline-state').textContent = 'Running';
  const body = { question: $('question').value.trim(), top_k: Number($('top-k').value), rerank_level: $('rerank-level').value, top_n: Number($('top-n').value), neighbors: $('rerank-level').value === 'sentence' && $('neighbors').checked, temperature: Number($('temperature').value), seed: Number($('seed').value), max_tokens: Number($('max-tokens').value), document_ids: [...selected] };
  try {
    if ($('execution-mode').value !== 'compare') {
      body.mode = $('execution-mode').value;
      log($('trace'), `Running ${body.mode}; the completed run will include its evidence and trace.`);
      const run = await api('/api/answer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      renderSingleRun(run);
      $('pipeline-state').textContent = run.status === 'failed' ? 'Failed' : 'Complete';
      await refreshHistory();
      return;
    }
    await stream('/api/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, event => {
      if (event.type === 'step') {
        log($('trace'), event.message);
        document.querySelectorAll('[data-stage]').forEach(node => { if (node.classList.contains('active')) node.classList.add('done'); node.classList.remove('active'); if (node.dataset.stage === event.stage) node.classList.add('active'); });
      }
      if (event.type === 'query_vector') log($('trace'), `Question mapped to a ${event.dimension}-dimensional vector.`, event.sample);
      if (event.type === 'generalized') log($('trace'), 'Generalized retrieval query', event.query);
      if (event.type === 'focus_chunks') log($('trace'), `${event.chunks.length} broad passages retrieved`, event.chunks);
      if (event.type === 'sentences') log($('trace'), `${event.count} unique sentence candidates`, event.candidates);
      if (event.type === 'passages') log($('trace'), `${event.count} whole-passage candidates`, event.candidates);
      if (event.type === 'focused') { showSources(event.sources, event.seconds, 'focus', event.rerank_level ?? 'sentence'); log($('trace'), `${event.rerank_level ?? 'sentence'} ranking and selected context`, { ranked: event.ranked, reranker: event.reranker }); }
      if (event.type === 'sources') { showSources(event.sources, event.seconds); log($('trace'), `${event.sources.length} passages retrieved. Inspect the evidence below.`); }
      if (event.type === 'prompt') { $(`${event.variant}-prompt`).textContent = JSON.stringify({ messages: event.messages, options: event.options, think: false }, null, 2); log($('trace'), `${{baseline: 'Without RAG', rag: 'Standard RAG', focus: 'Focus Mode'}[event.variant]} prompt prepared.`, event.messages); }
      if (event.type === 'token') $(`${event.variant}-answer`).append(document.createTextNode(event.text));
      if (event.type === 'answer_done') { showMetrics(event.variant, event.result); log($('trace'), `${event.variant} answer complete (${event.result.seconds}s; ${event.result.load_seconds}s model loading).`); }
      if (event.type === 'complete') { renderRun(event.run); $('pipeline-state').textContent = 'Complete'; log($('trace'), 'Three answers saved. Your assessment is ready.'); }
    }); await refreshHistory();
  } catch (error) { fail(error); $('pipeline-state').textContent = 'Failed'; log($('trace'), error.message); document.querySelectorAll('[data-stage]').forEach(n => n.classList.remove('active')); }
  finally { setBusy(false); }
});
$('feedback-form').addEventListener('submit', async e => {
  e.preventDefault(); if (!currentRun || busy) return;
  const feedback = { winner: $('winner').value, baseline_score: Number($('baseline-score').value), rag_score: Number($('rag-score').value), notes: $('notes').value };
  if (currentRun.answers.focus) feedback.focus_score = Number($('focus-score').value);
  try { await api(`/api/runs/${currentRun.id}/feedback`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(feedback) }); currentRun.feedback = feedback; $('feedback-status').textContent = 'Assessment saved'; await refreshHistory(); }
  catch (error) { fail(error); }
});
$('export').addEventListener('click', () => {
  if (!currentRun) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(currentRun, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = `comparison-${currentRun.id}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('copy-answers').addEventListener('click', async () => {
  if (!currentRun) return;
  const run = currentRun;
  const text = buildComparisonXray(run);
  $('copy-answers').disabled = true;
  $('copy-status').textContent = '';
  try {
    if (!navigator.clipboard?.writeText) throw new Error('Clipboard access is unavailable. Open the app at http://localhost:8000 and try again.');
    await navigator.clipboard.writeText(text);
    if (currentRun === run) $('copy-status').textContent = 'Full X-ray copied';
  } catch (error) {
    fail(new Error(`Could not copy the answers: ${error.message}`));
  } finally { $('copy-answers').disabled = false; }
});
$('pdf').addEventListener('change', e => upload([...e.target.files]));
$('dropzone').addEventListener('dragover', e => { e.preventDefault(); if (!busy) $('dropzone').classList.add('drag'); });
$('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('drag'));
$('dropzone').addEventListener('drop', e => { e.preventDefault(); $('dropzone').classList.remove('drag'); upload([...e.dataTransfer.files]); });
$('refresh-history').addEventListener('click', () => { if (!busy) refreshHistory().catch(fail); });
function updateRankingControls() {
  const passage = $('rerank-level').value === 'passage';
  const mode = $('execution-mode').value;
  const prefix = mode === 'tree' ? 'Tree' : 'Focus';
  $('top-n-label').textContent = passage ? `${prefix}: selected by context budget` : `${prefix}: top sentences`;
  $('top-n').disabled = passage || mode === 'baseline';
  $('neighbors').disabled = passage || mode === 'baseline';
  document.querySelector('[data-stage="sentences"] b').textContent = passage ? 'Coherent passages' : 'Sentences';
  $('focus-evidence-note').textContent = passage
    ? 'Coherent passages target 150–250 BGE tokens, preserving paragraphs, headings, and recognized conditional sequences. Ranked against the original question; selected within 18,000 characters with duplicate overlaps removed. Focus retrieves 20 chunks.'
    : 'Sentences ranked against your original question. Adjacent sentences can supply context; overlapping windows are merged. Cross-encoder scores are relevance scores, not confidence probabilities.';
}
$('rerank-level').addEventListener('change', updateRankingControls);
function renderSingleRun(run) {
  currentRun = run;
  $('single-result').hidden = false;
  $('execution-mode').value = run.execution_mode;
  updateModeControls();
  const result = Object.values(run.answers)[0];
  $('single-title').textContent = { baseline: 'Normal RAG', focus: 'Focus Mode', tree: 'Tree-RAG' }[run.execution_mode];
  $('single-status').textContent = `${run.status} · ${result.seconds.toFixed(1)}s · ${run.trace.llm_calls} model calls · ${run.trace.retrieval_calls} searches${run.error ? ` · ${run.error}` : ''}`;
  $('single-answer').textContent = result.text;
  $('single-evidence').textContent = JSON.stringify(run.sources, null, 2);
  $('single-trace').textContent = JSON.stringify({ analysis: run.analysis, plan: run.plan, node_results: run.node_results, trace: run.trace, fallback: run.fallback }, null, 2);
  $('evaluation').hidden = true;
  $('sources-section').hidden = true;
  $('focus-sources-section').hidden = true;
  document.querySelectorAll('.answer-panel,.answers').forEach(node => node.hidden = true);
  log($('trace'), `${run.execution_mode}: ${run.status}`, run.trace);
}
$('execution-mode').addEventListener('change', () => {
  const comparing = $('execution-mode').value === 'compare';
  updateModeControls();
  document.querySelectorAll('.answer-panel,.answers').forEach(node => node.hidden = !comparing);
  $('single-result').hidden = true;
});
function updateModeControls() {
  const mode = $('execution-mode').value;
  $('compare').firstChild.textContent = mode === 'compare' ? 'Compare answers ' : 'Run question ';
  $('mode-badge').textContent = { compare: 'One model. Three answers.', baseline: 'Normal RAG', focus: 'Focus Mode', tree: 'Question → prerequisites → answer' }[mode];
  $('mode-note').textContent = {
    compare: 'Same model, system prompt, and settings. Retrieved passages are the only prompt difference. Each question starts a fresh conversation.',
    baseline: 'Answer using the retrieved document passages. Each question starts a fresh conversation.',
    focus: 'Answer using evidence reranked against your question. Each question starts a fresh conversation.',
    tree: 'Resolve prerequisites, carry their verified evidence into dependent steps, and integrate the final answer.'
  }[mode];
  $('top-k').disabled = mode === 'focus' || mode === 'tree';
  $('rerank-level').disabled = mode === 'baseline';
  $('focus-limit').textContent = mode === 'tree' ? 'Tree: each search retrieves up to 20 chunks' : 'Focus retrieval: fixed Top-K 20';
  $('rerank-level').parentNode.firstChild.textContent = mode === 'tree' ? 'Tree ranking level' : 'Focus ranking level';
  updateRankingControls();
}
$('single-export').addEventListener('click', () => $('export').click());
updateModeControls();
Promise.all([refreshStatus(), refreshDocuments(), refreshHistory()]).catch(fail);
setInterval(refreshStatus, 15000);
