/* Theme preference is local to this browser; system preference is the default. */
(() => {
  const root = document.documentElement;
  const button = document.getElementById('theme-toggle');
  const system = window.matchMedia('(prefers-color-scheme: dark)');
  let preference;
  try { preference = localStorage.getItem('retrieval-demo-theme'); } catch {}
  if (preference === 'light' || preference === 'dark') root.dataset.theme = preference;
  function sync() {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : system.matches;
    button.textContent = dark ? '[ LIGHT ]' : '[ DARK ]';
    button.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} theme`);
    button.setAttribute('aria-pressed', String(dark));
  }
  button.addEventListener('click', () => {
    root.dataset.theme = (root.dataset.theme ? root.dataset.theme === 'dark' : system.matches) ? 'light' : 'dark';
    try { localStorage.setItem('retrieval-demo-theme', root.dataset.theme); } catch {}
    sync();
  });
  system.addEventListener('change', sync);
  sync();
})();
