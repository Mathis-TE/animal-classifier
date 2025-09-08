// script.js — front Live Server -> API FastAPI
(() => {
  'use strict';

  const API = 'http://localhost:8000'; // <—— route correcte pour l’API
  const $ = (id) => document.getElementById(id);

  const els = {
    file: $('file'),
    btnPredict: $('btnPredict'),
    btnReset: $('btnReset'),
    preview: $('preview'),
    probs: $('probs'),
    predBadge: $('predBadge'),
    jsonOut: $('jsonOut'),
    modelType: $('modelType'),
    modelDepth: $('modelDepth'),
    modelArch: $('modelArch'),
    status: $('status'),
    err: $('err'),
  };

  const fmtPct = (x) => (100 * x).toFixed(1) + '%';
  const setErr = (msg) => { if (els.err) els.err.textContent = msg || ''; };

  function renderProbs(probs, labels = ['ecureuil','hibou','renard']) {
    if (!els.probs) return;
    els.probs.innerHTML = '';
    const max = Math.max(...probs);
    probs.forEach((p, i) => {
      const wrap = document.createElement('div');
      wrap.className = 'mb-2';
      wrap.innerHTML = `
        <div class="d-flex justify-content-between">
          <div>${labels[i] ?? ('Classe ' + i)}</div>
          <div class="text-secondary">${fmtPct(p)}</div>
        </div>
        <div class="progress" role="progressbar" aria-label="${labels[i] || ('Classe '+i)}">
          <div class="progress-bar ${p===max ? 'bg-success' : ''}" style="width:${Math.max(5, p*100)}%"></div>
        </div>`;
      els.probs.appendChild(wrap);
    });
  }

  function renderModel(meta) {
    if (els.modelType)  els.modelType.textContent  = meta.type ?? 'Modèle';
    if (els.modelDepth && Array.isArray(meta.arch)) els.modelDepth.textContent = `${meta.arch.length} couches`;
    if (els.modelArch  && Array.isArray(meta.arch)) els.modelArch.textContent  = `[${meta.arch.join(' › ')}]`;
  }

  async function loadModel() {
    try {
      const r = await fetch(API + '/model');
      if (!r.ok) throw new Error('GET /model ' + r.status);
      const j = await r.json();
      renderModel(j);
      return j.labels || ['ecureuil','hibou','renard'];
    } catch (e) { setErr('Impossible de charger /model'); console.error(e); return ['ecureuil','hibou','renard']; }
  }

  async function predict(file, labels) {
    setErr('');
    if (els.status) els.status.textContent = 'classification…';
    if (els.btnPredict) els.btnPredict.disabled = true;
    try {
      const fd = new FormData(); fd.append('file', file);
      const r = await fetch(API + '/predict', { method: 'POST', body: fd });
      if (!r.ok) throw new Error('POST /predict ' + r.status);
      const j = await r.json();
      if (Array.isArray(j.probs)) renderProbs(j.probs, labels);
      if (els.predBadge && j.klass) { els.predBadge.className = 'badge text-bg-primary'; els.predBadge.textContent = j.klass; }
      if (els.jsonOut) els.jsonOut.textContent = JSON.stringify(j, null, 2);
      if (j.arch) renderModel({ arch: j.arch, type: els.modelType?.textContent || 'Modèle' });
    } catch (e) {
      setErr('La prédiction a échoué (backend indisponible ?).');
      console.error(e);
    } finally {
      if (els.status) els.status.textContent = '';
      if (els.btnPredict) els.btnPredict.disabled = false;
    }
  }

  document.addEventListener('DOMContentLoaded', async () => {
    console.log('script.js chargé');
    const labels = await loadModel();

    if (els.file) {
      els.file.addEventListener('change', (e) => {
        setErr('');
        const f = e.target.files?.[0];
        if (els.btnPredict) els.btnPredict.disabled = !f;
        if (!f) { if (els.preview) els.preview.classList.add('d-none'); return; }
        if (!f.type || !f.type.startsWith('image/')) {
          console.warn('Type MIME non image, on tente quand même.');
        }
        if (els.preview) {
          if (els.preview.src) URL.revokeObjectURL(els.preview.src);
          els.preview.src = URL.createObjectURL(f);
          els.preview.classList.remove('d-none');
        }
      });
    }

    if (els.btnPredict) {
      els.btnPredict.addEventListener('click', () => {
        const f = els.file?.files?.[0];
        if (f) predict(f, labels);
      });
    }

    if (els.btnReset) {
      els.btnReset.addEventListener('click', () => {
        setErr('');
        if (els.file) els.file.value = '';
        if (els.btnPredict) els.btnPredict.disabled = true;
        if (els.preview) els.preview.classList.add('d-none');
        if (els.probs) els.probs.innerHTML = '';
        if (els.predBadge) { els.predBadge.className = 'badge text-bg-secondary'; els.predBadge.textContent = '—'; }
        if (els.jsonOut) els.jsonOut.textContent = '—';
      });
    }
  });
})();
