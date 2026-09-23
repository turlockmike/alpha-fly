// The engine lives in this worker so the page stays responsive while the fly thinks.
importScripts("vendor/chess.js");

let model = null, gpu = null, fly = null, backend = "cpu", evalMs = 0, pool = null;

/** A pool of compute workers, each owning a slice of the neurons and only its slice of the synapses.  Every
 *  update step sends them the whole state (667 KB each) and gets their slice of the next state back. */
function makePool(m, k) {
  const slices = fly.rowSlices(m.crow, k);
  const workers = slices.map(([i0, count]) => {
    const worker = new Worker("compute.js", { type: "module" });
    const a = m.crow[i0], b = m.crow[i0 + count];
    const crow = new Uint32Array(count + 1);
    for (let r = 0; r <= count; r++) crow[r] = m.crow[i0 + r] - a;
    const vals = m.values.slice(a, b), col = m.col.slice(a, b);
    worker.postMessage({ type: "init", i0, count, vals, col, crow, alpha: m.alpha, hMax: m.hMax }, [vals.buffer, col.buffer, crow.buffer]);
    return { worker, i0, count };
  });
  const ask = (msg) => Promise.all(workers.map(({ worker }) => new Promise((resolve, reject) => {
    worker.onmessage = e => (e.data && e.data.error ? reject(new Error(e.data.error)) : resolve(e.data)); worker.onerror = e => reject(new Error(e.message));
    worker.postMessage(msg);
  })));
  return {
    size: k,
    begin: (biasTotal) => ask({ type: "begin", biasTotal }),
    async step(hIn, hOut) {
      const parts = await ask({ type: "step", h: hIn });
      for (let w = 0; w < workers.length; w++) hOut.set(parts[w], workers[w].i0);
    },
    close() { for (const { worker } of workers) worker.terminate(); },
  };
}

const post = (msg) => self.postMessage(msg);

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.arrayBuffer();
}

async function load(base) {
  fly = await import("./fly.js");
  const t0 = performance.now();
  model = await fly.loadModel(base, fetchBytes, (done, total) => post({ type: "progress", done, total }));
  const loadMs = performance.now() - t0;
  let gpuName = null;
  try { gpu = await fly.initGpu(model); } catch (e) { gpu = null; post({ type: "log", text: "WebGPU unavailable: " + e.message }); }
  if (gpu) {
    // check the GPU path against the CPU path on the first test position before trusting it
    const t = model.tests[0], feats = fly.encodeFen(t.fen, false);
    const a = fly.evaluateCpu(model, feats, t.idx), b = await fly.evaluateGpu(model, gpu, feats, t.idx);
    let d = 0; for (let i = 0; i < t.idx.length; i++) d = Math.max(d, Math.abs(a.logits[i] - b.logits[i]));
    if (d < 0.05 && Math.abs(a.value - b.value) < 0.01) { backend = "gpu"; gpuName = gpu.name; }
    else { post({ type: "log", text: `WebGPU result differed from CPU by ${d.toFixed(3)}; using the CPU` }); gpu = null; }
  }
  const t = model.tests[1], feats = fly.encodeFen(t.fen, false);
  const time = async (n) => { const t1 = performance.now(); for (let i = 0; i < n; i++) await evaluate(feats, t.idx); return (performance.now() - t1) / n; };
  let cores = 1;
  if (backend !== "gpu") {
    // CPU: split the matrix passes over the cores, and keep the pool only if it is actually faster here
    const single = await time(3);
    const k = Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 2) - 1));
    if (k > 1) {
      pool = makePool(model, k); backend = "pool";
      const ref = fly.evaluateCpu(model, feats, t.idx), got = await evaluate(feats, t.idx);
      let d = 0; for (let i = 0; i < t.idx.length; i++) d = Math.max(d, Math.abs(ref.logits[i] - got.logits[i]));
      const multi = d < 1e-3 ? await time(3) : Infinity;
      if (multi < single) { evalMs = multi; cores = k; }
      else { pool.close(); pool = null; backend = "cpu"; evalMs = single; }
    } else evalMs = single;
  } else evalMs = await time(3);
  post({ type: "ready", backend: backend === "pool" ? "cpu" : backend, cores, gpuName, loadMs: Math.round(loadMs), evalMs: Math.round(evalMs),
         neurons: model.n, edges: model.col.length, steps: model.steps, checkpoint: model.checkpoint });
}

async function evaluate(feats, idx) {
  if (backend === "gpu") return fly.evaluateGpu(model, gpu, feats, idx);
  if (backend === "pool") return fly.evaluatePool(model, pool, feats, idx);
  return fly.evaluateCpu(model, feats, idx);
}

async function think({ id, moves, sims, explore }) {
  const game = new Chess();
  for (const u of moves) game.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] });
  const t0 = performance.now();
  let r;
  if (sims <= 0) r = await fly.rawMove(game, evaluate, explore);
  else r = await fly.search(game, evaluate, { sims, maxConsidered: 16, cVisit: 50, cScale: 1, explore });
  if (!r) return post({ type: "thought", id, move: null });
  post({ type: "thought", id, move: { from: r.move.from, to: r.move.to, promotion: r.move.promotion, san: r.move.san },
         value: r.value, raw: r.raw, sims: r.sims, considered: r.considered.slice(0, 8), ms: Math.round(performance.now() - t0) });
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") await load(msg.base);
    else if (msg.type === "think") await think(msg);
  } catch (err) {
    post({ type: "error", id: msg.id, text: String(err && err.stack || err) });
  }
};
