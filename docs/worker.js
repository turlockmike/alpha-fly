// The engine lives in this worker so the page stays responsive while the fly thinks.
importScripts("vendor/chess.js");

let model = null, gpu = null, fly = null, backend = "cpu", evalMs = 0;

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
  // time one evaluation so the page can estimate how long a search will take
  const t = model.tests[1], feats = fly.encodeFen(t.fen, false);
  const t1 = performance.now();
  for (let i = 0; i < 3; i++) await evaluate(feats, t.idx);
  evalMs = (performance.now() - t1) / 3;
  post({ type: "ready", backend, gpuName, loadMs: Math.round(loadMs), evalMs: Math.round(evalMs), neurons: model.n,
         edges: model.col.length, steps: model.steps, checkpoint: model.checkpoint });
}

async function evaluate(feats, idx) {
  return backend === "gpu" ? fly.evaluateGpu(model, gpu, feats, idx) : fly.evaluateCpu(model, feats, idx);
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
