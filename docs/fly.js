// alpha-fly: the fly connectome chess network in the browser.  One file, no dependencies, runs in a Web Worker
// (or in Node for tests).  The recurrence is the same as chessfly/brain.py:
//     h <- h + alpha * (clamp(W h + bias + input, 0, h_max) - h)     for `steps` updates, from h = 0,
// followed by a per-neuron standardisation of the output neurons and linear heads (chessfly/model.py).
// Evaluation runs on the GPU through WebGPU when the browser has it, else in plain float32 JavaScript.

const PIECE = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

// ---- model ----------------------------------------------------------------------------------------------------

function f16to32(u16) {                        // IEEE half -> float, elementwise
  const out = new Float32Array(u16.length);
  const buf = new ArrayBuffer(4), f32 = new Float32Array(buf), u32 = new Uint32Array(buf);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) {
      if (m === 0) { u32[0] = s; out[i] = f32[0]; continue; }
      let mm = m, ee = -1;                     // subnormal
      do { ee++; mm <<= 1; } while ((mm & 0x400) === 0);
      u32[0] = s | ((127 - 15 - ee) << 23) | ((mm & 0x3ff) << 13);
    } else if (e === 31) u32[0] = s | 0x7f800000 | (m << 13);
    else u32[0] = s | ((e - 15 + 127) << 23) | (m << 13);
    out[i] = f32[0];
  }
  return out;
}

function halfAt(u16, i) {                      // one half -> float, for rows read on demand
  const h = u16[i], e = (h >> 10) & 0x1f, m = h & 0x3ff, sign = h & 0x8000 ? -1 : 1;
  if (e === 0) return sign * m * 2 ** -24;
  if (e === 31) return m ? NaN : sign * Infinity;
  return sign * (1 + m / 1024) * 2 ** (e - 15);
}

export async function loadModel(base, fetchBytes, onProgress) {
  const manifest = JSON.parse(new TextDecoder().decode(await fetchBytes(base + "manifest.json")));
  const total = Object.values(manifest.tensors).reduce((s, t) => s + t.bytes, 0);
  let done = 0;
  const raw = {};
  for (const [name, t] of Object.entries(manifest.tensors)) {
    const buf = new Uint8Array(t.bytes);
    let off = 0;
    for (const f of t.files) {
      const part = new Uint8Array(await fetchBytes(base + f));
      buf.set(part, off); off += part.length; done += part.length;
      if (onProgress) onProgress(done, total);
    }
    const b = buf.buffer;
    raw[name] = t.kind === "f32" ? new Float32Array(b) : t.kind === "u32" ? new Uint32Array(b)
              : t.kind === "f16" ? new Uint16Array(b) : buf;
  }
  const col = new Uint32Array(manifest.edges);
  const c = raw.col;
  for (let i = 0, j = 0; i < col.length; i++, j += 3) col[i] = c[j] | (c[j + 1] << 8) | (c[j + 2] << 16);
  const m = {
    n: manifest.neurons, steps: manifest.steps, hMax: manifest.h_max, inputGain: manifest.input_gain, nBase: manifest.n_base,
    clip: manifest.norm_clip, tests: manifest.tests, checkpoint: manifest.checkpoint,
    values: f16to32(raw.values), crow: raw.crow, col, bias: raw.bias, alpha: raw.alpha, inIdx: raw.in_idx, outIdx: raw.out_idx,
    encoderWT: raw.encoder_wT, encoderB: raw.encoder_b, normMean: raw.norm_mean, normStd: raw.norm_std,
    policyW: raw.policy_w, policyB: raw.policy_b, fromW: raw.from_w, fromB: raw.from_b, toW: raw.to_w, toB: raw.to_b,
    valueW: raw.value_w, valueB: raw.value_b, nIn: raw.in_idx.length, nOut: raw.out_idx.length,
  };
  m.biasTotal = new Float32Array(m.n); m.hA = new Float32Array(m.n); m.hB = new Float32Array(m.n);
  return m;
}

// ---- board encoding (chessfly/encoding.py, first N_BASE features; attack planes are not read by this network) ----

const FILES = "abcdefgh";
export function sq(name) { return (name.charCodeAt(1) - 49) * 8 + FILES.indexOf(name[0]); }   // a1 = 0, like python-chess

/** Active features of a FEN from the side to move's view.  repetition: has this position occurred before? */
export function encodeFen(fen, repetition) {
  const [placement, turn, castling, ep, halfmove] = fen.split(" ");
  const white = turn === "w", flip = !white;
  const feats = [];                            // [index, value] pairs
  let rank = 7, file = 0;
  for (const ch of placement) {
    if (ch === "/") { rank--; file = 0; continue; }
    if (ch >= "1" && ch <= "8") { file += +ch; continue; }
    const isWhite = ch === ch.toUpperCase(), mine = isWhite === white;
    let s = rank * 8 + file; if (flip) s ^= 56;
    feats.push([((mine ? 0 : 6) + PIECE[ch.toLowerCase()]) * 64 + s, 1]);
    file++;
  }
  if (ep !== "-") { let s = sq(ep); if (flip) s ^= 56; feats.push([12 * 64 + s, 1]); }
  const c = castling === "-" ? "" : castling;
  const rights = white ? ["K", "Q", "k", "q"] : ["k", "q", "K", "Q"];
  rights.forEach((r, i) => { if (c.includes(r)) feats.push([13 * 64 + i, 1]); });
  const hm = Math.min(+halfmove || 0, 100) / 100;
  if (hm > 0) feats.push([13 * 64 + 4, hm]);
  if (repetition) feats.push([13 * 64 + 5, 1]);
  return feats;
}

/** Policy index of a move (squares as 0..63, promotion as 'q' 'r' 'b' 'n' or undefined). */
export function moveIndex(from, to, promotion, white) {
  if (!white) { from ^= 56; to ^= 56; }
  if (promotion && promotion !== "q") {
    const ff = from & 7;
    return 4096 + (ff * 3 + ((to & 7) - ff + 1)) * 3 + { n: 0, b: 1, r: 2 }[promotion];
  }
  return from * 64 + to;
}

function moveSquares(i) {                      // from- and to-square of a policy index (side-to-move frame)
  if (i < 4096) return [i >> 6, i & 63];
  const k = i - 4096, ff = (k / 9) | 0, d = ((k % 9) / 3) | 0;
  return [48 + ff, 56 + Math.min(Math.max(ff + d - 1, 0), 7)];
}

// ---- network ---------------------------------------------------------------------------------------------------

function inputDrive(m, feats) {
  const bt = m.biasTotal; bt.set(m.bias);
  const enc = new Float32Array(m.nIn); enc.set(m.encoderB);
  for (const [f, x] of feats) {
    const row = f * m.nIn, w = m.encoderWT;
    for (let j = 0; j < m.nIn; j++) enc[j] += x * halfAt(w, row + j);
  }
  for (let j = 0; j < m.nIn; j++) bt[m.inIdx[j]] += enc[j] * m.inputGain;
  return bt;
}

/** The first update from h = 0 needs no matrix pass: W h is zero. */
export function firstStep(alpha, hMax, biasTotal, hOut) {
  for (let i = 0; i < hOut.length; i++) {
    let s = biasTotal[i];
    if (s < 0) s = 0; else if (s > hMax) s = hMax;
    hOut[i] = alpha[i] * s;
  }
}

/** One update of rows i0 .. i0+count-1.  vals/col hold those rows' synapses only, crow is local (count+1 entries). */
export function stepRows(vals, col, crow, alpha, hMax, biasTotal, hIn, hOut, i0, count) {
  for (let r = 0; r < count; r++) {
    const i = i0 + r;
    let s = biasTotal[i];
    const end = crow[r + 1];
    for (let k = crow[r]; k < end; k++) s += vals[k] * hIn[col[k]];
    if (s < 0) s = 0; else if (s > hMax) s = hMax;
    hOut[r] = hIn[i] + alpha[i] * (s - hIn[i]);
  }
}

function recurCpu(m) {
  const { n, values, crow, col, alpha, hMax, biasTotal } = m;
  let h = m.hA, h2 = m.hB;
  firstStep(alpha, hMax, biasTotal, h);
  for (let t = 1; t < m.steps; t++) {
    stepRows(values, col, crow, alpha, hMax, biasTotal, h, h2, 0, n);
    const tmp = h; h = h2; h2 = tmp;
  }
  m.lastH = h;
  const out = new Float32Array(m.nOut);
  for (let j = 0; j < m.nOut; j++) out[j] = h[m.outIdx[j]];
  return out;
}

/** Split the rows into `k` slices of roughly equal synapse count: [[i0, count], ...]. */
export function rowSlices(crow, k) {
  const n = crow.length - 1, total = crow[n], out = [];
  let i0 = 0;
  for (let s = 1; s <= k; s++) {
    const target = total * s / k;
    let i1 = s === k ? n : i0;
    while (i1 < n && crow[i1] < target) i1++;
    out.push([i0, i1 - i0]); i0 = i1;
  }
  return out;
}

/** The recurrence with the matrix passes done by `pool` (worker.js): returns the output neurons' activity. */
export async function recurPool(m, pool) {
  const { alpha, hMax, biasTotal } = m;
  let h = m.hA, h2 = m.hB;
  firstStep(alpha, hMax, biasTotal, h);
  await pool.begin(biasTotal);
  for (let t = 1; t < m.steps; t++) {
    await pool.step(h, h2);
    const tmp = h; h = h2; h2 = tmp;
  }
  m.lastH = h;
  const out = new Float32Array(m.nOut);
  for (let j = 0; j < m.nOut; j++) out[j] = h[m.outIdx[j]];
  return out;
}

function readout(m, outAct, idx) {
  const r = new Float32Array(m.nOut);
  for (let j = 0; j < m.nOut; j++) {
    const v = (outAct[j] - m.normMean[j]) / m.normStd[j];
    r[j] = v < -m.clip ? -m.clip : v > m.clip ? m.clip : v;
  }
  const dotF32 = (w, off) => { let s = 0; for (let j = 0; j < m.nOut; j++) s += w[off + j] * r[j]; return s; };
  const fromCache = new Map(), toCache = new Map();
  const logits = new Float32Array(idx.length);
  for (let a = 0; a < idx.length; a++) {
    const i = idx[a], [f, t] = moveSquares(i);
    let s = m.policyB[i];
    const off = i * m.nOut;
    for (let j = 0; j < m.nOut; j++) s += halfAt(m.policyW, off + j) * r[j];
    if (!fromCache.has(f)) fromCache.set(f, m.fromB[f] + dotF32(m.fromW, f * m.nOut));
    if (!toCache.has(t)) toCache.set(t, m.toB[t] + dotF32(m.toW, t * m.nOut));
    logits[a] = s + fromCache.get(f) + toCache.get(t);
  }
  const z = [0, 1, 2].map(k => m.valueB[k] + dotF32(m.valueW, k * m.nOut));
  const mx = Math.max(...z), e = z.map(v => Math.exp(v - mx)), Z = e[0] + e[1] + e[2];
  return { logits, value: (e[0] - e[2]) / Z, wdl: [e[0] / Z, e[1] / Z, e[2] / Z] };
}

/** Evaluate one position on the CPU: legal-move logits (in `idx` order) and the value for the side to move. */
export function evaluateCpu(m, feats, idx) {
  inputDrive(m, feats);
  return readout(m, recurCpu(m), idx);
}

// ---- WebGPU ------------------------------------------------------------------------------------------------------

const WGSL = `
struct Params { n: u32, hmax: f32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> vals: array<f32>;
@group(0) @binding(2) var<storage, read> col: array<u32>;
@group(0) @binding(3) var<storage, read> crow: array<u32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> alpha: array<f32>;
@group(0) @binding(6) var<storage, read> hin: array<f32>;
@group(0) @binding(7) var<storage, read_write> hout: array<f32>;
@compute @workgroup_size(128)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  var s = bias[i];
  let end = crow[i + 1u];
  for (var k = crow[i]; k < end; k = k + 1u) { s = s + vals[k] * hin[col[k]]; }
  s = clamp(s, 0.0, p.hmax);
  hout[i] = hin[i] + alpha[i] * (s - hin[i]);
}
`;
const WGSL_GATHER = `
struct GatherParams { n: u32 }
@group(0) @binding(0) var<uniform> gp: GatherParams;
@group(0) @binding(1) var<storage, read> outIdx: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(128)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= gp.n) { return; }
  dst[j] = src[outIdx[j]];
}
`;

export async function initGpu(m, extraIdx) {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  const gatherIdx = extraIdx ? Uint32Array.from([...m.outIdx, ...extraIdx]) : m.outIdx;
  const nGather = gatherIdx.length;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return null;
  const need = Math.max(m.values.byteLength, m.col.byteLength);
  if (adapter.limits.maxStorageBufferBindingSize < need || adapter.limits.maxBufferSize < need) return null;
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: need, maxBufferSize: need } });
  const S = GPUBufferUsage.STORAGE, C = GPUBufferUsage.COPY_DST, U = GPUBufferUsage.UNIFORM;
  const mk = (data, usage) => {
    const b = device.createBuffer({ size: Math.ceil(data.byteLength / 4) * 4, usage });
    device.queue.writeBuffer(b, 0, data);
    return b;
  };
  const params = new ArrayBuffer(8);
  new Uint32Array(params, 0, 1)[0] = m.n; new Float32Array(params, 4, 1)[0] = m.hMax;
  const bufs = {
    params: mk(new Uint8Array(params), U | C), vals: mk(m.values, S), col: mk(m.col, S), crow: mk(m.crow, S),
    bias: device.createBuffer({ size: m.n * 4, usage: S | C }), alpha: mk(m.alpha, S),
    hA: device.createBuffer({ size: m.n * 4, usage: S | C }), hB: device.createBuffer({ size: m.n * 4, usage: S | C }),
    gparams: mk(new Uint32Array([nGather]), U), outIdx: mk(gatherIdx, S),
    out: device.createBuffer({ size: nGather * 4, usage: S | GPUBufferUsage.COPY_SRC }),
    stage: device.createBuffer({ size: nGather * 4, usage: GPUBufferUsage.MAP_READ | C }),
  };
  const stepPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: WGSL }), entryPoint: "step" } });
  const gatherPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: WGSL_GATHER }), entryPoint: "gather" } });
  const bind = (hin, hout) => device.createBindGroup({ layout: stepPipe.getBindGroupLayout(0), entries: [
    bufs.params, bufs.vals, bufs.col, bufs.crow, bufs.bias, bufs.alpha, hin, hout].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const bgAB = bind(bufs.hA, bufs.hB), bgBA = bind(bufs.hB, bufs.hA);
  const gbg = (src) => device.createBindGroup({ layout: gatherPipe.getBindGroupLayout(0), entries: [
    bufs.gparams, bufs.outIdx, src, bufs.out].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const gA = gbg(bufs.hA), gB = gbg(bufs.hB);
  const zeros = new Float32Array(m.n);
  const groups = Math.ceil(m.n / 128), ggroups = Math.ceil(nGather / 128);
  let info = "WebGPU";
  try { const ai = adapter.info || (adapter.requestAdapterInfo && await adapter.requestAdapterInfo()); if (ai) info = ai.description || ai.device || ai.vendor || info; } catch (e) { /* not exposed */ }
  return {
    name: info,
    async recur(biasTotal) {
      device.queue.writeBuffer(bufs.bias, 0, biasTotal);
      device.queue.writeBuffer(bufs.hA, 0, zeros);
      const enc = device.createCommandEncoder();
      let cur = "A";
      for (let t = 0; t < m.steps; t++) {
        const pass = enc.beginComputePass();
        pass.setPipeline(stepPipe); pass.setBindGroup(0, cur === "A" ? bgAB : bgBA); pass.dispatchWorkgroups(groups); pass.end();
        cur = cur === "A" ? "B" : "A";
      }
      const pass = enc.beginComputePass();
      pass.setPipeline(gatherPipe); pass.setBindGroup(0, cur === "A" ? gA : gB); pass.dispatchWorkgroups(ggroups); pass.end();
      enc.copyBufferToBuffer(bufs.out, 0, bufs.stage, 0, nGather * 4);
      device.queue.submit([enc.finish()]);
      await bufs.stage.mapAsync(GPUMapMode.READ);
      const all = new Float32Array(bufs.stage.getMappedRange().slice(0));
      bufs.stage.unmap();
      m.lastExtra = all.subarray(m.nOut);                    // the extra neurons, for the activity display
      return all.subarray(0, m.nOut);
    },
  };
}

export async function evaluateGpu(m, gpu, feats, idx) {
  const bt = inputDrive(m, feats);
  return readout(m, await gpu.recur(bt), idx);
}

export async function evaluatePool(m, pool, feats, idx) {
  inputDrive(m, feats);
  return readout(m, await recurPool(m, pool), idx);
}

// ---- Gumbel AlphaZero search (chessfly/mcts.py) ----------------------------------------------------------------

export function consideredVisits(maxConsidered, sims) {
  if (maxConsidered <= 1) return Array.from({ length: sims }, (_, i) => i);
  const log2max = Math.ceil(Math.log2(maxConsidered));
  const seq = [], visits = new Array(maxConsidered).fill(0);
  let considered = maxConsidered;
  while (seq.length < sims) {
    const reps = Math.max(1, Math.floor(sims / (log2max * considered)));
    for (let r = 0; r < reps; r++) {
      for (let i = 0; i < considered; i++) seq.push(visits[i]);
      for (let i = 0; i < considered; i++) visits[i]++;
    }
    considered = Math.max(2, considered >> 1);
  }
  return seq.slice(0, sims);
}

class Node {
  constructor() { this.moves = null; this.terminal = null; this.v = 0; }
  expand(moves, idx, logits, v) {
    const mx = Math.max(...logits);
    this.moves = moves; this.idx = idx; this.v = v;
    this.logits = Float32Array.from(logits, z => z - mx);
    const e = Array.from(this.logits, Math.exp), Z = e.reduce((a, b) => a + b, 0);
    this.P = Float32Array.from(e, x => x / Z);
    this.N = new Float32Array(moves.length); this.W = new Float32Array(moves.length);
    this.children = new Array(moves.length).fill(null);
  }
  completedQ() {
    const n = this.N.length, out = new Float32Array(n);
    let total = 0, pq = 0, ps = 0, any = false;
    for (let a = 0; a < n; a++) if (this.N[a] > 0) { any = true; total += this.N[a]; pq += this.P[a] * this.W[a] / this.N[a]; ps += this.P[a]; }
    const vmix = any ? (this.v + total * (pq / ps)) / (1 + total) : this.v;
    for (let a = 0; a < n; a++) out[a] = this.N[a] > 0 ? this.W[a] / this.N[a] : vmix;
    return out;
  }
  sigmaQ(cfg) {
    let mx = 0; for (const v of this.N) if (v > mx) mx = v;
    return this.completedQ().map(q => (cfg.cVisit + mx) * cfg.cScale * (q + 1) / 2);
  }
  improvedPolicy(cfg) {
    const s = this.sigmaQ(cfg), z = this.logits.map((l, a) => l + s[a]);
    const mx = Math.max(...z), e = z.map(v => Math.exp(v - mx)), Z = e.reduce((a, b) => a + b, 0);
    return e.map(x => x / Z);
  }
  selectGumbel(cfg) {
    const pi = this.improvedPolicy(cfg), tot = this.N.reduce((a, b) => a + b, 0);
    let best = 0, bv = -Infinity;
    for (let a = 0; a < pi.length; a++) { const v = pi[a] - this.N[a] / (1 + tot); if (v > bv) { bv = v; best = a; } }
    return best;
  }
}

function gumbel(n) { const g = new Float32Array(n); for (let i = 0; i < n; i++) g[i] = -Math.log(-Math.log(Math.random() || 1e-12)); return g; }

/**
 * game: a chess.js instance at the root position (moves are pushed and undone on it during the search).
 * evaluate(feats, idx) -> {logits, value}.  cfg: {sims, maxConsidered, cVisit, cScale, explore}.
 * Returns the chosen move, the root's value, and the moves considered with their visits and values.
 */
export async function search(game, evaluate, cfg) {
  const root = new Node();
  const keys = [];                                  // position keys of the line pushed during the search, for repetitions
  const history = new Map();                        // game history position counts, for the repetition feature
  const posKey = g => g.fen().split(" ").slice(0, 4).join(" ");
  {
    const undone = [];
    let mv;
    while ((mv = game.undo())) undone.push(mv);
    history.set(posKey(game), 1);
    while (undone.length) { game.move(undone.pop()); const k = posKey(game); history.set(k, (history.get(k) || 0) + 1); }
  }
  const seen = k => (history.get(k) || 0) + keys.filter(x => x === k).length;
  const terminalValue = () => {
    if (game.in_checkmate()) return -1;
    if (game.in_stalemate() || game.insufficient_material()) return 0;
    if (+game.fen().split(" ")[4] >= 100) return 0;
    if (seen(posKey(game)) >= 3) return 0;
    return null;
  };
  const expand = async node => {
    const moves = game.moves({ verbose: true }), white = game.turn() === "w";
    const idx = moves.map(mv => moveIndex(sq(mv.from), sq(mv.to), mv.promotion, white));
    const feats = encodeFen(game.fen(), seen(posKey(game)) >= 2);
    const { logits, value } = await evaluate(feats, idx);
    node.expand(moves, idx, logits, value);
    return value;
  };
  let g = null, rootVisits = null, seq = null, tIdx = 0;
  const rootAction = () => {
    if (seq === null) {
      const n = root.moves.length;
      g = cfg.explore ? gumbel(n) : new Float32Array(n);
      rootVisits = new Float32Array(n);
      seq = consideredVisits(Math.min(cfg.maxConsidered, n), cfg.sims + 1); tIdx = 0;
    }
    const want = seq[Math.min(tIdx, seq.length - 1)]; tIdx++;
    const s = root.sigmaQ(cfg);
    let minV = Infinity; for (const v of rootVisits) if (v < minV) minV = v;
    let eligibleWant = false; for (const v of rootVisits) if (v === want) { eligibleWant = true; break; }
    let best = -1, bv = -Infinity;
    for (let a = 0; a < root.moves.length; a++) {
      if (rootVisits[a] !== (eligibleWant ? want : minV)) continue;
      const v = g[a] + root.logits[a] + s[a];
      if (v > bv) { bv = v; best = a; }
    }
    rootVisits[best]++;
    return best;
  };
  const backup = (path, v) => { for (let i = path.length - 1; i >= 0; i--) { v = -v; const [node, a] = path[i]; node.N[a]++; node.W[a] += v; } };

  root.terminal = terminalValue();
  if (root.terminal !== null) return null;
  await expand(root);
  for (let sim = 0; sim < cfg.sims; sim++) {
    let node = root; const path = [];
    while (node.terminal === null && node.moves !== null) {
      const a = node === root ? rootAction() : node.selectGumbel(cfg);
      path.push([node, a]);
      game.move(node.moves[a]); keys.push(posKey(game));
      if (node.children[a] === null) node.children[a] = new Node();
      node = node.children[a];
    }
    let v;
    if (node.terminal === null && node.moves === null) node.terminal = terminalValue();
    if (node.terminal !== null) v = node.terminal;
    else v = await expand(node);
    backup(path, v);
    for (let i = 0; i < path.length; i++) { game.undo(); keys.pop(); }
  }
  let best = 0, bv = -Infinity, maxV = 0;
  for (const v of rootVisits) if (v > maxV) maxV = v;
  const s = root.sigmaQ(cfg);
  for (let a = 0; a < root.moves.length; a++) {
    if (rootVisits[a] !== maxV) continue;
    const v = g[a] + root.logits[a] + s[a];
    if (v > bv) { bv = v; best = a; }
  }
  const total = root.N.reduce((a, b) => a + b, 0);
  const considered = Array.from(root.moves.keys()).filter(a => root.N[a] > 0)
    .sort((a, b) => root.N[b] - root.N[a])
    .map(a => ({ san: root.moves[a].san, uci: root.moves[a].from + root.moves[a].to + (root.moves[a].promotion || ""),
                 visits: root.N[a], q: root.W[a] / root.N[a], prior: root.P[a] }));
  return { move: root.moves[best], value: total ? root.W.reduce((a, b) => a + b, 0) / total : root.v, raw: root.v, sims: total, considered };
}

/** The raw policy alone: the move with the highest logit (or one sampled from the policy), no search. */
export async function rawMove(game, evaluate, sample) {
  const moves = game.moves({ verbose: true }), white = game.turn() === "w";
  const idx = moves.map(mv => moveIndex(sq(mv.from), sq(mv.to), mv.promotion, white));
  const { logits, value } = await evaluate(encodeFen(game.fen(), false), idx);
  const mx = Math.max(...logits), e = Array.from(logits, z => Math.exp(z - mx)), Z = e.reduce((a, b) => a + b, 0);
  let best = 0;
  if (sample) { let r = Math.random() * Z; for (best = 0; best < e.length - 1 && (r -= e[best]) > 0; best++); }
  else for (let a = 1; a < e.length; a++) if (e[a] > e[best]) best = a;
  const considered = Array.from(moves.keys()).sort((a, b) => e[b] - e[a]).slice(0, 8)
    .map(a => ({ san: moves[a].san, uci: moves[a].from + moves[a].to + (moves[a].promotion || ""), visits: 0, q: value, prior: e[a] / Z }));
  return { move: moves[best], value, raw: value, sims: 0, considered };
}
