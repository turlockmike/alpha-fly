// node docs/test_fly.mjs  - the JavaScript port against the reference logits exported by scripts/export_web.py
import { readFile } from "node:fs/promises";
import { loadModel, encodeFen, evaluateCpu, consideredVisits } from "./fly.js";

const fetchBytes = async p => (await readFile(p)).buffer;
const m = await loadModel("docs/model/", fetchBytes);
let worst = 0, worstV = 0;
for (const t of m.tests) {
  const feats = encodeFen(t.fen, false);
  const got = feats.map(([f]) => f).sort((a, b) => a - b).join(",");
  const want = [...t.features].sort((a, b) => a - b).join(",");
  if (got !== want) { console.log("FEATURE MISMATCH", t.fen, "\n got ", got, "\n want", want); process.exit(1); }
  const t0 = Date.now();
  const r = evaluateCpu(m, feats, t.idx);
  const dt = Date.now() - t0;
  let d = 0; for (let a = 0; a < t.idx.length; a++) d = Math.max(d, Math.abs(r.logits[a] - t.logits[a]));
  worst = Math.max(worst, d); worstV = Math.max(worstV, Math.abs(r.value - t.value));
  let best = 0; for (let a = 1; a < r.logits.length; a++) if (r.logits[a] > r.logits[best]) best = a;
  let ref = 0; for (let a = 1; a < t.logits.length; a++) if (t.logits[a] > t.logits[ref]) ref = a;
  console.log(`${t.fen.padEnd(72)} max|dlogit| ${d.toFixed(4)}  value ${r.value.toFixed(4)} vs ${t.value}  best ${t.moves[best]} (ref ${t.moves[ref]})  ${dt} ms`);
}
console.log("worst logit diff", worst.toFixed(4), "worst value diff", worstV.toFixed(5));
console.log("schedule 16/33:", consideredVisits(16, 33).join(""));
process.exit(worst < 0.05 && worstV < 0.01 ? 0 : 1);
