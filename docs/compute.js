// One slice of the fly's neurons: this worker owns rows i0 .. i0+count-1 and their synapses (see worker.js makePool).
// A module worker, so `stepRows` is loaded before the first message is handled.
import { stepRows } from "./fly.js?v=3";

let S = null, biasTotal = null, out = null;
self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === "init") { S = m; out = new Float32Array(m.count); }
    else if (m.type === "begin") { biasTotal = m.biasTotal; self.postMessage(null); }
    else if (m.type === "step") {
      stepRows(S.vals, S.col, S.crow, S.alpha, S.hMax, biasTotal, m.h, out, S.i0, S.count);
      self.postMessage(out);
    }
  } catch (err) {
    self.postMessage({ error: String(err && err.stack || err) });
  }
};
