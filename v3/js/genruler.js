/* General ruler tick detection — port of ruler_ticks.py to OpenCV.js.

   Handles any ruler with repeating marks (metric, imperial, plastic rule,
   dermatology card) with no assumption about colour, tick count, position or
   scale. Kept entirely separate from ruler.js, the coloured-swatch card
   detector, which is not touched by this file.

   Pipeline (same order as the Python):
     1. ink channel + map : illumination-flattened dark-mark response, on
                            whichever of {luma,B,G,R} shows the marks best
     2. stroke map        : opening with a vertical kernel, drops skin texture
     3. skew              : angle chosen by the quality of the FINAL tick fit,
                            not by a proxy like contrast
     4. band              : comb-energy rows, hysteresis thresholded
     5. peaks             : prominence picking, sub-pixel centroid; pitch from
                            autocorrelation PAST the central lobe
     6. levels            : long ticks found by their regular sub-lattice
     7. calibration       : local-pitch lattice indexing + projective fit

   Two things differ from the standalone tool, both because this runs inside a
   measurement app rather than as a diagnostic:

   - It is told where the lesion is, and the lesion is painted out of the frame
     before any band is looked for, with a REFUSAL still in place if a band
     lands on it anyway. The Python has no such context and on a small tilted
     photo (rl3) it locked onto hairs on the mole and reported a confident
     101 px/cm -- a scale error that squares into the area.
     (The "true ~60 px/cm" this comment used to quote for rl3 was wrong. The
     gap between its printed 1 and 2 is ~81 px, and the mm lattice reads 83.)
   - Anything short of a solid fit returns null with a reason, so the caller
     falls back to tapping two marks by hand. A wrong scale is worse than no
     scale here.

   Inclined rulers ARE in scope here, unlike the Python: the skew search spans
   +/-45 degrees in each of the two orientations, so the two sweeps meet and no
   angle is unreachable. See GEN_SKEW_SPAN.
*/

const GEN_MIN_TICKS = 10;      // fewer than this and the fit is not trusted
const GEN_WORK_MAX  = 1600;    // longest side analysed
const GEN_WORK_MIN  = 700;     // below this, upsample: strokes get too short
const GEN_SKEW_SPAN = 45.0;    // degrees either side of level. The Python uses
                               // 15 and calls inclined rulers out of scope; that
                               // leaves a dead zone, because the two orientations
                               // only cover 15 degrees around level and around
                               // upright. rl3 sits at ~41 degrees, squarely in
                               // the gap, and was refused outright. At 45 the h
                               // and v sweeps meet, so every angle is reachable.
                               // Widening is safe only because a bad angle still
                               // has to clear GEN_MIN_TICKS, the lattice fit and
                               // the on-lesion guard; the angle is scored by the
                               // quality of the tick fit, not by contrast.

// ---- small numeric helpers ------------------------------------------------
const _med = a => { if (!a.length) return 0;
  const s = Float64Array.from(a).sort(); const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const _mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
const _std = a => { if (a.length < 2) return 0; const m = _mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length); };
function _percentile(arr, p) {
  const s = Float64Array.from(arr).sort();
  const i = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
  return s[i];
}
// 1-D box-ish blur of a profile, standing in for GaussianBlur on a 1xN row
function _smooth1d(a, k) {
  const n = a.length, r = Math.max(1, k >> 1), out = new Float64Array(n);
  let sum = 0, cnt = 0;
  for (let i = 0; i < n; i++) {
    if (i === 0) { for (let j = 0; j <= r && j < n; j++) { sum += a[j]; cnt++; } }
    else {
      const add = i + r, rem = i - r - 1;
      if (add < n) { sum += a[add]; cnt++; }
      if (rem >= 0) { sum -= a[rem]; cnt--; }
    }
    out[i] = sum / Math.max(1, cnt);
  }
  return out;
}
// autocorrelation, lags 0..n-1, normalised to ac[0]=1
function _autocorr(p) {
  const n = p.length, m = _mean(Array.from(p));
  const q = new Float64Array(n);
  for (let i = 0; i < n; i++) q[i] = p[i] - m;
  const ac = new Float64Array(n);
  for (let lag = 0; lag < n; lag++) {
    let s = 0;
    for (let i = 0; i + lag < n; i++) s += q[i] * q[i + lag];
    ac[lag] = s;
  }
  const d = ac[0] || 1e-9;
  for (let i = 0; i < n; i++) ac[i] /= d;
  return ac;
}
// least squares for A x = y, A is m x k (k <= 3), via normal equations
function _lstsq(A, y, k) {
  const N = [], r = [];
  for (let i = 0; i < k; i++) { N.push(new Float64Array(k)); r.push(0); }
  for (let row = 0; row < y.length; row++) {
    for (let i = 0; i < k; i++) {
      r[i] += A[row][i] * y[row];
      for (let j = 0; j < k; j++) N[i][j] += A[row][i] * A[row][j];
    }
  }
  // gaussian elimination with partial pivoting
  for (let c = 0; c < k; c++) {
    let piv = c;
    for (let i = c + 1; i < k; i++) if (Math.abs(N[i][c]) > Math.abs(N[piv][c])) piv = i;
    if (Math.abs(N[piv][c]) < 1e-12) return null;
    if (piv !== c) { const t = N[piv]; N[piv] = N[c]; N[c] = t;
                     const tr = r[piv]; r[piv] = r[c]; r[c] = tr; }
    for (let i = c + 1; i < k; i++) {
      const f = N[i][c] / N[c][c];
      if (!f) continue;
      for (let j = c; j < k; j++) N[i][j] -= f * N[c][j];
      r[i] -= f * r[c];
    }
  }
  const x = new Float64Array(k);
  for (let i = k - 1; i >= 0; i--) {
    let s = r[i];
    for (let j = i + 1; j < k; j++) s -= N[i][j] * x[j];
    x[i] = s / N[i][i];
  }
  return x;
}

// ---- 1. ink map -----------------------------------------------------------
// float32 Mat in [0,1]: how locally dark each pixel is, flattened for lighting
function inkMap(grayMat) {
  const g = new cv.Mat(), bg = new cv.Mat(), gf = new cv.Mat();
  cv.GaussianBlur(grayMat, g, new cv.Size(0, 0), 1.0);
  g.convertTo(gf, cv.CV_32F);
  cv.GaussianBlur(gf, bg, new cv.Size(0, 0), 25);
  const n = gf.rows * gf.cols;
  const G = gf.data32F, B = bg.data32F;
  const ink = new cv.Mat(gf.rows, gf.cols, cv.CV_32F);
  const O = ink.data32F;
  for (let i = 0; i < n; i++) {
    let flat = G[i] / Math.max(B[i], 1e-3);
    if (flat > 2) flat = 2; else if (flat < 0) flat = 0;
    let v = 1 - flat;
    O[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
  }
  // stretch between the median and the 99.5th percentile, on a sample for speed
  const step = Math.max(1, Math.floor(n / 20000));
  const samp = [];
  for (let i = 0; i < n; i += step) samp.push(O[i]);
  const lo = _percentile(samp, 50);
  let hi = _percentile(samp, 99.5);
  if (hi - lo < 1e-6) hi = lo + 1e-6;
  const inv = 1 / (hi - lo);
  for (let i = 0; i < n; i++) {
    let v = (O[i] - lo) * inv;
    O[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
  }
  g.delete(); bg.delete(); gf.delete();
  return ink;
}

const _minStroke = h => Math.max(3, Math.min(40, Math.floor(h / 25)));

// keep only ink belonging to vertical strokes
function verticalStrokeMap(ink, minLen) {
  const k = cv.getStructuringElement(cv.MORPH_RECT,
                                     new cv.Size(1, Math.max(3, minLen)));
  const out = new cv.Mat();
  cv.morphologyEx(ink, out, cv.MORPH_OPEN, k);
  k.delete();
  return out;
}

// column sum of a Mat region -> profile
function _colProfile(mat, y0, y1) {
  const w = mat.cols, D = mat.data32F, stride = mat.cols;
  const prof = new Float64Array(w);
  for (let y = y0; y < y1; y++) {
    const row = y * stride;
    for (let x = 0; x < w; x++) prof[x] += D[row + x];
  }
  return prof;
}

// how comb-like (periodic) a region's column profile is
function _combScore(mat, y0, y1) {
  const prof = _colProfile(mat, y0, y1);
  if (prof.length < 12) return 0;
  const base = _smooth1d(prof, 61);
  const p = new Float64Array(prof.length);
  for (let i = 0; i < prof.length; i++) p[i] = prof[i] - base[i];
  const ac = _autocorr(p);
  const hi = Math.min(ac.length, Math.max(5, Math.floor(prof.length / 4)));
  let best = 0;
  for (let i = 3; i < hi; i++) if (ac[i] > best) best = ac[i];
  return best;
}
function _sharpness(mat, y0, y1) {
  const prof = _colProfile(mat, y0, y1);
  if (!prof.length) return 0;
  const base = _smooth1d(prof, 61);
  const hp = [];
  for (let i = 0; i < prof.length; i++) hp.push(prof[i] - base[i]);
  const absMean = _mean(hp.map(Math.abs));
  return _std(hp) / (absMean + 1e-9);
}

// ---- 3. ruler band --------------------------------------------------------
// rows carrying the periodic tick comb, seeded high and grown low
function findBand(vmap) {
  const h = vmap.rows, w = vmap.cols, D = vmap.data32F;
  const energy = new Float64Array(h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const line = new Float64Array(w);
    for (let x = 0; x < w; x++) line[x] = D[row + x];
    const base = _smooth1d(line, 31);
    let s = 0;
    for (let x = 0; x < w; x++) s += Math.abs(line[x] - base[x]);
    energy[y] = s / w;
  }
  const sm = _smooth1d(energy, 5);
  const med = _med(Array.from(sm));
  let peak = 0;
  for (let i = 0; i < h; i++) if (sm[i] > peak) peak = sm[i];
  if (peak - med < 1e-9) return [0, h];
  const norm = new Float64Array(h);
  for (let i = 0; i < h; i++) norm[i] = (sm[i] - med) / (peak - med);

  let bs = 0, be = 0, y = 0;
  while (y < h) {
    if (norm[y] > 0.55) {
      const a = y;
      while (y < h && norm[y] > 0.55) y++;
      if (y - a > be - bs) { bs = a; be = y; }
    }
    y++;
  }
  if (be === bs) return [0, h];
  while (bs > 0 && norm[bs - 1] > 0.18) bs--;
  while (be < h && norm[be] > 0.18) be++;
  if (be - bs < Math.max(4, Math.floor(h * 0.04))) return [0, h];
  const pad = Math.max(1, Math.floor(0.08 * (be - bs)));
  return [Math.max(0, bs - pad), Math.min(h, be + pad)];
}

// ---- 4. peaks -------------------------------------------------------------
function _prominencePeaks(prof, minDist, rel) {
  rel = rel === undefined ? 0.15 : rel;
  const n = prof.length;
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < n; i++) { if (prof[i] < mn) mn = prof[i]; if (prof[i] > mx) mx = prof[i]; }
  const floor = mn + rel * (mx - mn);
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => prof[b] - prof[a]);
  const taken = new Uint8Array(n), peaks = [];
  for (const i of order) {
    if (prof[i] < floor) break;
    const lo = Math.max(0, i - minDist), hi = Math.min(n, i + minDist + 1);
    let clash = false;
    for (let j = lo; j < hi; j++) if (taken[j]) { clash = true; break; }
    if (clash) continue;
    taken[i] = 1; peaks.push(i);
  }
  return peaks.sort((a, b) => a - b);
}
function _subpixel(prof, i, r) {
  r = r || 2;
  const lo = Math.max(0, i - r), hi = Math.min(prof.length, i + r + 1);
  let mn = Infinity;
  for (let j = lo; j < hi; j++) if (prof[j] < mn) mn = prof[j];
  let ws = 0, xs = 0;
  for (let j = lo; j < hi; j++) { const w = prof[j] - mn; ws += w; xs += w * j; }
  return ws > 0 ? xs / ws : i;
}
// Dominant pitch, taken past the autocorrelation's central lobe. A naive argmax
// locks onto lag ~2 on a blurry photo and splits every tick into duplicates.
function estimatePitch(prof) {
  const ac = _autocorr(prof);
  let i = 1;
  while (i < ac.length - 1 && ac[i] <= ac[i - 1]) i++;
  const lo = Math.max(2, i);
  const hi = Math.min(ac.length - 1, Math.max(lo + 2, Math.floor(prof.length / 2)));
  if (hi <= lo) return 3;
  let best = lo, bv = -Infinity;
  for (let j = lo; j < hi; j++) if (ac[j] > bv) { bv = ac[j]; best = j; }
  return best;
}

function detectTicks(vmap, band) {
  const [y0, y1] = band;
  if (y1 <= y0) return [];
  const prof = _colProfile(vmap, y0, y1);
  const base = _smooth1d(prof, 61);
  const hp = new Float64Array(prof.length);
  for (let i = 0; i < prof.length; i++) hp[i] = Math.max(0, prof[i] - base[i]);

  const pitch = estimatePitch(hp);
  const minDist = Math.max(1, Math.round(pitch * 0.55));
  const idx = _prominencePeaks(hp, minDist);

  // Otsu on the band so stroke lengths do not depend on exposure, but measured
  // over a taller window: the band is sized by the many SHORT ticks, so long
  // ticks run past its edge and would all clip to the same length, hiding the
  // cm/half-cm distinction.
  const sub = vmap.roi(new cv.Rect(0, y0, vmap.cols, y1 - y0));
  const u8 = new cv.Mat();
  sub.convertTo(u8, cv.CV_8U, 255, 0);
  const tmp = new cv.Mat();
  const thr = cv.threshold(u8, tmp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  sub.delete(); u8.delete(); tmp.delete();

  const ym = Math.min(vmap.rows, y0 + 3 * (y1 - y0));
  const D = vmap.data32F, W = vmap.cols;
  const bh = y1 - y0;
  const ticks = [];
  for (const i of idx) {
    let best = 0, run = 0;
    for (let y = y0; y < ym; y++) {
      let on = false;
      for (let x = Math.max(0, i - 1); x <= Math.min(W - 1, i + 1); x++) {
        if (D[y * W + x] * 255 > thr) { on = true; break; }
      }
      run = on ? run + 1 : 0;
      if (run > best) best = run;
    }
    if (best < Math.max(3, 0.15 * bh)) continue;
    ticks.push({ pos: _subpixel(hp, i), length: best, strength: hp[i], level: 0 });
  }
  ticks.sort((a, b) => a.pos - b.pos);
  return ticks;
}

// ---- 5. levels ------------------------------------------------------------
// Step tick to tick with a LOCAL pitch estimate. round((p-p0)/median) assumes a
// constant pitch, and under perspective drift the cumulative error eventually
// collapses two ticks onto one index and shifts everything after them.
function latticeIndex(pos) {
  if (pos.length < 2) return null;
  const d = [];
  for (let i = 1; i < pos.length; i++) {
    const g = pos[i] - pos[i - 1];
    if (g <= 0) return null;
    d.push(g);
  }
  const n = new Float64Array(pos.length);
  for (let i = 1; i < pos.length; i++) {
    const lo = Math.max(0, i - 4), hi = Math.min(d.length, i + 3);
    const local = _med(d.slice(lo, hi));
    if (local <= 0) return null;
    n[i] = n[i - 1] + Math.max(1, Math.round(d[i - 1] / local));
  }
  return n;
}
function _periodSearch(idx, length, periods) {
  let best = { P: null, ph: null, score: 0 };
  if (idx.length < 6) return best;
  const spread = _std(Array.from(length)) + 1e-6;
  const meanL = _mean(Array.from(length));
  for (const P of periods) {
    for (let ph = 0; ph < P; ph++) {
      const inL = [], outL = [];
      for (let i = 0; i < idx.length; i++) (idx[i] % P === ph ? inL : outL).push(length[i]);
      if (inL.length < 2 || outL.length < 2) continue;
      const sep = _mean(inL) - _mean(outL);
      if (sep <= 0.12 * meanL) continue;
      const score = sep / spread * Math.min(1, inL.length / 3);
      if (score > best.score) best = { P, ph, score };
    }
  }
  return best;
}
function classifyTicks(ticks) {
  if (ticks.length < 3) { ticks.forEach(t => t.level = 0); return [null, null]; }
  const pos = ticks.map(t => t.pos), length = ticks.map(t => t.length);
  const li = latticeIndex(pos);
  if (!li) { ticks.forEach(t => t.level = 0); return [null, null]; }
  const idx = Array.from(li, v => Math.round(v));
  const mn = Math.min(...length), mx = Math.max(...length);
  if (mx - mn < 0.15 * _mean(length)) { ticks.forEach(t => t.level = 0); return [null, null]; }

  const periods = []; for (let p = 2; p < 25; p++) periods.push(p);
  let { P, ph } = _periodSearch(idx, length, periods);
  if (P === null) {
    const thr = _mean(length) + 0.5 * _std(length);
    ticks.forEach((t, i) => t.level = length[i] >= thr ? 2 : 0);
    return [null, null];
  }
  // A period of 20 always "works" if 10 does, so walk down to the smallest
  // divisor -- but only when the ticks it newly admits really are majors.
  // Comparing separation scores cannot tell 20->10 (correct) from 10->5
  // (wrong, the admitted ticks are shorter half-cm marks); asking where the
  // admitted ticks sit on the minor->major length scale can.
  const divisors = [];
  for (let x = 2; x < P; x++) if (P % x === 0) divisors.push(x);
  for (const dv of divisors) {
    const mD = idx.map(v => v % dv === ph % dv);
    const mP = idx.map(v => v % P === ph);
    const added = mD.map((v, i) => v && !mP[i]);
    const nAdd = added.filter(Boolean).length;
    const nOut = mD.filter(v => !v).length;
    const nP = mP.filter(Boolean).length;
    if (nAdd < 1 || nOut < 2 || nP < 1) continue;
    const Lmaj = _mean(length.filter((_, i) => mP[i]));
    const Ladd = _mean(length.filter((_, i) => added[i]));
    const Lmin = _mean(length.filter((_, i) => !mD[i]));
    if (Lmaj - Lmin <= 1e-6) continue;
    if ((Ladd - Lmin) / (Lmaj - Lmin) >= 0.75) { P = dv; ph = ph % dv; break; }
  }
  const major = idx.map(v => v % P === ph);
  ticks.forEach((t, i) => t.level = major[i] ? 2 : 0);

  // a second, intermediate level among the rest (half-centimetre marks)
  let P2 = null;
  const subs = [];
  for (let p = 2; p < P; p++) if (P % p === 0) subs.push(p);
  const restI = [];
  for (let i = 0; i < idx.length; i++) if (!major[i]) restI.push(i);
  if (subs.length && restI.length >= 6) {
    const r = _periodSearch(restI.map(i => idx[i]), restI.map(i => length[i]), subs);
    if (r.P !== null && r.score > 0.25) {
      restI.forEach(i => { if (idx[i] % r.P === r.ph) ticks[i].level = 1; });
      P2 = r.P;
    }
  }
  return [P, P2];
}

// ---- 6. calibration -------------------------------------------------------
function robustPitch(pos) {
  if (pos.length < 2) return [0, 0];
  const n = latticeIndex(pos);
  if (!n) return [0, 0];
  const A = [], y = [];
  for (let i = 0; i < pos.length; i++) { A.push([n[i], 1]); y.push(pos[i]); }
  const sol = _lstsq(A, y, 2);
  if (!sol) return [0, 0];
  const [pitch, off] = sol;
  let s = 0;
  for (let i = 0; i < pos.length; i++) { const r = pos[i] - (pitch * n[i] + off); s += r * r; }
  return [pitch, Math.sqrt(s / pos.length)];
}
// pos = (a*n + b) / (c*n + 1): a photographed ruler is a perspective view of an
// even lattice, so the pitch drifts smoothly and a straight line leaves an
// arched residual that is geometry, not detection error.
function fitProjective(pos) {
  if (pos.length < 4) return null;
  const n = latticeIndex(pos);
  if (!n) return null;
  const A = [], y = [];
  for (let i = 0; i < pos.length; i++) { A.push([n[i], 1, -n[i] * pos[i]]); y.push(pos[i]); }
  const sol = _lstsq(A, y, 3);
  if (!sol) return null;
  const [a, b, c] = sol;
  let s = 0;
  for (let i = 0; i < pos.length; i++) {
    const den = c * n[i] + 1;
    if (Math.abs(den) < 1e-9) return null;
    const r = pos[i] - (a * n[i] + b) / den;
    s += r * r;
  }
  return { a, b, c, rms: Math.sqrt(s / pos.length), n };
}
const localPitch = (a, b, c, n) => (a - b * c) / Math.pow(c * n + 1, 2);

// px per minor tick at a position ALONG THE RUN OF TICKS, in ORIGINAL image
// pixels. `pos` is in de-skewed working-frame units -- the same units as
// tick.pos, which is what the perspective fit was fitted against. It is
// deliberately not an image x: the tick axis is only image x when the frame is
// unrotated, unskewed and uncropped.
function pitchAtPos(res, pos) {
  if (!res || !res.perspective) return res ? res.pxPerTick : 0;
  const { a, b, c } = res.perspective, k = res.workScale;
  const den = pos * c - a;
  if (Math.abs(den) < 1e-9) return res.pxPerTick;
  return localPitch(a, b, c, (b - pos) / den) / k;
}

// ---- 7. skew search -------------------------------------------------------
function _rotateMat(src, angle) {
  const out = new cv.Mat();
  const M = cv.getRotationMatrix2D(new cv.Point(src.cols / 2, src.rows / 2), angle, 1);
  cv.warpAffine(src, out, M, new cv.Size(src.cols, src.rows),
                cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());
  M.delete();
  return out;
}
// Score an angle by the quality of the tick fit it produces. Proxy objectives
// (contrast, autocorrelation) get fooled by skin and fabric; the end result
// does not -- a real ruler yields many ticks on a lattice with sub-pixel error.
function _evaluate(ink, angle) {
  const rot = angle === 0 ? ink.clone() : _rotateMat(ink, angle);
  const vm = verticalStrokeMap(rot, _minStroke(ink.rows));
  rot.delete();
  const band = findBand(vm);
  const frac = (band[1] - band[0]) / Math.max(1, ink.rows);
  if (frac >= 0.6 || frac < 0.02) { vm.delete(); return 0; }
  const ticks = detectTicks(vm, band);
  vm.delete();
  if (ticks.length < 5) return 0;
  const pos = ticks.map(t => t.pos);
  let [pitch, rms] = robustPitch(pos);
  if (pitch <= 0) return 0;
  const pf = fitProjective(pos);
  if (pf) rms = pf.rms;
  const fit = Math.exp(-rms / (0.06 * pitch));
  const span = (Math.max(...pos) - Math.min(...pos)) / Math.max(1, ink.cols);
  return ticks.length * fit * Math.min(1, span / 0.3);
}
function _searchAngle(ink) {
  // The coarse sweep only has to land in the right few degrees, and it does
  // not need full resolution to do that -- halving each side makes every
  // evaluation about 4x cheaper, which is what pays for the +/-45 span. The
  // refinement passes run on the full-size map, where the sub-pixel quality of
  // the fit is what is actually being compared.
  let bestA = 0, bestQ = -1;
  const half = new cv.Mat();
  cv.resize(ink, half, new cv.Size(Math.max(32, ink.cols >> 1),
                                   Math.max(32, ink.rows >> 1)), 0, 0, cv.INTER_AREA);
  const coarse = src => {
    let a0 = 0, q0 = -1;
    for (let a = -GEN_SKEW_SPAN; a <= GEN_SKEW_SPAN + 1e-9; a += 2.0) {
      const q = _evaluate(src, a);
      if (q > q0) { q0 = q; a0 = a; }
    }
    return [a0, q0];
  };
  [bestA, bestQ] = coarse(half);
  half.delete();
  // Ticks can be too fine to survive the downscale. If nothing scored there,
  // the cheap pass has told us nothing and the full-size sweep has to run.
  if (bestQ <= 0) [bestA, bestQ] = coarse(ink);
  bestQ = -1;
  for (const [step, rng] of [[0.5, 2.5], [0.1, 0.5]]) {
    let cand = bestA;
    for (let a = bestA - rng; a <= bestA + rng + 1e-9; a += step) {
      const q = _evaluate(ink, a);
      if (q > bestQ) { bestQ = q; cand = a; }
    }
    bestA = cand;
  }
  return [bestA, bestQ];
}

// pick the colour channel in which the marks are darkest
function inkChannel(rgbaMat) {
  const rgb = new cv.Mat(); cv.cvtColor(rgbaMat, rgb, cv.COLOR_RGBA2RGB);
  const ch = new cv.MatVector(); cv.split(rgb, ch);
  const luma = new cv.Mat(); cv.cvtColor(rgb, luma, cv.COLOR_RGB2GRAY);
  const cands = [luma, ch.get(0), ch.get(1), ch.get(2)];
  let best = null, bestS = -1, bestI = -1;
  for (let i = 0; i < cands.length; i++) {
    const ik = inkMap(cands[i]);
    const vm = verticalStrokeMap(ik, _minStroke(cands[i].rows));
    const band = findBand(vm);
    const frac = (band[1] - band[0]) / Math.max(1, cands[i].rows);
    if (frac < 0.6 && frac >= 0.02) {
      const s = _combScore(vm, band[0], band[1]) * _sharpness(vm, band[0], band[1]);
      if (s > bestS) { bestS = s; bestI = i; }
    }
    ik.delete(); vm.delete();
  }
  if (bestI < 0) bestI = 0;
  best = cands[bestI].clone();
  cands.forEach(m => m.delete());
  rgb.delete(); ch.delete();
  return best;
}

// ---- search region --------------------------------------------------------
// A ruler is laid beside the lesion, so look in a generous box around it. A
// multiple of the lesion alone is not enough: a small mole with the ruler a
// couple of centimetres away puts the ruler outside, and there is no scale yet
// to reason in millimetres.
function nearLesionRoi(imgW, imgH, bbox, factor) {
  const [lx, ly, lw, lh] = bbox, f = factor || 8;
  const cx = lx + lw / 2, cy = ly + lh / 2;
  const half = Math.max(Math.max(lw, lh) * f / 2, 0.5 * Math.min(imgW, imgH));
  const x0 = Math.max(0, Math.round(cx - half)), y0 = Math.max(0, Math.round(cy - half));
  return { x: x0, y: y0,
           w: Math.min(imgW, Math.round(cx + half)) - x0,
           h: Math.min(imgH, Math.round(cy + half)) - y0 };
}

/* Detect the ruler and its scale.
   src: full RGBA Mat. roi: {x,y,w,h}. lesionBox: [x,y,w,h] in full-image px.
   -> { pxPerTick, pxPerCm, marksPerMajor, majorUnit, count, ticks, roi, ... }
      or null, with the reason in window.__genDiag. */
function detectGenericRuler(src, roi, lesionBox) {
  const diag = window.__genDiag = { roi: [roi.x, roi.y, roi.w, roi.h],
    ticks: 0, need: GEN_MIN_TICKS, rms: 0, pitch: 0, minorPerMajor: null,
    angle: 0, band: null, stage: "", reason: "" };

  const rect = new cv.Rect(roi.x, roi.y, roi.w, roi.h);
  const t0 = src.roi(rect), full = t0.clone(); t0.delete();

  // normalise resolution: too small starves the band seeder, too large is slow
  let up = 1;
  const longest = Math.max(roi.w, roi.h);
  if (longest < GEN_WORK_MIN) up = Math.min(3, GEN_WORK_MIN / longest);
  else if (longest > GEN_WORK_MAX) up = GEN_WORK_MAX / longest;
  let work = full;
  if (up !== 1) {
    work = new cv.Mat();
    cv.resize(full, work, new cv.Size(Math.round(roi.w * up), Math.round(roi.h * up)),
              0, 0, up > 1 ? cv.INTER_CUBIC : cv.INTER_AREA);
    full.delete();
  }

  // Take the lesion out of the frame before anything goes looking for a band.
  // The guard further down rejects a band that landed on the lesion, but
  // rejecting is too late: on a small crop the mole's hairs out-score a thin
  // ruler, the search settles on them, and a photo with a perfectly good ruler
  // in it gets refused -- rl3 found 6 ticks, all on the mole, and never
  // examined the ruler at all. Painting the lesion flat leaves no dark marks
  // there, so both the channel choice above and the band search below have to
  // look where the ruler actually is. Padded, because hairs and the darker rim
  // reach past the traced outline.
  if (lesionBox) {
    const pad = 0.18;
    const lx = (lesionBox[0] - roi.x) * up, ly = (lesionBox[1] - roi.y) * up;
    const lw = lesionBox[2] * up, lh = lesionBox[3] * up;
    const x0 = Math.max(0, Math.round(lx - lw * pad));
    const y0 = Math.max(0, Math.round(ly - lh * pad));
    const x1 = Math.min(work.cols, Math.round(lx + lw * (1 + pad)));
    const y1 = Math.min(work.rows, Math.round(ly + lh * (1 + pad)));
    if (x1 - x0 > 1 && y1 - y0 > 1) {
      const m = cv.mean(work);
      cv.rectangle(work, new cv.Point(x0, y0), new cv.Point(x1, y1),
                   new cv.Scalar(m[0], m[1], m[2], 255), -1);
      diag.lesionMasked = [x0, y0, x1 - x0, y1 - y0];
    }
  }

  const gray0 = inkChannel(work);

  // Orientation AND skew, decided together. Ticks are found as vertical
  // strokes, so a ruler running down the photo is invisible unless the frame is
  // turned first -- a portrait phone shot of a ruler beside a mole detected 5
  // ticks instead of 51 until this was added. A few degrees of tilt also flips
  // which orientation scores better, so the two cannot be decided separately.
  // The search runs on a downscaled copy: at full working resolution it took
  // ~10 s on a 3000 px photo, and the angle does not need those pixels.
  let bestO = "h", bestAng = 0, bestQ = -1;
  for (const o of ["h", "v"]) {
    let g = gray0;
    if (o === "v") { g = new cv.Mat(); cv.rotate(gray0, g, cv.ROTATE_90_CLOCKWISE); }
    const f = Math.min(1, 700 / Math.max(g.cols, g.rows));
    let small = g;
    if (f < 1) { small = new cv.Mat();
      cv.resize(g, small, new cv.Size(Math.round(g.cols * f), Math.round(g.rows * f)),
                0, 0, cv.INTER_AREA); }
    const ik = inkMap(small);
    const [ang, q] = _searchAngle(ik);
    ik.delete();
    if (small !== g) small.delete();
    if (o === "v") g.delete();
    if (q > bestQ) { bestQ = q; bestO = o; bestAng = ang; }
  }
  const orientation = bestO, angle = bestAng;
  diag.angle = +angle.toFixed(2);
  diag.orientation = orientation;

  // measure at working resolution in the chosen orientation
  let gray = gray0;
  if (orientation === "v") { gray = new cv.Mat();
    cv.rotate(gray0, gray, cv.ROTATE_90_CLOCKWISE); }
  const ink0 = inkMap(gray);
  const ink = angle === 0 ? ink0.clone() : _rotateMat(ink0, angle);
  const vmap = verticalStrokeMap(ink, _minStroke(gray.rows));
  const band = findBand(vmap);
  diag.band = [Math.round(band[0] / up), Math.round(band[1] / up)];
  const ticks = detectTicks(vmap, band);
  diag.ticks = ticks.length;

  const cleanup = () => { work.delete(); gray0.delete(); if (gray !== gray0) gray.delete();
                          ink0.delete(); ink.delete(); vmap.delete(); };

  if (ticks.length < GEN_MIN_TICKS) {
    cleanup();
    diag.stage = "ticks";
    diag.reason = `Found the ruler area but only ${ticks.length} tick`
                + `${ticks.length === 1 ? "" : "s"} on it (need ${GEN_MIN_TICKS}). `
                + `Get closer, or make sure the markings are in focus.`;
    return null;
  }

  const [majorPeriod] = classifyTicks(ticks);
  const pos = ticks.map(t => t.pos);
  let [pitch, rmsLin] = robustPitch(pos);
  let rms = rmsLin, persp = null;
  const pf = fitProjective(pos);
  if (pf) { persp = { a: pf.a, b: pf.b, c: pf.c }; rms = pf.rms; }
  diag.pitch = +(pitch / up).toFixed(3);
  diag.rms = +(rms / up).toFixed(2);

  if (!(pitch > 0)) {
    cleanup(); diag.stage = "fit";
    diag.reason = "The marks found were not evenly spaced enough to give a scale.";
    return null;
  }
  // a lattice this loose is not a ruler
  if (rms > 0.25 * pitch) {
    cleanup(); diag.stage = "fit";
    diag.reason = `The marks found do not sit on a regular lattice `
                + `(residual ${(rms / pitch * 100).toFixed(0)}% of the spacing), `
                + `so the scale is not trustworthy. Tap two marks instead.`;
    return null;
  }

  // How many minor divisions make a major. classifyTicks has already decided
  // this: its period is counted in lattice indices, so it IS the answer, and
  // it is counted over every tick the comb should have rather than over the
  // ones that survived.
  //
  // Measuring the majors' own spacing and dividing by the pitch is the same
  // number only while the comb is complete. Let a stretch of ticks go missing
  // and the surviving majors are the pair either side of the hole: on rl3 the
  // lesion mask paints out the middle of the ruler, indices 8..16 disappear,
  // and the two majors left read 20 minors apart where the period is 10.
  // Twenty is in no MINOR table, so a photo whose scale was measured correctly
  // (8.34 px/tick either way) was sent to the "millimetres or centimetres?"
  // prompt for want of a name.
  const majors = ticks.filter(t => t.level === 2).map(t => t.pos);
  let minorPerMajor = majorPeriod || null;
  if (!minorPerMajor && majors.length >= 2) {   // no period found: fall back
    const pxPerMajor = robustPitch(majors)[0];  // to the spacing of the majors
    if (pxPerMajor > 0 && pitch > 0) minorPerMajor = Math.round(pxPerMajor / pitch);
  }
  diag.minorPerMajor = minorPerMajor;

  // The period says what the MINOR division is; that is the reading to trust,
  // not "the long mark is a centimetre". A long mark is whatever the ruler's
  // maker chose to emphasise, but the small division between marks is one of
  // a very short list of real things.
  //
  //   10 -> minor is a millimetre, long mark on the centimetre. The ruler
  //         everyone owns.
  //    5 -> minor is STILL a millimetre; the long mark is the half-centimetre.
  //         A rule divided into five 2 mm steps per centimetre barely exists,
  //         so 5 names the half-cm, not a centimetre. This is rl1, where
  //         calling the major a centimetre halved the scale to 118.97 against
  //         a true 237.9 -- and a 2x scale error is a 4x area error. Reading
  //         the minor instead gives 23.80 x 10 = 238.0, right on the nose.
  //  8/16 -> minor is an eighth or sixteenth of an inch, long mark on the
  //         inch. No metric rule puts a long mark every 8th or 16th. This is
  //         the dual-scale dermatology card, whose INCH edge the band search
  //         can settle on: period 8 used to reach a "1 mm or 1 cm?" prompt
  //         where NEITHER answer was true. As inches it reads 237.7 px/cm on
  //         l1-cropped against 234 from the card detector, 1.6% apart.
  //
  // Anything else still goes to the caller's confirmation rather than a guess.
  const MINOR = {
    5:  { cm: 0.1,        name: "millimetres",      major: "½ cm" },
    10: { cm: 0.1,        name: "millimetres",      major: "cm"        },
    8:  { cm: 2.54 / 8,   name: "eighth-inches",    major: "inch"      },
    16: { cm: 2.54 / 16,  name: "sixteenth-inches", major: "inch"      }
  };
  const minor = MINOR[minorPerMajor] || null;
  const unitConfident = minor !== null;

  // tick positions back to full-image coordinates. The band is horizontal in
  // the de-skewed frame; undo the rotation about the working centre.
  // tick positions back to full-image pixels: undo the skew rotation about the
  // analysed frame's centre, then the 90-degree turn if the frame was rotated
  // The skew was applied by _rotateMat(ink0, angle); undoing it means rotating
  // the point by +angle here, not -angle. The sign was wrong, and invisibly so:
  // at 0 degrees the two agree, and even at the old +/-15 cap the drawn ticks
  // were only slightly off. It also never moved the reported scale, which comes
  // from the pitch measured in the de-skewed frame. What it did move was the
  // overlay and the on-lesion guard -- at rl3's -42 degrees the mapped ticks
  // came out on a line roughly perpendicular to the ruler, straight across the
  // mole.
  const cxw = gray.cols / 2, cyw = gray.rows / 2;
  const rad = angle * Math.PI / 180, cs = Math.cos(rad), sn = Math.sin(rad);
  const bandMid = (band[0] + band[1]) / 2;
  const cropH = work.rows;
  const full_ticks = ticks.map(t => {
    const dx = t.pos - cxw, dy = bandMid - cyw;
    const X = cxw + dx * cs - dy * sn, Y = cyw + dx * sn + dy * cs;
    // cv.rotate 90 CW maps crop(r,c) -> rot(row=c, col=H-1-r), so invert it
    const cropX = orientation === "h" ? X : Y;
    const cropY = orientation === "h" ? Y : (cropH - 1 - X);
    return [roi.x + cropX / up, roi.y + cropY / up];
  });
  const workScale = up;

  // GUARD: the marks must not sit on the lesion. Without this the detector will
  // happily fit a lattice to hairs on a mole and report a confident scale --
  // 101 px/cm on rl3, where the ruler reads 83.
  //
  // Tested on the mapped tick positions, not on the band. The band is a strip
  // in the DE-SKEWED frame, so comparing it to an axis-aligned lesion box
  // quietly assumes the skew is small; at rl3's -42 degrees that assumption
  // inverts the answer, and a ruler correctly found off to the side was thrown
  // out for "64% overlap" with a lesion it never touched. These coordinates are
  // already back in the image, so they need no assumption at all.
  if (lesionBox && full_ticks.length) {
    const [lx, ly, lw, lh] = lesionBox;
    let on = 0;
    for (const [tx, ty] of full_ticks)
      if (tx >= lx && tx <= lx + lw && ty >= ly && ty <= ly + lh) on++;
    const frac = on / full_ticks.length;
    if (frac > 0.5) {
      cleanup();
      diag.stage = "onlesion";
      diag.onLesionFrac = +frac.toFixed(2);
      diag.reason = `The marks that were found sit on the lesion, not on a ruler `
                  + `(${Math.round(frac * 100)}% of them), so the scale would be wrong. `
                  + `Make sure the ruler is in the photo beside the lesion.`;
      return null;
    }
  }
  cleanup();

  const out = {
    pxPerTick: pitch / up,
    marksPerMajor: unitConfident ? minorPerMajor : null,
    majorUnit: unitConfident ? minor.major : null,   // what a long mark spans
    minorName: unitConfident ? minor.name : null,    // what one small step is
    minorPerMajor,                       // what the sub-lattice actually found
    pxPerCm: unitConfident ? (pitch / up) / minor.cm : null,
    unitUnknown: !unitConfident,
    count: ticks.length,
    rms: rms / up,
    perspective: persp,
    workScale,
    roi,
    ticks: full_ticks
  };
  diag.stage = "ok";
  diag.reason = unitConfident
    ? `${ticks.length} marks, every ${minorPerMajor}th a long one (1 ${minor.major}), `
      + `so the small marks are ${minor.name}.`
    : minorPerMajor
      ? `${ticks.length} marks at ${(pitch / up).toFixed(2)} px, with a long mark every `
        + `${minorPerMajor} — not a spacing that names a unit, so it needs confirming.`
      : `${ticks.length} marks at ${(pitch / up).toFixed(2)} px, but no long/short `
        + `pattern to say how many make a centimetre.`;
  // local scale at the lesion: with perspective the pitch drifts across the
  // frame, and using the global figure biased area by 12% on one test photo
  if (persp && lesionBox) {
    // Where the lesion falls along the run of ticks -- the only axis the
    // perspective fit is a function of. Reaching it means the trip the ticks
    // made, in reverse: image -> crop -> orientation -> de-skew. The previous
    // code passed the lesion's image x scaled by workScale, which quietly
    // assumed an unrotated, unskewed, un-offset frame. On rl3 ("v", -42
    // degrees) that is not even the right axis, and this number is what the
    // final measurement gets calibrated with.
    const lcx = (lesionBox[0] + lesionBox[2] / 2 - roi.x) * up;
    const lcy = (lesionBox[1] + lesionBox[3] / 2 - roi.y) * up;
    const LX = orientation === "h" ? lcx : (cropH - 1 - lcy);
    const LY = orientation === "h" ? lcy : lcx;
    const lpos = cxw + (LX - cxw) * cs + (LY - cyw) * sn;
    const lp = pitchAtPos(out, lpos);
    if (lp > 0 && unitConfident) out.pxPerCmAtLesion = lp / minor.cm;
    out.pxPerTickAtLesion = lp;
  }
  return out;
}
