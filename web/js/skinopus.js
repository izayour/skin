/* SkinOpus colour-calibration detection — port of skinopus_colorbox.py.

   Finds the colour calibration strips on a dermatology ruler card so the photo
   can later be colour-corrected against them. Stage 1 of two: this file
   locates the strips and their swatches; the correction maths comes next and
   is meaningless unless these come out right.

   Verification targets from the reference README, which is why swatch counts
   are reported rather than judged by eye:
       half.jpg 9 + 10 | l1-cropped.jpg 18 + 18 | l2.jpg 18 + 18

   Everything works in BGR to match the Python exactly: the Lab conversions and
   the channel indices in the scoring functions all assume that order, and
   quietly running them on RGB would shift every hue.
*/

// ---- numeric helpers ------------------------------------------------------
const _sMed = a => { if (!a.length) return 0;
  const s = Float64Array.from(a).sort(); const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const _sMean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
const _sStd = a => { if (a.length < 2) return 0; const m = _sMean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length); };
function _sPct(arr, p) {
  const s = Float64Array.from(arr).sort();
  if (!s.length) return 0;
  const i = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
  return s[i];
}
// polynomial least squares, returns coefficients highest power first (np.polyfit)
function _polyfit(x, y, deg) {
  const m = x.length, k = deg + 1;
  const A = [], b = [];
  for (let i = 0; i < m; i++) {
    const row = new Float64Array(k);
    for (let j = 0; j < k; j++) row[j] = Math.pow(x[i], deg - j);
    A.push(row); b.push(y[i]);
  }
  const N = [], r = new Float64Array(k);
  for (let i = 0; i < k; i++) N.push(new Float64Array(k));
  for (let row = 0; row < m; row++)
    for (let i = 0; i < k; i++) {
      r[i] += A[row][i] * b[row];
      for (let j = 0; j < k; j++) N[i][j] += A[row][i] * A[row][j];
    }
  for (let c = 0; c < k; c++) {
    let piv = c;
    for (let i = c + 1; i < k; i++) if (Math.abs(N[i][c]) > Math.abs(N[piv][c])) piv = i;
    if (Math.abs(N[piv][c]) < 1e-12) return null;
    if (piv !== c) { const t = N[piv]; N[piv] = N[c]; N[c] = t;
                     const tr = r[piv]; r[piv] = r[c]; r[c] = tr; }
    for (let i = c + 1; i < k; i++) {
      const f = N[i][c] / N[c][c]; if (!f) continue;
      for (let j = c; j < k; j++) N[i][j] -= f * N[c][j];
      r[i] -= f * r[c];
    }
  }
  const out = new Float64Array(k);
  for (let i = k - 1; i >= 0; i--) {
    let s = r[i];
    for (let j = i + 1; j < k; j++) s -= N[i][j] * out[j];
    out[i] = s / N[i][i];
  }
  return Array.from(out);
}
const _polyval = (p, x) => { let v = 0; for (let i = 0; i < p.length; i++) v = v * x + p[i]; return v; };
// moving average, 'same' length (np.convolve with a box kernel)
function _conv(a, k) {
  const n = a.length, r = k >> 1, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0, c = 0;
    for (let j = i - r; j <= i + r; j++) if (j >= 0 && j < n) { s += a[j]; c++; }
    out[i] = c ? s / c : 0;
  }
  return out;
}
// deterministic RNG so a run is reproducible (np.random.default_rng(0))
function _rng(seed) { let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

// ---- Lab parts ------------------------------------------------------------
// returns {L,a,b,C} as Float32Arrays over the whole Mat (a,b already centred)
function labParts(bgr) {
  const lab = new cv.Mat();
  cv.cvtColor(bgr, lab, cv.COLOR_BGR2Lab);
  const n = lab.rows * lab.cols, D = lab.data;
  const L = new Float32Array(n), a = new Float32Array(n),
        b = new Float32Array(n), C = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    L[i] = D[i * 3];
    a[i] = D[i * 3 + 1] - 128;
    b[i] = D[i * 3 + 2] - 128;
    C[i] = Math.hypot(a[i], b[i]);
  }
  lab.delete();
  return { L, a, b, C, rows: bgr.rows, cols: bgr.cols };
}

// ---- Stage 1: de-shadow ---------------------------------------------------
// A shadow multiplies luminance AND compresses chroma, so the same local gain
// is applied to a/b or shadowed swatches stay desaturated and fail the vivid
// test that seeds everything downstream.
function deshadow(bgr, sigmaFrac, chromaRegain) {
  sigmaFrac = sigmaFrac || 0.05; chromaRegain = chromaRegain || 1.3;
  const lab = new cv.Mat();
  cv.cvtColor(bgr, lab, cv.COLOR_BGR2Lab);
  const h = bgr.rows, w = bgr.cols, n = h * w, D = lab.data;

  const Lm = new cv.Mat(h, w, cv.CV_32F);
  const LF = Lm.data32F;
  for (let i = 0; i < n; i++) LF[i] = D[i * 3];

  const sigma = Math.max(9, Math.floor(sigmaFrac * Math.max(h, w)));
  const k = Math.max(3, (Math.floor(sigma / 6) * 2 + 1));
  const kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(k, k));
  const dil = new cv.Mat(), illum = new cv.Mat();
  cv.dilate(Lm, dil, kern);
  cv.GaussianBlur(dil, illum, new cv.Size(0, 0), sigma);
  const IF = illum.data32F;
  for (let i = 0; i < n; i++) if (IF[i] < 1) IF[i] = 1;

  const step = Math.max(1, Math.floor(n / 20000)), samp = [];
  for (let i = 0; i < n; i += step) samp.push(IF[i]);
  const target = _sPct(samp, 90);

  const out = new cv.Mat(h, w, cv.CV_8UC3);
  const O = out.data;
  for (let i = 0; i < n; i++) {
    let g = target / IF[i];
    if (g < 0.5) g = 0.5; else if (g > 3) g = 3;
    let L2 = D[i * 3] * g;
    L2 = L2 < 0 ? 0 : (L2 > 255 ? 255 : L2);
    const f = 1 + (g - 1) * chromaRegain;
    let a2 = (D[i * 3 + 1] - 128) * f, b2 = (D[i * 3 + 2] - 128) * f;
    a2 = a2 < -127 ? -127 : (a2 > 127 ? 127 : a2);
    b2 = b2 < -127 ? -127 : (b2 > 127 ? 127 : b2);
    O[i * 3] = L2; O[i * 3 + 1] = a2 + 128; O[i * 3 + 2] = b2 + 128;
  }
  const flat = new cv.Mat();
  cv.cvtColor(out, flat, cv.COLOR_Lab2BGR);
  lab.delete(); Lm.delete(); dil.delete(); illum.delete(); kern.delete(); out.delete();
  return flat;
}

// ---- Stage 2: vivid seeds -------------------------------------------------
function _seedsAt(Cf, h, w, thr) {
  const m = new cv.Mat(h, w, cv.CV_8U);
  const M = m.data;
  for (let i = 0; i < h * w; i++) M[i] = Cf[i] > thr ? 255 : 0;
  const kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(7, 7));
  const op = new cv.Mat();
  cv.morphologyEx(m, op, cv.MORPH_OPEN, kern);
  const lbl = new cv.Mat(), stats = new cv.Mat(), cent = new cv.Mat();
  const n = cv.connectedComponentsWithStats(op, lbl, stats, cent, 8);
  const lo = Math.max(400, 2e-5 * h * w), hi = 6e-3 * h * w;
  const seeds = [];
  const LB = lbl.data32S;
  for (let i = 1; i < n; i++) {
    const area = stats.intAt(i, 4);
    if (area < lo || area > hi) continue;
    // minAreaRect of this component, built over its BOUNDING BOX only.
    // A full-frame Mat per component exhausts the WASM heap on a 12 MP photo
    // (dozens of 12 MB allocations per threshold rung, eight rungs), after
    // which OpenCV calls start failing and the detector silently finds
    // nothing -- measured as 0 seeds on an image that yields 42.
    const bx = stats.intAt(i, 0), by = stats.intAt(i, 1),
          bw = stats.intAt(i, 2), bh = stats.intAt(i, 3);
    const one = new cv.Mat(bh, bw, cv.CV_8U);
    const O = one.data;
    for (let y = 0; y < bh; y++)
      for (let x = 0; x < bw; x++)
        O[y * bw + x] = LB[(by + y) * w + (bx + x)] === i ? 255 : 0;
    const cnts = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(one, cnts, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let big = null, bigA = -1;
    for (let c = 0; c < cnts.size(); c++) {
      const cc = cnts.get(c), a = cv.contourArea(cc);
      if (a > bigA) { bigA = a; if (big) big.delete(); big = cc; } else cc.delete();
    }
    if (big) {
      const rr = cv.minAreaRect(big);
      const rw = rr.size.width, rh = rr.size.height;
      if (!(rw < 5 || rh < 5 || area / (rw * rh) < 0.60))
        seeds.push({ cx: rr.center.x + bx, cy: rr.center.y + by,
                     h: Math.min(rw, rh), area });
      big.delete();
    }
    cnts.delete(); hier.delete(); one.delete();
  }
  m.delete(); op.delete(); kern.delete(); lbl.delete(); stats.delete(); cent.delete();
  return seeds;
}

// Print saturation varies hugely between cards (one peaks at chroma ~77,
// another barely reaches ~29), so walk a descending ladder and keep the level
// whose seeds are most COLLINEAR — a loose threshold pulls in background
// clutter that inflates the count but never lines up.
function vividSeeds(flat, minSeeds) {
  minSeeds = minSeeds || 6;
  const lp = labParts(flat);
  const ladder = [45, 40, 35, 30, 26, 22, 19, 16];
  let best = null;
  for (const thr of ladder) {
    const seeds = _seedsAt(lp.C, lp.rows, lp.cols, thr);
    if (seeds.length < 3) continue;
    const r = ransacAxis(seeds);
    const nIn = r ? r.inl.filter(Boolean).length : 0;
    const hs = seeds.map(s => s.h);
    const consist = _sMed(hs) / (_sStd(hs) + 1e-6);
    if (!best || nIn > best.nIn || (nIn === best.nIn && consist > best.consist))
      best = { nIn, consist, seeds, thr };
    if (nIn >= minSeeds && nIn >= 0.6 * seeds.length) break;
  }
  return best ? { seeds: best.seeds, thr: best.thr } : { seeds: [], thr: ladder[0] };
}

// ---- Stage 3: axis --------------------------------------------------------
function ransacAxis(seeds, tol, iters) {
  if (seeds.length < 3) return null;
  const P = seeds.map(s => [s.cx, s.cy]);
  if (tol == null) tol = Math.max(12, 0.75 * _sMed(seeds.map(s => s.h)));
  iters = iters || 4000;
  const rnd = _rng(0), n = P.length;
  let bestCount = 0, bestInl = null;
  for (let it = 0; it < iters; it++) {
    const i = Math.floor(rnd() * n);
    let j = Math.floor(rnd() * n);
    if (j === i) j = (j + 1) % n;
    const dx = P[j][0] - P[i][0], dy = P[j][1] - P[i][1];
    const nrm = Math.hypot(dx, dy);
    if (nrm < 30) continue;
    const nx = -dy / nrm, ny = dx / nrm;
    const inl = P.map(p => Math.abs((p[0] - P[i][0]) * nx + (p[1] - P[i][1]) * ny) < tol);
    const c = inl.filter(Boolean).length;
    if (c > bestCount) { bestCount = c; bestInl = inl; }
  }
  if (!bestInl || bestCount < 3) return null;
  return { pts: P.filter((_, i) => bestInl[i]), inl: bestInl };
}

function fitCentreline(pts, deg) {
  const x = pts.map(p => p[0]), y = pts.map(p => p[1]);
  if (pts.length < 4) deg = 1;
  if (Math.max(...x) - Math.min(...x) < 1e-6) return null;
  return _polyfit(x, y, deg == null ? 2 : deg);
}

// ---- Stage 4: band sampling ----------------------------------------------
// Unbend: sample rows parallel to the centreline y = f(x) + offset.
function sampleBand(img, poly, x0, x1, offset, halfH) {
  const h = img.rows, w = img.cols, ch = img.channels();
  const xa = Math.round(x0), xb = Math.round(x1);
  const nx = Math.max(1, xb - xa + 1), ny = 2 * halfH + 1;
  const xs = new Float64Array(nx), yc = new Float64Array(nx);
  for (let i = 0; i < nx; i++) { xs[i] = xa + i; yc[i] = _polyval(poly, xa + i) + offset; }
  const band = new cv.Mat(ny, nx, ch === 3 ? cv.CV_8UC3 : cv.CV_8U);
  const B = band.data, S = img.data;
  for (let r = 0; r < ny; r++) {
    const j = r - halfH;
    for (let i = 0; i < nx; i++) {
      let Y = Math.round(yc[i] + j), X = xa + i;
      const ok = Y >= 0 && Y < h && X >= 0 && X < w;
      Y = Y < 0 ? 0 : (Y >= h ? h - 1 : Y);
      X = X < 0 ? 0 : (X >= w ? w - 1 : X);
      const si = (Y * w + X) * ch, di = (r * nx + i) * ch;
      for (let c = 0; c < ch; c++) B[di + c] = ok ? S[si + c] : 0;
    }
  }
  return { band, xs, yc };
}

// column means over the vertical core of a band
function _coreMeans(lp, frac0, frac1) {
  const H = lp.rows, W = lp.cols;
  const r0 = Math.floor(frac0 * H), r1 = Math.min(H - 1, Math.floor(frac1 * H));
  const Lc = new Float64Array(W), ac = new Float64Array(W),
        bc = new Float64Array(W), Cc = new Float64Array(W);
  const rows = Math.max(1, r1 - r0 + 1);
  for (let x = 0; x < W; x++) {
    let sl = 0, sa = 0, sb = 0, sc = 0;
    for (let y = r0; y <= r1; y++) {
      const i = y * W + x;
      sl += lp.L[i]; sa += lp.a[i]; sb += lp.b[i]; sc += lp.C[i];
    }
    Lc[x] = sl / rows; ac[x] = sa / rows; bc[x] = sb / rows; Cc[x] = sc / rows;
  }
  return { Lc, ac, bc, Cc };
}

function bandColourScore(bandBgr) {
  const lp = labParts(bandBgr);
  const { Lc, Cc } = _coreMeans(lp, 0.25, 0.75);
  const Lpaper = _sPct(Array.from(lp.L), 92);
  let n = 0;
  for (let i = 0; i < Lc.length; i++) if (Cc[i] > 18 || Lc[i] < Lpaper - 55) n++;
  return n / Math.max(1, Lc.length);
}

// Lock the band onto the strip's true top and bottom edges.
function refineBand(flat, poly, x0, x1, offset, swH, grow) {
  grow = grow || 3.0;
  const tall = Math.round(swH * grow);
  const { band } = sampleBand(flat, poly, x0, x1, offset, tall);
  const lp = labParts(band);
  const Lpaper = _sPct(Array.from(lp.L), 92);
  const H = lp.rows, W = lp.cols;
  const rowFrac = new Float64Array(H);
  for (let y = 0; y < H; y++) {
    let n = 0;
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (lp.C[i] > 18 || lp.L[i] < Lpaper - 55) n++;
    }
    rowFrac[y] = n / W;
  }
  band.delete();
  const np = _conv(rowFrac, 5);
  let mid = tall; const thr = 0.45;
  if (np[mid] < thr) {
    let bestI = -1, bestD = Infinity;
    for (let i = 0; i < H; i++) if (np[i] >= thr && Math.abs(i - mid) < bestD) { bestD = Math.abs(i - mid); bestI = i; }
    if (bestI < 0) return { offset, half: Math.max(4, Math.round(swH * 0.45)) };
    mid = bestI;
  }
  let top = mid; while (top > 0 && np[top - 1] >= thr) top--;
  let bot = mid; while (bot < H - 1 && np[bot + 1] >= thr) bot++;
  const centre = (top + bot) / 2;
  return { offset: offset + (centre - tall),
           half: Math.max(4, Math.round((bot - top) / 2 * 0.88)) };
}

// Each strip bends differently, so fit a centreline to THIS strip.
function stripCentreline(flat, poly, x0, x1, offset, swH, grow) {
  grow = grow || 3.0;
  const tall = Math.round(swH * grow);
  const { band, xs, yc } = sampleBand(flat, poly, x0, x1, offset, tall);
  const lp = labParts(band);
  const Lpaper = _sPct(Array.from(lp.L), 92);
  const H = lp.rows, W = lp.cols;
  const npm = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) npm[i] = (lp.C[i] > 18 || lp.L[i] < Lpaper - 55) ? 1 : 0;
  band.delete();
  const cx = [], cy = [], hh = [];
  for (let i = 0; i < W; i++) {
    let near = false;
    for (let y = Math.max(0, tall - 3); y <= Math.min(H - 1, tall + 3); y++)
      if (npm[y * W + i]) { near = true; break; }
    if (!near) continue;
    let t = tall; while (t > 0 && npm[(t - 1) * W + i]) t--;
    let bm = tall; while (bm < H - 1 && npm[(bm + 1) * W + i]) bm++;
    if (bm - t < 0.3 * swH) continue;
    cx.push(xs[i]); cy.push(yc[i] + (t + bm) / 2 - tall); hh.push(bm - t);
  }
  if (cx.length < 20) return null;
  let p2 = _polyfit(cx, cy, 2);
  if (!p2) return null;
  for (let pass = 0; pass < 2; pass++) {          // robust IRLS passes
    const r = cx.map((x, i) => Math.abs(cy[i] - _polyval(p2, x)));
    const lim = Math.max(3, 2.5 * _sMed(r));
    const kx = [], ky = [];
    for (let i = 0; i < cx.length; i++) if (r[i] < lim) { kx.push(cx[i]); ky.push(cy[i]); }
    if (kx.length < 15) break;
    const np2 = _polyfit(kx, ky, 2);
    if (!np2) break;
    p2 = np2;
  }
  return { poly: p2, half: Math.max(4, Math.round(_sMed(hh) / 2 * 0.86)),
           x0: Math.min(...cx), x1: Math.max(...cx) };
}

// Sweep parallel offsets and keep the ones that look like colour strips.
function findStrips(flat, poly, x0, x1, swH, search) {
  search = search || 6.0;
  const half = Math.max(1, Math.round(swH * 0.45));
  const step = Math.max(2, swH * 0.1);
  const offs = [], scores = [];
  for (let o = -search * swH; o <= search * swH + 1e-9; o += step) {
    const { band } = sampleBand(flat, poly, x0, x1, o, half);
    scores.push(bandColourScore(band));
    band.delete();
    offs.push(o);
  }
  const order = offs.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const keep = [];
  for (const k of order) {
    if (scores[k] < 0.45) break;
    if (keep.every(m => Math.abs(offs[k] - offs[m]) >= swH * 1.2)) keep.push(k);
  }
  keep.sort((a, b) => offs[a] - offs[b]);
  return keep.map(k => ({ off: offs[k], score: scores[k] }));
}

// How far the paper card runs along the strip. Chroma alone is not enough --
// backlit skin measures chroma 16-24 and slips under any sensible threshold --
// so lightness against a LOCAL baseline is used as well (shadow varies the
// paper itself, so one global number mislabels genuine card margin).
function cardExtent(flat, poly, x0, x1, offset, half) {
  const Ls = [], Cs = [];
  for (const s of [-1, 1]) {
    const { band } = sampleBand(flat, poly, x0, x1, offset + s * 1.8 * half,
                                Math.max(3, Math.round(0.45 * half)));
    const lp = labParts(band);
    const H = lp.rows, W = lp.cols;
    const Lm = new Float64Array(W), Cm = new Float64Array(W);
    for (let x = 0; x < W; x++) {
      let sl = 0, sc = 0;
      for (let y = 0; y < H; y++) { sl += lp.L[y * W + x]; sc += lp.C[y * W + x]; }
      Lm[x] = sl / H; Cm[x] = sc / H;
    }
    Ls.push(Lm); Cs.push(Cm);
    band.delete();
  }
  const W = Ls[0].length;
  const Cmin = new Float64Array(W), Lmin = new Float64Array(W);
  for (let i = 0; i < W; i++) {
    Cmin[i] = Math.min(Cs[0][i], Cs[1][i]);
    Lmin[i] = Math.min(Ls[0][i], Ls[1][i]);
  }
  const C = _conv(Cmin, 15), L = _conv(Lmin, 15);
  let mid = W >> 1;
  const base = _sMed(Array.from(C.slice(Math.floor(0.25 * W), Math.floor(0.75 * W))));
  const cThr = Math.min(32, Math.max(12, base + 14));
  const wk = Math.max(31, ((W / 6) | 0) | 1);
  const local = _conv(L, wk);
  const paper = new Uint8Array(W);
  for (let i = 0; i < W; i++) paper[i] = (C[i] <= cThr && L[i] >= local[i] - 26) ? 1 : 0;
  if (!paper[mid]) {
    let bi = -1, bd = Infinity;
    for (let i = 0; i < W; i++) if (paper[i] && Math.abs(i - mid) < bd) { bd = Math.abs(i - mid); bi = i; }
    if (bi < 0) return [0, W - 1];
    mid = bi;
  }
  let lo = mid; while (lo > 0 && paper[lo - 1]) lo--;
  let hi = mid; while (hi < W - 1 && paper[hi + 1]) hi++;
  return [lo, hi];
}

// ---- Stage 6: split into swatches ----------------------------------------
function splitSwatches(bandBgr, cardCols, minFrac) {
  minFrac = minFrac == null ? 0.45 : minFrac;
  const lp = labParts(bandBgr);
  const H = lp.rows, W = lp.cols;
  const { Lc, ac, bc, Cc } = _coreMeans(lp, 0.22, 0.78);
  const Lpaper = _sPct(Array.from(lp.L), 92);

  // Pale swatches (near-white pink, periwinkle) fail an absolute chroma test,
  // so measure delta-E from the band's own paper reference instead.
  const Lq = _sPct(Array.from(Lc), 70), Cq = _sPct(Array.from(Cc), 40);
  const pale = [];
  for (let i = 0; i < W; i++) if (Lc[i] > Lq && Cc[i] < Cq) pale.push(i);
  let ref;
  if (pale.length >= 5) ref = [_sMean(pale.map(i => Lc[i])), _sMean(pale.map(i => ac[i])),
                               _sMean(pale.map(i => bc[i]))];
  else ref = [Lpaper, 0, 0];
  const nonpaper = new Uint8Array(W);
  for (let i = 0; i < W; i++) {
    const dE = Math.sqrt((Lc[i] - ref[0]) ** 2 + (ac[i] - ref[1]) ** 2 + (bc[i] - ref[2]) ** 2);
    nonpaper[i] = (dE > 10 || Cc[i] > 18 || Lc[i] < Lpaper - 55) ? 1 : 0;
  }
  const valid = new Uint8Array(W);
  for (let i = 0; i < W; i++)
    valid[i] = nonpaper[i] && (!cardCols || (i >= cardCols[0] && i <= cardCols[1])) ? 1 : 0;
  const idx = [];
  for (let i = 0; i < W; i++) if (valid[i]) idx.push(i);
  if (idx.length < 20) return [];
  const i0 = idx[0], i1 = idx[idx.length - 1], span = i1 - i0;
  if (span < 20) return [];

  // colour-change gradient along the band
  const sL = _conv(Lc, 5), sA = _conv(ac, 5), sB = _conv(bc, 5);
  const g = new Float64Array(W);
  for (let i = 0; i + 1 < W; i++)
    g[i] = Math.hypot(sL[i + 1] - sL[i], sA[i + 1] - sA[i], sB[i + 1] - sB[i]);
  const gg = _conv(g, 5);
  for (let i = 0; i < i0; i++) gg[i] = 0;
  for (let i = i1; i < W; i++) gg[i] = 0;

  // pitch by autocorrelation: touching swatches of similar hue give a weak
  // gradient, so peak-picking alone loses boundaries
  const seg = Array.from(gg.slice(i0, i1));
  const m = _sMean(seg);
  for (let i = 0; i < seg.length; i++) seg[i] -= m;
  const ac_ = new Float64Array(seg.length);
  for (let lag = 0; lag < seg.length; lag++) {
    let s = 0;
    for (let i = 0; i + lag < seg.length; i++) s += seg[i] * seg[i + lag];
    ac_[lag] = s;
  }
  if (!(ac_[0] > 0)) return [];
  for (let i = 0; i < ac_.length; i++) ac_[i] /= ac_[0];
  const lo = Math.max(4, Math.floor(span / 22));
  const hi = Math.min(Math.max(6, Math.floor(span / 3)), ac_.length - 1);
  if (hi <= lo) return [];

  // Take the FUNDAMENTAL, not the tallest peak: autocorrelation also fires at
  // integer multiples, and a harmonic outscoring the fundamental would merge
  // several swatches into one.
  const cand = [];
  for (let k = lo + 1; k < hi - 1; k++)
    if (ac_[k] >= ac_[k - 1] && ac_[k] >= ac_[k + 1] && ac_[k] > 0) cand.push(k);
  let pitch;
  if (!cand.length) {
    let bi = lo, bv = -Infinity;
    for (let k = lo; k < hi; k++) if (ac_[k] > bv) { bv = ac_[k]; bi = k; }
    pitch = bi;
  } else {
    const best = Math.max(...cand.map(k => ac_[k]));
    pitch = Math.min(...cand.filter(k => ac_[k] >= 0.72 * best));
    for (const div of [2, 3]) {
      const sub = pitch / div;
      if (sub < lo) continue;
      const j = Math.round(sub);
      let mx = -Infinity;
      for (let t = Math.max(lo, j - 2); t < Math.min(hi, j + 3); t++) if (ac_[t] > mx) mx = ac_[t];
      if (mx >= 0.72 * best) pitch = sub;
    }
  }
  if (!(pitch > 0)) return [];

  let bestPh = 0, bestSc = -Infinity;
  for (let ph = 0; ph < pitch; ph += 0.5) {
    const pos = [];
    for (let k = 0; k < Math.floor(span / pitch) + 2; k++) {
      const p = i0 + ph + k * pitch;
      if (p >= i0 && p <= Math.min(i1, W - 1)) pos.push(p);
    }
    if (pos.length < 3) continue;
    const sc = _sMean(pos.map(p => gg[Math.round(p)]));
    if (sc > bestSc) { bestSc = sc; bestPh = ph; }
  }
  if (bestSc === -Infinity) return [];

  const pos = [];
  for (let k = 0; k < Math.round(span / pitch) + 2; k++) {
    const p = i0 + bestPh + k * pitch;
    if (p >= i0 && p <= Math.min(i1, W - 1)) pos.push(p);
  }
  const win = Math.max(2, Math.floor(0.22 * pitch));
  const snapped = pos.map(p => {
    const a = Math.max(0, Math.floor(p - win)), b = Math.min(W - 1, Math.floor(p + win));
    let bi = Math.round(p), bv = -Infinity;
    for (let i = a; i <= b; i++) if (gg[i] > bv) { bv = gg[i]; bi = i; }
    return bv > 0 ? bi : Math.round(p);
  });
  const edges = Array.from(new Set([i0, ...snapped, i1])).sort((a, b) => a - b);
  let segs = [];
  for (let k = 0; k + 1 < edges.length; k++) segs.push([edges[k], edges[k + 1]]);
  segs = segs.filter((s, k) => {
    const edge = (k === 0 || k === segs.length - 1);
    return (s[1] - s[0]) >= (edge ? 0.22 : minFrac) * pitch;
  });

  const segStat = s => {
    const c0 = Math.floor(s[0] + 0.25 * (s[1] - s[0]));
    const c1 = Math.max(c0 + 1, Math.floor(s[1] - 0.25 * (s[1] - s[0])));
    let sc = 0, sl = 0, n = 0;
    for (let i = c0; i < c1 && i < W; i++) { sc += Cc[i]; sl += Lc[i]; n++; }
    return n ? [sc / n, sl / n] : [0, 0];
  };
  // bright + achromatic = card margin. The black swatch is achromatic too but
  // very dark, so lightness saves it.
  const isBorder = s => { const [c, l] = segStat(s); return c < 12 && l > 70; };
  while (segs.length > 3 && (isBorder(segs[0]) || isBorder(segs[segs.length - 1]))) {
    if (isBorder(segs[0])) segs.shift(); else segs.pop();
  }
  return segs;
}

// ---- validation helpers ---------------------------------------------------
function _rgbToLab(rgbs) {
  const m = new cv.Mat(1, rgbs.length, cv.CV_8UC3);
  const D = m.data;
  rgbs.forEach((c, i) => { D[i * 3] = c[0]; D[i * 3 + 1] = c[1]; D[i * 3 + 2] = c[2]; });
  const lab = new cv.Mat();
  cv.cvtColor(m, lab, cv.COLOR_RGB2Lab);
  const out = [];
  for (let i = 0; i < rgbs.length; i++)
    out.push([lab.data[i * 3], lab.data[i * 3 + 1] - 128, lab.data[i * 3 + 2] - 128]);
  m.delete(); lab.delete();
  return out;
}
// A genuine strip is a run of deliberately DIFFERENT colours, so consecutive
// swatches sit far apart in Lab. Skin gradients and shadow do not.
function stripIsColourful(rgbs, minAdjDe, minSpread) {
  minAdjDe = minAdjDe == null ? 18 : minAdjDe;
  minSpread = minSpread == null ? 12 : minSpread;
  if (rgbs.length < 3) return { ok: false, medAdj: 0, spread: 0 };
  const lab = _rgbToLab(rgbs);
  const adj = [];
  for (let i = 1; i < lab.length; i++)
    adj.push(Math.hypot(lab[i][0] - lab[i - 1][0], lab[i][1] - lab[i - 1][1],
                        lab[i][2] - lab[i - 1][2]));
  const medAdj = _sMed(adj);
  const va = _sStd(lab.map(l => l[1])) ** 2, vb = _sStd(lab.map(l => l[2])) ** 2;
  const spread = Math.sqrt(va + vb);
  return { ok: medAdj >= minAdjDe && spread >= minSpread, medAdj, spread };
}
function trimEnds(segs, rgbs, polys, centres, deThr) {
  deThr = deThr == null ? 14 : deThr;
  while (segs.length > 3) {
    const lab = _rgbToLab(rgbs);
    const d0 = Math.hypot(lab[0][0] - lab[1][0], lab[0][1] - lab[1][1], lab[0][2] - lab[1][2]);
    const n = lab.length;
    const d1 = Math.hypot(lab[n - 1][0] - lab[n - 2][0], lab[n - 1][1] - lab[n - 2][1],
                          lab[n - 1][2] - lab[n - 2][2]);
    let k;
    if (d0 < deThr) k = 0; else if (d1 < deThr) k = segs.length - 1; else break;
    [segs, rgbs, polys, centres].forEach(arr => arr.splice(k, 1));
  }
  return { segs, rgbs, polys, centres };
}
function swatchPolygon(poly, xs, offset, halfH, c0, c1, n) {
  n = n || 12;
  const pts = [], bot = [];
  for (let i = 0; i < n; i++) {
    const idx = Math.min(xs.length - 1, Math.max(0, Math.round(c0 + (c1 - c0) * i / (n - 1))));
    const x = xs[idx], y = _polyval(poly, x) + offset;
    pts.push([x, y - halfH]); bot.push([x, y + halfH]);
  }
  return pts.concat(bot.reverse());
}
function meanColour(bandBgr, c0, c1) {
  const H = bandBgr.rows, W = bandBgr.cols, D = bandBgr.data;
  const y0 = Math.floor(0.3 * H), y1 = Math.min(H - 1, Math.floor(0.7 * H));
  const x0 = Math.floor(c0 + 0.25 * (c1 - c0)), x1 = Math.min(W - 1, Math.floor(c1 - 0.25 * (c1 - c0)));
  let sb = 0, sg = 0, sr = 0, n = 0;
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) {
      const i = (y * W + x) * 3;
      sb += D[i]; sg += D[i + 1]; sr += D[i + 2]; n++;
    }
  if (!n) return [0, 0, 0];
  return [Math.round(sr / n), Math.round(sg / n), Math.round(sb / n)];   // RGB
}

// ---- public: detect the calibration strips --------------------------------
const CB_WORK_MAX = 1400;      // longest side actually analysed

function detectColourStrips(bgr0) {
  const diag = window.__cbDiag = { seeds: 0, thr: 0, inliers: 0, candidates: 0,
                                   strips: 0, counts: [], scale: 1,
                                   stage: "", reason: "" };
  // Work at a capped resolution. Every stage here walks pixels in JS, and a
  // 12 MP photo is minutes of that; the swatches are centimetres across so
  // they survive the downscale easily. Geometry is scaled back at the end.
  const f = Math.min(1, CB_WORK_MAX / Math.max(bgr0.cols, bgr0.rows));
  let bgr = bgr0;
  if (f < 1) {
    bgr = new cv.Mat();
    cv.resize(bgr0, bgr, new cv.Size(Math.round(bgr0.cols * f), Math.round(bgr0.rows * f)),
              0, 0, cv.INTER_AREA);
  }
  diag.scale = f;
  const flat = deshadow(bgr);
  const vs = vividSeeds(flat);
  diag.seeds = vs.seeds.length; diag.thr = vs.thr;
  if (vs.seeds.length < 3) {
    flat.delete(); if(bgr!==bgr0) bgr.delete(); diag.stage = "seeds";
    diag.reason = "No colour swatches found. This needs a photo with the "
                + "calibration card's colour strips in frame.";
    return { strips: [], flat: null };
  }
  const r = ransacAxis(vs.seeds);
  if (!r) { flat.delete(); if(bgr!==bgr0) bgr.delete(); diag.stage = "axis";
    diag.reason = "Found colour patches but they do not line up as a strip.";
    return { strips: [], flat: null }; }
  diag.inliers = r.pts.length;
  const poly = fitCentreline(r.pts, 2);
  if (!poly) { flat.delete(); if(bgr!==bgr0) bgr.delete(); diag.stage = "axis";
    diag.reason = "Could not fit the strip axis."; return { strips: [], flat: null }; }

  const swH = _sMed(vs.seeds.map(s => s.h));
  const pad = 12 * swH;
  const xsIn = r.pts.map(p => p[0]);
  const x0 = Math.max(0, Math.min(...xsIn) - pad);
  const x1 = Math.min(bgr.cols - 1, Math.max(...xsIn) + pad);

  const found = findStrips(flat, poly, x0, x1, swH);
  diag.candidates = found.length;
  const accepted = [], strips = [];
  for (const f of found) {
    const rb = refineBand(flat, poly, x0, x1, f.off, swH);
    let off = rb.offset, half = rb.half, polyS = poly, xa = x0, xb = x1;
    const sc = stripCentreline(flat, poly, x0, x1, off, swH);
    if (sc) {
      polyS = sc.poly; half = sc.half;
      xa = Math.max(x0, sc.x0 - 1.4 * swH);
      xb = Math.min(x1, sc.x1 + 1.4 * swH);
      off = 0;
    }
    const xm = 0.5 * (xa + xb), ym = _polyval(polyS, xm);
    if (accepted.some(a => Math.abs(_polyval(a.poly, xm) - ym) < 0.9 * (half + a.half))) continue;

    const sb = sampleBand(flat, polyS, xa, xb, off, half);
    const ccols = cardExtent(flat, polyS, xa, xb, off, half);
    let segs = splitSwatches(sb.band, ccols);
    if (segs.length < 3) { sb.band.delete(); continue; }

    let polys = [], rgbs = [], centres = [];
    for (const [c0, c1] of segs) {
      polys.push(swatchPolygon(polyS, sb.xs, off, half, c0, c1));
      const mi = Math.min(Math.floor((c0 + c1) / 2), sb.xs.length - 1);
      centres.push([sb.xs[mi], sb.yc[mi]]);
      rgbs.push(meanColour(sb.band, c0, c1));
    }
    const tr = trimEnds(segs, rgbs, polys, centres);
    segs = tr.segs; rgbs = tr.rgbs; polys = tr.polys; centres = tr.centres;
    // same colour-diversity gate as the reference, or skin bands and
    // three-segment fragments would poison the result
    if (!stripIsColourful(rgbs).ok) { sb.band.delete(); continue; }

    accepted.push({ poly: polyS, half });
    strips.push({ poly: polyS, half, off, xa, xb, segs, polys, centres, rgbs });
    sb.band.delete();
  }
  // geometry back to original-image pixels
  if (f < 1) {
    const inv = 1 / f;
    strips.forEach(s => {
      s.polys = s.polys.map(p => p.map(pt => [pt[0] * inv, pt[1] * inv]));
      s.centres = s.centres.map(c => [c[0] * inv, c[1] * inv]);
    });
  }
  if (bgr !== bgr0) bgr.delete();
  diag.strips = strips.length;
  diag.counts = strips.map(s => s.segs.length);
  if (!strips.length) {
    diag.stage = "strips";
    diag.reason = "Found colour patches but none formed a usable strip of "
                + "distinct swatches.";
  } else {
    diag.stage = "ok";
    diag.reason = `${strips.length} strip${strips.length === 1 ? "" : "s"}: `
                + strips.map(s => s.segs.length + " swatches").join(", ") + ".";
  }
  return { strips, flat };
}
