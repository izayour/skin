/* Generic ruler calibration — for a plain ruler with mm/cm markings and no
   coloured swatch strip.

   Deliberately separate from ruler.js, which localises one specific printed
   card (paperness -> perspective warp -> tick band above the first colourful
   row). None of that applies here. ruler.js is not touched by this file.

   The first version of this hunted for tick-like blobs across the whole area
   around the lesion and hoped ticks would outvote everything else. Measured on
   real photos they do not: on rl3 every surviving candidate lay inside the
   lesion (hair and texture), none on the ruler. So the order is inverted here:

     1. find the ruler as a region  — bright, desaturated, elongated
     2. find marks only inside it   — skin texture is excluded by construction
     3. fit a lattice to the marks  — tolerates missed marks
     4. read the unit off the mark LENGTHS — every 10th mark is the long one,
        so the number of steps between long marks is how many marks make a
        centimetre. That settles mm-vs-cm from the image instead of guessing
        from spacing arithmetic, where a factor of 2 is a 4x area error.
*/

const GEN_MIN_TICKS = 10;      // marks needed before a run is trusted
const GEN_GAP_TOL = 0.25;      // a mark may sit this far off its lattice slot
const GEN_WORK_MAX = 1200;     // longest side actually analysed, in pixels

// Search region: a plain ruler is laid beside the lesion, so look in a
// generous box around it rather than only below, as the card detector does.
function nearLesionRoi(imgW, imgH, bbox, factor) {
  const [lx, ly, lw, lh] = bbox;
  const f = factor || 8;
  const cx = lx + lw / 2, cy = ly + lh / 2;
  // A multiple of the lesion alone is not enough: a small mole with the ruler
  // laid a couple of centimetres away would put the ruler outside the box, and
  // there is no scale yet to reason in millimetres.
  const half = Math.max(Math.max(lw, lh) * f / 2, 0.5 * Math.min(imgW, imgH));
  return {
    x: Math.max(0, Math.round(cx - half)), y: Math.max(0, Math.round(cy - half)),
    w: Math.min(imgW, Math.round(cx + half)) - Math.max(0, Math.round(cx - half)),
    h: Math.min(imgH, Math.round(cy + half)) - Math.max(0, Math.round(cy - half))
  };
}

const medianOf = a => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

/* ---- stage 1: where is the ruler? --------------------------------------
   A ruler body is bright and close to colourless, and skin is neither. Both
   tests are relative to this photo, so exposure and skin tone do not matter.
   Returns a mask of the chosen region plus its long axis. */
function findRulerRegion(cropRGBA, roiW, roiH) {
  const rgb = new cv.Mat(), hsv = new cv.Mat(), ch = new cv.MatVector();
  cv.cvtColor(cropRGBA, rgb, cv.COLOR_RGBA2RGB);
  cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
  cv.split(hsv, ch);
  const S = ch.get(1), V = ch.get(2);
  const bright = new cv.Mat(), plain = new cv.Mat(), mask = new cv.Mat();
  cv.threshold(V, bright, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  cv.threshold(S, plain, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
  cv.bitwise_and(bright, plain, mask);
  // close over the marks so the body reads as one region
  const kk = Math.max(3, Math.round(Math.min(roiW, roiH) / 40)) | 1;
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kk, kk));
  const closed = new cv.Mat();
  cv.morphologyEx(mask, closed, cv.MORPH_CLOSE, k, new cv.Point(-1, -1), 2);

  const labels = new cv.Mat(), stats = new cv.Mat(), cent = new cv.Mat();
  const n = cv.connectedComponentsWithStats(closed, labels, stats, cent, 8);
  const frame = roiW * roiH;
  let bestIdx = -1, bestScore = 0;
  for (let i = 1; i < n; i++) {
    const w = stats.intAt(i, 2), h = stats.intAt(i, 3), a = stats.intAt(i, 4);
    if (a < 0.02 * frame) continue;
    // a ruler is long and straight: prefer big and elongated
    const elong = Math.max(w, h) / Math.max(1, Math.min(w, h));
    const score = a * Math.min(elong, 6);
    if (score > bestScore) { bestScore = score; bestIdx = i; }
  }
  let out = null;
  if (bestIdx > 0) {
    const region = new cv.Mat();
    const want = new cv.Mat(labels.rows, labels.cols, labels.type(),
                            new cv.Scalar(bestIdx));
    cv.compare(labels, want, region, cv.CMP_EQ);
    want.delete();
    // long axis from the region's own minimum-area rectangle
    const cnts = new cv.MatVector(), hi = new cv.Mat();
    cv.findContours(region, cnts, hi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let big = null, bigA = -1;
    for (let i = 0; i < cnts.size(); i++) {
      const c = cnts.get(i), a = cv.contourArea(c);
      if (a > bigA) { bigA = a; if (big) big.delete(); big = c; } else c.delete();
    }
    if (big) {
      const rr = cv.minAreaRect(big);
      let ang = rr.angle * Math.PI / 180;
      if (rr.size.width < rr.size.height) ang += Math.PI / 2;   // along the long side
      out = { mask: region, ux: Math.cos(ang), uy: Math.sin(ang),
              area: bigA, frac: bigA / frame };
      big.delete();
    } else region.delete();
    cnts.delete(); hi.delete();
  }
  rgb.delete(); hsv.delete(); ch.delete(); S.delete(); V.delete();
  bright.delete(); plain.delete(); mask.delete(); closed.delete(); k.delete();
  labels.delete(); stats.delete(); cent.delete();
  return out;
}

/* ---- stage 2: marks inside the ruler, as a 1-D profile -----------------
   Connected components are the wrong tool here. On a real ruler the marks are
   a dense comb, and once the millimetre lines touch or blur together the
   components merge into one long blob that every shape filter then rejects --
   measured on rl2, the tick band yielded ZERO components while the skin above
   it yielded hundreds.

   Instead, walk the ruler pixels once and accumulate darkness into bins along
   the ruler's own axis. Marks become peaks in that 1-D signal whether or not
   they touch, and the vertical spread of dark pixels in each bin gives the
   mark's length for free -- which is what identifies centimetre marks later. */
function markProfile(gray, rulerMask, ux, uy, roiW, roiH) {
  const nx = -uy, ny = ux;                       // across the ruler
  // bin along the axis at 1px; track darkness and the across-ruler extent
  const corners = [[0, 0], [roiW, 0], [0, roiH], [roiW, roiH]];
  let tMin = Infinity, tMax = -Infinity;
  for (const [x, y] of corners) {
    const t = x * ux + y * uy; tMin = Math.min(tMin, t); tMax = Math.max(tMax, t);
  }
  const nBins = Math.max(16, Math.ceil(tMax - tMin) + 1);
  const dark = new Float64Array(nBins);
  const sMin = new Float64Array(nBins).fill(Infinity);
  const sMax = new Float64Array(nBins).fill(-Infinity);
  const cnt = new Float64Array(nBins);
  const xs = new Float64Array(nBins), ys = new Float64Array(nBins);

  // Direct typed-array access: ucharAt() per pixel is an emscripten call and
  // hangs outright on a multi-megapixel crop.
  const G = gray.data, M = rulerMask.data, stride = gray.cols;

  // mean brightness of the ruler, to measure "how dark" relative to the body
  let sum = 0, num = 0;
  for (let y = 0; y < roiH; y++) { const row = y * stride;
    for (let x = 0; x < roiW; x++) {
      if (!M[row + x]) continue;
      sum += G[row + x]; num++;
    } }
  if (!num) return { bins: dark, tMin, nBins, sMin, sMax, cnt, xs, ys, ok: false };
  const mean = sum / num;

  for (let y = 0; y < roiH; y++) { const row = y * stride;
  for (let x = 0; x < roiW; x++) {
    if (!M[row + x]) continue;
    const g = G[row + x];
    if (g >= mean * 0.85) continue;              // not ink
    const t = x * ux + y * uy, s = x * nx + y * ny;
    const b = Math.round(t - tMin);
    if (b < 0 || b >= nBins) continue;
    dark[b] += (mean - g);
    cnt[b] += 1; xs[b] += x; ys[b] += y;
    if (s < sMin[b]) sMin[b] = s;
    if (s > sMax[b]) sMax[b] = s;
  } }
  return { bins: dark, tMin, nBins, sMin, sMax, cnt, xs, ys, ok: true };
}

// local maxima of the darkness profile -> one entry per ruler mark
function peaksFromProfile(prof) {
  const { bins, nBins, cnt } = prof;
  let peak = 0;
  for (let i = 0; i < nBins; i++) peak = Math.max(peak, bins[i]);
  if (peak <= 0) return [];
  const floor = peak * 0.15;                     // ignore paper grain
  const out = [];
  for (let i = 1; i < nBins - 1; i++) {
    if (bins[i] < floor) continue;
    if (bins[i] < bins[i - 1] || bins[i] < bins[i + 1]) continue;
    // centre of mass over the local bump, so 2px-wide marks are not biased
    let w = 0, wt = 0, lo = i, hi = i;
    while (lo > 0 && bins[lo - 1] > floor && bins[lo - 1] <= bins[lo]) lo--;
    while (hi < nBins - 1 && bins[hi + 1] > floor && bins[hi + 1] <= bins[hi]) hi++;
    for (let j = lo; j <= hi; j++) { w += bins[j]; wt += bins[j] * j; }
    if (!w) continue;
    const c = wt / w;
    if (out.length && c - out[out.length - 1].t < 1) continue;   // same bump
    let len = 0, px = 0, py = 0, n = 0;
    for (let j = lo; j <= hi; j++) {
      if (prof.sMax[j] > -Infinity) len = Math.max(len, prof.sMax[j] - prof.sMin[j]);
      if (cnt[j]) { px += prof.xs[j]; py += prof.ys[j]; n += cnt[j]; }
    }
    if (!n) continue;
    out.push({ t: c + prof.tMin, len, x: px / n, y: py / n });
    i = hi;
  }
  return out;
}

/* ---- stage 3: fit a regular lattice to the projected positions ---------
   A missing mark is just an unused index, so holes do not end the fit. */
function fitTickLattice(sorted) {
  if (sorted.length < GEN_MIN_TICKS) return null;
  const gaps = [];
  for (let i = 1; i < sorted.length; i++) gaps.push(sorted[i] - sorted[i - 1]);
  const g0 = medianOf(gaps);
  if (!(g0 > 0)) return null;
  const single = gaps.filter(g => g > 0.6 * g0 && g < 1.4 * g0);
  const sp0 = single.length ? medianOf(single) : g0;

  const span = sorted[sorted.length - 1] - sorted[0];
  const seeds = [];
  for (const div of [1, 2, 3, 4]) if (sp0 / div >= 2) seeds.push(sp0 / div);
  if (span > 0) seeds.push(span / (sorted.length - 1));

  const fits = [];
  for (const seed of seeds) {
    for (let a = 0; a < sorted.length; a++) {
      let t0 = sorted[a], cur = seed, keep = null;
      for (let iter = 0; iter < 5; iter++) {
        keep = [];
        for (const t of sorted) {
          const kk = Math.round((t - t0) / cur);
          if (Math.abs(t - (t0 + kk * cur)) <= GEN_GAP_TOL * cur) keep.push({ k: kk, t });
        }
        if (keep.length < GEN_MIN_TICKS) break;
        const m = keep.length;
        let sk = 0, st = 0, skk = 0, skt = 0;
        for (const q of keep) { sk += q.k; st += q.t; skk += q.k * q.k; skt += q.k * q.t; }
        const den = m * skk - sk * sk;
        if (!den) break;
        const B = (m * skt - sk * st) / den;
        if (!(B > 0)) break;
        t0 = (st - B * sk) / m; cur = B;
      }
      if (keep && keep.length >= GEN_MIN_TICKS) fits.push({ gap: cur, keep });
    }
  }
  if (!fits.length) return null;
  // Coarsest lattice that still explains most marks: halving a spacing fits
  // everything the original did, so maximising inliers alone always drifts to
  // the finest subdivision and reports a fraction of the true scale.
  const most = Math.max(...fits.map(f => f.keep.length));
  const good = fits.filter(f => f.keep.length >= 0.8 * most);
  good.sort((x, y) => y.gap - x.gap);
  const best = good[0];
  const ks = best.keep.map(q => q.k);
  return { gap: best.gap, keep: best.keep,
           occupancy: best.keep.length / (Math.max(...ks) - Math.min(...ks) + 1),
           explained: best.keep.length / sorted.length };
}

/* ---- stage 4: how many marks make a centimetre? ------------------------
   Rulers mark every 10th division with the longest line. The number of
   lattice steps between long marks is therefore marks-per-centimetre, which
   settles the unit from the picture rather than from a guess. */
function marksPerCm(keep) {
  const lens = keep.map(q => q.len).filter(v => v > 0);
  if (lens.length < GEN_MIN_TICKS) return null;
  const sorted = [...lens].sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.25)];
  const hi = sorted[Math.floor(sorted.length * 0.95)];
  if (!(hi > lo * 1.25)) return null;              // all marks alike: no pattern
  const cut = (lo + hi) / 2;
  const longKs = keep.filter(q => q.len >= cut).map(q => q.k).sort((a, b) => a - b);
  if (longKs.length < 2) return null;
  const steps = [];
  for (let i = 1; i < longKs.length; i++) {
    const d = longKs[i] - longKs[i - 1];
    if (d > 0) steps.push(d);
  }
  if (!steps.length) return null;
  const p = medianOf(steps);
  // only trust a plausible ruler period, and only if it actually repeats
  if (![2, 4, 5, 10].includes(p)) return null;
  const agree = steps.filter(s => s === p).length / steps.length;
  if (agree < 0.6) return null;
  return { perCm: p, agree, longCount: longKs.length };
}

/* ---- public entry point ------------------------------------------------
   src: full RGBA Mat. roi: {x,y,w,h}.
   -> { pxPerCm, pxPerTick, marksPerCm, count, ticks:[[x,y]...], roi } or null.
   Diagnostics for the UI land in window.__genDiag either way. */
function detectGenericRuler(src, roi) {
  const diag = window.__genDiag = { roi: [roi.x, roi.y, roi.w, roi.h],
    rulerFound: false, rulerFrac: 0, marks: 0, onLattice: 0, need: GEN_MIN_TICKS,
    occupancy: 0, explained: 0, perCm: null, stage: "", reason: "" };

  const rect = new cv.Rect(roi.x, roi.y, roi.w, roi.h);
  const tmp = src.roi(rect); const full = tmp.clone(); tmp.delete();
  // Work at a capped resolution. A phone photo's ROI is millions of pixels and
  // the profile pass walks every one of them; ruler marks stay resolvable well
  // below full size, and every measurement is scaled back at the end.
  const scale = Math.min(1, GEN_WORK_MAX / Math.max(roi.w, roi.h));
  let crop;
  if (scale < 1) {
    crop = new cv.Mat();
    cv.resize(full, crop, new cv.Size(Math.max(8, Math.round(roi.w * scale)),
                                      Math.max(8, Math.round(roi.h * scale))),
              0, 0, cv.INTER_AREA);
    full.delete();
  } else crop = full;
  const cw = crop.cols, chh = crop.rows;
  const gray = new cv.Mat();
  cv.cvtColor(crop, gray, cv.COLOR_RGBA2GRAY);

  const ruler = findRulerRegion(crop, cw, chh);
  if (!ruler) {
    crop.delete(); gray.delete();
    diag.stage = "ruler";
    diag.reason = "Couldn't find a ruler near the lesion. The ruler needs to be "
                + "in the same photo, reasonably flat and brighter than the skin.";
    return null;
  }
  diag.rulerFound = true; diag.rulerFrac = +ruler.frac.toFixed(2);

  const prof = markProfile(gray, ruler.mask, ruler.ux, ruler.uy, cw, chh);
  const dedup = prof.ok ? peaksFromProfile(prof) : [];
  diag.marks = dedup.length;
  crop.delete(); gray.delete(); ruler.mask.delete();
  if (dedup.length < GEN_MIN_TICKS) {
    diag.stage = "marks";
    diag.reason = `Found the ruler but only ${dedup.length} mark${dedup.length === 1 ? "" : "s"} `
                + `on it (need ${GEN_MIN_TICKS}). Get closer, or make sure the markings are in focus.`;
    return null;
  }

  const lat = fitTickLattice(dedup.map(q => q.t));
  if (!lat) {
    diag.stage = "spacing";
    diag.reason = `${dedup.length} marks were found on the ruler, but fewer than `
                + `${GEN_MIN_TICKS} sit at a regular spacing. Blurred or partly hidden `
                + `markings will do this — try a sharper, straighter view.`;
    return null;
  }
  diag.onLattice = lat.keep.length;
  diag.occupancy = +lat.occupancy.toFixed(2);
  diag.explained = +lat.explained.toFixed(2);

  // attach each lattice member's mark length, for the period test
  const byT = new Map(dedup.map(q => [q.t, q]));
  const keep = lat.keep.map(q => { const p = byT.get(q.t) || {};
    return { k: q.k, t: q.t, len: p.len || 0, m: p }; });
  const per = marksPerCm(keep);
  diag.perCm = per ? per.perCm : null;

  // Confidence first: a shaky lattice must not be offered to the user at all,
  // whether or not the unit could be read. Otherwise a 54%-occupancy fit gets
  // presented as a spacing to confirm, and confirming it bakes in the error.
  if (lat.explained < 0.85 || lat.occupancy < 0.6) {
    diag.stage = "ambiguous";
    diag.reason = `Found ${dedup.length} marks on the ruler but couldn't pin the spacing `
                + `confidently (${Math.round(lat.explained * 100)}% fit one spacing, `
                + `${Math.round(lat.occupancy * 100)}% of positions filled). A wrong guess here `
                + `would be off by a whole factor, so it's safer to set the scale by hand.`;
    return null;
  }

  if (!per) {
    // No usable long/short pattern -> the unit is genuinely unknown from the
    // image. Report the spacing and let the caller ask, rather than guess.
    diag.stage = "unit";
    diag.reason = `Found ${lat.keep.length} evenly spaced marks (${lat.gap.toFixed(1)} px apart) `
                + `but couldn't tell millimetre marks from centimetre marks — the long `
                + `and short markings don't form a clear pattern.`;
    return { pxPerTick: lat.gap / scale, pxPerCm: null, marksPerCm: null,
             count: lat.keep.length, unitUnknown: true, roi, ticks: toFull(keep) };
  }

  diag.stage = "ok";
  diag.reason = `${lat.keep.length} marks, ${per.perCm} per centimetre.`;
  return { pxPerTick: lat.gap / scale, pxPerCm: (lat.gap * per.perCm) / scale,
           marksPerCm: per.perCm, count: lat.keep.length, roi, ticks: toFull(keep) };

  // measurements are made on the scaled crop; report them in full-image pixels
  function toFull(list) {
    return list.filter(q => q.m && q.m.x != null)
               .map(q => [q.m.x / scale + roi.x, q.m.y / scale + roi.y]);
  }
}
