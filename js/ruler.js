/* Ruler / centimetre-tick detection ported from ruler_detector_interactive.py
   (+ the app's 2-nearest-cm-tick calibration and blur guard).
   Lean port: only what's needed for the mm scale — no swatches/digits. */

const BLUR_THRESHOLD = 0.18;              // matches app.py

// ---- numeric helpers -----------------------------------------------------
function median(a) {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
// Otsu split of a small number array; returns boolean[] (true = value > threshold)
function otsuSplitArray(vals) {
  const mn = Math.min(...vals), mx = Math.max(...vals);
  if (mx <= mn) return vals.map(() => true);
  const norm = vals.map(v => Math.round((v - mn) / (mx - mn) * 255));
  const hist = new Array(256).fill(0);
  norm.forEach(v => hist[v]++);
  const total = norm.length;
  let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, maxVar = -1, thr = 0;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (!wB) continue;
    const wF = total - wB; if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) { maxVar = between; thr = t; }
  }
  return norm.map(v => v > thr);
}

// ---- ruler localization --------------------------------------------------
function paperness(rgb) {                  // -> 8U mask input (normalized score)
  const lab = new cv.Mat(), chans = new cv.MatVector();
  cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
  cv.split(lab, chans);
  const a = new cv.Mat(), b = new cv.Mat();
  chans.get(1).convertTo(a, cv.CV_32F);
  chans.get(2).convertTo(b, cv.CV_32F);
  // score = -(a-128) - (b-128) = 256 - a - b
  const score = new cv.Mat();
  cv.add(a, b, score);                     // a + b
  const c256 = new cv.Mat(score.rows, score.cols, cv.CV_32F, new cv.Scalar(256));
  cv.subtract(c256, score, score);         // 256 - (a+b)
  const norm = new cv.Mat();
  cv.normalize(score, norm, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
  lab.delete(); chans.delete(); a.delete(); b.delete(); score.delete(); c256.delete();
  return norm;
}

function cleanRuler(mask, ksize) {         // close x2, open x1 (ellipse ksize)
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(ksize, ksize));
  const t = new cv.Mat(), out = new cv.Mat(), anch = new cv.Point(-1, -1);
  cv.morphologyEx(mask, t, cv.MORPH_CLOSE, k, anch, 2);
  cv.morphologyEx(t, out, cv.MORPH_OPEN, k, anch, 1);
  k.delete(); t.delete();
  return out;
}

// collect points of ruler-like contours (big + rectangular), ROI-local
function pickRulerPoints(mask, opts) {
  const h = mask.rows, w = mask.cols, imgArea = h * w;
  const cx0 = w / 2, cy0 = h / 2, diag = Math.hypot(cx0, cy0);
  const minArea = Math.max(200, opts.minAreaFrac * imgArea);
  const contours = new cv.MatVector(), hier = new cv.Mat();
  cv.findContours(mask, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  const pts = [];
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const area = cv.contourArea(c), frac = area / imgArea;
    if (area >= minArea && frac <= opts.maxAreaFrac) {
      const rr = cv.minAreaRect(c);
      const rectArea = rr.size.width * rr.size.height;
      const rectangularity = rectArea > 0 ? area / rectArea : 0;
      if (rectangularity >= 0.35) {
        const d = c.data32S;
        for (let j = 0; j < d.length; j += 2) pts.push([d[j], d[j + 1]]);
      }
    }
    c.delete();
  }
  contours.delete(); hier.delete();
  return pts.length ? pts : null;
}

function orderCorners(p) {                  // p: [{x,y}*4] -> [tl,tr,br,bl]
  const s = p.map(q => q.x + q.y), d = p.map(q => q.y - q.x);
  const amin = a => a.indexOf(Math.min(...a)), amax = a => a.indexOf(Math.max(...a));
  return [p[amin(s)], p[amin(d)], p[amax(s)], p[amax(d)]];
}

function sharpnessOf(warpedRGBA) {          // normalized gradient energy
  const gray = new cv.Mat();
  cv.cvtColor(warpedRGBA, gray, cv.COLOR_RGBA2GRAY);
  const gn = new cv.Mat();
  const nw = Math.max(2, Math.round(gray.cols * 400 / gray.rows));
  cv.resize(gray, gn, new cv.Size(nw, 400));
  gn.convertTo(gn, cv.CV_64F);
  const gx = new cv.Mat(), gy = new cv.Mat();
  cv.Sobel(gn, gx, cv.CV_64F, 1, 0, 3);
  cv.Sobel(gn, gy, cv.CV_64F, 0, 1, 3);
  cv.multiply(gx, gx, gx); cv.multiply(gy, gy, gy);
  cv.add(gx, gy, gx);
  const meanGrad = cv.mean(gx)[0];
  const meanInt = cv.mean(gn)[0];
  gray.delete(); gn.delete(); gx.delete(); gy.delete();
  return meanGrad / (meanInt * meanInt + 1e-6);
}

// first "colorful" row (majority of pixels saturated) — bottom of the tick band
function firstColorfulRow(warpedRGB) {
  const hsv = new cv.Mat(), chans = new cv.MatVector();
  cv.cvtColor(warpedRGB, hsv, cv.COLOR_RGB2HSV);
  cv.split(hsv, chans);
  const sat = chans.get(1), colorful = new cv.Mat();
  cv.threshold(sat, colorful, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  const rowAvg = new cv.Mat();
  cv.reduce(colorful, rowAvg, 1, cv.REDUCE_AVG, cv.CV_32F);  // h x 1, 0..255
  const bin = new cv.Mat();
  cv.threshold(rowAvg, bin, 127.5, 255, cv.THRESH_BINARY);
  const closed = new cv.Mat();
  const k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, 7));
  cv.morphologyEx(bin, closed, cv.MORPH_CLOSE, k);
  const h = closed.rows;
  let y1 = 0; const d = closed.data32F;
  while (y1 < h && !(d[y1] > 0)) y1++;
  hsv.delete(); chans.delete(); sat.delete(); colorful.delete();
  rowAvg.delete(); bin.delete(); closed.delete(); k.delete();
  return { y1, h };
}

// black-ink tick components in the band above the swatch strip (warped coords)
function tickRowComponents(warpedRGB) {
  const { y1, h } = firstColorfulRow(warpedRGB);
  if (y1 < 0.02 * h) return [];
  const band = warpedRGB.roi(new cv.Rect(0, 0, warpedRGB.cols, y1)).clone();
  const hsv = new cv.Mat(), hch = new cv.MatVector();
  cv.cvtColor(band, hsv, cv.COLOR_RGB2HSV); cv.split(hsv, hch);
  const v = hch.get(2), dark = new cv.Mat();
  cv.adaptiveThreshold(v, dark, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                       cv.THRESH_BINARY_INV, 63, 25);
  const lab = new cv.Mat(), lch = new cv.MatVector();
  cv.cvtColor(band, lab, cv.COLOR_RGB2Lab); cv.split(lab, lch);
  const neutral = new cv.Mat();
  cv.threshold(lch.get(1), neutral, 132, 255, cv.THRESH_BINARY_INV); // a < 133
  const black = new cv.Mat();
  cv.bitwise_and(dark, neutral, black);
  const labels = new cv.Mat(), stats = new cv.Mat(), cent = new cv.Mat();
  const n = cv.connectedComponentsWithStats(black, labels, stats, cent, 8);
  const boxes = [];
  for (let i = 1; i < n; i++) {
    const x = stats.intAt(i, 0), y = stats.intAt(i, 1);
    const cw = stats.intAt(i, 2), ch = stats.intAt(i, 3);
    if (cw > 0 && ch > 0) boxes.push([x, y, cw, ch]);
  }
  band.delete(); hsv.delete(); hch.delete(); v.delete(); dark.delete();
  lab.delete(); lch.delete(); neutral.delete(); black.delete();
  labels.delete(); stats.delete(); cent.delete();
  return boxes;
}

function classifyLongTicks(boxes) {        // -> [x,y,w,h] long ticks, l-to-r
  const cand = boxes.filter(b => b[3] / b[2] >= 2.5);
  if (!cand.length) return [];
  let keep;
  if (cand.length === 1) keep = cand;
  else {
    const heights = cand.map(c => c[3]);
    const groupA = otsuSplitArray(heights);
    const meanA = mean(heights.filter((_, i) => groupA[i]));
    const meanB = mean(heights.filter((_, i) => !groupA[i]));
    const takeA = meanA >= meanB;
    keep = cand.filter((_, i) => groupA[i] === takeA);
  }
  return keep.sort((a, b) => a[0] - b[0]);
}
function mean(a) { return a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0; }

// apply a 3x3 perspective matrix (Mat CV_64F) to a point
function applyPersp(M, x, y) {
  const m = M.data64F;
  const w = m[6] * x + m[7] * y + m[8];
  return [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w];
}

// ---- public: detect ruler + cm ticks in an ROI --------------------------
// src: full RGBA Mat. roi:{x,y,w,h}. returns {corners:[4 full pts], sharpness,
//   cmTicks:[{corners:[4 full pts]}...]} or null.
function detectRuler(src, roi, opts) {
  opts = Object.assign({ minAreaFrac: 0.05, maxAreaFrac: 0.98, closeKsize: 25 }, opts || {});
  const rect = new cv.Rect(roi.x, roi.y, roi.w, roi.h);
  const crop = src.roi(rect).clone();
  const rgb = new cv.Mat(); cv.cvtColor(crop, rgb, cv.COLOR_RGBA2RGB);
  const den = new cv.Mat(); cv.bilateralFilter(rgb, den, 9, 75, 75, cv.BORDER_DEFAULT);
  const paper = paperness(den);
  const otsu = new cv.Mat();
  cv.threshold(paper, otsu, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  const mask = cleanRuler(otsu, opts.closeKsize | 1);

  let out = null;
  const pts = pickRulerPoints(mask, opts);
  if (pts) {
    const flat = []; for (const [x, y] of pts) flat.push(x, y);
    const pm = cv.matFromArray(pts.length, 1, cv.CV_32SC2, flat);
    const rr = cv.minAreaRect(pm);
    const box = cv.RotatedRect.points(rr);              // [{x,y}*4], ROI-local
    const full = box.map(p => ({ x: p.x + roi.x, y: p.y + roi.y }));
    const o = orderCorners(full);                       // [tl,tr,br,bl]
    const W = Math.round(Math.max(Math.hypot(o[1].x - o[0].x, o[1].y - o[0].y),
                                  Math.hypot(o[2].x - o[3].x, o[2].y - o[3].y)));
    const H = Math.round(Math.max(Math.hypot(o[3].x - o[0].x, o[3].y - o[0].y),
                                  Math.hypot(o[2].x - o[1].x, o[2].y - o[1].y)));
    if (W >= 2 && H >= 2) {
      const srcCorners = cv.matFromArray(4, 1, cv.CV_32FC2,
        [o[0].x, o[0].y, o[1].x, o[1].y, o[2].x, o[2].y, o[3].x, o[3].y]);
      const dstCorners = cv.matFromArray(4, 1, cv.CV_32FC2,
        [0, 0, W - 1, 0, W - 1, H - 1, 0, H - 1]);
      const M = cv.getPerspectiveTransform(srcCorners, dstCorners);
      const warped = new cv.Mat();
      cv.warpPerspective(src, warped, M, new cv.Size(W, H));
      const Minv = new cv.Mat(); cv.invert(M, Minv);
      const sharp = sharpnessOf(warped);
      const wrgb = new cv.Mat(); cv.cvtColor(warped, wrgb, cv.COLOR_RGBA2RGB);
      const longTicks = classifyLongTicks(tickRowComponents(wrgb));
      const cmTicks = longTicks.map(([x, y, w, h]) => ({
        corners: [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
          .map(([px, py]) => applyPersp(Minv, px, py))
      }));
      out = { corners: o.map(p => [p.x, p.y]), sharpness: sharp, cmTicks };
      srcCorners.delete(); dstCorners.delete(); M.delete(); Minv.delete();
      warped.delete(); wrgb.delete();
    }
    pm.delete();
  }
  crop.delete(); rgb.delete(); den.delete(); paper.delete(); otsu.delete(); mask.delete();
  return out;
}

// ---- 2 cm ticks nearest the lesion -> px/cm ------------------------------
function tickH(t) { const ys = t.corners.map(p => p[1]); return Math.max(...ys) - Math.min(...ys); }
function tickCx(t) { const xs = t.corners.map(p => p[0]); return xs.reduce((s, v) => s + v, 0) / xs.length; }

function twoCmTicksNear(cmTicks, lesionX) {
  if (cmTicks.length < 2) return null;
  const hmax = Math.max(...cmTicks.map(tickH));
  const cm = cmTicks.filter(t => tickH(t) >= 0.6 * hmax).sort((a, b) => tickCx(a) - tickCx(b));
  if (cm.length < 2) return null;
  const xs = cm.map(tickCx);
  const gaps = []; for (let i = 1; i < xs.length; i++) gaps.push(xs[i] - xs[i - 1]);
  const med = median(gaps);
  const pairs = [];
  for (let i = 0; i < gaps.length; i++)
    if (Math.abs(gaps[i] - med) <= 0.3 * med) pairs.push([cm[i], cm[i + 1], gaps[i]]);
  if (!pairs.length) return null;
  pairs.sort((p, q) =>
    Math.abs((tickCx(p[0]) + tickCx(p[1])) / 2 - lesionX) -
    Math.abs((tickCx(q[0]) + tickCx(q[1])) / 2 - lesionX));
  const [a, b, g] = pairs[0];
  return { a, b, pxPerCm: g };
}

function belowLesionRoi(imgW, imgH, bbox) {  // bbox [x,y,w,h]
  const [lx, ly, lw] = bbox, lh = bbox[3];
  const y0 = Math.min(ly + lh, imgH - 1);
  const x0 = Math.max(0, lx - 3 * lw);
  const x1 = Math.min(imgW, lx + 4 * lw);
  return { x: x0, y: y0, w: x1 - x0, h: imgH - y0 };
}
