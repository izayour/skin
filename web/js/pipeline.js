/* Classical lesion segmentation ported from lesion_detector_interactive.py.
   Works on a cv.Mat (RGBA, as read from a canvas). All intermediate Mats are
   explicitly freed — OpenCV.js has no garbage collection. */

// ---- small helpers -------------------------------------------------------
function cntPoints(cnt) {                 // contour Mat (CV_32SC2) -> [[x,y],...]
  const d = cnt.data32S, p = [];
  for (let i = 0; i < d.length; i += 2) p.push([d[i], d[i + 1]]);
  return p;
}
function pointsToContour(pts) {           // [[x,y],...] -> contour Mat
  const flat = [];
  for (const [x, y] of pts) flat.push(Math.round(x), Math.round(y));
  return cv.matFromArray(pts.length, 1, cv.CV_32SC2, flat);
}
function fillPoly(mask, pts, val) {       // fill one polygon into an 8U mask
  const c = pointsToContour(pts);
  const mv = new cv.MatVector(); mv.push_back(c);
  cv.drawContours(mask, mv, 0, new cv.Scalar(val), -1);
  c.delete(); mv.delete();
}
function polyArea(pts) {                   // shoelace |area|
  let a = 0;
  for (let i = 0, n = pts.length; i < n; i++) {
    const [x1, y1] = pts[i], [x2, y2] = pts[(i + 1) % n];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}
function ellipseKernel(n) {
  return cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(n, n));
}

// ---- segmentation stages (mirror the Python) ----------------------------
function localDarkness(gray, k) {
  const bg = new cv.Mat();
  if (k <= 255) {
    cv.medianBlur(gray, bg, k | 1);
  } else {                                 // downscale-median-upscale (large k)
    const s = 101.0 / k;
    const sw = Math.max(1, Math.round(gray.cols * s));
    const sh = Math.max(1, Math.round(gray.rows * s));
    const small = new cv.Mat(), sb = new cv.Mat();
    cv.resize(gray, small, new cv.Size(sw, sh), 0, 0, cv.INTER_AREA);
    cv.medianBlur(small, sb, 101);
    cv.resize(sb, bg, new cv.Size(gray.cols, gray.rows), 0, 0, cv.INTER_LINEAR);
    small.delete(); sb.delete();
  }
  const dark = new cv.Mat(), norm = new cv.Mat();
  cv.subtract(bg, gray, dark);
  cv.normalize(dark, norm, 0, 255, cv.NORM_MINMAX, cv.CV_8U);
  bg.delete(); dark.delete();
  return norm;
}

function cleanMask(mask) {                 // close x2 then open x1
  const k = ellipseKernel(7), out = new cv.Mat(), tmp = new cv.Mat();
  const anchor = new cv.Point(-1, -1);
  cv.morphologyEx(mask, tmp, cv.MORPH_CLOSE, k, anchor, 2);
  cv.morphologyEx(tmp, out, cv.MORPH_OPEN, k, anchor, 1);
  k.delete(); tmp.delete();
  return out;
}

function initialMask(darkness) {
  const otsu = new cv.Mat(), adap = new cv.Mat(), out = new cv.Mat();
  cv.threshold(darkness, otsu, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  cv.adaptiveThreshold(darkness, adap, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                       cv.THRESH_BINARY, 51, -5);
  cv.bitwise_or(otsu, adap, out);
  otsu.delete(); adap.delete();
  return out;
}

function smoothMask(mask) {                // round jagged staircase edges
  const blur = new cv.Mat(), out = new cv.Mat();
  cv.GaussianBlur(mask, blur, new cv.Size(9, 9), 0);
  cv.threshold(blur, out, 127, 255, cv.THRESH_BINARY);
  blur.delete();
  return out;
}

// pick the most lesion-like blob: weighted size + compactness + centrality
function pickBlob(mask, opts) {
  const h = mask.rows, w = mask.cols, imgArea = h * w;
  const cx0 = w / 2, cy0 = h / 2, diag = Math.hypot(cx0, cy0);
  const minArea = Math.max(80, opts.minAreaFrac * imgArea);
  const contours = new cv.MatVector(), hier = new cv.Mat();
  cv.findContours(mask, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  let best = null, bestScore = -1;
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const area = cv.contourArea(c), frac = area / imgArea;
    if (area >= minArea && frac <= opts.maxAreaFrac) {
      const perim = cv.arcLength(c, true) + 1e-6;
      const compactness = 4 * Math.PI * area / (perim * perim);
      const m = cv.moments(c);
      const cx = m.m10 / (m.m00 + 1e-6), cy = m.m01 / (m.m00 + 1e-6);
      const centrality = 1 - Math.hypot(cx - cx0, cy - cy0) / diag;
      const score = (1 - opts.compactnessWeight - opts.centerWeight) * frac
                  + opts.compactnessWeight * compactness
                  + opts.centerWeight * centrality;
      if (score > bestScore) { bestScore = score; best = cntPoints(c); }
    }
    c.delete();
  }
  contours.delete(); hier.delete();
  return best;                             // [[x,y],...] in ROI coords, or null
}

function largestContour(mask) {            // -> points (ROI coords) or null
  const contours = new cv.MatVector(), hier = new cv.Mat();
  cv.findContours(mask, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  let best = null, bestA = -1;
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i), a = cv.contourArea(c);
    if (a > bestA) { bestA = a; best = cntPoints(c); }
    c.delete();
  }
  contours.delete(); hier.delete();
  return best;
}

// local Otsu inside the blob's padded bbox, then cleaned
function localOtsuRefine(darkness, blobPts) {
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
  for (const [x, y] of blobPts) {
    minX = Math.min(minX, x); minY = Math.min(minY, y);
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
  }
  const pad = 20;
  const x0 = Math.max(minX - pad, 0), y0 = Math.max(minY - pad, 0);
  const x1 = Math.min(maxX + pad, darkness.cols), y1 = Math.min(maxY + pad, darkness.rows);
  const rect = new cv.Rect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
  const window = darkness.roi(rect).clone();
  const th = new cv.Mat();
  cv.threshold(window, th, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  const cleaned = cleanMask(th);
  const refined = cv.Mat.zeros(darkness.rows, darkness.cols, cv.CV_8UC1);
  const dstRoi = refined.roi(rect);
  cleaned.copyTo(dstRoi);
  window.delete(); th.delete(); cleaned.delete(); dstRoi.delete();
  return refined;
}

// ---- public entry point --------------------------------------------------
// src: cv.Mat RGBA (full image). roi: {x,y,w,h} full-image px. bgKsize: int.
// returns {points:[[x,y]... full coords], areaPx, centroid:[x,y], bbox:[x,y,w,h]} or null
function segmentLesion(src, roi, bgKsize, opts) {
  opts = Object.assign({ blur: 5, minAreaFrac: 0.002, maxAreaFrac: 0.95,
                         compactnessWeight: 0.35, centerWeight: 0.25 }, opts || {});
  const rect = new cv.Rect(roi.x, roi.y, roi.w, roi.h);
  const work = src.roi(rect).clone();
  const rgb = new cv.Mat(), lab = new cv.Mat(), chans = new cv.MatVector();
  cv.cvtColor(work, rgb, cv.COLOR_RGBA2RGB);
  cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
  cv.split(lab, chans);
  const gray = new cv.Mat();
  cv.GaussianBlur(chans.get(0), gray, new cv.Size(opts.blur, opts.blur), 0);

  const darkness = localDarkness(gray, bgKsize);
  const im = initialMask(darkness);
  const seed = cleanMask(im);
  im.delete();

  let result = null;
  const blob = pickBlob(seed, opts);
  if (blob) {
    const refined = localOtsuRefine(darkness, blob);
    let cpts = largestContour(refined);
    if (cpts) {
      const filled = cv.Mat.zeros(refined.rows, refined.cols, cv.CV_8UC1);
      fillPoly(filled, cpts, 255);
      const smooth = smoothMask(filled);
      const finalPts = largestContour(smooth);
      if (finalPts && cv.countNonZero(smooth) > 0) {
        // offset ROI-local -> full-image coords
        const pts = finalPts.map(([x, y]) => [x + roi.x, y + roi.y]);
        const area = polyArea(pts);
        let sx = 0, sy = 0;
        for (const [x, y] of pts) { sx += x; sy += y; }
        const cx = Math.round(sx / pts.length), cy = Math.round(sy / pts.length);
        let mnx = 1e9, mny = 1e9, mxx = -1e9, mxy = -1e9;
        for (const [x, y] of pts) {
          mnx = Math.min(mnx, x); mny = Math.min(mny, y);
          mxx = Math.max(mxx, x); mxy = Math.max(mxy, y);
        }
        result = { points: pts, areaPx: area, centroid: [cx, cy],
                   bbox: [mnx, mny, mxx - mnx, mxy - mny] };
      }
      filled.delete(); smooth.delete();
    }
    refined.delete();
  }

  work.delete(); rgb.delete(); lab.delete(); chans.delete();
  gray.delete(); darkness.delete(); seed.delete();
  return result;
}
