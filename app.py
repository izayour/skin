"""Phone-facing web app for the lesion cascade.

Serves a mobile page where the user snaps/uploads a photo; the server runs
the same tiers as lesion_cascade.py with the user approving or rejecting
each result in the browser:
  1. U-Net (automatic)          -> Accept / Reject
  2. classical CV in a drawn box -> Accept / Reject
  3. hand-tapped lesion border   -> Accept / Redraw

Run:  python app.py   then open http://<this-PC's-IP>:5050 on the phone
(phone and PC on the same network; allow it through the firewall prompt).
"""
import base64
import io
import os
import uuid
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from lesion_cascade import detect_unet, UNET_WEIGHTS_DEFAULT, MIN_AREA_PX
from lesion_detector_interactive import LesionDetector
from ruler_detector_interactive import RulerDetector

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True  # else edits need a server restart

DISPLAY_MAX = 1000
STORE = {}  # sid -> {"image", "scale", "result", "contour", "method"}


def _store(image_bgr: np.ndarray) -> str:
    sid = uuid.uuid4().hex[:12]
    h, w = image_bgr.shape[:2]
    STORE[sid] = {"sid": sid, "image": image_bgr,
                  "scale": min(1.0, DISPLAY_MAX / float(max(h, w)))}
    return sid


def _stats(contour, px_per_cm=None) -> dict:
    area = cv2.contourArea(contour)
    M = cv2.moments(contour)
    cx = int(M["m10"] / (M["m00"] + 1e-6))
    cy = int(M["m01"] / (M["m00"] + 1e-6))
    st = {"area_px": float(area),
          "centroid": [cx, cy],
          "bbox": list(cv2.boundingRect(contour))}
    if px_per_cm:
        mm = 10.0 / px_per_cm
        st["px_per_cm"] = round(px_per_cm, 1)
        st["area_mm2"] = round(area * mm * mm, 1)
        st["equiv_diameter_mm"] = round(2.0 * np.sqrt(area / np.pi) * mm, 1)
        st["bbox_mm"] = [round(st["bbox"][2] * mm, 1),
                         round(st["bbox"][3] * mm, 1)]
    return st


def _jpeg_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _draw_lesion(overlay: np.ndarray, contour, method: str,
                 px_per_cm: Optional[float] = None) -> dict:
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)
    st = _stats(contour, px_per_cm)
    cx, cy = st["centroid"]
    cv2.circle(overlay, (cx, cy), 8, (0, 0, 255), -1)
    bx, by = st["bbox"][:2]
    # once calibrated, label the lesion with its real area (and diameter);
    # otherwise just tag which detector found it
    if px_per_cm:
        label = f"{st['area_mm2']} mm2  (D {st['equiv_diameter_mm']} mm)"
    else:
        label = method
    cv2.putText(overlay, label, (bx, max(by - 15, 34)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    return st


def _draw_tick_pair(overlay, pa, pb):
    """Draw the two cm-tick markers and the 1 cm segment joining them."""
    pa = (int(pa[0]), int(pa[1]))
    pb = (int(pb[0]), int(pb[1]))
    for p in (pa, pb):
        cv2.circle(overlay, p, 10, (0, 220, 255), -1)
    cv2.line(overlay, pa, pb, (0, 220, 255), 3)
    cv2.putText(overlay, "1 cm", (min(pa[0], pb[0]), min(pa[1], pb[1]) - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 3)


def _result_json(entry, overlay, px_per_cm):
    st = _stats(entry["contour"], px_per_cm)
    entry.update(result=overlay, px_per_cm=px_per_cm)
    s = entry["scale"]
    disp = cv2.resize(overlay, None, fx=s, fy=s) if s < 1.0 else overlay
    return {"ok": True, "id": entry["sid"], "method": entry["method"],
            "stats": st, "px_per_cm": round(px_per_cm, 1),
            "preview": _jpeg_b64(disp)}


def _respond_with_contour(sid: str, contour, method: str):
    entry = STORE[sid]
    overlay = entry["image"].copy()
    st = _draw_lesion(overlay, contour, method)
    entry.update(result=overlay, contour=contour, method=method,
                 px_per_cm=None)
    s = entry["scale"]
    disp = cv2.resize(overlay, None, fx=s, fy=s) if s < 1.0 else overlay
    return jsonify({"ok": True, "id": sid, "method": method, "stats": st,
                    "preview": _jpeg_b64(disp)})


def _below_lesion_roi(image, lesion_bbox):
    """A search box below the lesion, where the ruler usually sits.

    Generous horizontally (the ruler is wider than the lesion) and running
    to the bottom edge, so a loosely-placed sticker is still enclosed.
    """
    H, W = image.shape[:2]
    lx, ly, lw, lh = lesion_bbox
    y0 = min(ly + lh, H - 1)
    x0 = max(0, lx - 3 * lw)
    x1 = min(W, lx + 4 * lw)
    return (x0, y0, x1 - x0, H - y0)


def _tick_h(t):
    ys = [p[1] for p in t.corners]
    return max(ys) - min(ys)


def _tick_cx(t):
    xs = [p[0] for p in t.corners]
    return sum(xs) / len(xs)


def _two_cm_ticks_near(ruler, lesion_x):
    """Return the two adjacent centimeter ticks nearest the lesion.

    The detector's long-tick list can include half-centimeter ticks
    depending on the crop, which are spaced 0.5 cm apart and would halve
    the scale. Centimeter ticks are physically the tallest, so keep only
    the tallest cluster (>= 0.6 of the max tick height), then take the
    consecutive pair whose gap matches the median cm spacing (rejecting a
    doubled gap from a locally missed tick) and whose midpoint is closest
    to the lesion -- the nearest ticks are also the least perspective-
    distorted, so their spacing is the most trustworthy scale.
    """
    ticks = ruler.cm_ticks
    if len(ticks) < 2:
        return None
    hmax = max(_tick_h(t) for t in ticks)
    cm = sorted((t for t in ticks if _tick_h(t) >= 0.6 * hmax), key=_tick_cx)
    if len(cm) < 2:
        return None
    xs = [_tick_cx(t) for t in cm]
    gaps = np.diff(xs)
    med = float(np.median(gaps))
    pairs = [(cm[i], cm[i + 1], gaps[i]) for i in range(len(gaps))
             if abs(gaps[i] - med) <= 0.3 * med]
    if not pairs:
        return None
    a, b, g = min(pairs, key=lambda p: abs(
        (_tick_cx(p[0]) + _tick_cx(p[1])) / 2 - lesion_x))
    return a, b, float(g)


# Below this normalized-gradient-energy score the ruler crop is too out of
# focus to read ticks reliably (blurry small1-2 ~= 0.10; sharp rulers >= 0.32).
BLUR_THRESHOLD = 0.18


def _measure_with_ruler(entry, roi):
    """Detect the ruler in `roi` and calibrate from the 2 cm ticks nearest the
    lesion.

    Returns (result_dict, reason). reason is "ok" on success; otherwise
    result_dict is None and reason is one of "no_ruler", "no_ticks", or
    "blurry" (ruler located but too out of focus to read).
    """
    ruler = RulerDetector(verbose=False).detect(entry["image"], roi=roi)
    if ruler is None or not ruler.corners:
        return None, "no_ruler"
    blurry = ruler.sharpness < BLUR_THRESHOLD
    M = cv2.moments(entry["contour"])
    lesion_x = M["m10"] / (M["m00"] + 1e-6)
    pair = _two_cm_ticks_near(ruler, lesion_x)
    if pair is None:
        return None, ("blurry" if blurry else "no_ticks")
    a, b, px_per_cm = pair
    if px_per_cm <= 0:
        return None, ("blurry" if blurry else "no_ticks")

    overlay = entry["image"].copy()
    _draw_lesion(overlay, entry["contour"], entry["method"], px_per_cm)
    for tick in (a, b):
        pts = np.array(tick.corners, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 220, 255),
                     thickness=4)
    ca = (_tick_cx(a), np.mean([p[1] for p in a.corners]))
    cb = (_tick_cx(b), np.mean([p[1] for p in b.corners]))
    _draw_tick_pair(overlay, ca, cb)
    return _result_json(entry, overlay, px_per_cm), "ok"


BLURRY_MSG = "Photo is too blurry to detect the ruler — retake in focus."


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def api_upload():
    f = request.files.get("photo")
    if f is None:
        return jsonify({"ok": False, "error": "no file"}), 400
    data = np.frombuffer(f.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"ok": False, "error": "could not decode image"}), 400
    sid = _store(image)
    contour = detect_unet(image, UNET_WEIGHTS_DEFAULT)
    if contour is None:
        s = STORE[sid]["scale"]
        disp = cv2.resize(image, None, fx=s, fy=s) if s < 1.0 else image
        return jsonify({"ok": False, "id": sid, "method": "unet",
                        "display": _jpeg_b64(disp)})
    return _respond_with_contour(sid, contour, "unet")


@app.get("/api/display/<sid>")
def api_display(sid):
    entry = STORE.get(sid)
    if entry is None:
        return jsonify({"ok": False, "error": "unknown id"}), 404
    s = entry["scale"]
    img = entry["image"]
    disp = cv2.resize(img, None, fx=s, fy=s) if s < 1.0 else img
    return jsonify({"ok": True, "display": _jpeg_b64(disp)})


@app.post("/api/classical")
def api_classical():
    body = request.get_json(force=True)
    entry = STORE.get(body.get("id"))
    if entry is None:
        return jsonify({"ok": False, "error": "unknown id"}), 404
    s = entry["scale"]
    x, y, w, h = (int(round(v / s)) for v in body["roi"])
    H, W = entry["image"].shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    if w < 5 or h < 5:
        return jsonify({"ok": False, "error": "box too small"})
    # Background window: bigger than the lesion but strictly smaller than
    # the crop, so the lesion never dominates its own background estimate.
    bg_ksize = max(31, int(min(w, h) * 0.9)) | 1
    det = LesionDetector(local_bg_ksize=bg_ksize)
    det.select_roi = lambda *a, **k: (x, y, w, h)
    result = det.detect(entry["image"])
    area = None if (result is None or result.contour is None) else result.area_px
    print(f"[classical] img={W}x{H} box=({x},{y},{w},{h}) bg_ksize={bg_ksize} "
          f"area_px={area} min={MIN_AREA_PX}", flush=True)
    if (result is None or result.contour is None
            or result.area_px < MIN_AREA_PX):
        return jsonify({"ok": False, "id": body["id"], "method": "classical"})
    return _respond_with_contour(body["id"], result.contour, "classical")


@app.post("/api/manual")
def api_manual():
    body = request.get_json(force=True)
    entry = STORE.get(body.get("id"))
    if entry is None:
        return jsonify({"ok": False, "error": "unknown id"}), 404
    pts = body.get("points", [])
    if len(pts) < 3:
        return jsonify({"ok": False, "error": "need at least 3 points"})
    s = entry["scale"]
    contour = (np.array(pts, np.float32) / s).round().astype(np.int32)
    return _respond_with_contour(body["id"], contour.reshape(-1, 1, 2),
                                 "manual")


@app.post("/api/finalize")
def api_finalize():
    """Automatic ruler pass: look for the ruler below the accepted lesion.

    On success returns the calibrated result with the 2 cm ticks drawn.
    On failure returns ruler:"manual" so the client can ask the user to
    draw a box around the ruler.
    """
    body = request.get_json(force=True)
    sid = body.get("id")
    entry = STORE.get(sid)
    if entry is None or "contour" not in entry:
        return jsonify({"ok": False, "error": "no accepted lesion yet"}), 404

    roi = _below_lesion_roi(entry["image"], _stats(entry["contour"])["bbox"])
    res, reason = _measure_with_ruler(entry, roi)
    if res is not None:
        return jsonify(res)
    if reason == "blurry":
        return jsonify({"ok": False, "id": sid, "blurry": True,
                        "message": BLURRY_MSG})
    # keep the lesion-only overlay as the result if the user skips the ruler
    entry.setdefault("px_per_cm", None)
    return jsonify({"ok": True, "id": sid, "method": entry["method"],
                    "ruler": "manual"})


@app.post("/api/ruler")
def api_ruler():
    """Manual fallback: user drew a box around the ruler."""
    body = request.get_json(force=True)
    sid = body.get("id")
    entry = STORE.get(sid)
    if entry is None or "contour" not in entry:
        return jsonify({"ok": False, "error": "no accepted lesion yet"}), 404
    s = entry["scale"]
    x, y, w, h = (int(round(v / s)) for v in body["roi"])
    H, W = entry["image"].shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    if w < 20 or h < 20:
        return jsonify({"ok": False, "error": "box too small"})

    res, reason = _measure_with_ruler(entry, (x, y, w, h))
    if res is not None:
        return jsonify(res)
    if reason == "blurry":
        return jsonify({"ok": False, "id": sid, "blurry": True,
                        "message": BLURRY_MSG})
    return jsonify({"ok": False, "id": sid, "method": "ruler"})


@app.post("/api/manual_ticks")
def api_manual_ticks():
    """Last-resort fallback: user tapped the two 1 cm marks themselves.

    Their pixel separation is exactly one centimeter, giving the scale
    directly -- no ruler detection involved.
    """
    body = request.get_json(force=True)
    sid = body.get("id")
    entry = STORE.get(sid)
    if entry is None or "contour" not in entry:
        return jsonify({"ok": False, "error": "no accepted lesion yet"}), 404
    pts = body.get("points", [])
    if len(pts) != 2:
        return jsonify({"ok": False, "error": "need exactly 2 points"})
    s = entry["scale"]
    (ax, ay), (bx, by) = ((p[0] / s, p[1] / s) for p in pts)
    px_per_cm = float(np.hypot(bx - ax, by - ay))
    if px_per_cm < 1:
        return jsonify({"ok": False, "error": "points too close"})

    overlay = entry["image"].copy()
    _draw_lesion(overlay, entry["contour"], entry["method"], px_per_cm)
    _draw_tick_pair(overlay, (ax, ay), (bx, by))
    return jsonify(_result_json(entry, overlay, px_per_cm))


@app.get("/result/<sid>")
def result(sid):
    entry = STORE.get(sid)
    if entry is None or "result" not in entry:
        return jsonify({"ok": False, "error": "no result"}), 404
    ok, buf = cv2.imencode(".jpg", entry["result"],
                           [cv2.IMWRITE_JPEG_QUALITY, 92])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg",
                     download_name=f"lesion_{sid}.jpg")


if __name__ == "__main__":
    # PORT is provided by most cloud hosts; defaults to 5050 for local use.
    port = int(os.environ.get("PORT", 5050))
    try:
        from waitress import serve  # production WSGI server if available
        print(f"Serving on http://0.0.0.0:{port}")
        serve(app, host="0.0.0.0", port=port)
    except ImportError:
        app.run(host="0.0.0.0", port=port)
