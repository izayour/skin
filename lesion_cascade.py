"""Tiered lesion detection: U-Net first, then classical CV in a user box,
then a hand-drawn border as the last resort.

Usage:
    python lesion_cascade.py IMAGE [--out overlay.jpg] [--roi x,y,w,h]
                                   [--weights unet_lesion.pt]
                                   [--force-tier unet|classical|manual]

Tier 1 (automatic): U-Net segmentation, coarse full-frame pass to find the
lesion, then a second pass on a zoomed-in crop for a sharp boundary.
Tier 2: classical darkness-based detector inside a box the user draws
(or --roi when given).
Tier 3: the user clicks the lesion border point by point.
"""
import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

from lesion_detector_interactive import LesionDetector

UNET_SIZE = 256


def _default_weights() -> str:
    """Locate the U-Net weights: $UNET_WEIGHTS, else unet_lesion.pt next to
    this file (how the app ships it)."""
    return os.environ.get("UNET_WEIGHTS") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "unet_lesion.pt")


UNET_WEIGHTS_DEFAULT = _default_weights()
MIN_AREA_PX = 80.0


def _unet_predict(model, torch, dev, bgr: np.ndarray) -> np.ndarray:
    """Probability map at the input's resolution."""
    h, w = bgr.shape[:2]
    inp = cv2.resize(bgr, (UNET_SIZE, UNET_SIZE))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = (inp - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.from_numpy(inp.transpose(2, 0, 1)[None]).float().to(dev)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return cv2.resize(prob, (w, h))


def _dark_candidates(image_bgr: np.ndarray, max_candidates: int = 10):
    """Propose candidate lesion locations as dark, compact blobs.

    A lesion is darker than the surrounding skin, so subtract a heavily
    blurred (local-background) copy of the L channel and threshold. This is
    done on a downscaled copy -- we only need approximate centres; the U-Net
    confirms and segments each at full resolution afterwards. Returns
    (cx, cy, size) in full-image pixels, most lesion-like first.
    """
    H, W = image_bgr.shape[:2]
    scale = min(1.0, 1000.0 / max(H, W))
    small = (cv2.resize(image_bgr, None, fx=scale, fy=scale)
             if scale < 1.0 else image_bgr)
    sh, sw = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2Lab)[:, :, 0],
                            (5, 5), 0)
    k = (int(min(sh, sw) * 0.10) | 1)  # background window ~10% of frame, odd
    bg = cv2.medianBlur(gray, min(k, 151))
    darkness = cv2.normalize(cv2.subtract(bg, gray), None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(darkness, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # drop thin hair
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame = float(sh * sw)
    cand = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.0002 * frame or a > 0.15 * frame:
            continue
        perim = cv2.arcLength(c, True) + 1e-6
        comp = 4.0 * np.pi * a / (perim * perim)
        if comp < 0.35:
            continue
        M = cv2.moments(c)
        cx, cy = M["m10"] / (M["m00"] + 1e-6), M["m01"] / (M["m00"] + 1e-6)
        _, _, w, h = cv2.boundingRect(c)
        m = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 1, cv2.FILLED)
        score = float(darkness[m > 0].mean()) * comp
        cand.append((score, cx / scale, cy / scale, max(w, h) / scale))
    cand.sort(reverse=True)
    return [(cx, cy, sz) for _, cx, cy, sz in cand[:max_candidates]]


def detect_unet(image_bgr: np.ndarray, weights: str) -> Optional[np.ndarray]:
    """Localize dark candidates classically, then confirm/segment each with
    the U-Net at native scale. Returns a full-image contour or None.

    The U-Net is trained on zoomed-in dermoscopy: it is accurate on a tight
    crop but unreliable when the whole frame is squashed to 256px (a small
    mole becomes ~10-20px and either vanishes or is out-competed by
    background). So instead of localizing by downscaling the frame, a cheap
    classical dark-blob detector proposes locations and the U-Net is run
    zoomed on each -- where it scores near 1.0 on real lesions -- picking the
    best-confirmed one.
    """
    try:
        import torch
        import segmentation_models_pytorch as smp
    except ImportError as e:
        print(f"[unet] unavailable ({e}); falling back")
        return None
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = smp.Unet("efficientnet-b0", encoder_weights=None,
                         in_channels=3, classes=1).to(dev)
        model.load_state_dict(torch.load(weights, map_location=dev))
        model.eval()
    except Exception as e:
        print(f"[unet] failed to load weights {weights} ({e}); falling back")
        return None

    H, W = image_bgr.shape[:2]
    candidates = _dark_candidates(image_bgr)
    print(f"[unet] {len(candidates)} dark candidate(s) to confirm")

    best, best_prob = None, -1.0
    for cx, cy, sz in candidates:
        r = int(max(sz * 1.5, 120))          # zoom to ~3x the blob
        x0, y0 = max(int(cx - r), 0), max(int(cy - r), 0)
        x1, y1 = min(int(cx + r), W), min(int(cy + r), H)
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        prob = _unet_predict(model, torch, dev, crop)
        mask = (prob > 0.5).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= MIN_AREA_PX]
        if not cnts:
            continue
        # the blob nearest the crop centre (that's where the candidate is)
        ctr = np.array([(x1 - x0) / 2.0, (y1 - y0) / 2.0])
        c = min(cnts, key=lambda c: np.linalg.norm(
            np.array(cv2.minEnclosingCircle(c)[0]) - ctr))
        a = cv2.contourArea(c)
        if a / float((x1 - x0) * (y1 - y0)) > 0.9:
            continue  # fills the whole crop -> skin/background, not a lesion
        perim = cv2.arcLength(c, True) + 1e-6
        if 4.0 * np.pi * a / (perim * perim) < 0.30:
            continue  # not compact enough
        m = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 1, cv2.FILLED)
        mp = float(prob[m > 0].mean())
        if mp >= 0.75 and mp > best_prob:
            best_prob, best = mp, c + [x0, y0]

    if best is None:
        print("[unet] no candidate confirmed by the U-Net")
        return None
    print(f"[unet] confirmed lesion (mean prob {best_prob:.2f})")
    return best


def detect_classical(image_bgr: np.ndarray,
                     roi: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
    """Classical darkness-based detector inside a user box. Contour or None."""
    if roi is None:
        roi = LesionDetector.select_roi(image_bgr, window="Draw box around lesion")
    if roi is None:
        print("[classical] ROI selection cancelled")
        return None
    # Background window must be wider than the lesion, or the lesion's own
    # interior dominates its background estimate and vanishes from the
    # darkness map -- but strictly smaller than the crop, or medianBlur
    # gets a kernel wider than its input and dies.
    bg_ksize = max(31, int(min(roi[2], roi[3]) * 0.9)) | 1
    det = LesionDetector(local_bg_ksize=bg_ksize)
    det.select_roi = lambda *a, **k: roi
    result = det.detect(image_bgr)
    if result is None or result.contour is None or result.area_px < MIN_AREA_PX:
        print("[classical] no lesion found in the box")
        return None
    return result.contour


def detect_manual(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Let the user click the lesion border point by point.

    Left-click adds a point, right-click removes the last one,
    ENTER/SPACE closes the polygon (needs >= 3 points), ESC cancels.
    """
    h, w = image_bgr.shape[:2]
    max_side = 1000
    scale = min(1.0, max_side / float(max(h, w)))
    disp_base = (cv2.resize(image_bgr, None, fx=scale, fy=scale)
                 if scale < 1.0 else image_bgr.copy())
    points = []
    window = "Click lesion border (ENTER done, right-click undo, ESC cancel)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, disp_base.shape[1], disp_base.shape[0])
    cv2.setMouseCallback(window, on_mouse)
    while True:
        disp = disp_base.copy()
        for p in points:
            cv2.circle(disp, p, 4, (0, 0, 255), -1)
        if len(points) >= 2:
            cv2.polylines(disp, [np.array(points)], False, (0, 255, 0), 2)
        cv2.imshow(window, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            print("[manual] cancelled")
            return None
        if key in (13, 32) and len(points) >= 3:
            break
    cv2.destroyWindow(window)
    pts = (np.array(points, dtype=np.float32) / scale).round().astype(np.int32)
    return pts.reshape(-1, 1, 2)


def review_result(image_bgr: np.ndarray, contour: np.ndarray,
                  method: str) -> bool:
    """Show the tier's contour and let the user accept or reject it.

    Y/ENTER/SPACE accepts; N/ESC rejects, which sends the cascade to the
    next tier.
    """
    h, w = image_bgr.shape[:2]
    max_side = 1000
    scale = min(1.0, max_side / float(max(h, w)))
    disp = image_bgr.copy()
    cv2.drawContours(disp, [contour], -1, (0, 255, 0), max(int(3 / scale), 3))
    if scale < 1.0:
        disp = cv2.resize(disp, None, fx=scale, fy=scale)
    window = f"[{method}] result -- Y/ENTER accept, N/ESC reject"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, disp.shape[1], disp.shape[0])
    cv2.imshow(window, disp)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("y"), ord("Y"), 13, 32):
            cv2.destroyWindow(window)
            return True
        if key in (ord("n"), ord("N"), 27):
            cv2.destroyWindow(window)
            print(f"[{method}] result rejected by user; falling back")
            return False


def main():
    ap = argparse.ArgumentParser(description="Tiered lesion detector")
    ap.add_argument("image", help="Path to the image file")
    ap.add_argument("--out", default=None, help="Output overlay path")
    ap.add_argument("--roi", default=None,
                    help="Classical-tier box x,y,w,h (skips drawing it)")
    ap.add_argument("--weights", default=UNET_WEIGHTS_DEFAULT,
                    help="U-Net weights (.pt)")
    ap.add_argument("--force-tier", choices=["unet", "classical", "manual"],
                    default=None, help="Skip straight to one tier (testing)")
    ap.add_argument("--no-review", action="store_true",
                    help="Skip the interactive accept/reject step (testing)")
    args = ap.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"Error: Could not load image {args.image}")
        sys.exit(1)

    roi = None
    if args.roi:
        try:
            x, y, w, h = (int(v) for v in args.roi.split(","))
            roi = (x, y, w, h)
        except ValueError:
            print(f"Error: --roi must be x,y,w,h integers, got {args.roi!r}")
            sys.exit(1)

    contour, method = None, None
    tiers = [("unet", lambda: detect_unet(image_bgr, args.weights)),
             ("classical", lambda: detect_classical(image_bgr, roi)),
             ("manual", lambda: detect_manual(image_bgr))]
    if args.force_tier:
        tiers = [t for t in tiers if t[0] == args.force_tier]
    for name, run in tiers:
        contour = run()
        if contour is None:
            continue
        if not args.no_review and not review_result(image_bgr, contour, name):
            contour = None
            continue
        method = name
        break

    if contour is None:
        print("No lesion detected by any tier.")
        sys.exit(1)

    area = cv2.contourArea(contour)
    M = cv2.moments(contour)
    cx = int(M["m10"] / (M["m00"] + 1e-6))
    cy = int(M["m01"] / (M["m00"] + 1e-6))
    bbox = cv2.boundingRect(contour)
    print(f"Method: {method}")
    print(f"Area (pixels): {area:.2f}")
    print(f"Centroid: ({cx}, {cy})")
    print(f"Bounding box: {bbox}")

    out_path = args.out or "lesion_cascade_result.jpg"
    overlay = image_bgr.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)
    cv2.circle(overlay, (cx, cy), 8, (0, 0, 255), -1)
    cv2.putText(overlay, method, (bbox[0], max(bbox[1] - 15, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imwrite(out_path, overlay)
    print(f"Overlay saved to {out_path}")


if __name__ == "__main__":
    main()
