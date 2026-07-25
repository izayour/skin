"""
lesion_detector_interactive.py — Interactive-only lesion detector for SkinOpus.

Draw a box around the lesion on your real image; it segments the lesion
inside that box, measures its area, and overlays the result on the photo.

Run:
    python lesion_detector_interactive.py l1.jpg
    python lesion_detector_interactive.py l1.jpg --px-per-cm 125 --out overlay.png

ROI window controls:
    drag a rectangle -> ENTER/SPACE to confirm -> C to cancel
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LesionResult:
    mask: np.ndarray
    contour: Optional[np.ndarray] = None
    area_px: float = 0.0
    area_cm2: Optional[float] = None
    diameter_mm: Optional[float] = None
    centroid: Tuple[int, int] = (0, 0)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    debug: dict = field(default_factory=dict)

    @property
    def area_mm2(self) -> Optional[float]:
        return None if self.area_cm2 is None else self.area_cm2 * 100.0


class LesionDetector:
    """Segments a single pigmented lesion inside an interactively drawn box."""

    def __init__(
        self,
        blur_ksize: int = 5,
        local_bg_ksize: int = 101,
        min_area_frac: float = 0.002,
        max_area_frac: float = 0.95,
        compactness_weight: float = 0.35,
        center_weight: float = 0.25,
    ):
        self.blur_ksize = blur_ksize
        self.local_bg_ksize = local_bg_ksize | 1
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.compactness_weight = compactness_weight
        self.center_weight = center_weight

    # ---------------------- interactive ROI ---------------------- #
    @staticmethod
    def select_roi(
        image_bgr: np.ndarray, window: str = "Draw box around lesion"
    ) -> Optional[Tuple[int, int, int, int]]:
        """Draggable box on the real image -> (x, y, w, h) in full-res coords."""
        h, w = image_bgr.shape[:2]
        max_side = 1000
        scale = min(1.0, max_side / float(max(h, w)))
        disp = (cv2.resize(image_bgr, None, fx=scale, fy=scale)
                if scale < 1.0 else image_bgr)

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, disp.shape[1], disp.shape[0])
        r = cv2.selectROI(window, disp, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window)

        if r is None or r[2] == 0 or r[3] == 0:
            return None
        x, y, bw, bh = (int(round(v / scale)) for v in r)
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        return (x, y, bw, bh)

    # ---------------------- segmentation stages ---------------------- #
    def _local_darkness(self, gray: np.ndarray) -> np.ndarray:
        k = self.local_bg_ksize
        if k <= 255:
            bg = cv2.medianBlur(gray, k)
        else:
            # OpenCV 5's medianBlur rejects large kernels on smooth data
            # (data-dependent assertion) and is slow regardless, so
            # approximate a k-wide median: downscale until a 101-px kernel
            # covers the same footprint, median there, scale back up. The
            # background estimate is smooth by definition, so the resampling
            # loses nothing that matters.
            h, w = gray.shape
            s = 101.0 / k
            small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))),
                               interpolation=cv2.INTER_AREA)
            bg = cv2.medianBlur(small, 101)
            bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
        darkness = cv2.subtract(bg, gray)
        return cv2.normalize(darkness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _initial_mask(self, darkness: np.ndarray) -> np.ndarray:
        _, otsu = cv2.threshold(darkness, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adap = cv2.adaptiveThreshold(darkness, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, blockSize=51, C=-5)
        return cv2.bitwise_or(otsu, adap)

    def _local_otsu_refine(self, darkness: np.ndarray,
                           seed_mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(seed_mask > 0)
        if len(xs) == 0:
            return seed_mask
        pad = 20
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, darkness.shape[1])
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, darkness.shape[0])
        window = darkness[y0:y1, x0:x1]
        _, local = cv2.threshold(window, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        local = self._clean(local)  # close+open: strips the speckle noise
                                     # that raw Otsu leaves on real skin texture
        refined = np.zeros_like(seed_mask)
        refined[y0:y1, x0:x1] = local
        return refined

    def _pick_blob(self, mask: np.ndarray) -> Optional[np.ndarray]:
        h, w = mask.shape
        img_area = float(h * w)
        cx0, cy0 = w / 2.0, h / 2.0
        # Absolute floor in addition to the fractional one: on a very small
        # ROI, min_area_frac * img_area can drop to a handful of pixels,
        # letting pure noise blobs through. 80px mirrors the floor used in
        # skinopus_process_interactive_roi.py's segment_lesion_in_roi().
        min_area_px = max(80.0, self.min_area_frac * img_area)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, -1.0
        for c in contours:
            area = cv2.contourArea(c)
            frac = area / img_area
            if area < min_area_px or frac > self.max_area_frac:
                continue
            perim = cv2.arcLength(c, True) + 1e-6
            compactness = 4.0 * np.pi * area / (perim * perim)
            M = cv2.moments(c)
            cx = M["m10"] / (M["m00"] + 1e-6)
            cy = M["m01"] / (M["m00"] + 1e-6)
            dist = np.hypot(cx - cx0, cy - cy0) / np.hypot(cx0, cy0)
            centrality = 1.0 - dist
            score = ((1.0 - self.compactness_weight - self.center_weight) * frac
                     + self.compactness_weight * compactness
                     + self.center_weight * centrality)
            if score > best_score:
                best, best_score = c, score
        return best

    @staticmethod
    def _clean(mask: np.ndarray) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    @staticmethod
    def _smooth_mask(mask: np.ndarray, blur_ksize: int = 9) -> np.ndarray:
        """Round off the jagged pixel-staircase edges left by thresholding.

        Blurring the binary mask turns hard, jagged edges into a soft gray
        gradient; re-thresholding at the midpoint collapses that gradient
        back into a smooth binary boundary that still hugs the original
        shape (rather than a polygon approximation, which can visibly
        distort concave lesion margins).
        """
        k = blur_ksize | 1  # must be odd
        blurred = cv2.GaussianBlur(mask, (k, k), 0)
        _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        return smoothed

    # ---------------------- public entry point ---------------------- #
    def detect(self, image_bgr: np.ndarray,
               px_per_cm: Optional[float] = None) -> Optional[LesionResult]:
        """Ask for a box interactively, then segment inside it.

        Returns None if the user cancels the box selection.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image passed to detect()")

        roi = self.select_roi(image_bgr)
        if roi is None:
            return None

        ox, oy, bw, bh = roi
        work = image_bgr[oy:oy + bh, ox:ox + bw]
        if work.size == 0:
            raise ValueError(f"ROI {roi} produced an empty crop")

        lab = cv2.cvtColor(work, cv2.COLOR_BGR2Lab)
        gray = cv2.GaussianBlur(lab[:, :, 0],
                                (self.blur_ksize, self.blur_ksize), 0)

        darkness = self._local_darkness(gray)
        seed = self._clean(self._initial_mask(darkness))

        full_h, full_w = image_bgr.shape[:2]
        blob = self._pick_blob(seed)
        if blob is None:
            return LesionResult(mask=np.zeros((full_h, full_w), np.uint8),
                                debug={"roi": roi})

        seed_mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(seed_mask, [blob], -1, 255, cv2.FILLED)

        refined = self._local_otsu_refine(darkness, seed_mask)
        final_mask = np.zeros((full_h, full_w), np.uint8)
        final_mask[oy:oy + bh, ox:ox + bw] = refined

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out noise: keep only the largest contour
        contour = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            final_mask = np.zeros((full_h, full_w), np.uint8)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, cv2.FILLED)

            # Round off the jagged pixel-staircase boundary left by
            # thresholding, then re-extract the contour from the smoothed
            # mask so area/centroid/bbox are all computed from the same
            # smooth shape that gets drawn.
            final_mask = LesionDetector._smooth_mask(final_mask)
            smoothed_contours, _ = cv2.findContours(
                final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if smoothed_contours:
                contour = max(smoothed_contours, key=cv2.contourArea)

        if contour is None or cv2.countNonZero(final_mask) == 0:
            print("WARNING: local Otsu refinement produced an empty mask "
                  "inside the ROI; no lesion boundary found.")
            return LesionResult(mask=final_mask, contour=None,
                                debug={"roi": roi, "reason": "empty_refined_mask"})

        area_px = cv2.countNonZero(final_mask)
        M = cv2.moments(final_mask)
        cx = int(M["m10"] / (M["m00"] + 1e-6))
        cy = int(M["m01"] / (M["m00"] + 1e-6))

        x, y, w, h = cv2.boundingRect(final_mask)
        diameter_px = max(w, h) if max(w, h) > 0 else 0.0

        area_cm2 = None
        diameter_mm = None
        if px_per_cm is not None and px_per_cm > 0:
            area_cm2 = area_px / (px_per_cm ** 2)
            diameter_mm = (diameter_px / px_per_cm) * 10.0

        result = LesionResult(
            mask=final_mask,
            contour=contour,
            area_px=float(area_px),
            area_cm2=area_cm2,
            diameter_mm=diameter_mm,
            centroid=(cx, cy),
            bbox=(x, y, w, h),
        )

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive lesion detector")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--px-per-cm", type=float, default=None, help="Pixels per centimeter for scale calibration")
    parser.add_argument("--out", type=str, default=None, help="Output image path for the overlay")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"Error: Could not load image {args.image}")
        sys.exit(1)

    detector = LesionDetector()
    result = detector.detect(image_bgr, px_per_cm=args.px_per_cm)

    if result is None:
        print("User cancelled ROI selection.")
        sys.exit(0)

    # Draw annotation directly on the input image
    annotated_image = image_bgr.copy()

    # Fill the lesion area with semi-transparent dark color
    lesion_fill = np.zeros_like(annotated_image)
    lesion_fill[result.mask > 0] = [50, 0, 100]  # Dark red/purple
    annotated_image = cv2.addWeighted(annotated_image, 0.6, lesion_fill, 0.4, 0)

    # Draw bright green contour
    if result.contour is not None:
        cv2.drawContours(annotated_image, [result.contour], -1, [0, 255, 0], 3)

    # Print results
    print(f"Area (pixels): {result.area_px:.2f}")
    if result.area_cm2 is not None:
        print(f"Area (cm²): {result.area_cm2:.4f}")
    if result.diameter_mm is not None:
        print(f"Diameter (mm): {result.diameter_mm:.2f}")
    print(f"Centroid: {result.centroid}")
    print(f"Bounding box: {result.bbox}")

    if args.out:
        cv2.imwrite(args.out, annotated_image)
        print(f"Overlay saved to {args.out}")
    else:
        # Scale for display if too large
        h, w = annotated_image.shape[:2]
        max_display = 800
        scale = min(1.0, max_display / float(max(h, w)))
        if scale < 1.0:
            display_img = cv2.resize(annotated_image, None, fx=scale, fy=scale)
        else:
            display_img = annotated_image

        # Use the same window name to replace the ROI window
        cv2.namedWindow("Draw box around lesion", cv2.WINDOW_NORMAL)
        cv2.imshow("Draw box around lesion", display_img)
        print("Annotation complete. Close the window to continue.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
