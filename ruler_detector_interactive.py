"""
ruler_detector_interactive.py — Interactive-only ruler-patch detector for SkinOpus.

Draw a box around the calibration ruler sticker on your real image; it
locates the ruler inside that box and marks its 4 corners.

Draw the box snugly around the ruler. With a lot of extra skin margin,
the paper-vs-skin threshold can misclassify some nearby bright skin as
part of the ruler, pulling the fitted corners outward.

Run:
    python ruler_detector_interactive.py l1.jpg
    python ruler_detector_interactive.py l1.jpg --out overlay.png

ROI window controls:
    drag a rectangle -> ENTER/SPACE to confirm -> C to cancel
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from lesion_detector_interactive import LesionDetector


@dataclass
class Swatch:
    index: int  # position in reading order: row-major, top-to-bottom then left-to-right
    corners: List[Tuple[int, int]]  # 4 corner points, full-image coords
    color_bgr: Tuple[int, int, int]  # sampled representative color


@dataclass
class Tick:
    index: int  # left-to-right position among the long cm ticks
    corners: List[Tuple[int, int]]  # 4 corner points, full-image coords


@dataclass
class DigitTick:
    index: int  # left-to-right position among paired digit/tick labels
    digit_corners: List[Tuple[int, int]]  # 4 corner points, full-image coords
    tick_corners: List[Tuple[int, int]]  # the paired long tick's 4 corner points


@dataclass
class RulerResult:
    corners: List[Tuple[int, int]]  # 4 points, full-image coords, clockwise
    area_px: float = 0.0
    centroid: Tuple[int, int] = (0, 0)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    swatches: List[Swatch] = field(default_factory=list)
    cm_ticks: List[Tick] = field(default_factory=list)
    digits: List[Tick] = field(default_factory=list)
    digit_ticks: List[DigitTick] = field(default_factory=list)
    sharpness: float = 0.0  # normalized gradient energy of the ruler crop;
                            # low => out of focus. 0 when no ruler was warped.
    debug: dict = field(default_factory=dict)


class RulerDetector:
    """Locates the paper calibration ruler sticker inside an interactively
    drawn box and returns its 4 corners.

    The ruler is a bright, low-saturation, rigid rectangular sticker on
    top of warmer/more saturated skin, so detection finds "paper-like"
    pixels (high value, low saturation), merges same-object fragments (a
    shadow band from the sticker curving over skin can split it into
    disconnected mask pieces), and fits the smallest enclosing rotated
    rectangle around them. That directly gives the 4 corners without
    needing a pixel-precise boundary trace, which is both simpler and
    more robust than segmenting the exact edge.
    """

    def __init__(
        self,
        min_area_frac: float = 0.05,
        max_area_frac: float = 0.98,
        rect_weight: float = 0.5,
        center_weight: float = 0.2,
        close_ksize: int = 25,
        denoise_d: int = 9,
        denoise_sigma_color: float = 75.0,
        denoise_sigma_space: float = 75.0,
        shadow_blur_frac: float = 0.25,
        verbose: bool = True,
    ):
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.rect_weight = rect_weight
        self.center_weight = center_weight
        self.close_ksize = close_ksize | 1
        self.denoise_d = denoise_d
        self.denoise_sigma_color = denoise_sigma_color
        self.denoise_sigma_space = denoise_sigma_space
        self.shadow_blur_frac = shadow_blur_frac
        self.verbose = verbose

    # Reuse the generic drag-a-box ROI picker; it has nothing lesion-specific in it.
    select_roi = staticmethod(LesionDetector.select_roi)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[detect] {message}")

    # ---------------------- segmentation stages ---------------------- #
    def _denoise(self, image_bgr: np.ndarray) -> np.ndarray:
        """Smooth sensor/JPEG noise before computing paperness.

        Paperness is a per-pixel V-S difference, so speckle noise in
        either channel shows up directly as holes/islands in the mask.
        A bilateral filter is used instead of a plain blur because it
        preserves the ruler's edges and swatch/text boundaries while
        still flattening noise within otherwise-uniform regions.
        """
        return cv2.bilateralFilter(
            image_bgr, self.denoise_d,
            self.denoise_sigma_color, self.denoise_sigma_space,
        )

    def _flatten_shadow(self, v: np.ndarray) -> np.ndarray:
        """Divide out a soft shadow's smooth brightness gradient from V.

        A cast shadow darkens a broad, smoothly-varying area roughly
        multiplicatively -- unlike the ruler's small, sharp-edged swatches,
        text, and ticks. Dividing V by a heavily blurred copy of itself
        removes that broad gradient while keeping the local contrast
        (paper vs. ink, paper vs. skin) that the paper/skin threshold
        actually depends on, so the same cutoff applies inside a shadowed
        patch and outside it.
        """
        h, w = v.shape[:2]
        k = int(round(min(h, w) * self.shadow_blur_frac)) | 1
        k = max(k, 3)
        background = cv2.GaussianBlur(v, (k, k), 0)
        flat = v / (background + 1e-3) * float(background.mean())
        return np.clip(flat, 0, 255)

    def _paperness(self, bgr: np.ndarray) -> np.ndarray:
        """High on cool/neutral white paper, low on warm-toned skin.

        Brightness/saturation (V-S) can't separate paper from pale, brightly
        lit skin -- both are bright and desaturated, and skin can even score
        higher. What does hold across skin tones is color *temperature*: skin
        is warm (Lab a and b above neutral 128) while white paper sits at or
        below neutral, so the score is the summed distance below neutral on
        both opponent axes.
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        a = lab[:, :, 1].astype(np.float32)
        b = lab[:, :, 2].astype(np.float32)
        score = -(a - 128.0) - (b - 128.0)
        return cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def _initial_mask(paperness: np.ndarray) -> np.ndarray:
        _, otsu = cv2.threshold(paperness, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu

    @staticmethod
    def _clean(mask: np.ndarray, ksize: int) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        # Close first with a fairly large kernel: the printed color swatches
        # and black text on the ruler locally look nothing like blank paper
        # and would otherwise punch holes through the sticker's silhouette.
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    def _pick_points(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Return points covering the ruler, merging same-object fragments.

        A ruler wrapped over curved skin often gets cut by a shadow band
        into two or more disconnected mask pieces even after closing. Any
        single piece would only be half the sticker, so every piece that
        plausibly belongs to the ruler (big enough, rectangular enough) is
        combined rather than picking just the best one.
        """
        h, w = mask.shape
        img_area = float(h * w)
        cx0, cy0 = w / 2.0, h / 2.0
        min_area_px = max(200.0, self.min_area_frac * img_area)
        min_rectangularity = 0.35
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            frac = area / img_area
            if area < min_area_px or frac > self.max_area_frac:
                continue
            rect = cv2.minAreaRect(c)
            rect_area = rect[1][0] * rect[1][1]
            rectangularity = area / rect_area if rect_area > 0 else 0.0
            if rectangularity < min_rectangularity:
                continue
            M = cv2.moments(c)
            cx = M["m10"] / (M["m00"] + 1e-6)
            cy = M["m01"] / (M["m00"] + 1e-6)
            dist = np.hypot(cx - cx0, cy - cy0) / np.hypot(cx0, cy0)
            centrality = 1.0 - dist
            score = ((1.0 - self.rect_weight - self.center_weight) * frac
                     + self.rect_weight * rectangularity
                     + self.center_weight * centrality)
            candidates.append((score, c))
        if not candidates:
            return None
        return np.vstack([c for _, c in candidates])

    # ---------------------- color-strip detection ---------------------- #
    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """Order 4 points as [top-left, top-right, bottom-right, bottom-left].

        Needed because cv2.boxPoints' starting corner/winding depends on
        the rectangle's rotation angle, but a perspective warp needs a
        consistent mapping from source corner to destination corner.
        """
        pts = pts.astype(np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def _warp_ruler(image_bgr: np.ndarray, corners: np.ndarray):
        """Perspective-warp the ruler to a straightened, axis-aligned view.

        Detecting the color swatches directly on the skewed photo means
        fighting the same rotation/perspective distortion the ruler
        detection itself had to work around. Once flattened, each swatch
        is just an axis-aligned block, which is far easier to find.
        """
        tl, tr, br, bl = corners
        width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
        height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
        width, height = max(width, 1), max(height, 1)
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                        [0, height - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(image_bgr, M, (width, height))
        return warped, M

    @staticmethod
    def _sharpness(warped_bgr: np.ndarray) -> float:
        """Focus measure of the (raw) warped ruler: normalized gradient energy.

        Mean squared Sobel gradient divided by mean intensity squared, so it
        reflects edge crispness independent of exposure and resolution. Sharp
        rulers score high (printed ticks/edges are crisp); an out-of-focus
        photo smears those edges and the score collapses. Measured on a
        fixed-height resize so a small ruler isn't unfairly penalized.
        """
        g = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = g.shape[:2]
        if h < 2 or w < 2:
            return 0.0
        gn = cv2.resize(g, (max(2, int(w * 400 / h)), 400))
        gx = cv2.Sobel(gn, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gn, cv2.CV_64F, 0, 1, ksize=3)
        return float((gx * gx + gy * gy).mean() / (gn.mean() ** 2 + 1e-6))

    @staticmethod
    def _otsu_split(values: np.ndarray) -> np.ndarray:
        """Boolean split of a 1-D profile into low/high groups via Otsu.

        Used instead of a fixed saturation cutoff because "how saturated
        is skin" varies by skin tone and lighting; row/column saturation
        here is clearly bimodal (paper/skin vs. printed color), which is
        exactly what Otsu is for.
        """
        img = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8).reshape(-1, 1)
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask.ravel().astype(bool)

    @staticmethod
    def _colorful_row_mask(warped_bgr: np.ndarray) -> np.ndarray:
        """Per-row boolean: True where a majority of pixels are individually colorful.

        A row counts as colorful only if a *majority* of its pixels are
        individually saturated, not just a high row-mean saturation. That
        distinguishes a solid swatch strip (~100% colorful) from the
        cm-tick-mark row, which is mostly blank paper with sparse thin red
        dashes -- those dashes are saturated enough to skew the row mean,
        but they cover only a small fraction of the row.
        """
        h, w = warped_bgr.shape[:2]
        hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        pixel_colorful = RulerDetector._otsu_split(sat.reshape(-1)).reshape(h, w)
        row_colorful_frac = pixel_colorful.mean(axis=1)
        row_is_colorful = row_colorful_frac > 0.5
        # Bridge tiny gaps (anti-aliasing at a divider row) between colorful rows.
        row_is_colorful = cv2.morphologyEx(
            row_is_colorful.astype(np.uint8).reshape(-1, 1),
            cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8)
        ).ravel().astype(bool)
        return row_is_colorful

    @staticmethod
    def _find_swatch_boxes(warped_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find each color swatch's (x, y, w, h) in the warped ruler image.

        Swatches sit in one or more horizontal strips and butt directly up
        against their neighbors, separated only by a thin dark divider
        line -- a plain saturation threshold would merge an entire strip
        into one blob. This finds the colorful strip rows first, then
        within each row splits it into individual swatches by column-mean
        *brightness* rather than saturation: a pale swatch (e.g. lavender)
        can have saturation as low as the background, but every divider is
        consistently near-black, so brightness reliably tells divider from
        swatch regardless of the swatch's own color.
        """
        h, w = warped_bgr.shape[:2]
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        row_is_colorful = RulerDetector._colorful_row_mask(warped_bgr)

        boxes = []
        y = 0
        while y < h:
            if not row_is_colorful[y]:
                y += 1
                continue
            y0 = y
            while y < h and row_is_colorful[y]:
                y += 1
            y1 = y
            if (y1 - y0) < 0.02 * h:
                continue  # too thin to be a real strip row

            band = gray[y0:y1, :]
            col_brightness = band.mean(axis=0)
            col_is_swatch = RulerDetector._otsu_split(col_brightness)
            x = 0
            while x < w:
                if not col_is_swatch[x]:
                    x += 1
                    continue
                x0 = x
                while x < w and col_is_swatch[x]:
                    x += 1
                x1 = x
                if (x1 - x0) >= 0.01 * w:
                    boxes.append((x0, y0, x1 - x0, y1 - y0))
        return boxes

    @staticmethod
    def _sample_color(image_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """Return a swatch box's representative color as median BGR.

        Samples an inset region so the thin divider-line pixels right at
        the box edges don't pull the average color toward black.
        """
        x, y, w, h = box
        inset_x, inset_y = max(1, w // 6), max(1, h // 6)
        x0, y0 = x + inset_x, y + inset_y
        x1, y1 = x + w - inset_x, y + h - inset_y
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = x, y, x + w, y + h
        region = image_bgr[y0:y1, x0:x1]
        if region.size == 0:
            region = image_bgr[y:y + h, x:x + w]
        b, g, r = np.median(region.reshape(-1, 3), axis=0)
        return (int(round(b)), int(round(g)), int(round(r)))

    def _extract_swatches(
        self,
        color_source: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        to_full_image: "callable",
    ) -> List[Swatch]:
        """Build ordered Swatch records from raw boxes plus a coordinate mapper.

        `boxes` is already in reading order (row-major, top-to-bottom then
        left-to-right) because `_find_swatch_boxes` scans rows top-down and
        columns left-to-right within each row.
        """
        swatches = []
        for i, (sx, sy, sw, sh) in enumerate(boxes):
            color = self._sample_color(color_source, (sx, sy, sw, sh))
            local_corners = np.array(
                [[sx, sy], [sx + sw, sy], [sx + sw, sy + sh], [sx, sy + sh]],
                dtype=np.float32)
            full_corners = to_full_image(local_corners)
            swatches.append(Swatch(
                index=i,
                corners=[(int(round(px)), int(round(py))) for px, py in full_corners],
                color_bgr=color,
            ))
        return swatches

    @staticmethod
    def _tick_row_black_components(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Connected black-ink (x, y, w, h) boxes in the ruler's top tick row.

        The tick row sits above the first color-swatch strip, so that
        strip's leading edge marks the row's bottom boundary. Black ink is
        told apart from the row's red mm dashes by saturation -- red is
        strongly saturated, black ink isn't -- which is what leaves ticks,
        digits, and letters ("CM") as the surviving components here.
        """
        row_is_colorful = RulerDetector._colorful_row_mask(image_bgr)
        h = image_bgr.shape[0]
        y1 = 0
        while y1 < h and not row_is_colorful[y1]:
            y1 += 1
        if y1 < 0.02 * h:
            return []  # no real tick-mark band before the first color row
        band = image_bgr[0:y1, :]

        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        # Ink is darker than the paper *around* it, so threshold V locally
        # (adaptive) rather than at a fixed cutoff: under uneven lighting a
        # fixed cutoff swallows whole shadowed paper regions, merging every
        # tick there into one giant non-tick-shaped blob.
        dark = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 63, 25)
        # Exclude the red/pink millimetre dashes. Saturation alone doesn't do
        # it -- faded dashes sit right at the S cutoff -- but on the Lab
        # red-green axis they are firmly red (a ~= 138+) while black cm ink is
        # neutral (a ~= 128), a clean gap. Keep only neutral (non-red) ink.
        lab = cv2.cvtColor(band, cv2.COLOR_BGR2LAB)
        neutral = (lab[:, :, 1] < 133).astype(np.uint8) * 255
        black_mask = cv2.bitwise_and(dark, neutral)

        n, _, stats, _ = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
        return [(int(x), int(y), int(cw), int(ch)) for x, y, cw, ch, area in stats[1:]
                if cw > 0 and ch > 0]

    @staticmethod
    def _classify_long_ticks(
        components: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        """Pick out the long tick lines from a list of black-ink (x, y, w, h) boxes.

        Long tick lines are told apart from shorter secondary ticks and
        squat digit/letter glyphs by shape: a tick is a tall, thin
        connected component, so a low aspect-ratio (height/width) cutoff
        drops the glyphs, and an Otsu split on the remaining components'
        heights separates the long ticks from the shorter half/mm ticks.
        """
        candidates = [b for b in components if b[3] / float(b[2]) >= 2.5]
        if not candidates:
            return []
        if len(candidates) == 1:
            long_ticks = candidates
        else:
            heights = np.array([c[3] for c in candidates], dtype=np.float32)
            is_group_a = RulerDetector._otsu_split(heights)
            # keep whichever Otsu group has the greater mean height -- the long ticks
            keep = is_group_a if heights[is_group_a].mean() >= heights[~is_group_a].mean() else ~is_group_a
            long_ticks = [c for c, k in zip(candidates, keep) if k]
        return sorted(long_ticks, key=lambda b: b[0])  # left-to-right

    @staticmethod
    def _classify_digit_boxes(
        components: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        """Pick out digit/letter glyphs from a list of black-ink (x, y, w, h) boxes.

        These are the components that *aren't* long or short tick lines:
        squat rather than tall-and-thin, so the same aspect-ratio cutoff
        used to pick out ticks (but inverted) isolates them, with a
        minimum size to drop stray noise specks. Shape alone can't tell a
        digit ("1", "2", "3") apart from a letter ("CM"), so this also
        returns letter glyphs -- there's no clean geometric signal to
        split them without OCR.
        """
        min_w, min_h = 8, 15
        candidates = [
            b for b in components
            if b[2] >= min_w and b[3] >= min_h and b[3] / float(b[2]) < 2.5
        ]
        return sorted(candidates, key=lambda b: b[0])  # left-to-right

    @staticmethod
    def _find_cm_ticks(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find the long black centimeter tick marks' (x, y, w, h) boxes
        among the ruler's top tick row's black-ink components."""
        return RulerDetector._classify_long_ticks(
            RulerDetector._tick_row_black_components(image_bgr))

    @staticmethod
    def _find_digit_boxes(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find the digit/letter glyphs' (x, y, w, h) boxes among the
        ruler's top tick row's black-ink components."""
        return RulerDetector._classify_digit_boxes(
            RulerDetector._tick_row_black_components(image_bgr))

    @staticmethod
    def _pair_digit_ticks(
        digit_boxes: List[Tuple[int, int, int, int]],
        tick_boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
        """Pair each digit/letter glyph with the long tick line to its right.

        Each numbered tick's label sits just to its left in the same row,
        so a tick's match is the nearest digit box that ends at or before
        it and vertically overlaps it. A max-gap cutoff keeps this from
        reaching all the way across the row -- which also happens to
        reject the "CM" label, since its nearest tick (the "0" mark) sits
        far to its right, past any real digit-to-tick gap. The cutoff is
        scaled from the ticks' own height, not any individual digit's:
        digit boxes come from a much noisier detector and an abnormally
        large false "digit" blob would otherwise buy itself an abnormally
        generous gap allowance.

        Matching is greedy one-to-one over all (digit, tick) candidates
        sorted by gap, smallest first, so a digit or tick already claimed
        can't be claimed again. With a noisy digit detector, several stray
        glyph-like blobs can each sit within range of the same real tick
        (or vice versa); without one-to-one matching a single tick or
        digit could get reported as paired more than once.
        """
        tick_heights = [t[3] for t in tick_boxes]
        max_gap = 2.0 * float(np.median(tick_heights)) if tick_heights else 0.0

        candidates = []  # (gap, digit_box, tick_box)
        for d in digit_boxes:
            dx, dy, dw, dh = d
            d_y0, d_y1 = dy, dy + dh
            for t in tick_boxes:
                tx, ty, tw, th = t
                overlap = min(d_y1, ty + th) - max(d_y0, ty)
                if overlap <= 0:
                    continue  # not in the same row
                gap = tx - (dx + dw)
                if gap < -dw or gap > max_gap:
                    continue
                candidates.append((gap, d, t))

        candidates.sort(key=lambda c: c[0])
        used_digits, used_ticks = set(), set()
        pairs = []
        for gap, d, t in candidates:
            if d in used_digits or t in used_ticks:
                continue
            used_digits.add(d)
            used_ticks.add(t)
            pairs.append((d, t))
        return sorted(pairs, key=lambda p: p[0][0])  # left-to-right by digit x

    def _extract_digit_ticks(
        self,
        pairs: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]],
        to_full_image: "callable",
    ) -> List[DigitTick]:
        """Build ordered DigitTick records, left-to-right, from raw pairs."""
        def corners_of(box):
            bx, by, bw, bh = box
            local = np.array(
                [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]],
                dtype=np.float32)
            full = to_full_image(local)
            return [(int(round(px)), int(round(py))) for px, py in full]

        return [
            DigitTick(index=i, digit_corners=corners_of(d), tick_corners=corners_of(t))
            for i, (d, t) in enumerate(pairs)
        ]

    def _extract_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        to_full_image: "callable",
    ) -> List[Tick]:
        """Build ordered Tick records, left-to-right, from raw boxes.

        Used for both cm ticks and digit/letter glyphs -- both are just an
        ordered sequence of boxes with no extra per-box data attached.
        """
        items = []
        for i, (bx, by, bw, bh) in enumerate(boxes):
            local_corners = np.array(
                [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]],
                dtype=np.float32)
            full_corners = to_full_image(local_corners)
            items.append(Tick(
                index=i,
                corners=[(int(round(px)), int(round(py))) for px, py in full_corners],
            ))
        return items

    # ---------------------- public entry point ---------------------- #
    def detect(
        self,
        image_bgr: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[RulerResult]:
        """Locate the ruler's 4 corners inside `roi` (x, y, w, h).

        If `roi` is None, asks for the box interactively; returns None if the
        user cancels that selection.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image passed to detect()")

        if roi is None:
            roi = self.select_roi(image_bgr, window="Draw box around ruler")
        if roi is None:
            self._log("ROI selection cancelled by user.")
            return None

        ox, oy, bw, bh = roi
        self._log(f"ROI selected: x={ox}, y={oy}, w={bw}, h={bh}")
        work = image_bgr[oy:oy + bh, ox:ox + bw]
        if work.size == 0:
            raise ValueError(f"ROI {roi} produced an empty crop")

        denoised = self._denoise(work)
        self._log("Denoised ROI crop (bilateral filter).")
        paperness = self._paperness(denoised)
        seed = self._clean(self._initial_mask(paperness), self.close_ksize)
        paper_frac = float((seed > 0).mean())
        self._log(f"Computed paperness mask (Lab cool-neutrality, Otsu + morph clean): "
                  f"{paper_frac:.1%} of ROI classified as paper.")

        points = self._pick_points(seed)
        if points is None:
            self._log("No ruler-shaped contour found in the mask -- falling back to "
                      "swatch/tick/digit detection directly on the (unwarped) ROI crop.")
            print("WARNING: no ruler found inside the ROI; "
                 "looking for color swatches directly in the ROI instead.")
            # No ruler outline to warp flat, so look for swatches directly
            # on the (unwarped) ROI crop -- worse if the ruler is heavily
            # skewed, but still useful, and doesn't depend on the ruler
            # detection succeeding first.
            to_full_image = lambda pts: pts + np.array([ox, oy], dtype=np.float32)
            swatch_boxes = self._find_swatch_boxes(denoised)
            self._log(f"Found {len(swatch_boxes)} color swatch box(es).")
            swatches = self._extract_swatches(denoised, swatch_boxes, to_full_image=to_full_image)
            tick_boxes = self._find_cm_ticks(denoised)
            self._log(f"Found {len(tick_boxes)} long cm tick(s).")
            cm_ticks = self._extract_boxes(tick_boxes, to_full_image=to_full_image)
            digit_boxes = self._find_digit_boxes(denoised)
            self._log(f"Found {len(digit_boxes)} digit/letter glyph candidate(s).")
            digits = self._extract_boxes(digit_boxes, to_full_image=to_full_image)
            digit_tick_pairs = self._pair_digit_ticks(digit_boxes, tick_boxes)
            self._log(f"Paired {len(digit_tick_pairs)} digit(s) to a tick.")
            digit_ticks = self._extract_digit_ticks(digit_tick_pairs, to_full_image=to_full_image)
            return RulerResult(corners=[], swatches=swatches, cm_ticks=cm_ticks,
                              digits=digits, digit_ticks=digit_ticks, debug={"roi": roi})

        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)  # 4x2 float32, ROI-local coords, clockwise
        box[:, 0] += ox
        box[:, 1] += oy
        corners = [(int(round(x)), int(round(y))) for x, y in box]
        self._log(f"Fitted ruler rectangle from mask contours. Corners: {corners}")

        rw, rh = rect[1]
        area_px = float(rw * rh)
        cx = int(round(rect[0][0] + ox))
        cy = int(round(rect[0][1] + oy))
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

        ordered = self._order_corners(box)
        warped, M = self._warp_ruler(image_bgr, ordered)
        sharp = self._sharpness(warped)
        self._log(f"Perspective-warped ruler to a flat {warped.shape[1]}x{warped.shape[0]} view "
                  f"(sharpness {sharp:.3f}).")
        swatch_boxes = self._find_swatch_boxes(warped)
        self._log(f"Found {len(swatch_boxes)} color swatch box(es).")
        M_inv = np.linalg.inv(M)

        def to_full_image(pts: np.ndarray) -> np.ndarray:
            return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M_inv).reshape(-1, 2)

        swatches = self._extract_swatches(warped, swatch_boxes, to_full_image=to_full_image)
        tick_boxes = self._find_cm_ticks(warped)
        self._log(f"Found {len(tick_boxes)} long cm tick(s).")
        cm_ticks = self._extract_boxes(tick_boxes, to_full_image=to_full_image)
        digit_boxes = self._find_digit_boxes(warped)
        self._log(f"Found {len(digit_boxes)} digit/letter glyph candidate(s).")
        digits = self._extract_boxes(digit_boxes, to_full_image=to_full_image)
        digit_tick_pairs = self._pair_digit_ticks(digit_boxes, tick_boxes)
        self._log(f"Paired {len(digit_tick_pairs)} digit(s) to a tick.")
        digit_ticks = self._extract_digit_ticks(digit_tick_pairs, to_full_image=to_full_image)

        return RulerResult(
            corners=corners,
            area_px=area_px,
            centroid=(cx, cy),
            bbox=bbox,
            swatches=swatches,
            cm_ticks=cm_ticks,
            digits=digits,
            digit_ticks=digit_ticks,
            sharpness=sharp,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive ruler corner detector")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--out", type=str, default=None, help="Output image path for the overlay")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step pipeline log lines")
    parser.add_argument("--roi", type=str, default=None,
                        help="Skip the interactive picker and use this ROI: x,y,w,h in full-image pixels")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"Error: Could not load image {args.image}")
        sys.exit(1)

    roi_arg = None
    if args.roi:
        try:
            x, y, w, h = (int(v) for v in args.roi.split(","))
        except ValueError:
            print(f"Error: --roi must be x,y,w,h integers, got {args.roi!r}")
            sys.exit(1)
        roi_arg = (x, y, w, h)

    detector = RulerDetector(verbose=not args.quiet)
    result = detector.detect(image_bgr, roi=roi_arg)

    if result is None:
        print("User cancelled ROI selection.")
        sys.exit(0)

    annotated_image = image_bgr.copy()
    if result.corners:
        pts = np.array(result.corners, dtype=np.int32)
        cv2.polylines(annotated_image, [pts], isClosed=True,
                     color=(0, 255, 0), thickness=3)
        for (x, y) in result.corners:
            cv2.circle(annotated_image, (x, y), 12, (0, 0, 255), -1)
    else:
        print("WARNING: no ruler found inside the ROI.")
    for swatch in result.swatches:
        spts = np.array(swatch.corners, dtype=np.int32)
        cv2.polylines(annotated_image, [spts], isClosed=True,
                     color=(0, 255, 255), thickness=2)
        label_pos = spts[0]
        cv2.putText(annotated_image, str(swatch.index), tuple(label_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    for tick in result.cm_ticks:
        tpts = np.array(tick.corners, dtype=np.int32)
        cv2.polylines(annotated_image, [tpts], isClosed=True,
                     color=(255, 0, 0), thickness=2)
    for digit in result.digits:
        dpts = np.array(digit.corners, dtype=np.int32)
        cv2.polylines(annotated_image, [dpts], isClosed=True,
                     color=(0, 128, 255), thickness=2)
    for pair in result.digit_ticks:
        d_center = np.mean(pair.digit_corners, axis=0).astype(np.int32)
        t_center = np.mean(pair.tick_corners, axis=0).astype(np.int32)
        cv2.line(annotated_image, tuple(d_center), tuple(t_center),
                color=(255, 0, 255), thickness=2)

    print(f"Area (pixels): {result.area_px:.2f}")
    print(f"Centroid: {result.centroid}")
    print(f"Bounding box: {result.bbox}")
    print(f"Corners: {result.corners}")
    print(f"Swatches found: {len(result.swatches)}")
    for swatch in result.swatches:
        b, g, r = swatch.color_bgr
        print(f"  [{swatch.index}] BGR=({b}, {g}, {r})  corners={swatch.corners}")
    print(f"CM ticks found: {len(result.cm_ticks)}")
    for tick in result.cm_ticks:
        print(f"  [{tick.index}] corners={tick.corners}")
    print(f"Digits/letters found: {len(result.digits)}")
    for digit in result.digits:
        print(f"  [{digit.index}] corners={digit.corners}")
    print(f"Digit-tick pairs found: {len(result.digit_ticks)}")
    for pair in result.digit_ticks:
        print(f"  [{pair.index}] digit={pair.digit_corners}  tick={pair.tick_corners}")

    if args.out:
        cv2.imwrite(args.out, annotated_image)
        print(f"Overlay saved to {args.out}")
    else:
        h, w = annotated_image.shape[:2]
        max_display = 800
        scale = min(1.0, max_display / float(max(h, w)))
        if scale < 1.0:
            display_img = cv2.resize(annotated_image, None, fx=scale, fy=scale)
        else:
            display_img = annotated_image

        cv2.namedWindow("Draw box around ruler", cv2.WINDOW_NORMAL)
        cv2.imshow("Draw box around ruler", display_img)
        print("Annotation complete. Close the window to continue.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
