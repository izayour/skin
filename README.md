# Skin Lesion Scanner

A phone-friendly web app that measures a skin lesion from a photo. Take a
picture of the lesion with a calibration ruler sticker in frame; the app
segments the lesion, finds the two nearest centimetre ticks on the ruler,
and reports the lesion's real size in millimetres.

> ⚠️ **Not a medical device.** This is an experimental tool for
> demonstration only. It does not diagnose anything and must not be used
> for medical decisions. Uploaded images are processed on the server that
> hosts the app.

## How it works

1. **Lesion** — a U-Net locates and segments the mole (with a classical
   dark-spot detector to find candidates), falling back to a user-drawn box
   and then a hand-tapped border.
2. **Ruler** — the ruler is located below the lesion; the two centimetre
   ticks nearest the lesion give the pixels-per-centimetre scale. Fallbacks:
   draw a box around the ruler, or tap the two 1 cm marks yourself.
3. **Result** — the lesion is drawn with its area (mm²) and diameter (mm).

If the photo is too out of focus to read the ruler, the app says so and
asks for a sharper retake.

## Run it locally

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5050** on the same computer, or
**http://<that-computer's-LAN-IP>:5050** on a phone on the same Wi-Fi.

The model weights (`unet_lesion.pt`, ~25 MB) ship in this repo. To point at
a different file, set the `UNET_WEIGHTS` environment variable.

## Deploy (public link, no PC required)

The app reads `$PORT` and serves via `waitress`, so it runs on any Python
host — e.g. Hugging Face Spaces (Docker), Render, Railway, or Fly.io. A
sample `half.jpg` is included for a quick demo; users upload their own
photos.

## Files

- `app.py` — Flask server + web UI endpoints
- `templates/index.html` — the mobile web UI
- `lesion_cascade.py` — U-Net + classical + manual lesion detection
- `lesion_detector_interactive.py` — classical darkness-based segmentation
- `ruler_detector_interactive.py` — ruler / centimetre-tick detection
- `unet_lesion.pt` — trained U-Net weights
- `half.jpg` — sample demo image
