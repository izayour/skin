# Body Map

A standalone app: photograph a skin lesion, and see **where on the body it is**
on a 3D model you can spin.

Open `index.html` from any static server. It shares nothing with the Skin
Lesion Scanner in this repo except `../vendor/opencv.js`.

## How it works

1. **Close-up** of the lesion, with the ruler in shot. Tap the lesion — it is
   large and easy to hit here.
2. **Wide shot** of the same spot, ruler left where it is, with a joint or two
   in frame.
3. The **ruler is found in both photos**. One flat rectangle seen twice gives
   four point correspondences, hence an exact homography, so the tap from the
   close-up maps itself into the wide shot. Skin has no features to match
   across a 20-50x change of scale; a printed card does.
4. **MediaPipe Pose** finds the joints in the wide shot, and the mapped point is
   resolved onto a limb: which one, how far along it, and which face of it.
5. The mark is drawn on a **procedurally generated 3D body**.

The ruler also gives true pixels-per-centimetre in the wide shot, which is what
lets a mark be reported as "6.4 cm from the elbow" rather than "about halfway",
and replaces an assumption that everyone is 1.75 m tall.

If the ruler cannot be found, press **Point at the ruler** and tap roughly on
it; if that fails too, tap the lesion in the wide shot directly. Tapping the 3D
body always works as a manual override.

## Files

| | |
|---|---|
| `js/anatomy.js` | the canonical body. ONE table drives both the 3D mesh and the photo matcher, so a marker cannot drift from the part it was matched to |
| `js/pose.js` | MediaPipe Pose, against vendored wasm |
| `js/locate.js` | photo point + pose -> body segment, position along it, angle round it. Also builds the per-limb search boxes |
| `js/pair.js` | ruler matching across the two photos: correspondence, homography, local scale |
| `js/body3d.js` | the 3D body and its markers, generated from `anatomy.js` |
| `js/app.js` | the flow |
| `js/pipeline.js`, `js/ruler.js` | **copies** of the scanner's lesion segmenter and ruler detector. Deliberately copies: this app is standalone. Re-copy from `../js/` to pick up improvements |

## Testing

`pairtest.html` runs itself and checks the ruler matching against a **known**
homography, so there is ground truth. It composites the close-up onto a real
room photo, then measures how far the mapped lesion lands from where it should.
Current result: **0.92 cm on skin**, judged against a 1.5 cm limit — the scale
that matters, since a limb is 5-10 cm across.

## Known limits

- The close-up ruler search takes ~20 s at 1100 px. Correct, but slow.
- The full flow has not yet been run on a real close-up/wide photo pair; the
  geometry is proven in `pairtest.html`, the wiring around it is not.
- A single photo cannot tell the near side of a limb from the far side. Which
  way the subject faces settles it globally, from the pose, rather than being
  guessed per limb — and when that is uncertain, confidence drops.
- Prototype. Not a medical device.
