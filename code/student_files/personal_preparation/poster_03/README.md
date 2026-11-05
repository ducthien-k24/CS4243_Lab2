# Third capture: whole-source retiming before annotation

The user chose to compress the full new source to exactly **15 seconds**,
rather than use 1.2x and produce about 13.8 seconds. [AI-CODE]

- Source: `data/video/new_video.mp4`, 496 decoded frames, approximately
  16.566667 seconds. Source hash and exact command are in `preparation.json`.
- Speed: **1.1044444444444443x**, uniform timestamp compression of the whole
  source. No source-time interval was selected or cut out. CFR sampling may
  skip/repeat individual frames; one terminal clone is available for EOF
  rounding, with the output capped at its declared 450-frame interval.
- Prepared video: `data/personal/poster_03/clip.mp4`, original **1280x720**,
  **30 fps CFR**, **450 frames**, **15.000 seconds**. Audio, chapters and
  inherited recording metadata were removed; technical container tags remain.
- All PNG frames: `data/personal/poster_03/frames/frame_000000.png` through
  `frame_000449.png`. Start numbering at zero for this new clip.
- Approximate camera-source time = `1.1044444444444443 * frame_index / 30`.
  **All audit/evaluation/model timestamps must use the prepared clip instead.**
- Reference: same unchanged `new_reference.jpg`, rectified again with the
  previously inspected photo corners to **400x564**. The resulting PNG is
  byte-identical to poster_02's rectification. No video audit corners were used.
- Artwork: retain `data/artwork/ad4.png`, **1200x1692**, matching the board's
  ratio exactly; orange side padding remains. No artwork change is necessary.
- Existing poster_01/poster_02 videos, annotations and outputs are preserved.
  Their annotations must not be reused for poster_03.

`verification.json` records a complete output decode, frame counts, CFR
timestamps, first/last comparison with source, metadata and artwork checks.
No classical or generative inference was run. People are visible in the
background; privacy/consent for this replacement capture remains for the
student to review and document, rather than inheriting an earlier statement.

## Human annotation

Selection was made from the prepared source only, before method outputs.
See `audit_plan.json` and `audit_selection_preview.jpg`. [AI-DESIGN] [AI-VISION]

| Frame | Prepared time (s) | Purpose | Mask |
|---:|---:|---|---|
| 0 | 0.000 | Clean comparison; initial pose | Empty |
| 45 | 1.500 | Occluded comparison; lower crossing | Draw foreground |
| 75 | 2.500 | Clear pose after first crossing | Not requested |
| 150 | 5.000 | Intermediate clear pose | Not requested |
| 195 | 6.500 | Upper crossing | Draw foreground |
| 240 | 8.000 | Tilted/viewpoint comparison | Empty |
| 300 | 10.000 | Opposite tilt; full poster inside image | Not requested |
| 390 | 13.000 | Clear late pose; temporal pair begins | Not requested |
| 420 | 14.000 | Late clear pose | Not requested |
| 449 | 14.967 | Final upper crossing | Draw foreground |

Temporal pair **390,391** stays in VACE segment **300-449** and maps to
distinct joined 16-fps output indices **208,209**. Frame 391 needs no separate
corner row; use the same full audit dictionary with the lab evaluator.

Run from the repository root:

```powershell
.venv/Scripts/python.exe code/student_files/annotate_poster_03.py
```

Or, from `code/`:

```powershell
../.venv/Scripts/python.exe student_files/annotate_poster_03.py
```

The supplied GUI opens for corners first, then masks, then checks the result.
Click the printed poster **TL -> TR -> BR -> BL**, then **Enter**. Do not
click the black display-board corners. Where the case hides a corner, estimate
it manually from visible edges and neighbouring source frames; disclose this
uncertainty. Never use a case corner in place of a poster corner.

For masks **45,195,449**, click a polygon around foreground overlapping the
poster, press **Enter** to fill and **s** to save. For **0,240**, leave it
empty and press **s**. The black display board is not foreground occlusion.
`u`: undo; `r`: reset; `b`: brush; `e`: eraser; `q`: save/quit. Rerun the
same helper to resume missing labels. Do not modify clip timing after this.

Audit labels were manually completed by Hoang Bach: all 10 corner rows and
5 masks (3 nonempty) pass the supplied checker and the previews were reviewed.
Active `manifests/genai_conditions.json` now selects **poster_03** and **ad4**;
the capture manifest has a new row. Run both classical variants with ad4 and
frozen thresholds, then export fresh board masks in Notebook 4. See
**STAGE4_SETUP.md** for exact next commands. The 3x5-second A100 segment boundaries
remain valid. Do not use earlier capture masks as model inputs.

## Reproduce clip/reference preparation

From the repo root, with a fresh destination:

```powershell
.venv/Scripts/python.exe code/student_files/prepare_personal_clip.py --video data/video/new_video.mp4 --photo data/video/new_reference.jpg --clip-id poster_03 --start 0 --target-duration 15 --fps 30 --reference-width 400 --reference-corners 49 64 881 44 880 1194 90 1200
```
