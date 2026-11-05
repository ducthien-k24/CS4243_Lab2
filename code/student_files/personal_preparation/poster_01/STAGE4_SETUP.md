# Stage 4 input handoff: poster_01

All capture-manifest and generation-condition fields are in English.

## Prepared artwork

`data/artwork/ad3.png`: RGB PNG, 960x1362, exactly the aspect ratio of the
320x454 reference. The complete 1200x1600 source JPEG was resized to 960x1280
and centred with 41 black pixels above and below. No crop, nonuniform stretch,
face removal or generative pixel editing was performed. The small cartoon
burger mascot remains. Original artwork authorship is not established by
this preparation; the source file was supplied by the user. [AI-CODE]

## Capture manifest

`manifests/capture_manifest.csv` records role B, student ID A0360382H,
camera operator/annotator Hoang Bach, and iPhone (model unspecified).
The user reports no visible hands or body parts, so body-part consent is
recorded as not applicable. Physical board width/height are blank because
they have not been measured. The annotated board occupies 31.3850% of
frame 0. Event examples are explicitly distinguished from measured event
start/end ranges. The video remains 854x480, 450 frames, 30 fps, 15 seconds.
[HUMAN-DATA]

## Frozen classical configuration

The saved Notebook 3 validation-search result agrees with
`outputs/thresholds.json`: strong=0.25, weak=0.12, min_area=152 at original
video resolution. Stage 1/2 settings have not been changed in this setup.
The Notebook 3 file currently has no saved execution counts/outputs even
though its metric/figure files exist: save the executed notebook in the
notebook editor before submission. No pipeline is executed by this handoff.

Run from `code/` when ready:

```powershell
../.venv/Scripts/python.exe -m overlaylab.run_pipeline --clip poster_01 --variant independent --artwork ad3 --thresholds outputs/thresholds.json --no-webm
../.venv/Scripts/python.exe -m overlaylab.run_pipeline --clip poster_01 --variant flow_assisted --artwork ad3 --thresholds outputs/thresholds.json --no-webm
```

Then run Notebook 4 Part I for classical evaluation/Scorecard, and Part II's
input-export cell to produce the frozen-pipeline board masks for VACE.

## Declared generative configuration

`manifests/genai_conditions.json` identifies poster_01/ad3/reference and
clean=0, occluded=45, perspective=398, temporal pair=390,391. These selected
indices are unchanged from the pre-annotation source-inspection plan.

The declared source interval is still the full 0-449 range. The active
A100 40-GB profile now uses three consecutive 5-second segments, source
0-149, 150-299, 300-449, at 832x480 with sequential CPU offload. Each raw
model output is 81 frames; preserve all raw outputs, and keep only indices
0-79 of each in the joined video. This excludes predeclared VAE padding,
giving exactly 240 frames at 16 fps (15 seconds), without accumulated seam
retiming. All three segments keep the same model/settings/seed 2026.

Use `student_files/vace/segmented_infer.py`, not the old 241-frame command.
Full cluster commands, verification and technical resume behaviour are in
`student_files/vace/A100_40GB.md`. These settings have not been GPU-benchmarked.
Source frames 390 and 391 remain together in segment 3 and map to joined
output indices 208 and 209. Notebook 4's extraction formula is unchanged.
No generative invocation has been performed by this preparation.

`stage4_setup.json` preserves artwork conversion details and SHA256 hashes.
