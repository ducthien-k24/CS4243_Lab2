# Active capture: poster_03, artwork ad4

Hoang Bach completed all 10 corner frames and 5 masks (3 nonempty). The
supplied checker passed and the corner/mask previews were reviewed.
`annotation_review.json` records the checks. [HUMAN-DATA]

Active `manifests/genai_conditions.json` now uses poster_03 and ad4, clean
frame 0, occluded 45, perspective 240 and pair 390/391. At 16 fps, these map
to output indices 0,24,128,208,209. The pair stays in segment 300-449.
Model settings and the three 5-second segments remain fixed. [AI-DESIGN]

The camera source was accelerated to 15 seconds **before annotation**. All
following method inputs, audit indices and timestamps refer to the prepared
30-fps clip. Do not retime or resize that clip now.

## Run classical locally

From **code/**:

```powershell
../.venv/Scripts/python.exe -m overlaylab.run_pipeline --clip poster_03 --variant independent --artwork ad4 --thresholds outputs/thresholds.json --no-webm
../.venv/Scripts/python.exe -m overlaylab.run_pipeline --clip poster_03 --variant flow_assisted --artwork ad4 --thresholds outputs/thresholds.json --no-webm
```

Outputs use prefixes `poster_03_independent` and `poster_03_flow_assisted`
under `code/outputs/videos/`, with MP4s, stats and mask folders. Keep common
validation thresholds frozen. Notebook 3 need not be rerun solely for this
replacement capture. No pipeline run was performed by the annotation setup.

## Notebook 4 and cluster

1. Restart the Notebook 4 kernel. Run setup and Part I classical evaluation;
   setup should print `clip: poster_03` and `artwork: ad4`.
2. Run Part II section **1. Choose the frames (and a cut if your model needs
   one), export the inputs**. Confirm it uses
   `poster_03_flow_assisted_stats.json` and writes **450 new board masks** to
   `outputs/genai/inputs/masks/`. Old-capture masks are not valid inputs.
3. Follow the updated `student_files/vace/A100_40GB.md` on the cluster. Its
   commands now use `data/personal/poster_03/clip.mp4` and `data/artwork/ad4.png`.
   Run preflight before the actual segmented inference in a GPU job.
4. Return `outputs/genai/output.mp4`, then continue Notebook 4's extraction
   and generative comparison. Keep raw outputs and all invocation logs.

The capture manifest includes poster_03; the first-frame board fraction is
**0.190151**, from the human quadrilateral. Physical dimensions, replacement
capture's exact device and background-person consent/privacy details remain
unconfirmed in that row. Prior captures and their records are preserved.
