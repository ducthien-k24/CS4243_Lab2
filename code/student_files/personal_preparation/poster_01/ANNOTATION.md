# Human annotation session: Hoang Bach

Selected source-frame indices refer to the normalised 450-frame clip.
Selection was made by inspecting the source, before model outputs. [AI-DESIGN]

| Frame | Time (s) | Purpose |
|---|---:|---|
| 0 | 0.000 | clean comparison, empty mask |
| 30 | 1.000 | foreground crossing, mask |
| 45 | 1.500 | occluded comparison, mask |
| 90 | 3.000 | distance/scale change |
| 145 | 4.833 | second foreground crossing, mask |
| 210 | 7.000 | intermediate pose |
| 240 | 8.000 | viewpoint change |
| 360 | 12.000 | late pose |
| 390 | 13.000 | late pose; temporal pair begins |
| 398 | 13.267 | strong perspective/roll comparison, empty mask |

Temporal pair: **390,391**, mapped to distinct 16-fps output indices 208,209
when the generative interval starts at 0. Frame 391 need not have a separate
corner row; the evaluator interpolates with the full audit dictionary.

Run from `code/`:

```powershell
../.venv/Scripts/python.exe student_files/annotate_poster_01.py
```

The helper opens the supplied GUI, resumes missing frames, then opens masks
and checks the files. It never supplies coordinates or paints foreground.
Actual annotation must be done by Hoang Bach. [AI-CODE]

Corners: click **TL -> TR -> BR -> BL**, then **Enter** to save. `u` undoes,
`r` restarts. The folder hides a poster corner in some occluded frames; use
the visible poster edges and neighbouring source frames to manually estimate
its position, and record uncertainty. Do not click the folder's corner.
Frame 398's upper right corner is near the image boundary: zoom/magnify and
inspect carefully; do not invent an off-screen point if it is not visible.

Masks: outline only the folder where it covers the board. Click polygon
vertices, **Enter** fills, **s** saves and proceeds. `b` brush, `e` eraser,
`+/-` brush size, `u` undo, `r` clear. The supplied tool clips the mask to
your manually annotated board. On frames **0 and 398**, leave the mask empty
and press **s**. Frames **30,45,145** must have the actual foreground painted.

Closing/skipping a window leaves the session incomplete; rerun the helper to
continue missing labels. To correct an accepted label, call the supplied
`overlaylab.annotate corners` or `masks` command with that frame and the same
`--annotator "Hoang Bach"`. Never change the video after these annotations.

No `[HUMAN-DATA]` or `[HUMAN-CHECK]` claim is made until the student actually
performs and verifies the annotations. Selection metadata is separate from
the audit CSVs; `audit_plan.json` is a plan, not completed ground truth.
