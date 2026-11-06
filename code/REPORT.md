# CS4243 Lab 2 report

<!--
DRAFT STATUS (delete this comment before submission)
- Written as a draft of Stages 1-2 (Person A). Stages 3-4 are left for Person B.
- Numbers marked (prelim.) were computed with the SAME functions the official tables use
  (overlaylab.report.stage1_table / report.summarise_tracking on track_board + failure_monitor runs)
  at full resolution with the frozen outputs/thresholds.json, BEFORE run_pipeline --all existed.
  After `python -m overlaylab.run_pipeline --all ...` and `python -m overlaylab.report`, paste
  outputs/stress_tables.md unchanged into the three "Stress table" sections and check every (prelim.) number.
- [Figure: ...] = insert the named notebook figure.
- AI-use markers ([AI-CODE], [AI-LANGUAGE], [HUMAN-CHECK], ...) still have to be added after the paragraphs they cover,
  and the AI-use table filled in.
-->

This compact report records how every `YOUR CODE HERE` section is exercised. Elaborate on your implementation and the results you observed; add figures, plots and grounded reasoning that show your understanding. Every reported number must name its clip (common / viewpoint / scale / blur / exit / combined / your clip), frame subset (validation / test / audit) and method variant, and must come from `overlaylab.metrics` or `overlaylab.evaluate`. Thresholds are frozen on the common clip's validation frames; each Stage's **stress table** (`python -m overlaylab.report`, `outputs/stress_tables.md`) reports the common clip (validation and test) and the Stage's two stress clips with that frozen configuration. Paste the generated table into the Stage's section unchanged and explain the row the Stage's question points at; the other rows need no interpretation.

**Conventions used below.** "Full-res" means the native 1280×720 clips, which are used by `run_pipeline` and by the stress tables. "NB-640" means the deterministic 640-px notebook subset (63 frames, `data.load_notebook_subset`). Corner error is `metrics.corner_error`: the mean corner distance divided by the longer ground-truth diagonal. All settings and seeds are stored in `outputs/thresholds.json` (RANSAC seed 0).

## Stage 1 — Localise
<!-- harris_response, detect_corners, describe_keypoints, match_descriptors; bilinear_sample, normalise_points, dlt_homography, symmetric_transfer_error, ransac_homography, transfer_points, warp_artwork; localise_board, artwork_to_frame, overlay_frame, evaluate_audit_frames; frozen success threshold -->

### Implementation

**Keypoints (`keypoints.py`).**

- **`harris_response`.** Builds the structure tensor from the supplied normalised Sobel gradients. Each product image (`gx²`, `gy²`, `gx·gy`) is smoothed with a Gaussian of scale `sigma_w`; the products are formed *before* smoothing. The function then returns R = det M − κ (tr M)². R is positive at corners, negative on edges and about 0 on flat regions; all three unit tests check this.
- **`detect_corners`.** Discards R below `response_fraction · max R` and removes the border band *before* NMS, so that suppressed border corners cannot take `max_corners` slots. Two NMS modes are available:
  - *window* keeps a pixel equal to the maximum of its (2r+1)² neighbourhood. This is a vectorised max-filter using `sliding_window_view` over an image padded with −∞.
  - *radius* is greedy: candidates are sorted by R, and each accepted corner stamps a disc of radius r into a `blocked` mask. Every pair of accepted corners is therefore more than r apart, at a cost of O(N + K r²).

  Output is (x, y), sorted by decreasing R.
- **`describe_keypoints`.** A SIFT-like descriptor at a fixed scale.
  1. A 16×16 sampling grid with spacing `patch_scale` is centred on the keypoint.
  2. The dominant orientation θ is the peak of a 36-bin, magnitude- and Gaussian-weighted gradient-orientation histogram, refined by a parabolic fit through the peak and its two neighbours.
  3. The grid is rotated by θ, and gradients are sampled with `bilinear_sample`. Each gradient's angle is taken relative to θ and soft-binned into 2 of 8 orientation bins of its 4×4 cell, giving 128 values.
  4. The vector is L2-normalised, clamped at 0.2 and renormalised.

  Keypoints are dropped when the circumscribed circle of the grid (radius 7.5·√2·`patch_scale`) leaves the image. This test does not depend on θ, so the kept set is the same for an image and its rotated copy.
- **`match_descriptors`.** Computes all pairwise distances at once (‖a‖² + ‖b‖² − 2ab). It finds the first and second neighbours with `argpartition`, accepts a match when d1/d2 < `ratio`, and can optionally require the match to be mutual. Two exact duplicates (d1 = d2 = 0) are treated as ambiguous and rejected.

**Geometry (`geometry.py`).**

- **`normalise_points`.** Hartley normalisation: the points are translated to zero centroid and scaled so that their mean distance from the origin is √2. Returns T and the transformed points.
- **`dlt_homography`.**
  - Builds the 2n×9 system from normalised points and takes h from the last row of Vᵀ.
  - Returns `status = False` and `H = None` when σ₈/σ₁ < 1e-8, i.e. when rank(A) < 8. Collinear points give σ₈/σ₁ ≈ 1e-17, while a generic 4-point set gives about 0.19.
  - De-normalises with H = T′⁻¹ Ĥ T and scales so that H₃₃ = 1.
- **`transfer_points` / `symmetric_transfer_error`.** π(Hx), and eᵢ = ‖x′ − π(Hx)‖² + ‖x − π(H⁻¹x′)‖² in px². Points sent to infinity get e = ∞, so they can never be counted as inliers.
- **`ransac_homography`.**
  - Draws 4-point samples from `config.rng()`, which is a fresh generator per call, so results are deterministic given the seed.
  - Rejects samples with three collinear points (`_three_collinear`) but still counts them as iterations.
  - Scores each model by symmetric transfer error against `threshold`. The best model has the most inliers; ties go to the smaller inlier residual sum.
  - Whenever the best model improves, the iteration count is updated to N = log(1−p)/log(1−w⁴), capped by `max_iterations`.
  - Finally refits by DLT on all inliers and recomputes inliers and residuals with the refit H.
  - NB1 §7 confirms the adaptive count: for true outlier fractions of 0.50 and 0.67, the predicted N(0.995) was 83 and 427, and RANSAC used 83 and 427 iterations.
- **`warp_artwork`.** Destination-to-source warping. Every frame pixel inside the bounding box of the projected artwork quadrilateral is mapped back through H⁻¹. It is covered when it lands in [0, W−1]×[0, H−1] (pixel-centre convention), and it is then sampled bilinearly. NB1 §8: the coverage bounding box matches the expected (60, 300, 40, 150), and the mean |ours − `cv2.warpPerspective`| inside the coverage is 0.0015.

**Localisation (`localisation.py`).**

- **`localise_board`.** Detects and describes keypoints on both images, keeps only keypoints that have a descriptor (so that match indices refer to `keypoints_ref` / `keypoints_frame`), runs the ratio test (reference = A, frame = B), runs RANSAC, and projects the reference rectangle corners TL, TR, BR, BL. `median_residual` is the median over inliers of √(eᵢ/2), the RMS one-way transfer error in pixels. On failure it returns `H = None`, NaN corners and `status = False`.
- **`artwork_to_frame`.** Composes H_art→frame = H_ref→frame · S_art→ref. Here S = `rectangle_to_rectangle((W_a, H_a), (W_r, H_r))` maps the artwork's pixel rectangle onto the reference rectangle (`warp.reference_shape`). The transformation applied first stands on the right. On common frame 112 (full-res), the artwork corners mapped by the composed matrix coincide with the localised board corners to within 0.0 px.
- **`overlay_frame`.** Warps the artwork with the composed matrix and alpha-blends it with α = coverage. Feathering and gain matching are available but switched off in the frozen settings. Pixels outside the quadrilateral are unchanged (mean |Δ| = 0 on common frame 112).
- **`evaluate_audit_frames`.** Runs `localise_board` on each annotated frame. A failed localisation, or a frame missing from the input, gets error ∞. The function returns per-frame diagnostics and `metrics.summarise_errors`.

### Frozen Stage 1 settings and success threshold

All settings were chosen on the 13 **common · validation** frames only.

| setting | default | frozen |
|---|---|---|
| `detector.max_corners` | 600 | **1000** |
| `matcher.ratio` | 0.80 | **0.85** |
| `ransac.min_inliers` | 12 | 12 |
| detector (radius NMS, r = 5, σ_w = 2.0, κ = 0.05), descriptor, RANSAC threshold 9 px² / p = 0.995 / seed 0 | | unchanged |
| `success_threshold` | | **0.02** (2 % of the diagonal) |

Validation sweep (common · validation, full-res, `evaluate_audit_frames`):

| max_corners / ratio / min_inliers | success | median err | worst frame | fewest inliers |
|---|---|---|---|---|
| 600 / 0.80 / 12 (default) | 10/13 | 0.40 % | fails (80, 448, +1) | 6 |
| **1000 / 0.85 / 12 (frozen)** | **13/13** | **0.38 %** | **0.95 %** | **16** |
| 1500 / 0.85 / 12 | 13/13 | 0.50 % | 0.95 % | 16 |
| 1000 / 0.90 / 12 | 13/13 | 0.53 % | 1.01 % | 28 |

- **Why more corners and a looser ratio.** At full resolution the board covers about 4× more pixels than in the notebook subset, so 600 corners leave too few on the board. On the hard frames 80 and 448, only 6–7 inliers survived and the localisation failed.
- **Why `min_inliers` stays at 12.** With the frozen settings every validation frame keeps ≥ 16 inliers, so lowering `min_inliers` would only make it easier to accept a wrong homography from a few accidental matches.
- **Why not 1500 or 0.90.** 1500 corners gave no gain. Ratio 0.90 admitted more wrong matches: the median inlier ratio fell from 0.78 to 0.62, and the errors rose.
- **NMS mode.** Radius NMS was kept because of NB1 §3 (NB-640, validation frames 0 → 32): its repeatability (0.86 at σ_w = 2 for κ ∈ {0.04, 0.06}) is at least as high as window NMS (≤ 0.84 in the sweep), and it places about 4× more corners on the board (432 against 101 at σ_w = 2, κ = 0.06).
- **Success threshold.** 2 % is kept as an application-level tolerance: about 10 px on a 500 px board diagonal. With the frozen settings, every common · validation frame at full-res is below 0.95 %. The threshold was not raised to rescue the hardest NB-640 frame (448: 3.69 %).

### Stress table (Stage 1)
<!-- generated rows: common · validation, common · test, viewpoint, scale — median / p90 corner error, success at the frozen threshold, inliers, inlier ratio, affine median error / success -->

<!-- PASTE the Stage 1 table from outputs/stress_tables.md here, unchanged. -->

(prelim., computed with `report.stage1_table` and the frozen config at full-res; must equal the pasted table)

| row | frames | median err | p90 err | success | inliers (median) | inlier ratio | affine median err | affine success |
|---|---|---|---|---|---|---|---|---|
| common · validation | 13 | 0.37 % | 0.75 % | 100 % | 131 | 0.80 | 0.94 % | 92 % |
| common · test | 13 | 0.37 % | 0.50 % | 92 % | 144 | 0.80 | 0.86 % | 85 % |
| viewpoint (`coke_perspective`) | 32 | 0.57 % | 1.32 % | 69 % | 37 | 0.47 | 3.38 % | 19 % |
| scale (`coke_scale`) | 32 | 0.36 % | 0.85 % | 28 % | 5 | 0.20 | 0.66 % | 28 % |

### Evidence

- **Inlier count and ratio against frame index** (common · validation, NB-640, NB2 §2; common · test plot in NB2 §3). The inlier count follows the board's on-screen size and texture visibility: 192 inliers on frame 0, down to 42 on frame 448. The ratio stays at 0.78–0.98 on validation. [Figure: NB2 §3 inlier count / ratio plot]
- **RANSAC versus least squares on all ratio-test matches** (common · test, NB-640, NB2 §4, same matches and settings). RANSAC gives 0.37–0.95 % on every test frame. Least-squares DLT on all matches is pulled away by the few outliers: 34.2 % on frame 16, 144.4 % on frame 48, 57.4 % on frame 96, 29.6 % on frame 368. It is competitive only on frames where nearly every match is correct (224: 0.52 %; 256: 0.60 %). A handful of wrong matches is enough to ruin an unweighted least-squares fit, because their squared residuals dominate the cost.
- **Two worst cases** (common clip, full-res, frozen config). Only **one** common frame fails the frozen threshold:
  1. **common · test frame 48** fails. RANSAC finds only 5 inliers (ratio 0.24 of the ratio-test matches), below `min_inliers = 12`, so no model is returned.
  2. **common · validation frame 448** *passes* with 0.94 %. It is nevertheless the worst validation frame: 17 inliers, inlier ratio 0.36, the lowest in the subset.

  In both frames the board is small or far, and few descriptors survive the ratio test. [Figure: for frames 48 and 448 — matches (inliers green / outliers red), histogram of residuals `diag["residuals"]`, projected quadrilateral against the GT quadrilateral]

### Question 1 — Matching
<!-- a frame pair on the viewpoint or scale clip where the matching rate collapses; rotation absorbed by the dominant orientation vs viewpoint / scale not absorbed; which descriptor component fails; the labelled SIFT cell -->

**Frame pair.** Scale clip (`coke_scale`, annotated frames, full-res, frozen config). The matching rate is the number of ratio-test matches divided by the 760 reference descriptors. The reference image is 396 px wide.

| frame | board width on screen | width / reference width | matching rate | RANSAC inliers | corner error |
|---|---|---|---|---|---|
| 112 | 346 px | 0.87 | 0.287 | 183 | 0.20 % |
| 224 | 220 px | 0.56 | 0.024 | 4 | fail |

Between the two frames the camera only moves away. In-plane rotation stays within [−7°, 1°] and the keystone is 1.06, yet the matching rate drops by a factor of 12 and localisation fails.

Along the clip, localisation succeeds on every annotated frame but one (frame 96) while the board is ≥ 0.75× the reference width (and up to 1.36× when the board is larger). It fails on every annotated frame from 160 on (≤ 0.70×), where the matching rate stays at 0.02–0.06.

**What fails.** The descriptor has a fixed spatial support: a 16×16 grid at 1 px spacing, with no scale selection. The Harris detector also uses a fixed σ_w.
- When the board is 0.56× smaller on screen, the same 16-px window covers about 1.8× more board in reference coordinates. The 4×4 cells and their gradient histograms therefore summarise *different* image content in the two images, and nearest neighbours stop being correct.
- Fine texture also falls below the detector scale, so different corners are detected in the first place (lower repeatability).

The orientation histogram is computed over the same mis-scaled window, so θ itself becomes unreliable.

**What dominant-orientation normalisation absorbs, and what it does not.**
- **Absorbed: in-plane rotation.** NB1 §5 (NB-640): matching rate 0.510 on the original frame against 0.501 after a 30° rotation. The grid and the gradient angles are both expressed relative to θ. Normalisation and clamping also absorb affine brightness changes: an offset does not change gradients, a gain is removed by L2 normalisation, and clamping limits a few saturated gradients.
- **Not absorbed:** scale (same cell, downscaled ×0.6: 0.072), anisotropic foreshortening and shear (viewpoint), and blur. None of these is a rotation of the patch.

On the viewpoint clip, the matching rate falls from 0.255 (frame 128, keystone 1.09) to 0.034 (frame 256, keystone 1.57). In that clip the board also shrinks (0.69× the reference width), so viewpoint and scale are confounded there. That is why the scale clip is used for the pair above.

<!-- TODO: viewpoint frames 464/480 also fail at low keystone (1.01–1.07) and width ~1.2× while the in-plane angle changes by ~13° over 16 frames; check visually (motion blur?) before mentioning. -->

**Scale row.** The scale row reports only **28 % success** (9 of 32 annotated frames; 23 failed), with a median of 5 inliers and an inlier ratio of 0.20. The median error of the frames that do succeed is low (0.36 %): when the descriptor matches, the geometry is fine; the failures are all-or-nothing.

Because the reference has a fixed size, a descriptor without scale invariance works only inside a narrow band of on-screen sizes around the reference size, here roughly 0.75–1.36×. The pipeline's `_scale_reference` therefore resizes the reference with the processing width. For a clip that zooms, one would need either a reference pyramid (match against several reference scales) or a scale-covariant detector.

**SIFT cell** (NB1 §5, the only cell where `cv2.SIFT` is used). On the hardest common subset frame (448, NB-640), our matching rate is 0.149 and SIFT's is 0.179. SIFT detects keypoints in a difference-of-Gaussian scale space and sizes each descriptor window to the detected scale, so the window covers the same physical patch at every zoom. The rest of the pipeline is the same idea as ours (orientation-normalised 4×4×8 gradient histograms). This scale selection is exactly the component our descriptor lacks, and it is what fails on the scale clip.

### Question 2 — Geometry
<!-- the affine column on the viewpoint clip: why the affine fit breaks as the camera orbits, related to the keystone; whole-frame warp and parallax; why parallax does not harm the overlay and one capture condition under which it would -->

**Affine against homography on the viewpoint clip.**

- **Stage 1 table** (viewpoint · `coke_perspective`, full-res; affine = 6-parameter least-squares fit to the *same* RANSAC inliers): median error **0.57 %** for the homography against **3.38 %** for the affine fit; success 69 % against 19 %.
- **Per-frame errors** (NB2 §5, NB-640): affine and homography start close (frame 0: 0.81 % against 0.46 %). As the camera orbits, the affine error grows steadily: 3.42 % at frame 64, 6.86 % at 192, 9.51 % at 224, 12.15 % at 240. Over the same frames the homography stays at 0.4–1.1 %. The affine error falls back (2.57 % at 432, 1.38 % at 448) when the camera returns towards a fronto-parallel view.
- **Ground truth alone** (dataset summary): the best affine fit to the GT quadrilaterals has a residual of 9.8 % on the viewpoint clip, against 1.6 % on the scale clip.

**Why, from the camera motion.**

- **The camera orbits the poster,** so one side of the board moves closer to the camera than the other. Image size scales as 1/depth, so the near edge appears longer than the far edge. The keystone, the ratio of the left and right edge lengths, grows from 1.09 (frame 128) to 1.57 (frame 256). The board's parallel edges now converge towards a finite vanishing point.
- **An affine map cannot represent this.** It preserves parallelism and length ratios along parallel lines, so it can only produce a parallelogram. The best affine fit to a keystoned quadrilateral therefore leaves an error that grows with the keystone.
- **A homography can.** Its last row (h₃₁, h₃₂) makes the projective scale depend on position, which is the perspective division by depth. A plane imaged by a pinhole camera from *any* viewpoint is exactly related to the reference by a homography, so the homography error does not grow as the camera orbits. It fails only when matching fails (frames 256–352, where the board is also small).

**Whole-frame warp and parallax** (NB2 §6). [Figure: NB2 §6 — a frame warped onto another frame with the board's homography; board aligned, background misaligned]

- **What a homography represents.** The board's homography is the plane-induced homography of the board plane. For two camera positions with relative motion (R, t) and board plane (n, d): H = K(R + t nᵀ/d)K⁻¹. A point at depth offset δ from the plane lands away from its true position by an amount proportional to |t|·δ/(depth·(depth ± δ)). The shift grows with the camera translation and with the distance from the plane, and it vanishes for points *on* the plane or for a purely rotating camera (t = 0).
- **Why parallax does not harm the overlay.**
  - The artwork is painted only inside the projected board quadrilateral, and it lives on the board plane, where the homography is exact by construction.
  - Every pixel outside the quadrilateral is copied from the live frame (`overlay_frame` leaves it unchanged), so the misregistered background in the whole-frame warp is never displayed.
- **Under which capture condition it would harm the overlay.** When the surface being replaced is not a plane, and the camera translates.
  - Examples: a poster wrapped around a column, a curled or wavy banner, or a signboard whose painted area has raised letters or a thick frame.
  - Off-plane parts of the surface then shift relative to the overlay as the camera moves sideways, and the artwork appears to slide over them.
  - The same holds for anything on the board that is not on its plane, for example an object hanging just in front of it, when the Stage 3 compensated difference compares the warped reference with the frame.
  - A pure camera rotation (a tripod pan) causes no parallax even then.

## Stage 2 — Track
<!-- lk_flow, track_board (both variants on the common clip), failure_monitor; jitter and drift; frozen monitor thresholds -->

### Implementation

- **`lk_flow`** (Person B). Iterative single-level Lucas–Kanade over a (2w+1)² window (w = 7).
  - The structure tensor is built from the Sobel gradients of the previous frame and normalised by the number of window pixels. The point is invalid when λ₂ < `min_eigenvalue`.
  - At each iteration the window is resampled bilinearly in the next frame, and the update δ = G⁻¹ Σ ∇I (I − J) is added.
  - The point is valid only if the iteration converged, the mean absolute residual is ≤ `max_residual`, and the window stays inside the image.
  - Unit tests: sub-pixel translation is recovered; flat and edge-only windows are invalid. Checked on a known synthetic shift (+1.3, −0.7) px of a real NB-640 frame: median estimate (1.31, −0.69), 76/77 points valid.
- **`track_board`.** Two variants.
  - **`independent`:** `localise_board` on every frame.
  - **`flow_assisted`:** localises frame 0 and keeps the RANSAC inliers as fixed **reference points** and moving **frame points**. On each following frame:
    1. The frame points are propagated with `lk_flow`, and invalid points are dropped.
    2. RANSAC is run between the *reference* points and the propagated points. H is therefore always estimated from the fixed reference to the current frame and never chained frame-to-frame, so errors do not compound multiplicatively.
    3. Only valid inliers are carried to the next frame.
    4. The tracker reinitialises from the reference when fewer than `reinit_min_points` = 20 points remain, every `reinit_every` = 60 frames, or when RANSAC fails.
  - Per-frame statistics use the supplied `_frame_stats`: inlier ratio = inliers / valid propagated points; median residual = median √(e/2) px; mean flow over valid points. Flow records (points, vectors, eigenvalues, residuals, valid) are stored for every tracked frame, and `{}` on (re)initialised frames.
  - Frames are consumed once, so the generator used by `run_pipeline` works.
- **`failure_monitor`.** Implements the required policy:
  1. `status != "ok"` → HIDE (`no_model`).
  2. Too few inliers (count or ratio) → REINIT, escalating to HIDE after `hide_after_consecutive` = 2 consecutive troubled frames (rule-1 frames count towards the run).
  3. Otherwise the first of median residual, corner displacement, mean flow and brightness change that exceeds its maximum → REINIT.
  4. Otherwise OK, which resets the run.

  NaN or missing statistics never fire. Only projected-corner displacement and reprojection error are compared, never homography matrices.

### Frozen Stage 2 settings

Defaults, frozen in NB2 §2 together with Stage 1:

| group | settings |
|---|---|
| flow | window 7, 20 iterations, λ₂ ≥ 2e-4, residual ≤ 0.08, convergence 0.01 px, 1 pyramid level |
| tracking | `reinit_every` 60, `reinit_min_points` 20, propagate inliers |
| monitor | inliers ≥ 12, ratio ≥ 0.40, median residual ≤ 3 px, corner displacement ≤ 60 px, mean flow ≤ 25 px, brightness change ≤ 0.15, HIDE after 2 frames |

On the NB-640 validation segment both variants tracked 65–68 of 70 frames.

<!-- TODO (A): add one sentence on why the defaults were kept, from the validation frames only. -->

### Stress table (Stage 2)
<!-- generated rows: common · validation / test for both variants, blur, exit — corner metrics, jitter, drift, frames tracked, REINIT / HIDE counts -->

<!-- PASTE the Stage 2 table from outputs/stress_tables.md here, unchanged. -->

(prelim., `report.summarise_tracking` on full-res `track_board` + `failure_monitor` runs with the frozen config; must equal the pasted table)

| row | frames | median err | p90 err | success | jitter | drift (≥16 fr) | longest run | tracked | (re)localisations | REINIT | HIDE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| common · validation · independent | 13 | 0.34 % | 0.71 % | 92 % | 6.64 px | n/a | 1 | 472/500 | 500 | 123 | 37 |
| common · validation · flow_assisted | 13 | 0.49 % | 0.72 % | 92 % | 6.72 px | 0.46 % | 28 | 472/500 | 170 | 121 | 36 |
| common · test · independent | 13 | 0.37 % | 0.50 % | 92 % | 6.64 px | n/a | 1 | 472/500 | 500 | 123 | 37 |
| common · test · flow_assisted | 13 | 0.42 % | 0.83 % | 92 % | 6.72 px | 0.74 % | 28 | 472/500 | 170 | 121 | 36 |
| blur (`coke_blur`) · flow_assisted | 32 | 0.33 % | 1.13 % | 12 % | 3.97 px | n/a | 14 | 44/500 | 480 | 16 | 461 |
| exit (`coke_out_of_view`) · flow_assisted | 30 | 0.88 % | 1.86 % | 73 % | 7.84 px | n/a | 13 | 406/500 | 291 | 176 | 111 |

### Evidence

- **A sequence in which optical flow stabilises the overlay.** Combined clip · flow_assisted (full-res): the tracker is initialised at frame 299 and LK carries the track through frames 300–307 without relocalisation. The inlier ratio is 0.93–0.98, the median residual is 0.92–1.21 px, and the corner displacement follows the camera (2.5–11.8 px per frame). [Figure: overlay frames 299–307 or `viz.plot_monitor_timeline` crop]
- **A sequence in which it fails.**
  - *Blur clip* · flow_assisted. LK tracks frames 1–22. The median λ₂ at the tracked points falls from 3.1e-3 (frame 1) to 3.5e-4 (frame 12) as motion blur smears the texture. At frame 12, 11 of 68 points fall below the λ₂ threshold and 22 fail the residual test. Only 33 valid points remain. After frame 22 the track is lost, and relocalisation succeeds only sporadically: 44/500 frames are tracked in total.
  - *Large motion* (NB2 §7, common · NB-640, frames 170→171). The camera moves about 27 px in one frame: the `cv2.calcOpticalFlowPyrLK` median is 27.2 px with 2 pyramid levels. Our single-level LK with a 15×15 window accepts only 6 of 78 points; most of the others fail the residual test. The displacement is outside the basin of convergence of single-level LK, which the optional pyramid would extend.

### Question 3 — Tracking
<!-- flicker (independent) vs drift (flow-assisted); eigenvalues: aperture vs brightness constancy, where each kind lies on the board (blur clip); monitor timeline of the combined clip: which statistic fired first at the card crossing and at the fast rotation, occlusion vs drift; what the exit clip's HIDE / REINIT sequence shows -->

**Does independent registration jitter more?** Not measurably in our runs.
- On the common clip (full-res), jitter is **6.64 px independent** against **6.72 px flow_assisted**. On the NB-640 segment of NB2 §8 it is 2.48 px against 2.63 px.
- **Why this happens.**
  - Our flow-assisted variant does not smooth over time: it re-estimates H by RANSAC on every frame, from the same kind of Harris points, so its frame-to-frame noise is similar to independent registration.
  - It is also relocalised often: **170 localiser calls in 500 frames** (34 %), against 500 for independent, and the longest run without reinitialisation is only 28 frames. Most tracks end because fewer than 20 valid inliers remain: invalid points are dropped every frame and never replenished. Frames where the localiser is lost (28 HIDE `no_model`) are retried every frame.
- **Caveat on the measure.** Jitter is computed after subtracting ground-truth motion interpolated linearly between annotations 16 frames apart. On a hand-held clip, part of the 6.6–6.7 px is that interpolation error, which affects both variants equally.

**Does flow-assisted registration drift?**
- Very little. Drift, the median corner error on annotated frames at least 16 frames after the last (re)initialisation, is 0.46 % (common · validation, 1 such frame) and 0.74 % (common · test, 2 frames). Both are well below the 2 % threshold and comparable to the independent variant's errors (0.34–0.37 %).
- Drift is small because:
  - RANSAC always maps the *fixed* reference points to the tracked points, so only the tracked positions can drift, not a chained homography.
  - Points whose LK residual or RANSAC residual grows are discarded every frame.
  - Tracks are short (≤ 28 frames), because of the 170 reinitialisations.

  The price is that this variant gains little speed or smoothness over independent registration at full resolution.

**Eigenvalues on the blur clip: aperture against brightness constancy.**
- **Brightness constancy is the main LK failure.** Over the 20 flow-tracked frames of the blur clip, **1.1 %** of tracked points fail the λ₂ test (aperture-type) and **5 %** fail the residual test. On the common clip the figures are 0.5 % and 21 %; the large residual share there comes from fast hand-held motion (median flow 7 px/frame).
- **Blur lowers λ₂.** It removes gradient energy along the motion direction: the median λ₂ falls about 9× between frames 1 and 12, and aperture-type failures appear only once the blur is strong (frame 12: 11 points). A corner smeared along the motion direction becomes edge-like, with a large λ₁ across the smear and a small λ₂ along it.
- **Where the failures lie.** Our tracked points are Harris inliers, which are corners of the poster's interior texture, so the board's long straight edges are almost never tracked. Only about 2 % of tracked points lie within 15 px of the board boundary, and none of the blur clip's aperture failures do. Aperture failures therefore appear inside the board, where blur turns texture corners into edges, rather than along the board's straight outline. Brightness-constancy failures are spread over the board wherever the blurred appearance in frame t+1 no longer matches the template from frame t.
- **The blur clip's dominant failure is upstream of flow.** 456 HIDE `no_model` frames: once the board is blurred, Harris and descriptor matching cannot relocalise it (480 localiser calls, 44/500 frames tracked).

**Monitor timeline of the combined clip** (`coke_unconstrained` · flow_assisted, full-res). [Figure: `viz.plot_monitor_timeline` on `coke_unconstrained_flow_assisted_stats.json`]

- **Card crossings (frames 210–222 and 302–314).** No REINIT statistic fires *first*. The tracker keeps an OK model while LK simply discards the points under the card:
  - At the second crossing the valid-point count falls from 219 (frame 300) to 62 (frame 307), and the inlier count from 214 to 59. Meanwhile the inlier ratio over the surviving valid points stays at 0.93–0.98 and the median residual at 0.9–1.2 px. At frame 308 only 41 valid points remain (28 inliers, ratio 0.68). Even then no rule fires: 28 ≥ 12 inliers and 0.68 ≥ 0.40.
  - At frame 309 too few points remain, RANSAC fails, relocalisation also fails, and the first flag is **HIDE `no_model`**.
  - The first crossing behaves the same way. The inlier count falls between relocalisations, from 62 to 19 over frames 214–217 and from 75 to 25 over frames 218–222, without any statistic firing; frame 223 is HIDE `no_model`.
- **Fast rotation** (GT rotation of 5.2°/frame over frames 80–96 and 5.0°/frame over 448–464). The first non-OK statistic is **`brightness_change`** (REINIT at frames 76 and 96), followed by HIDE `no_model`. Only 4/17 and 2/33 frames are tracked in these intervals. Large inter-frame motion breaks single-level LK, and motion blur defeats relocalisation.
- **How the monitor separates occlusion from drift.**
  - *Occlusion* removes points: the valid-point and inlier counts fall, while the ratio and residual of the survivors stay good.
  - *Drift* would keep the count but degrade the fit: the inlier ratio falls and the median residual rises as the tracked points slide off their reference positions.
  - In our runs the median-residual rule never fires and the ratio rule fires rarely (5–10 REINIT and 8–22 HIDE frames per clip); occlusion ends in the count-based failure through `no_model`.
- **Caveat on `brightness_change`.** It is the most frequent REINIT trigger: 113 frames (common · independent), 116 (common · flow_assisted), 63 (combined) and 168 (exit). The statistic is the mean |I_t − I_{t−1}| inside the board quadrilateral **without motion compensation**, so camera motion over a textured board exceeds the 0.15 threshold even under constant illumination. REINIT flags do not hide the overlay; only HIDE does.

**Exit clip** (`coke_out_of_view` · flow_assisted). In 17 of the 30 annotated frames, two board corners lie outside the image. The timeline alternates between three states:
- **HIDE** (111 frames): `no_model` 94, `inlier_ratio` 17. These occur when too little of the board remains visible to relocalise, for example frames 53–71 and 362–373.
- **REINIT** (176 frames): mostly `brightness_change`, as the board content slides in and out of the frame.
- **OK** runs while enough of the board is visible.

Because the four corners are always projected from the reference rectangle, the overlay stays geometrically consistent even when half the board is outside the image: 406/500 frames tracked, 73 % success on the annotated frames, median error 0.88 %. The HIDE frames are where the monitor correctly refuses to paint without a model.

## Stage 3 — Composite
<!-- hysteresis, compensated_difference, occlusion_mask, select_difference_thresholds, composite; mask IoU / F1 / leakage; frozen thresholds -->

We warp the board reference into each frame using the estimated homography. `compensated_difference` fits one global gain and offset to the lower-residual covered pixels, then measures the mean absolute RGB residual. `hysteresis` keeps weak residuals connected to strong seeds. `occlusion_mask` opens that mask with a disk and removes small connected components. `composite` feathers the visible artwork coverage, leaving the observed pixels in areas marked as foreground. `select_difference_thresholds` selects the difference settings using the three common validation masks, with the Stage 1–2 settings held fixed. The saved settings are strong 0.25, weak 0.12, opening radius 2 px and minimum area 152 pixels at full resolution (38 processing pixels at width 640). [AI-LANGUAGE]
### Stress table (Stage 3)
<!-- generated rows: common · validation / test, viewpoint, combined — corner error, mask IoU / F1 / boundary F1 / leakage, without occlusion handling, frames hidden / scored -->
<!-- PASTE the Stage 3 table from outputs/stress_tables.md here, unchanged. -->
### Question 4 — Occlusion
<!-- strong threshold and minimum area; hysteresis on a thin occluder vs opening; one false-occlusion mechanism on the viewpoint clip and the violated assumption -->

**Which occluders the residual recovers.** On the common clip at width 640 with independent localisation, the frozen mask recovers most of the occluder on the three validation masks (mean IoU 0.8971 and F1 0.9456). On the three held-out test masks, mean IoU is 0.8330, F1 is 0.9051 and boundary F1 is 0.5675. All three masks in each split were scored, with zero hidden or missing. On test, mean alpha leakage drops from 84.67% with no occlusion handling to 0.70% with the mask. Frame 448 has IoU 0.6892 and boundary F1 0.3944, illustrating that protecting the foreground can coexist with imperfect boundary localisation. High-contrast foreground with connected residuals is well supported. Foreground close in colour to the underlying board or reduced to a narrow edge can lack strong seeds or be removed by opening. [Validation evidence](student_files/notebook3/common_validation_metrics.json) · [Test evidence](student_files/notebook3/common_test_metrics.json)

**Strong threshold and minimum area.** Raising the strong threshold requires more evidence to seed a region. Weak pixels can still be retained if connected to a surviving seed. Raising the minimum area removes small surviving components and can also discard genuine small occluders. In the width-640 common validation ablation, raising strong from 0.25 to 0.375 or minimum area from 38 to 228 processing pixels left mean IoU at 0.8971 and leakage at 1.36% on all three masks. These changes did not alter the measured result for this validation subset. [Validation ablations](student_files/notebook3/validation_ablations.json)

**Hysteresis and opening.** On those same three common validation masks, setting weak equal to strong drops mean IoU to 0.6778 and increases leakage to 19.85%. Weak but connected foreground pixels therefore matter to the composite. With opening enabled, mean IoU rises from 0.8863 to 0.8971 while leakage rises slightly from 1.34% to 1.36%. In a controlled residual-map diagnostic, a 120-pixel-long horizontal strip has residual 0.20 with a 5-pixel-long seed of residual 0.50. Using the Notebook 3 processing settings, widths 1–4 px are removed by the radius-2 disk and width 5 px is the thinnest tested strip that survives (588 of 600 pixels). With radius 0, even the 1-pixel strip survives because its area exceeds 38 pixels. This is a measurement for a straight synthetic strip, where length and orientation are controlled.

**False occlusion under viewpoint change.** In the width-640 independent diagnostic on `coke_perspective`, frame 224 uses the estimated frozen homography and has an empty foreground audit mask. The predicted mask still covers 4.5% of the board. Frame 288 has 2.7% false occlusion even with the explicitly labelled GT-fallback homography. The latter is diagnostic evidence only. A homography aligns the plane, but perspective changes the sampling of high-contrast edges and may change local appearance. One global gain and offset cannot account for spatially varying residuals. Strong residuals can therefore seed connected false regions in the weak mask. The evidence identifies the residual mechanism, while the images alone do not separate resampling, blur and illumination effects. [Frame 224 residual and mask](student_files/notebook3/viewpoint_false_occlusion_000224.png) · [Frame 288 GT-fallback diagnostic](student_files/notebook3/viewpoint_false_occlusion_000288.png) [AI-LANGUAGE]

## Stage 4 — Your clip
### Capture and audit
<!-- capture conditions, privacy, annotation (10 corner frames, 5 masks, annotator), the manifest -->
<!-- TODO (Person B) -->
### Overlay Scorecard
<!-- outputs/scorecards/<clip_id>_scorecard.md for both variants; one representative frame + one failure frame -->
The [poster_03 Overlay Scorecard](outputs/scorecards/poster_03_scorecard.md) covers both classical variants on the full-resolution personal audit. Frame 240 illustrates an overlay that follows the tilted board. Flow-assisted frame 195 is the placement failure at the frozen threshold despite the monitor reporting OK. [Audit examples](student_files/personal_preparation/poster_03/scorecard_audit_examples.png) [AI-LANGUAGE]
### Generative comparison
<!-- the open-weight video model you chose (direct pixel generation; inference scripts permitted), its settings, the video you gave it (whole clip, a cut, or joined segments), and why (cite the model); the seed, the log, the overlaylab.evaluate --frames report, comparison table against the classical output on the same frames, [AI-GENERATION] markers -->
<!-- TODO (Person B) -->
### Question 5 — Reflection
<!-- classical vs your generative model on your clip: geometric placement, artwork identity, occlusion order, background preservation (what the background PSNR says about generated vs composited pixels); controllability, cost; what the model generated outside the board and in front of it; whether it kept the clip's timing (compare with the staff MiniMax-H3 example in data/genai_oracle/) -->
<!-- TODO (Person B) -->

## Evidence table
| Student file | Function | Report section | Notebook figure / table / video |
|---|---|---|---|
| keypoints.py | harris_response, detect_corners, describe_keypoints, match_descriptors | Stage 1 Implementation; Frozen settings (NMS choice); Q1 (descriptor support, rotation vs scale, SIFT) | NB1 §3 (response map, NMS modes, repeatability sweep), §4 (ratio test), §5 (matching rate, rotation/scale test, SIFT cell); Stage 1 stress table (scale row) |
| geometry.py | bilinear_sample, normalise_points, dlt_homography, symmetric_transfer_error, ransac_homography, transfer_points, warp_artwork | Stage 1 Implementation; Evidence (RANSAC vs least squares); Q2 | NB1 §6 (normalised vs raw DLT, degenerate case, `cv2.findHomography` check), §7 (RANSAC threshold / N(p,ε)), §8 (warp convention, `cv2.warpPerspective` comparison); NB2 §4 |
| localisation.py | localise_board, artwork_to_frame, overlay_frame, evaluate_audit_frames | Stage 1 Implementation; Frozen settings; Stress table; Evidence (inliers, worst cases); Q2 (affine vs homography, parallax) | NB2 §2–§6; Stage 1 stress table; `outputs/videos/*.mp4` |
| temporal.py | lk_flow, track_board, failure_monitor | Stage 2 Implementation; Stress table; Evidence; Q3 | NB2 §7 (LK flow, eigenvalues, `cv2.calcOpticalFlowPyrLK` cell), §8 (two variants, jitter/drift), §9 (monitor); Stage 2 stress table; combined-clip monitor timeline |
| compositing.py | hysteresis, compensated_difference, occlusion_mask, select_difference_thresholds, composite | Stage 3 Composite and Q4 | NB3 validation threshold search, held-out mask evaluation, hysteresis/opening ablations and viewpoint residual panels |

## AI-use disclosure table
| Tool / model | Affected artefacts (file, cell, figure, section) | Markers | What the tool contributed | How it was verified |
|---|---|---|---|---|
| | | [AI-CODE] [AI-DESIGN] [AI-VISION] [AI-GENERATION] [AI-LANGUAGE] [HUMAN-DATA] [HUMAN-CHECK] | | |
| Codex | Stage 3/Q4 and poster_03 Scorecard | [AI-LANGUAGE] [AI-CODE] | Drafted analysis and assembled the audit comparison and diagnostic measurements | Checked saved Notebook 3 outputs and recomputed mask overlap, frame-0 localisation and same-pose alpha leakage from the frozen settings and saved audit data |
