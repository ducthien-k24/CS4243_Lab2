# Stage 3 implementation and synthetic debugging

The four functions in `overlaylab/compositing.py` implement colour gain/offset
compensation, warped-reference differencing, hysteresis plus disc opening and
component filtering, validation threshold selection, and alpha compositing.
Public signatures and the existing `hysteresis` implementation are preserved.
The optional colour appearance model is not implemented. [AI-CODE]

Run from `code/` with the repository virtual environment:

```powershell
../.venv/Scripts/python.exe -m unittest discover -s tests -v
../.venv/Scripts/python.exe student_files/debug_compositing.py
```

To recreate compatible dependencies, install
`student_files/requirements-stage3.txt` into a Python 3.12 virtual environment.
The versions used are also recorded in `stage3_debug/summary.json`.
OpenCV 5 rejects the supplied float-image text renderer; NumPy 2.5 rejects
the supplied annotation tool's cross products of two-dimensional vectors.
The compatibility pins avoid changing those supplied modules. [AI-CODE]

`debug_compositing.py` runs six additional regression checks for exact gain/offset
recovery with an occluder, constant-reference and empty-coverage behaviour,
uint8 input, empty warps, foreground preservation during feathering, optional
brightness matching, immutable configuration and the threshold tie rule.
It then uses the supplied `synthetic.py` renderer for six scenes and writes
six PNG montages plus metrics and the search log in `stage3_debug/`.
Each montage shows source, difference, synthetic truth, predicted mask,
composite and alpha. All are synthetic development examples with known
homographies; they do not measure localisation or temporal tracking. [AI-CODE]

Threshold selection uses only its supplied examples. For synthetic debugging,
two separate ball scenes (seeds 10/11) select one configuration, then the six
scenes (seeds 100..105) use it unchanged. The script never loads real validation,
test, stress or personal audit frames and never writes `outputs/thresholds.json`.
Final common-clip validation tuning remains a Notebook 3 task. [AI-DESIGN]

Search grid: input strong times 0.6/0.8/1/1.2/1.4, weak times
0.5/0.75/1/1.25, min_area times 0.5/1/2. Keep weak < strong. Score by
mean finite `metrics.mask_iou`, retaining undefined entries as NaN and logging
`n_scored`. Among maximum-IoU settings, choose the strong value nearest the
midpoint of the tied strong range; remaining ties prefer weak and min_area
closest to the input config, then smaller values. Return a new config. [AI-CODE]

The blurred-ball case deliberately applies blur as a separate diagnostic,
without retuning the selected configuration. Differences around lettering
and texture can become false foreground even with exact geometry: a global
gain/offset cannot compensate spatial blur. Inspect its PNG and metrics rather
than interpreting low leakage alone as a good mask. [AI-LANGUAGE]

AI provenance: Codex generated the four implementations, the private shared
mask helper, this debug script and these notes. Verification consists of the
provided unit tests, the six additional regression checks and inspection of
synthetic montages. A student should independently review these results before
adding any `[HUMAN-CHECK]` marker to the final disclosure. [AI-CODE] [AI-LANGUAGE]
