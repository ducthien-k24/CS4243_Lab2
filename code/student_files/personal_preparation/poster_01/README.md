# Prepared personal capture: poster_01

The user selected original video seconds **0.3 to 15.3**. The prepared
`data/personal/poster_01/clip.mp4` contains **450 frames at 30 fps**,
854x480, duration 15 seconds. Source images/video were retained unchanged.
Audio, subtitles, data streams, chapters and inherited metadata were removed;
technical MP4 format/encoder tags remain. [AI-CODE]

The source reference photograph's poster boundary was detected against its
blue wall, visually inspected and mapped to a **320x454** rectangle with the
lab's normalised DLT and inverse bilinear warp. The width approximates its
on-screen size. Its aspect ratio is estimated from the nearly frontal photo;
physical dimensions were not supplied. Printed content and reflections were
preserved. These corners are reference-image preparation, not audit labels
and not pipeline initialisation. [AI-CODE] [AI-VISION]

`preparation.json` records source hashes, corner positions, transform and the
FFmpeg command. `verification.json` records an independent FFmpeg decode and
CFR timestamp check. All 450 frames have been extracted under the personal
clip's `frames/`; `masks/` is empty pending human audit annotation.

All annotation indices must refer to the prepared clip: index 0 starts the
new clip, index 449 is its last frame. Nominal original source time is
`0.3 + index / 30` seconds. After audit annotation, do not change the prepared
video, its frame rate, spatial size or frame indices.

`ad1.jpg` (Shrek) and `ad2.jpg` (Spider-Man) were inspected but not modified
or selected as submission artwork: both contain third-party branding and
faces, which conflict with the handout's personal-artwork requirements.
Create a compliant PNG in the same board aspect ratio before Stage 4 runs.

No audit labels were generated, and no classical or generative method was run
on this personal clip. Record capture device, consent and annotator identity
from the students' own information in `capture_manifest.csv`.
