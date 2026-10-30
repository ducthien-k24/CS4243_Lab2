# Staff oracle: MiniMax-H3 on the common clip

One run (seed 1000) of the Stage 4 generative edit with the generic prompt, by the staff on a cut of the
common clip (interval 160-335) with artwork ad.png, scored with `overlaylab.evaluate --frames` on audit frames
160 / 161 / 192 / 320. An example of what a 33-billion-parameter open video model does, not a target.

| output | placement median | artwork found | fidelity NCC | bg PSNR (dB) | bg drift (px) | leakage | coverage | flicker | warping error | time |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniMax-H3 | 2.7 % | 100.0 % | 0.32 | 20.04 | 264.89 | 8.3 % | 90.0 % | 0.14 | 0.08 | 35 min |
