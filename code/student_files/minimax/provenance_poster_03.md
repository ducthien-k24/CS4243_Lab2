# Generation provenance for poster_03

## MiniMax experiment [AI-GENERATION]

Job 1083102 ran on 5 November 2026 on one NVIDIA A100 80 GB PCIe. MiniMax-H3 Ref2VA used checkpoint revision `42ed227ee7df40d41602854ae760620d6eb651fe`, seed 2026 for each segment, 30 steps, an 864 × 480 canvas at 24 fps, bfloat16, ComponentsManager auto CPU offload and reference image short edge 768 px. The exact runtime versions and hashes are in the run manifest. [Attempt manifest](../../outputs/genai/attempt.json) · [Segment log](../../manifests/genai_log.csv) · [Slurm console](h3-poster03-1083102.log) · [Inference script](run_ref2va.py)

Prompt

> Replace the advertisement on the board in <Video 1> with the attached artwork <Picture 1>.

The runner decoded consecutive ranges from the prepared 450-frame clip at 30 fps and passed those frame sequences and the artwork to Ref2VA. The Notebook 4 interval export and board masks were prepared for the earlier mask-conditioned workflow. MiniMax Ref2VA took its video reference directly from the prepared source clip and its picture reference directly from `ad4.png`. [Source clip](../../../data/personal/poster_03/clip.mp4) · [Artwork](../../../data/artwork/ad4.png)

| Source frames | Output frames kept | Raw file | Measured segment time |
|---|---|---|---:|
| 0–149 | 0–119 | [segment_00_raw.mp4](../../outputs/genai/segment_00_raw.mp4) | 1,759.177 s |
| 150–299 | 120–239 | [segment_01_raw.mp4](../../outputs/genai/segment_01_raw.mp4) | 1,498.549 s |
| 300–449 | 240–359 | [segment_02_raw.mp4](../../outputs/genai/segment_02_raw.mp4) | 1,509.196 s |

Each raw file retains all 124 generated frames. The joined file contains the first 120 frames from each segment in source order. The mapping from source audit index to joined index is 0→0, 45→36, 240→192, 390→312 and 391→313. The last pair lies in segment 3. The output hashes and durations identify these files as the artifacts from the logged job. The saved MiniMax evidence shows one completed invocation per segment and records no technical retry for that job. [Joined output](../../outputs/genai/output.mp4) · [Shared audit reports](../../outputs/evaluation/stage4_genai_reports.json)

## Earlier VACE run and submission status

The earlier Wan2.1-VACE-14B experiment was a separate model run. An initial Slurm submission failed at the nested `srun` launch after preflight, before inference. A subsequent VACE submission completed three segments. The MiniMax files and figures document job 1083102 only. The official choice of output remains subject to TA confirmation under the one-model, one-attempt protocol. Confirm artwork eligibility with the TA if the small marks visible at the bottom of `ad4.png` count as logos under the lab rule.

[AI-LANGUAGE] Assembled from the saved MiniMax artifacts and the recorded VACE run history.
