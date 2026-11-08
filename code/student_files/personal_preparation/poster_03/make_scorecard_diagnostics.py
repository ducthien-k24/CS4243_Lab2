"""Render saved poster_03 audit diagnostics, without rerunning tracking. [AI-CODE]"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
from overlaylab import data, metrics, supplied as sp, visualize as viz


def main():
    destination = Path(__file__).resolve().parent
    video, _ = data.clip_paths("poster_03")
    variants = ("independent", "flow_assisted")
    runs = {
        variant: sp.read_json(CODE / "outputs/videos" / f"poster_03_{variant}_stats.json")
        for variant in variants
    }
    corners = data.load_audit_corners("poster_03")
    masks = data.load_audit_masks("poster_03")
    frames = {index: frame for index, frame in sp.iter_video(video, stop=196)
              if index in (0, 195)}
    summary = {"clip": "poster_03", "resolution": [1280, 720], "frames": {}}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    fig.suptitle("poster_03 | full-resolution audit | saved tracking and occlusion predictions")
    for row, index in enumerate((0, 195)):
        frame = frames[index]
        truth = np.asarray(corners[index])
        bounds = np.vstack([truth] + [runs[v]["corners"][index] for v in variants])
        left, top = np.maximum(np.floor(bounds.min(axis=0)).astype(int) - 35, 0)
        right, bottom = np.minimum(
            np.ceil(bounds.max(axis=0)).astype(int) + 35,
            [frame.shape[1] - 1, frame.shape[0] - 1],
        )
        ax = axes[row, 0]
        ax.imshow(frame)
        closed = np.vstack([truth, truth[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="lime", lw=2)
        record = {}
        for variant, colour in zip(variants, ("deepskyblue", "magenta")):
            run = runs[variant]
            assert run["resize_width"] is None
            predicted = np.asarray(run["corners"][index])
            closed = np.vstack([predicted, predicted[0]])
            ax.plot(closed[:, 0], closed[:, 1], color=colour, lw=1.5, ls="--")
            record[variant] = {
                "saved_corner_error": metrics.corner_error(predicted, truth),
                "monitor_flag": run["flags"][index],
            }
        ax.set_title(f"Frame {index} | GT and saved tracking corners")
        ax.legend(handles=[
            Line2D([], [], color="lime", label="Human GT"),
            Line2D([], [], color="deepskyblue", ls="--", label="Independent"),
            Line2D([], [], color="magenta", ls="--", label="Flow-assisted"),
        ], fontsize=8, loc="lower left")
        for col, variant in enumerate(variants, 1):
            predicted = sp.read_mask(
                CODE / "outputs/videos" / f"poster_03_{variant}_masks" / f"mask_{index:06d}.png"
            )
            gt = masks[index]
            assert predicted.shape == gt.shape == frame.shape[:2]
            false_positive = predicted & ~gt
            missed = gt & ~predicted
            overlap = predicted & gt
            display = sp.to_float(frame).copy()
            for region, colour in [
                (overlap, (1.0, 0.85, 0.0)),
                (false_positive, (1.0, 0.0, 0.0)),
                (missed, (0.0, 0.8, 1.0)),
            ]:
                display[region] = 0.4 * display[region] + 0.6 * np.asarray(colour)
            record[variant].update(
                predicted_mask_pixels=int(predicted.sum()),
                gt_mask_pixels=int(gt.sum()),
                false_positive_pixels=int(false_positive.sum()),
                missed_pixels=int(missed.sum()),
            )
            label = f"false mask {false_positive.sum():,} px"
            if gt.any():
                iou = metrics.mask_iou(predicted, gt)
                record[variant]["mask_iou"] = iou
                label = f"mask IoU {iou:.3f}"
            axes[row, col].imshow(display)
            axes[row, col].set_title(f"{variant} | frame {index}\n{label}")
        for ax in axes[row]:
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)
            ax.axis("off")
        summary["frames"][str(index)] = record
    axes[0, 1].legend(handles=[
        Patch(color="gold", label="GT and predicted"),
        Patch(color="red", label="False foreground"),
        Patch(color="deepskyblue", label="Missed foreground"),
    ], fontsize=8, loc="lower left")
    fig.savefig(destination / "scorecard_intermediate_evidence.png", dpi=150)
    plt.close(fig)

    run = runs["flow_assisted"]
    errors = {index: metrics.corner_error(np.asarray(run["corners"][index]), gt)
              for index, gt in corners.items()}
    monitor = sp.read_json(CODE / "outputs/thresholds.json")["track"]["monitor"]
    thresholds = {
        "inlier_ratio": monitor["min_inlier_ratio"],
        "mean_flow": monitor["max_mean_flow"],
        "brightness_change": monitor["max_brightness_change"],
    }
    fig = viz.plot_monitor_timeline(
        run["stats"], flags=[tuple(flag) for flag in run["flags"]],
        errors=errors, thresholds=thresholds,
        title="poster_03 | flow-assisted | full-resolution audit | saved tracking diagnostics",
    )
    for ax in fig.axes:
        ax.axvline(195, color="purple", ls=":", lw=1)
    threshold = sp.read_json(CODE / "outputs/thresholds.json")["localisation"]["success_threshold"]
    fig.axes[-1].axhline(threshold * 100, color="black", ls="--", lw=1)
    fig.axes[-1].set_title("Frame 195 remains OK despite audit error above 2%", fontsize=9)
    fig.tight_layout()
    fig.savefig(destination / "scorecard_monitor_timeline.png", dpi=150)
    plt.close(fig)
    summary["flow_assisted_audit_errors_from_saved_corners"] = errors
    (destination / "scorecard_diagnostics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
