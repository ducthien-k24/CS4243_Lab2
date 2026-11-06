"""Figures for REPORT.md, Stages 1-2 (Person A).

Run from code/:  python student_files/report_figures/make_report_figures.py

Every figure uses the frozen outputs/thresholds.json at full resolution and the saved
run statistics (outputs/videos/*_stats.json, outputs/stress_tables.json). No setting is
changed and nothing is tuned here; ground truth is used only to draw / score.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from overlaylab import data, metrics as M, supplied as sp, visualize as viz          # noqa: E402
from overlaylab.config import PipelineConfig                                         # noqa: E402
from overlaylab.localisation import localise_board                                   # noqa: E402
from overlaylab.geometry import warp_artwork                                         # noqa: E402

OUT = Path(__file__).resolve().parent
CFG = PipelineConfig.from_json((CODE / "outputs" / "thresholds.json").read_text())
LOC = CFG.localisation
VID = CODE / "outputs" / "videos"


def save(fig, name):
    fig.savefig(OUT / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def frames_at(video, wanted):
    wanted = set(int(w) for w in wanted); got = {}
    for t, f in sp.iter_video(video):
        if t in wanted:
            got[t] = f
        if len(got) == len(wanted):
            break
    return got


def stats(name):
    return json.loads((VID / f"{name}_stats.json").read_text())


def crop_box(corners, shape, margin=60):
    c = np.asarray(corners, float)
    x0 = int(max(np.nanmin(c[:, 0]) - margin, 0)); x1 = int(min(np.nanmax(c[:, 0]) + margin, shape[1]))
    y0 = int(max(np.nanmin(c[:, 1]) - margin, 0)); y1 = int(min(np.nanmax(c[:, 1]) + margin, shape[0]))
    return x0, x1, y0, y1


# ---------------------------------------------------------------- Stage 1: inliers against frame index
def fig_inliers():
    t = json.loads((CODE / "outputs" / "stress_tables.json").read_text())["tables"]["1"]
    rows = t["rows"] if isinstance(t, dict) else t
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    for r, mk in zip(rows[:2], ("o", "s")):
        pf = {int(k): v for k, v in r["per_frame"].items()}
        fr = sorted(pf)
        inl = [pf[f]["inliers"] for f in fr]; rat = [pf[f]["inlier_ratio"] for f in fr]
        bad = [f for f in fr if not np.isfinite(pf[f]["error"] if pf[f]["error"] is not None else np.inf)
               or (pf[f]["error"] or 0) > LOC.success_threshold]
        ax[0].plot(fr, inl, mk + "-", ms=4, lw=1, label=r["label"])
        ax[1].plot(fr, rat, mk + "-", ms=4, lw=1, label=r["label"])
        for f in bad:
            ax[0].plot(f, pf[f]["inliers"], "rx", ms=10, mew=2)
            ax[1].plot(f, pf[f]["inlier_ratio"], "rx", ms=10, mew=2)
        print(r["label"], "inliers", min(inl), "-", max(inl), "ratio", round(min(rat), 2), "-", round(max(rat), 2),
              "frame0", pf.get(0, {}).get("inliers"), "frame448", pf.get(448, {}).get("inliers"))
    ax[0].axhline(LOC.ransac.min_inliers, color="k", ls="--", lw=0.8)
    ax[0].set_ylabel("RANSAC inliers"); ax[1].set_ylabel("inlier ratio"); ax[1].set_xlabel("frame index")
    ax[0].legend(fontsize=8); ax[0].set_title("Common clip, full resolution, frozen config (red x: fails the 2% threshold)")
    save(fig, "s1_inliers_vs_frame.png")


# ---------------------------------------------------------------- Stage 1: the two worst common frames
def fig_worst():
    ref = data.load_reference()
    cfg = LOC; cfg.warp.reference_shape = ref.shape[:2]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    for row, (f, split) in enumerate(((48, "test"), (448, "validation"))):
        gt = np.asarray(data.load_gt_corners(split=split)[f])
        frame = data.load_frame(f)
        H, corners, pairs, inl, diag = localise_board(ref, frame, cfg)
        err = M.corner_error(corners, gt) if diag["status"] else float("inf")
        m = viz.draw_matches(ref, frame, diag["keypoints_ref"], diag["keypoints_frame"], pairs, inl)
        axes[row, 0].imshow(np.clip(m, 0, 1)); axes[row, 0].axis("off")
        axes[row, 0].set_title(f"common · {split} · frame {f}: {diag['n_matches']} matches, {diag['n_inliers']} inliers (green)", fontsize=9)
        r = np.sqrt(np.asarray(diag["residuals"], float) / 2.0)
        ok = np.isfinite(r)
        if ok.any():
            bins = np.linspace(0, max(np.percentile(r[ok], 98), 4), 40)
            axes[row, 1].hist(r[ok & inl], bins=bins, color="green", alpha=0.7, label="inliers")
            axes[row, 1].hist(r[ok & ~inl], bins=bins, color="red", alpha=0.5, label="outliers")
            axes[row, 1].legend(fontsize=8)
        axes[row, 1].axvline(np.sqrt(cfg.ransac.threshold / 2.0), color="k", ls="--", lw=0.8)
        axes[row, 1].set_xlabel("sqrt(e_i / 2)  [px]"); axes[row, 1].set_ylabel("matches")
        axes[row, 1].set_title("residuals of the best RANSAC model", fontsize=9)
        img = viz.draw_quad(frame, gt, viz.GREEN, 2)
        if diag["status"]:
            img = viz.draw_quad(img, corners, viz.RED, 2)
        x0, x1, y0, y1 = crop_box(gt, frame.shape)
        axes[row, 2].imshow(np.clip(img[y0:y1, x0:x1], 0, 1)); axes[row, 2].axis("off")
        txt = f"error {err * 100:.2f}%" if np.isfinite(err) else "no model (status False)"
        axes[row, 2].set_title(f"GT quad (green) vs estimate (red): {txt}", fontsize=9)
        print("worst", f, split, diag["n_matches"], diag["n_inliers"], round(diag["inlier_ratio"], 3), txt)
    fig.tight_layout()
    save(fig, "s1_worst_cases.png")


# ---------------------------------------------------------------- Q2: whole-frame warp and parallax
def fig_parallax(clip="coke_perspective", fa=0, fb=192):
    video, refp = data.clip_paths(clip)
    ref = sp.read_image(refp); cfg = LOC; cfg.warp.reference_shape = ref.shape[:2]
    fr = frames_at(video, (fa, fb))
    Ha, ca, *_ = localise_board(ref, fr[fa], cfg)
    Hb, cb, *_ = localise_board(ref, fr[fb], cfg)
    H_ba = Ha @ np.linalg.inv(Hb)                       # frame-b pixel -> frame-a pixel (board plane)
    warped, cov = warp_artwork(fr[fb], H_ba, fr[fa].shape)
    # smooth before differencing so that a few pixels of registration noise do not dominate;
    # what survives is the large, depth-dependent shift (parallax) of off-plane structure
    ga = sp.gaussian_smooth(sp.to_gray(fr[fa]), 4.0); gw = sp.gaussian_smooth(sp.to_gray(warped), 4.0)
    diff = np.abs(ga - gw) * cov
    board = sp.quad_mask(fr[fa].shape[:2], ca)
    print(f"parallax: mean smoothed |diff| on board {diff[board & cov].mean():.4f}, off board {diff[~board & cov].mean():.4f}, "
          f"p95 off board {np.percentile(diff[~board & cov], 95):.4f}")
    blend = 0.5 * fr[fa] + 0.5 * warped * cov[..., None] + 0.5 * fr[fa] * (~cov)[..., None]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].imshow(np.clip(warped, 0, 1)); ax[0].set_title(f"{clip} frame {fb} warped into frame {fa} with the board homography", fontsize=9)
    ax[1].imshow(viz.draw_quad(np.clip(blend, 0, 1), ca, viz.GREEN, 2))
    ax[1].set_title(f"50/50 blend with frame {fa} (board outlined): board sharp, structure off the board doubled", fontsize=9)
    for a_ in ax:
        a_.axis("off")
    fig.tight_layout()
    save(fig, "q2_parallax_whole_frame_warp.png")


# ---------------------------------------------------------------- Stage 2: a flow-stabilised run (combined clip)
def fig_flow_run(clip="coke_unconstrained", ts=(301, 304, 307)):
    s = stats(f"{clip}_flow_assisted")
    video, _ = data.clip_paths(clip)
    fr = frames_at(video, [t - 1 for t in ts])
    fig, ax = plt.subplots(1, len(ts), figsize=(16, 3.4))
    for a, t in zip(ax, ts):
        rec = s["flow"][t]
        img = viz.draw_flow(fr[t - 1], np.array(rec["points"]), np.array(rec["vectors"]), np.array(rec["valid"], bool), scale=3)
        img = viz.draw_quad(img, np.array(s["corners"][t]), viz.BLUE, 2)
        st = s["stats"][t]
        a.imshow(np.clip(img, 0, 1)); a.axis("off")
        a.set_title(f"frame {t}: {st['n_valid_points']} valid, {st['inlier_count']} inliers, ratio {st['inlier_ratio']:.2f}, flag {s['flags'][t][0]}", fontsize=9)
    fig.suptitle(f"{clip} · flow_assisted, full resolution: LK vectors x3 (green valid / red invalid), tracked quad (blue)", fontsize=10)
    fig.tight_layout()
    save(fig, "s2_flow_stabilised_combined_300s.png")


# ---------------------------------------------------------------- Stage 2: blur clip eigenvalues
def fig_blur(clip="coke_blur", t_detail=12):
    s = stats(f"{clip}_flow_assisted")
    mi = CFG.track.flow.min_eigenvalue
    ts = [t for t, r in enumerate(s["flow"]) if r]
    lam2 = [np.median(np.array(s["flow"][t]["eigenvalues"])[:, 1]) for t in ts]
    nval = [int(np.sum(s["flow"][t]["valid"])) for t in ts]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].semilogy(ts, lam2, "o-", ms=4, label="median lambda_2"); ax[0].axhline(mi, color="k", ls="--", lw=0.8, label="min_eigenvalue")
    ax2 = ax[0].twinx(); ax2.plot(ts, nval, "s--", color="tab:orange", ms=3, label="valid points"); ax2.set_ylabel("valid points", color="tab:orange")
    ax[0].set_xlabel("frame index (flow-tracked frames)"); ax[0].set_ylabel("median lambda_2"); ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title(f"{clip} · flow_assisted: structure tensor at the tracked points", fontsize=9)
    rec = s["flow"][t_detail]; e = np.array(rec["eigenvalues"]); v = np.array(rec["valid"], bool)
    res = np.array(rec["residuals"], float)
    ap = e[:, 1] < mi; rf = (~ap) & (res > CFG.track.flow.max_residual)
    ax[1].loglog(e[v, 0] + 1e-9, e[v, 1] + 1e-9, ".", color="green", label=f"valid ({v.sum()})")
    ax[1].loglog(e[rf, 0] + 1e-9, e[rf, 1] + 1e-9, "^", color="red", ms=5, label=f"residual fail ({rf.sum()})")
    ax[1].loglog(e[ap, 0] + 1e-9, e[ap, 1] + 1e-9, "x", color="purple", ms=6, label=f"lambda_2 < min ({ap.sum()})")
    ax[1].axhline(mi, color="k", ls="--", lw=0.8); ax[1].set_xlabel("lambda_1"); ax[1].set_ylabel("lambda_2"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"frame {t_detail}: aperture-type vs brightness-constancy failures", fontsize=9)
    fig.tight_layout()
    save(fig, "q3_blur_eigenvalues.png")


# ---------------------------------------------------------------- Q3: monitor timeline of the combined clip
def fig_timeline(clip="coke_unconstrained"):
    s = stats(f"{clip}_flow_assisted")
    gt = data.load_diagnostic_corners(clip)
    hidden = {i for i, f in enumerate(s["flags"]) if f[0] == "HIDE"}
    errs = {f: M.corner_error(np.array(s["corners"][f]), gt[f]) for f in gt
            if f < len(s["status"]) and s["status"][f] == "ok" and f not in hidden}
    errs = {f: e for f, e in errs.items() if np.isfinite(e)}
    mon = CFG.track.monitor
    fig = viz.plot_monitor_timeline(
        s["stats"], [tuple(f) for f in s["flags"]],
        events={"card crossing 1": (210, 222), "card crossing 2": (302, 314), "fast rotation A": (80, 96), "fast rotation B": (448, 464)},
        errors=errs,
        thresholds={"inlier_ratio": mon.min_inlier_ratio, "mean_flow": mon.max_mean_flow, "brightness_change": mon.max_brightness_change},
        title=f"{clip} · flow_assisted, full resolution (red: HIDE, orange: REINIT; dashed: frozen thresholds)")
    save(fig, "q3_monitor_timeline_combined.png")


# ---------------------------------------------------------------- Q1 follow-up: viewpoint frames 448 / 464 / 480
def fig_viewpoint_late(clip="coke_perspective", fs=(448, 464, 480)):
    video, refp = data.clip_paths(clip)
    ref = sp.read_image(refp); cfg = LOC; cfg.warp.reference_shape = ref.shape[:2]
    gt = data.load_diagnostic_corners(clip)
    fr = frames_at(video, fs)
    import cv2
    fig, ax = plt.subplots(1, len(fs), figsize=(15, 4))
    for a, f in zip(ax, fs):
        g = sp.to_gray(fr[f]); m = sp.quad_mask(g.shape, gt[f])
        sharp = float(cv2.Laplacian(g.astype(np.float64), cv2.CV_64F)[m].var())
        H, c, pairs, inl, diag = localise_board(ref, fr[f], cfg)
        x0, x1, y0, y1 = crop_box(gt[f], g.shape, 20)
        a.imshow(np.clip(fr[f][y0:y1, x0:x1], 0, 1)); a.axis("off")
        a.set_title(f"frame {f}: {diag['n_inliers']} inliers, board sharpness {sharp * 1e4:.1f}e-4", fontsize=9)
        print("viewpoint", f, "inliers", diag["n_inliers"], "sharpness", sharp)
    fig.suptitle(f"{clip}: board crops of three consecutive annotated frames (full resolution, frozen config)", fontsize=10)
    fig.tight_layout()
    save(fig, "q1_viewpoint_448_464_480.png")


if __name__ == "__main__":
    for fn in sys.argv[1:] or ["fig_inliers", "fig_worst", "fig_parallax", "fig_flow_run", "fig_blur", "fig_timeline", "fig_viewpoint_late"]:
        globals()[fn]()
