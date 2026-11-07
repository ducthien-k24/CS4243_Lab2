#!/usr/bin/env python3
"""Check the CS4243 Lab 2 submission and pack it into one zip.

Run from the repository root (the folder holding code/ and data/), with the
course environment (Python 3.11 + code/requirements.txt):

    python make_submission.py                    # check everything, then zip
    python make_submission.py --check-only       # check, do not zip
    python make_submission.py --skip-tests       # skip the unit-test run
    python make_submission.py --clip poster_03 --artwork ad4.png --out CS4243_Lab2.zip

The zip has two top-level folders, as in the release (handout "Submission
Checklist", code/README.md "Submission"):

    code/   the whole folder, minus caches and editor/OS files
    data/   only data/personal/<clip>/ and data/artwork/<artwork>; the supplied
            data (common_clip/, diagnostic/, genai_oracle/, artwork/ad.png) is
            left out.

Any failed check stops the zip unless --force is given. Warnings never stop it.

Note: the stress-table rebuild runs the supplied ``overlaylab.report``, which
re-evaluates every video and rewrites code/outputs/evaluation/*_report.json in
place (same videos -> same reports). It takes several minutes; --skip-rebuild
skips it.
"""
from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
DATA = ROOT / "data"

ASSESSED = {
    "keypoints.py": ["harris_response", "detect_corners", "describe_keypoints", "match_descriptors"],
    "geometry.py": ["bilinear_sample", "normalise_points", "dlt_homography", "symmetric_transfer_error",
                    "ransac_homography", "transfer_points", "warp_artwork"],
    "localisation.py": ["localise_board", "artwork_to_frame", "overlay_frame", "evaluate_audit_frames"],
    "temporal.py": ["lk_flow", "track_board", "failure_monitor"],
    "compositing.py": ["hysteresis", "compensated_difference", "occlusion_mask",
                       "select_difference_thresholds", "composite"],
}
SUPPLIED_VIDEOS = [
    "common_clip_independent", "common_clip_flow_assisted", "common_clip_flow_assisted_no_occlusion",
    "coke_perspective_flow_assisted", "coke_scale_flow_assisted", "coke_blur_flow_assisted",
    "coke_out_of_view_flow_assisted", "coke_unconstrained_flow_assisted",
]
OVERLAYLAB = ("config data keypoints geometry localisation temporal compositing metrics evaluate report "
              "pipelines run_pipeline extract_frames annotate to_webm supplied synthetic visualize __init__").split()
TESTS = ("test_keypoints test_geometry test_flow test_compositing test_metrics test_evaluate test_report "
         "test_real_data test_annotate").split()
MANIFESTS = ("dataset_summary.json common_clip_corners.csv common_clip_masks.csv diagnostic_corners.csv "
             "diagnostic_masks.csv capture_manifest_template.csv audit_corners_template.csv "
             "audit_masks_template.csv genai_conditions.json genai_log_template.csv capture_manifest.csv "
             "audit_corners.csv audit_masks.csv genai_log.csv").split()
NOTEBOOKS = ["01_keypoints_and_geometry", "02_localisation_and_temporal",
             "03_occlusion_and_compositing", "04_personal_clips_and_genai_analysis"]

# Never packed, anywhere in the tree.
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", ".git", ".venv", "venv",
                ".vscode", ".idea"}
EXCLUDE_FILES = ["*.pyc", "*.pyo", ".DS_Store", "._*", "Thumbs.db", "*.webm", "*.safetensors", "*.ckpt",
                 "*.pt", "*.pth", "*.zip"]
# Already-compressed formats are stored, not deflated (faster, same size).
STORED_EXT = {".mp4", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".npz"}


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def ok(self, msg):
        print(f"  [ OK ] {msg}")

    def fail(self, msg):
        self.errors.append(msg)
        print(f"  [FAIL] {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  [WARN] {msg}")

    def check(self, cond, ok_msg, fail_msg):
        (self.ok if cond else self.fail)(ok_msg if cond else fail_msg)
        return cond


def section(title):
    print(f"\n== {title}")


# --------------------------------------------------------------------------- checks

def check_required_files(r: Report, clip: str):
    section("Required files (code/README.md, Submission)")
    req = [f"manifests/{m}" for m in MANIFESTS]
    req += [f"notebooks/{n}.ipynb" for n in NOTEBOOKS]
    req += [f"overlaylab/{m}.py" for m in OVERLAYLAB]
    req += [f"tests/{t}.py" for t in TESTS]
    req += ["README.md", "REPORT.md", "REPORT_TEMPLATE.md", "requirements.txt",
            "outputs/stress_tables.md", "outputs/stress_tables.json", "outputs/thresholds.json",
            "outputs/stage1_settings.json", f"outputs/scorecards/{clip}_scorecard.md",
            "outputs/genai/output.mp4", "outputs/genai/inputs",
            "outputs/evaluation/genai_frames_report.json", "outputs/evaluation/stage4_genai_reports.json"]
    for v in SUPPLIED_VIDEOS + [f"{clip}_independent", f"{clip}_flow_assisted"]:
        req += [f"outputs/videos/{v}.mp4", f"outputs/videos/{v}_stats.json", f"outputs/evaluation/{v}_report.json"]
    missing = [p for p in req if not (CODE / p).exists()]
    r.check(not missing, f"all {len(req)} required paths present", f"missing: {missing}")

    # A *_masks/ folder exists only for runs that have ground-truth mask frames.
    for v in SUPPLIED_VIDEOS + [f"{clip}_independent", f"{clip}_flow_assisted"]:
        stats = CODE / f"outputs/videos/{v}_stats.json"
        if not stats.exists():
            continue
        s = json.loads(stats.read_text())
        if s.get("mask_frames") and not (CODE / "outputs/videos" / f"{v}_masks").is_dir():
            r.fail(f"{v}: stats list mask frames but outputs/videos/{v}_masks/ is missing")

    audit = sorted((CODE / "outputs/genai").glob("frame_*.png"))
    r.check(bool(audit), f"{len(audit)} generative audit frames in outputs/genai/",
            "no frame_XXXXXX.png audit extracts in outputs/genai/")


def check_assessed_code(r: Report):
    section("Assessed functions (no stubs left)")
    for fname, funcs in ASSESSED.items():
        path = CODE / "overlaylab" / fname
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for f in funcs:
            node = defs.get(f)
            if node is None:
                r.fail(f"{fname}: {f} not found (interface renamed?)")
                continue
            body = [s for s in node.body
                    if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))]
            stub = (not body or all(isinstance(s, ast.Pass) for s in body)
                    or (len(body) == 1 and isinstance(body[0], ast.Raise)
                        and "NotImplementedError" in ast.unparse(body[0])))
            if stub:
                r.fail(f"{fname}: {f} is still a stub")
        # Imports beyond numpy / cv2 / the standard helpers look like editor auto-imports or forbidden libraries.
        allowed = {"__future__", "typing", "dataclasses", "itertools", "math", "functools", "numpy", "cv2"}
        for n in ast.walk(tree):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else [n.module or ""] if isinstance(n, ast.ImportFrom) and n.level == 0 else [])
            for mod in mods:
                if mod.split(".")[0] not in allowed:
                    r.warn(f"{fname}:{n.lineno}: unexpected import {ast.unparse(n)!r}")
    if not any(e.startswith(tuple(ASSESSED)) for e in r.errors):
        r.ok(f"all {sum(map(len, ASSESSED.values()))} assessed functions implemented")


def check_notebooks(r: Report):
    section("Notebooks (run top to bottom, no errors)")
    for n in NOTEBOOKS:
        nb = json.loads((CODE / "notebooks" / f"{n}.ipynb").read_text(encoding="utf-8"))
        cells = [c for c in nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip()]
        counts = [c.get("execution_count") for c in cells]
        errors = [i for i, c in enumerate(cells) for o in c.get("outputs", []) if o.get("output_type") == "error"]
        problems = []
        if None in counts:
            problems.append(f"{counts.count(None)} unexecuted cells")
        elif counts != list(range(1, len(counts) + 1)):
            problems.append(f"execution counts not 1..{len(counts)} (not a fresh top-to-bottom run)")
        if errors:
            problems.append(f"error output in code cells {errors}")
        r.check(not problems, f"{n}: {len(cells)} cells, fresh run, no errors", f"{n}: {'; '.join(problems)}")

        # Required (non-optional) YOUR CODE HERE prompts that have nothing after them in the cell.
        for i, c in enumerate(cells):
            lines = "".join(c["source"]).splitlines()
            for j, line in enumerate(lines):
                if "YOUR CODE HERE" in line and "(optional)" not in line:
                    rest = [x for x in lines[j + 1:] if x.strip() and not x.strip().startswith("#")]
                    if not rest:
                        r.warn(f"{n} code cell {i}: required prompt has no code after it: "
                               f"{line.strip()[:110]}")


def check_tests(r: Report, python: str):
    section("Unit tests (python -m unittest discover -s tests)")
    p = subprocess.run([python, "-m", "unittest", "discover", "-s", "tests"], cwd=CODE,
                       capture_output=True, text=True)
    tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
    r.check(p.returncode == 0, f"tests pass: {' | '.join(tail)}", f"tests failed: {' | '.join(tail)}")


def check_stress_tables(r: Report, python: str):
    section("Stress tables (rebuilt from the saved runs, must be unedited)")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "stress_tables.md"
        p = subprocess.run([python, "-m", "overlaylab.report", "--out", str(out), "--quiet"], cwd=CODE,
                           capture_output=True, text=True)
        if p.returncode != 0:
            r.fail(f"overlaylab.report failed: {(p.stderr or p.stdout).strip().splitlines()[-1:]}")
            return
        same = out.read_text() == (CODE / "outputs/stress_tables.md").read_text()
        r.check(same, "outputs/stress_tables.md identical to a fresh rebuild",
                "outputs/stress_tables.md differs from a fresh rebuild (edited, or runs changed)")
    tables = [l for l in (CODE / "outputs/stress_tables.md").read_text().splitlines() if l.startswith("|")]
    report = set((CODE / "REPORT.md").read_text(encoding="utf-8").splitlines())
    missing = [l for l in tables if l not in report]
    r.check(not missing, f"all {len(tables)} stress-table lines pasted unchanged in REPORT.md",
            f"{len(missing)} stress-table lines not pasted verbatim in REPORT.md, e.g. {missing[:1]}")


def check_report(r: Report, clip: str, artwork: str):
    section("REPORT.md")
    text = (CODE / "REPORT.md").read_text(encoding="utf-8")
    # Draft notes and placeholder text outside the template's guidance comments.
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    for pat in [r"\bTODO\b", r"\bTBD\b", r"\bFIXME\b", r"\bXXX\b", r"DRAFT STATUS", r"\?\?\?"]:
        hits = [i + 1 for i, l in enumerate(body.splitlines()) if re.search(pat, l)]
        if hits:
            r.fail(f"placeholder {pat!r} in REPORT.md text")
    # Relative links must resolve; links into the supplied data/ resolve for the grader but are not packed.
    links = re.findall(r"\]\(([^)\s#]+)(?:#[^)]*)?\)", text)
    broken, supplied = [], []
    for link in links:
        if re.match(r"[a-z]+://", link):
            continue
        target = (CODE / link).resolve()
        if not target.exists():
            broken.append(link)
        elif DATA in target.parents and not _is_student_data(target, clip, artwork):
            supplied.append(link)
    r.check(not broken, f"{len(links)} links, none broken", f"broken links: {broken}")
    if supplied:
        r.warn(f"links into supplied data (present on the grader's side, not in the zip): {supplied}")
    # AI-use disclosure table.
    m = re.search(r"## AI-use disclosure table\n(.*?)(?:\n## |\Z)", text, flags=re.S)
    if not r.check(m is not None, "AI-use disclosure table present", "AI-use disclosure table missing"):
        return
    rows = [l for l in m.group(1).splitlines() if l.startswith("|")][2:]
    cells = [[c.strip() for c in row.strip("|").split("|")] for row in rows]
    empty = [i + 1 for i, c in enumerate(cells) if not c[0] or not c[1] or not c[3] or not c[4]]
    r.check(rows and not empty, f"{len(rows)} AI-use rows, all filled",
            f"AI-use rows with empty tool/artefact/contribution/verification: {empty}")
    markers = set(re.findall(r"\[(?:AI-[A-Z]+|HUMAN-[A-Z]+)\]", text))
    r.ok(f"markers used in REPORT.md: {' '.join(sorted(markers))}")


def _is_student_data(path: Path, clip: str, artwork: str) -> bool:
    return (DATA / "personal" / clip) in [path, *path.parents] or path == DATA / "artwork" / artwork


def check_student_data(r: Report, clip: str, artwork: str):
    section(f"Student data (data/personal/{clip}/, data/artwork/{artwork})")
    d = DATA / "personal" / clip
    for p in ["clip.mp4", "reference.png", "frames", "masks"]:
        r.check((d / p).exists(), f"data/personal/{clip}/{p}", f"missing data/personal/{clip}/{p}")
    r.check((DATA / "artwork" / artwork).exists(), f"data/artwork/{artwork}", f"missing data/artwork/{artwork}")
    if artwork == "ad.png":
        r.fail("ad.png is the supplied artwork; pass your own artwork with --artwork")

    # Every file the audit manifests point at must be inside what we pack.
    for man, cols in [("audit_corners.csv", ["frame_path"]), ("audit_masks.csv", ["frame_path", "mask_path"])]:
        rows = list(csv.DictReader(open(CODE / "manifests" / man, newline="", encoding="utf-8")))
        bad = []
        for row in rows:
            if row.get("clip_id") != clip:
                bad.append(f"clip_id {row.get('clip_id')}")
                continue
            for col in cols:
                target = (CODE / "manifests" / row[col]).resolve()
                if not target.exists() or not _is_student_data(target, clip, artwork):
                    bad.append(row[col])
        r.check(not bad, f"{man}: {len(rows)} rows, all paths resolve inside the packed data",
                f"{man}: bad rows {bad[:5]}")

    with open(CODE / "manifests" / "capture_manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next((x for x in rows if x["clip_id"] == clip), None)
    if r.check(row is not None, f"capture_manifest.csv has {clip}", f"capture_manifest.csv has no row for {clip}"):
        for key in ("consent_obtained", "metadata_removed"):
            r.check(row.get(key, "").strip().lower() == "yes", f"capture manifest {key} = yes",
                    f"capture manifest {key} is {row.get(key)!r}")

    others = sorted(p.name for p in (DATA / "personal").iterdir() if p.is_dir() and p.name != clip)
    if others:
        r.warn(f"other personal clips not packed: {others}")
    extra_art = sorted(p.name for p in (DATA / "artwork").iterdir()
                       if p.is_file() and p.name not in (artwork, "ad.png") and not p.name.startswith("."))
    if extra_art:
        r.warn(f"other artwork files not packed: {extra_art}")


# --------------------------------------------------------------------------- packing

def _excluded(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    # Retired experiment files and diagnostic previews stay outside the submission.
    if any("vace" in part.lower() or "before_" in part.lower() for part in rel.parts):
        return True
    if rel.parts[:2] == ("outputs", "annotation_check"):
        return True
    if rel.parts[:3] == ("student_files", "personal_preparation", "poster_03") and rel.name in {
        "README.md", "STAGE4_SETUP.md"
    }:
        return True
    return any(fnmatch.fnmatch(rel.name, pat) for pat in EXCLUDE_FILES)


def collect(clip: str, artwork: str, other_clips: list[str]) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    skipped_stale: list[str] = []
    for p in sorted(CODE.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(CODE)
        if _excluded(rel):
            continue
        # Evaluator reports / videos of personal clips that are not submitted.
        if any(part == o or part.startswith(f"{o}_") for o in other_clips for part in rel.parts):
            skipped_stale.append(str(rel))
            continue
        files.append((p, f"code/{rel.as_posix()}"))
    for base in [DATA / "personal" / clip]:
        for p in sorted(base.rglob("*")):
            if p.is_file() and not _excluded(p.relative_to(DATA)):
                files.append((p, f"data/{p.relative_to(DATA).as_posix()}"))
    files.append((DATA / "artwork" / artwork, f"data/artwork/{artwork}"))
    if skipped_stale:
        print(f"  [INFO] left out outputs of personal clips that are not submitted: {skipped_stale}")
    return files


def write_zip(files: list[tuple[Path, str]], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    total = sum(src.stat().st_size for src, _ in files)
    done = 0
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for k, (src, arc) in enumerate(files, 1):
            method = zipfile.ZIP_STORED if src.suffix.lower() in STORED_EXT else zipfile.ZIP_DEFLATED
            z.write(src, arc, compress_type=method)
            done += src.stat().st_size
            if k % 200 == 0 or k == len(files):
                print(f"\r  packed {k}/{len(files)} files, {done / 2**20:.0f}/{total / 2**20:.0f} MB",
                      end="", flush=True)
    print()
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad:
            raise SystemExit(f"zip verification failed at {bad}")
        tops = sorted({n.split("/")[0] for n in z.namelist()})
        if tops != ["code", "data"]:
            raise SystemExit(f"unexpected top-level folders {tops}")
    tmp.replace(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="poster_03", help="personal clip id under data/personal/ (default poster_03)")
    ap.add_argument("--artwork", default="ad4.png", help="your artwork under data/artwork/ (default ad4.png)")
    ap.add_argument("--out", default=str(ROOT / "CS4243_Lab2_submission.zip"), help="output zip path")
    ap.add_argument("--python", default=sys.executable, help="interpreter for tests and the stress-table rebuild")
    ap.add_argument("--skip-tests", action="store_true", help="do not run the unit tests")
    ap.add_argument("--skip-rebuild", action="store_true", help="do not rebuild the stress tables")
    ap.add_argument("--check-only", action="store_true", help="run the checks, write no zip")
    ap.add_argument("--force", action="store_true", help="write the zip even if a check fails")
    args = ap.parse_args(argv)

    r = Report()
    check_required_files(r, args.clip)
    check_assessed_code(r)
    check_notebooks(r)
    if not args.skip_tests:
        check_tests(r, args.python)
    if not args.skip_rebuild:
        check_stress_tables(r, args.python)
    check_report(r, args.clip, args.artwork)
    check_student_data(r, args.clip, args.artwork)

    section("Summary")
    print(f"  {len(r.errors)} failed, {len(r.warnings)} warnings")
    if args.check_only:
        return 1 if r.errors else 0
    if r.errors and not args.force:
        print("  zip NOT written (fix the failures or pass --force)")
        return 1

    section("Packing")
    others = [p.name for p in (DATA / "personal").iterdir() if p.is_dir() and p.name != args.clip]
    files = collect(args.clip, args.artwork, others)
    out = Path(args.out).resolve()
    write_zip(files, out)
    print(f"  wrote {out} ({out.stat().st_size / 2**20:.0f} MB, {len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
