"""Download the official Diffusers checkpoint without loading it. [AI-CODE]"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

MODEL = "Wan-AI/Wan2.1-VACE-14B-diffusers"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--revision", default="main", help="Hub commit/tag; resolved and recorded once")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    marker = args.weights / "lab_snapshot.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["repo_id"] != MODEL:
            raise ValueError("Existing directory belongs to a different model")
        if args.revision not in ("main", saved["revision"]):
            raise ValueError("Use a different directory for a different checkpoint revision")
        revision = saved["revision"]
    else:
        revision = HfApi().model_info(MODEL, revision=args.revision).sha
    snapshot_download(repo_id=MODEL, revision=revision, local_dir=args.weights,
                      allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*",
                                      "text_encoder/*", "transformer/*", "vae/*", "README.md", "LICENSE*"])
    index = json.loads((args.weights / "model_index.json").read_text(encoding="utf-8"))
    if index.get("_class_name") != "WanVACEPipeline":
        raise ValueError("Unexpected model format; require the official Diffusers checkpoint")
    marker.write_text(json.dumps({"repo_id": MODEL, "revision": revision,
                                 "downloaded_at": datetime.now(timezone.utc).isoformat()}, indent=2),
                      encoding="utf-8")
    print(f"Downloaded {MODEL}@{revision} to {args.weights.resolve()}")


if __name__ == "__main__":
    main()

