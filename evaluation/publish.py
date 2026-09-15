"""Copy an eval run into evaluation/results/published/ - the only results that
are committed and shipped in the image, so what the UI shows is an explicit,
reviewable choice.

    python -m evaluation.publish evaluation/results/<run>.json [--label "..."]
    python -m evaluation.publish evaluation/results/compare_retrieval_<ts>.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PUBLISHED_DIR = Path("evaluation/results/published")


def publish(path: Path, label: str | None = None) -> Path:
    from evaluation.run_eval import EvalResult

    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    target = PUBLISHED_DIR / path.name
    if path.suffix == ".json":
        result = EvalResult.model_validate_json(path.read_text(encoding="utf-8"))  # must parse
        if label:
            result.config["label"] = label
        target.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    elif path.suffix == ".md":
        shutil.copyfile(path, target)
    else:
        raise SystemExit(f"unsupported file type: {path.name}")
    return target


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+")
    p.add_argument("--label", default=None, help="short human label stored in the run's config")
    args = p.parse_args()
    for raw in args.paths:
        out = publish(Path(raw), args.label)
        print(f"published {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
