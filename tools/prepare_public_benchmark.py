"""Prepare a frozen BIRD support/split manifest; no model calls or downloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataverify.evaluation.public_benchmark import prepare_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a prepared manifest; choose a new output.")
    manifest = prepare_manifest(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "total",
                    "eligible",
                    "eligible_by_database",
                    "excluded_reasons",
                    "test_ids",
                    "development_ids",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
