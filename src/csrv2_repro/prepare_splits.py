"""E3: freeze query partitions and export qrels for both LoRA and SAE preparation.

    uv run --frozen python -m csrv2_repro.prepare_splits --config configs/E3_splits.toml

Downloads qrels only; does not encode data, train models, or modify upstream code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

from csrv2_repro.common import OUTPUTS, REPO_ROOT, load_config, provenance, write_json
from csrv2_repro.splits import (
    PARTITIONS,
    build_task_split,
    load_manifest,
    load_source_qrels,
    seal_manifest,
    select_qrels,
)


def export_qrels(manifest: dict, sources: dict, directory: Path) -> dict:
    """Export complete judgments, never collapse a multi-positive query to one document."""
    checksums = {}
    for name, entry in manifest["tasks"].items():
        for partition in PARTITIONS:
            selected = select_qrels(entry, sources[name], partition)
            path = directory / name / f"{partition}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                for qid, docs in selected.items():
                    for did, score in sorted(docs.items()):
                        f.write(
                            json.dumps({"query-id": qid, "corpus-id": did, "score": score}) + "\n"
                        )
            checksums[str(path.relative_to(directory))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "qrels_checksums.json", checksums)
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/E3_splits.toml")
    parser.add_argument("--output", default="results/E3/split_manifest.json")
    parser.add_argument("--export-dir", help="default: scratch/outputs data_splits/<manifest SHA>")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(REPO_ROOT / args.config)
    options = {k: cfg[k] for k in ("seed", "test_fraction", "validation_fraction")}
    sources, entries = {}, {}
    for name, source in cfg["sources"].items():
        logging.info("Loading pinned qrels for %s", name)
        sources[name] = load_source_qrels(source)
        entries[name] = build_task_split(sources[name], source, **options)
        logging.info(
            "%s: %s", name, {p: s["n_queries"] for p, s in entries[name]["partitions"].items()}
        )
    manifest = seal_manifest(entries, **options)
    output = REPO_ROOT / args.output
    if output.exists() and load_manifest(output)["sha256"] != manifest["sha256"]:
        raise ValueError(
            f"Refusing to replace a different frozen manifest: {output}; choose --output"
        )
    directory = (
        Path(args.export_dir)
        if args.export_dir
        else Path(os.environ.get("SCRATCH_ROOT", OUTPUTS)) / "data_splits" / manifest["sha256"]
    )
    checksums = export_qrels(manifest, sources, directory)
    write_json(output, manifest)
    audit = {
        "manifest_sha256": manifest["sha256"],
        "source_code_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("splits.py", "prepare_splits.py")
        },
        "export_dir": str(directory.resolve()),
        "export_sha256": checksums,
        "tasks": {
            name: {
                "strategy": entry["strategy"],
                "counts": {
                    p: {k: v for k, v in spec.items() if k != "query_ids"}
                    for p, spec in entry["partitions"].items()
                },
                "query_overlap": 0,
            }
            for name, entry in entries.items()
        },
        "provenance": provenance(cfg),
    }
    write_json(output.with_name(output.stem + "_audit.json"), audit)
    logging.info("Frozen manifest: %s (%s)", output, manifest["sha256"])


if __name__ == "__main__":
    main()
