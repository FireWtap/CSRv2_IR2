"""Pre-download E9's datasets and released checkpoints into $HF_HOME on a login node (ledger D-008).

E9 jobs run with HF_HUB_OFFLINE=1: many parallel jobs calling the HF API anonymously get HTTP 429.
Retries with backoff, since the login node gets rate-limited too. Safe to rerun (all cached).

    source jobs/snellius/env.sh && uv run --frozen python scripts/prefetch_E9.py [task_type ...]
"""

from __future__ import annotations

import sys
import time
import tomllib

import mteb
from huggingface_hub import snapshot_download

from csrv2_repro.common import HF_SUBSETS, REPO_ROOT, TASK_TYPES


def retry(what: str, fn, tries: int = 8):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - HF raises many types for rate limits / timeouts
            wait = 120 * (attempt + 1)
            print(f"FAIL {what} (attempt {attempt}, retry in {wait}s): {e!r:.200}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {what}")


def main() -> None:
    types = sys.argv[1:] or [t for t in TASK_TYPES if t != "retrieval"]
    with open(REPO_ROOT / "configs" / "E9_csrv2_released.toml", "rb") as f:
        models = tomllib.load(f)["models"]
    with open(REPO_ROOT / "configs" / "E9_dense.toml", "rb") as f:
        dense = tomllib.load(f)
    retry("backbone", lambda: snapshot_download(dense["model"], revision=dense["revision"]))
    for t in types:
        m = models[t]
        print("ckpt", t, retry(m["model"], lambda: snapshot_download(m["model"], revision=m["revision"])))
        for name in TASK_TYPES[t]:
            def load(name=name):
                task = mteb.get_task(name, languages=["eng"], hf_subsets=HF_SUBSETS.get(name))
                task.load_data()
                return task.hf_subsets

            print("ok", t, name, retry(name, load), flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
