"""Write the eight retrieval test tasks to an E7 manifest."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from csrv2_repro.common import RETRIEVAL_TASKS
from csrv2_repro.e7_profiling import TextRow, task_rows


def select_rows() -> list[TextRow]:
    """Keep every test query and corpus row for the eight cached retrieval tasks."""
    return [row for task in RETRIEVAL_TASKS for row in task_rows(task)]


def main() -> None:
    """Write the ordered manifest without replacing an existing one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError(output)

    rows = select_rows()
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        output.hardlink_to(stream.name)
    documents = sum(row["role"] == "document" for row in rows)
    print(f"{output}: {documents:,} documents, {len(rows) - documents:,} queries")


if __name__ == "__main__":
    main()
