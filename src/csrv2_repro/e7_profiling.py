"""E7: profile raw-text encoding and exact search over prepared embeddings.

Both cache sources return NumPy matrices keyed by (model, role). Rows are texts,
columns are features. Shape notation below:

- Q: number of queries; N: number of corpus documents.
- B: queries in one timed batch; C: documents in one corpus block.
- D: backbone width (2,560 for Qwen3); H: sparse-head width (10,240 for CSRv2).
- F: width used by the search case (D for dense, k for MRL, H for CSRv2).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, SparseEncoder
from sentence_transformers.sparse_encoder.models import SparseAutoEncoder
from torch.nn import functional as F

from csrv2_repro.common import (
    OUTPUTS,
    RESULTS,
    load_config,
    provenance,
    upstream_prompts,
    write_json,
)
from csrv2_repro.encoders import embedding_cache_path

logger = logging.getLogger(__name__)
ROLES = ("query", "document")
MODELS = ("dense", "csrv2")
TextRow = dict[str, str]
EmbeddingsByModelAndRole = dict[tuple[str, str], np.ndarray]
EncodingOptions = tuple[str | None, int, int]


def main() -> None:
    """Parse E7 options and run one independent profiling stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("encode", "search"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    positive_keys = (
        ("encoding_batch_size", "encoding_k")
        if args.command == "encode"
        else ("query_batch_size", "search_rounds", "corpus_block_size", "top_n")
    )
    for key in positive_keys:
        if cfg[key] <= 0:
            parser.error(f"{key} must be positive")
    if cfg["prompt_mode"] not in ("both", "query"):
        parser.error("prompt_mode must be both or query")
    stride = cfg.get("encoding_profile_stride", 1)
    if type(stride) is not int or stride < 0:
        parser.error("encoding_profile_stride must be a nonnegative integer")
    root = Path(os.environ.get("SCRATCH_ROOT", OUTPUTS))
    run_dir = root / "e7" / cfg["run_name"]
    out = RESULTS / "E7" / cfg["run_name"]
    if args.command == "search":
        if "manifest" not in cfg:
            parser.error("search requires manifest in the config")
        source = cfg.get("embedding_source")
        if source not in ("cache", "encoded"):
            parser.error("search requires embedding_source to be cache or encoded")
        if source == "cache" and "cache_root" not in cfg:
            parser.error("search with embedding_source=cache requires cache_root")
        if source == "cache" and cfg["cache_corpus_chunk_size"] <= 0:
            parser.error("cache_corpus_chunk_size must be positive")
        if not cfg["ks"] or min(cfg["ks"]) <= 0:
            parser.error("ks must contain positive values")
    rows = load_rows(cfg)
    all_prompts = upstream_prompts()
    prompts = {task: all_prompts[task] for task in sorted({row["task"] for row in rows})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.command == "encode":
        measurements = encode_rows(cfg, rows, prompts, run_dir, device)
        write_json(
            out / "encoding.json",
            dict(
                measurements=measurements,
                embeddings=str(run_dir),
                provenance=provenance(cfg),
            ),
        )
    else:
        vectors, paths = (
            load_cached(cfg, rows, prompts, Path(cfg["cache_root"]))
            if source == "cache"
            else load_encoded(cfg, rows, prompts, run_dir)
        )
        for variant, indices, distinct_documents in search_corpora(cfg, rows):
            label = f"{source}_{variant}" if variant else source
            logger.info("Benchmarking corpus %s", label)
            trace_path = root / "e7" / "traces" / cfg["run_name"] / f"{label}.json"
            results, trace = run_search(cfg, vectors, device, trace_path, indices)
            write_json(
                out / f"retrieval_{label}.json",
                dict(
                    measurements=results,
                    source=source,
                    corpus_variant=variant,
                    distinct_document_texts=distinct_documents,
                    manifest=cfg["manifest"],
                    input_files=paths,
                    trace=trace,
                    provenance=provenance(cfg),
                ),
            )


def search_corpora(
    cfg: dict, rows: list[TextRow]
) -> Iterator[tuple[str, np.ndarray | None, int | None]]:
    """Choose document row indices while leaving all queries in manifest order."""
    repeat_to = cfg.get("repeat_corpus_to")
    duplicate_control = cfg.get("duplicate_control", False)
    if repeat_to is None and not duplicate_control:
        yield "", None, None
        return
    if repeat_to is not None and duplicate_control:
        raise ValueError("Choose either repeat_corpus_to or duplicate_control")

    texts = [row["text"].strip() for row in rows if row["role"] == "document"]
    rng = np.random.default_rng(cfg["corpus_seed"])
    if repeat_to is not None:
        if type(repeat_to) is not int or repeat_to < len(texts):
            raise ValueError("repeat_corpus_to must be an integer at least as large as the corpus")
        # Repeat every original row evenly, then spread copies across search blocks.
        indices = np.resize(rng.permutation(len(texts)), repeat_to)
        rng.shuffle(indices)
        yield "repeated", indices, len(set(texts))
        return

    first_by_text = {}
    for index, text in enumerate(texts):
        first_by_text.setdefault(text, index)
    unique = rng.permutation(list(first_by_text.values()))
    pool = unique[: (len(unique) + 3) // 4]
    repeated = np.resize(pool, len(unique))
    rng.shuffle(repeated)
    yield "unique", unique, len(unique)
    yield "repeated", repeated, len(pool)


# Inputs and saved embeddings


def task_rows(name: str, split: str = "test") -> list[TextRow]:
    """Read queries and corpus in the original MTEB cache order."""
    import mteb

    task = mteb.get_task(name)
    task.load_data(eval_splits=[split])
    queries = task.queries[split]
    corpus = task.corpus[split]
    # MTEB 1.38.39 sorts document IDs in reverse order before creating cache chunks.
    rows = [
        {"id": key, "text": text, "role": "query", "task": task.metadata.name}
        for key, text in queries.items()
    ]
    rows += [
        {"id": key, "text": corpus[key], "role": "document", "task": task.metadata.name}
        for key in sorted(corpus, reverse=True)
    ]
    return rows


def load_rows(cfg: dict) -> list[TextRow]:
    """Read and validate an E7 manifest or load a single task for encoding."""
    if "manifest" in cfg:
        with Path(cfg["manifest"]).open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        rows = task_rows(cfg["task"], cfg["split"])
    required_fields = {"id", "text", "role", "task"}
    if any(
        not isinstance(row, dict)
        or row.keys() != required_fields
        or any(not isinstance(value, str) for value in row.values())
        for row in rows
    ):
        raise ValueError("Each input row requires exactly id, text, role and task.")
    keys = [(row["role"], row["task"], row["id"]) for row in rows]
    if len(keys) != len(set(keys)) or {row["role"] for row in rows} != set(ROLES):
        raise ValueError("Inputs require both roles and unique (role, task, id) keys.")
    return rows


def model_tag(model: dict, cfg: dict) -> str:
    """Match the E1/E6b cache identity, including precision and tokenizer settings."""
    return "|".join(
        str(value)
        for value in (
            model["model"],
            model["revision"],
            cfg["dtype"],
            cfg["max_seq_length"],
            cfg["padding_side"],
        )
    )


def read_vectors(path: Path, count: int, dim: int) -> np.ndarray:
    """Read a cache matrix and check its row count, width and storage dtype."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding file: {path}")
    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (count, dim) or vectors.dtype != np.float32:
        raise ValueError(
            f"{path}: expected float32 [{count}, {dim}], got {vectors.dtype} {vectors.shape}"
        )
    logger.info("Read %s (%d rows)", path, count)
    return vectors


def cached_chunks(
    cfg: dict, rows: list[TextRow], prompts: dict[str, str], root: Path
) -> Iterator[tuple[tuple[str, str], Path, np.ndarray, np.ndarray, np.ndarray]]:
    """Resolve cache headers and yield matching source and manifest row indices."""
    groups = defaultdict(list)
    counts = dict.fromkeys(ROLES, 0)
    for row in rows:
        role = row["role"]
        groups[row["task"], role].append((counts[role], row))
        counts[role] += 1

    for task in dict.fromkeys(row["task"] for row in rows):
        original = task_rows(task, cfg.get("split", "test"))
        for role in ROLES:
            selection = groups.get((task, role), [])
            if not selection:
                continue
            full = [row for row in original if row["role"] == role]
            positions = {row["id"]: index for index, row in enumerate(full)}
            for _, row in selection:
                if row["id"] not in positions or full[positions[row["id"]]] != row:
                    raise ValueError(f"Manifest row does not match {task}/{role}/{row['id']}")
            source_indices = np.array([positions[row["id"]] for _, row in selection])
            target_indices = np.array([index for index, _ in selection])
            texts = [row["text"] for row in full]
            prompt = prompts[task] if cfg["prompt_mode"] == "both" or role == "query" else None
            chunk_size = len(texts) if role == "query" else cfg["cache_corpus_chunk_size"]
            for start in range(0, len(texts), chunk_size):
                end = start + chunk_size
                selected = (source_indices >= start) & (source_indices < end)
                if not selected.any():
                    continue
                for name in MODELS:
                    model = cfg[name]
                    path = embedding_cache_path(
                        root / "emb_cache" / model["cache_run"],
                        model_tag(model, cfg),
                        texts[start:end],
                        prompt,
                        task,
                        role,
                    )
                    chunk = read_vectors(path, min(end, len(texts)) - start, cfg["embedding_dim"])
                    yield (
                        (name, role), path, chunk, source_indices[selected] - start,
                        target_indices[selected],
                    )


def load_cached(
    cfg: dict, rows: list[TextRow], prompts: dict[str, str], root: Path
) -> tuple[EmbeddingsByModelAndRole, list[str]]:
    """Gather the requested cache rows into four matrices in manifest order."""
    vectors = {}
    for role in ROLES:
        count = sum(row["role"] == role for row in rows)
        for name in MODELS:
            vectors[name, role] = np.empty((count, cfg["embedding_dim"]), dtype=np.float32)
    paths = []
    for key, path, chunk, source_indices, target_indices in cached_chunks(cfg, rows, prompts, root):
        vectors[key][target_indices] = chunk[source_indices]
        paths.append(str(path))
    return vectors, paths


def model_inputs(cfg: dict) -> dict:
    """Record the settings that must match between encoding and retrieval."""
    return {
        key: cfg[key]
        for key in (
            "dense",
            "csrv2",
            "dtype",
            "max_seq_length",
            "padding_side",
            "prompt_mode",
            "embedding_dim",
        )
    }


def load_encoded(
    cfg: dict,
    rows: list[TextRow],
    prompts: dict[str, str],
    run_dir: Path,
) -> tuple[EmbeddingsByModelAndRole, list[str]]:
    """Return the same [Q, D] and [N, D] matrices from an E7 encoding run."""
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata != {"models": model_inputs(cfg), "prompts": prompts}:
        raise ValueError(f"Embedding settings do not match {run_dir / 'metadata.json'}")
    with (run_dir / "manifest.jsonl").open() as stream:
        for row in rows:
            if row != json.loads(next(stream, "null")):
                raise ValueError(f"Input mismatch at {row['role']}/{row['task']}/{row['id']}")
        if next(stream, None) is not None:
            raise ValueError("The encoded manifest has extra rows.")
    vectors, paths = {}, []
    for name in MODELS:
        for role in ROLES:
            path = run_dir / "embeddings" / f"{name}_{role}.npy"
            count = sum(row["role"] == role for row in rows)
            vectors[name, role] = read_vectors(path, count, cfg["embedding_dim"])
            paths.append(str(path))
    return vectors, paths


# Encoding benchmark


def load_backbone(
    cfg: dict,
    name: str,
    device: torch.device,
) -> tuple[SentenceTransformer, SparseAutoEncoder | None]:
    """Load the dense encoder or split the released model into backbone and sparse head."""
    spec = cfg[name]
    kwargs = dict(
        revision=spec["revision"],
        device=str(device),
        model_kwargs={"torch_dtype": getattr(torch, cfg["dtype"])},
        tokenizer_kwargs={"padding_side": cfg["padding_side"]},
    )
    if name == "dense":
        backbone, sae = SentenceTransformer(spec["model"], **kwargs), None
    else:
        model = SparseEncoder(spec["model"], **kwargs)
        index, sae = next((i, m) for i, m in enumerate(model) if isinstance(m, SparseAutoEncoder))
        backbone = SentenceTransformer(modules=list(model)[:index], device=str(device))
    backbone.max_seq_length = cfg["max_seq_length"]
    return backbone.eval(), sae


def sync(device: torch.device) -> None:
    """Wait for queued CUDA work before reading the wall clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def encode_batch(
    backbone: SentenceTransformer,
    sae: SparseAutoEncoder | None,
    texts: list[str],
    prompt: str | None,
    batch_size: int,
    k: int,
    markers: tuple[torch.cuda.Event, ...] = (),
) -> torch.Tensor:
    """Include sparse projection in the forward and return its reusable dense input."""
    if markers:
        markers[0].record()
    # One row per input text: [texts in this batch, D].
    dense = backbone.encode(
        texts,
        prompt=prompt,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    if markers:
        markers[1].record()
    if sae is not None:
        # [texts in this batch, H], with up to k nonzeros per row. Include this
        # work in the timer but return the pre-head matrix for later k sweeps.
        sae({"sentence_embedding": dense}, max_active_dims=k)
        if markers:
            markers[2].record()
    return dense


def time_encode_batch(
    backbone: SentenceTransformer,
    sae: SparseAutoEncoder | None,
    texts: list[str],
    options: EncodingOptions,
    device: torch.device,
    markers: tuple[torch.cuda.Event, ...] = (),
) -> tuple[torch.Tensor, float]:
    """Time one encoding forward using the existing batch-boundary synchronization."""
    sync(device)
    start = time.perf_counter()
    dense = encode_batch(backbone, sae, texts, *options, markers=markers)
    sync(device)
    return dense, time.perf_counter() - start


@torch.inference_mode()
def encode_rows(
    cfg: dict,
    rows: list[TextRow],
    prompts: dict[str, str],
    run_dir: Path,
    device: torch.device,
) -> dict:
    """Time raw-text encoding and publish all four pre-head arrays as one completed run."""
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists. Choose a new run_name to encode again.")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    timings = {}
    batch_size = cfg["encoding_batch_size"]
    profile_stride = cfg.get("encoding_profile_stride", 1)
    with TemporaryDirectory(prefix=f".{run_dir.name}-", dir=run_dir.parent) as tmp:
        staging = Path(tmp) / "run"
        (staging / "embeddings").mkdir(parents=True)
        for name in MODELS:
            backbone, sae = load_backbone(cfg, name, device)
            markers = ()
            if device.type == "cuda" and profile_stride:
                marker_count = 3 if sae is not None else 2
                markers = tuple(torch.cuda.Event(enable_timing=True) for _ in range(marker_count))
                # CUDA events are initialized lazily. Do that before the timed forwards.
                for marker in markers:
                    marker.record()
            timings[name] = {}
            for role in ROLES:
                rows_by_task = defaultdict(list)
                role_rows = [row for row in rows if row["role"] == role]
                for row_index, row in enumerate(role_rows):
                    rows_by_task[row["task"]].append((row_index, row["text"]))
                # Rows follow the manifest, even though the encoder batches by task.
                output = np.lib.format.open_memmap(
                    staging / "embeddings" / f"{name}_{role}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(role_rows), cfg["embedding_dim"]),
                )
                elapsed = 0.0
                profiled_batches = profiled_texts = 0
                backbone_stream_ms = sae_stream_ms = 0.0
                for task, items in rows_by_task.items():
                    prompt = (
                        prompts[task] if cfg["prompt_mode"] == "both" or role == "query" else None
                    )

                    options = (prompt, batch_size, cfg["encoding_k"])
                    warmup_texts = [text for _, text in items[:batch_size]]
                    for _ in range(cfg["encoding_warmup"]):
                        encode_batch(backbone, sae, warmup_texts, *options)
                    for offset in range(0, len(items), batch_size):
                        indices, texts = zip(*items[offset : offset + batch_size])
                        batch_markers = ()
                        if markers and profile_stride and offset // batch_size % profile_stride == 0:
                            batch_markers = markers
                        dense, seconds = time_encode_batch(
                            backbone, sae, list(texts), options, device, batch_markers
                        )
                        elapsed += seconds
                        if batch_markers:
                            profiled_batches += 1
                            profiled_texts += len(texts)
                            backbone_stream_ms += markers[0].elapsed_time(markers[1])
                            if sae is not None:
                                sae_stream_ms += markers[1].elapsed_time(markers[2])
                        # Store the intermediates from the timed forward, outside the timer.
                        output[list(indices)] = dense.float().cpu().numpy()
                output.flush()
                del output
                timings[name][role] = dict(
                    texts=len(role_rows),
                    seconds=elapsed,
                    texts_per_second=len(role_rows) / elapsed,
                    profiled_batches=profiled_batches,
                    profiled_texts=profiled_texts,
                    backbone_stream_ms_per_batch_mean=(
                        backbone_stream_ms / profiled_batches if profiled_batches else None
                    ),
                    sae_stream_ms_per_batch_mean=(
                        sae_stream_ms / profiled_batches if profiled_batches and sae is not None else None
                    ),
                )
                logger.info("%s %s: %.3f seconds for %d texts", name, role, elapsed, len(role_rows))
            del backbone, sae
        with (staging / "manifest.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(staging / "metadata.json", {"models": model_inputs(cfg), "prompts": prompts})
        staging.rename(run_dir)
    return timings


# Retrieval benchmark


def prepare_vectors(
    array: np.ndarray,
    device: torch.device,
    k: int | None = None,
    sae: SparseAutoEncoder | None = None,
) -> torch.Tensor:
    """Convert a pre-head matrix [rows, D] to normalized search vectors [rows, F]."""
    if sae is None and k is not None:
        # MRL keeps the first k coordinates: [rows, D] -> [rows, k].
        array = array[:, :k]
    vectors = torch.from_numpy(np.array(array, copy=True, order="C")).to(device)
    if sae is not None:
        # CSRv2 projects to [rows, H]. TopK zeros entries without shrinking H.
        vectors = sae(
            {"sentence_embedding": vectors.to(next(sae.parameters()))},
            max_active_dims=k,
        )["sentence_embedding"].float()
    return F.normalize(vectors, dim=1)


def representation(
    vectors: EmbeddingsByModelAndRole,
    name: str,
    k: int | None,
    sae: SparseAutoEncoder | None,
    block_size: int,
    device: torch.device,
    corpus_indices: np.ndarray | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return dense queries [Q, F] and device-resident corpus blocks [C, F]."""
    queries = prepare_vectors(vectors[name, "query"], device, k, sae)
    corpus = vectors[name, "document"]
    blocks = []
    count = len(corpus) if corpus_indices is None else len(corpus_indices)
    for start in range(0, count, block_size):
        selection = slice(start, start + block_size)
        rows = selection if corpus_indices is None else corpus_indices[selection]
        block = prepare_vectors(corpus[rows], device, k, sae)
        # CSR changes storage, not the logical shape [C, H].
        blocks.append(block.to_sparse_csr() if sae is not None else block)
    return queries, blocks


def score_block(block: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Multiply [C, F] by [F, B] to get cosine scores [C, B]."""
    if block.layout == torch.sparse_csr:
        return torch.sparse.mm(block, queries.T)
    return block @ queries.T


def search_topk(
    blocks: list[torch.Tensor],
    queries: torch.Tensor,
    top_n: int,
    score_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scores, corpus row indices), each of shape [up to top_n, B]."""
    block_top_scores = []
    block_top_indices = []
    corpus_offset = 0
    for index, block in enumerate(blocks):
        if score_events is not None:
            start, end = score_events[index]
            start.record()
        scores = score_block(block, queries)
        if score_events is not None:
            end.record()
        # Scores are [C, B]. Rank along the document axis to get [min(top_n, C), B].
        top_scores, local_indices = scores.topk(min(top_n, len(block)), dim=0)
        del scores
        block_top_scores.append(top_scores)
        block_top_indices.append(local_indices + corpus_offset)
        corpus_offset += len(block)
    if len(block_top_scores) == 1:
        return block_top_scores[0], block_top_indices[0]

    # Each block contributes up to top_n rows, giving [total candidates, B].
    candidate_scores = torch.cat(block_top_scores)
    best_scores, candidate_indices = candidate_scores.topk(min(top_n, corpus_offset), dim=0)
    document_indices = torch.cat(block_top_indices).gather(0, candidate_indices)
    return best_scores, document_indices


def verify_sparse(blocks: list[torch.Tensor], queries: torch.Tensor, top_n: int) -> None:
    """Check sparse scores and selected IDs on a small multi-block sample, allowing ties."""
    sample = [block.to_sparse_coo().coalesce() for block in blocks[:2]]
    small_blocks = []
    for block in sample:
        # Slice while sparse so verification never densifies a large production block.
        rows = min(64, block.shape[0])
        keep = block.indices()[0] < rows
        small_blocks.append(
            torch.sparse_coo_tensor(
                block.indices()[:, keep],
                block.values()[keep],
                (rows, block.shape[1]),
            )
            .coalesce()
            .to_sparse_csr()
        )
    # Add an all-zero query to exercise ties. Both score matrices are [sample docs, sample queries].
    sample_queries = torch.cat((queries[:4], torch.zeros_like(queries[:1])))
    dense_scores = torch.cat([block.to_dense() @ sample_queries.T for block in small_blocks])
    sparse_scores = torch.cat([score_block(block, sample_queries) for block in small_blocks])
    torch.testing.assert_close(sparse_scores, dense_scores, atol=1e-6, rtol=1e-5)
    actual, document_indices = search_topk(small_blocks, sample_queries, top_n)
    expected = dense_scores.topk(min(top_n, len(dense_scores)), dim=0).values
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    # Compare scores at the returned indices, allowing different IDs at a tied cutoff.
    torch.testing.assert_close(
        dense_scores.gather(0, document_indices), expected, atol=1e-6, rtol=1e-5
    )


def query_batches(queries: torch.Tensor, size: int) -> Iterator[torch.Tensor]:
    """Cycle through ordered full-size batches, wrapping deterministically at the end."""
    size = min(size, len(queries))
    offsets = torch.arange(size, device=queries.device)
    start = 0
    while True:
        yield queries[(offsets + start) % len(queries)]
        start = (start + size) % len(queries)


def storage_bytes(tensor: torch.Tensor) -> int:
    """Count payload bytes, including CSR column indices and row pointers."""
    parts = (
        (tensor.values(), tensor.col_indices(), tensor.crow_indices())
        if tensor.layout == torch.sparse_csr
        else (tensor,)
    )
    return sum(part.numel() * part.element_size() for part in parts)


def search_cuda_memory(
    blocks: list[torch.Tensor], batch: torch.Tensor, top_n: int
) -> dict[str, int] | None:
    """Measure the PyTorch allocator peak for one search with vectors already on device."""
    device = batch.device
    if device.type != "cuda":
        return None
    sync(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    search_topk(blocks, batch, top_n)
    sync(device)
    peak = torch.cuda.max_memory_allocated(device)
    return dict(
        allocated_before_bytes=before,
        peak_allocated_bytes=peak,
        extra_peak_bytes=peak - before,
    )


def benchmark(
    blocks: list[torch.Tensor],
    queries: torch.Tensor,
    cfg: dict,
) -> dict:
    """Measure synchronized search only, excluding input selection and preparation."""
    batches = query_batches(queries, cfg["query_batch_size"])
    for _ in range(cfg["search_warmup"]):
        search_topk(blocks, next(batches), cfg["top_n"])
    durations = []
    for _ in range(cfg["search_rounds"]):
        batch = next(batches)  # [B, F], already on the device.
        sync(queries.device)
        start = time.perf_counter()
        search_topk(blocks, batch, cfg["top_n"])
        sync(queries.device)
        durations.append(time.perf_counter() - start)
    return dict(
        ms_per_batch_mean=1000 * float(np.mean(durations)),
        ms_per_batch_p95=1000 * float(np.percentile(durations, 95)),
        queries_per_second=len(batch) * len(durations) / sum(durations),
        query_batch_size=len(batch),
        queries=len(queries),
        documents=sum(len(block) for block in blocks),
        corpus_bytes=sum(map(storage_bytes, blocks)),
        query_bytes=storage_bytes(queries),
        cuda_memory=search_cuda_memory(blocks, batch, cfg["top_n"]),
        top_n=min(cfg["top_n"], sum(len(block) for block in blocks)),
        kernel="torch.sparse.mm" if blocks[0].layout == torch.sparse_csr else "torch.matmul",
        dtype=str(queries.dtype),
        device=str(queries.device),
    )


def profile_search_phases(
    blocks: list[torch.Tensor], queries: torch.Tensor, cfg: dict
) -> dict:
    """Time scoring and the remaining search work in separate CUDA-instrumented rounds."""
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in blocks
    ]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for begin_score, end_score in events:
        begin_score.record()
        end_score.record()
    start.record()
    end.record()
    batches = query_batches(queries, cfg["query_batch_size"])
    samples = []
    for _ in range(cfg["search_phase_rounds"]):
        batch = next(batches)
        sync(queries.device)
        start.record()
        search_topk(blocks, batch, cfg["top_n"], events)
        end.record()
        sync(queries.device)
        scoring = sum(first.elapsed_time(last) for first, last in events)
        samples.append((scoring, start.elapsed_time(end) - scoring))
    scoring, remaining = np.mean(samples, axis=0)
    return dict(
        rounds=len(samples),
        scoring_ms_per_batch_mean=float(scoring),
        selection_and_other_ms_per_batch_mean=float(remaining),
    )


def profile_sparse(
    blocks: list[torch.Tensor],
    queries: torch.Tensor,
    cfg: dict,
    path: Path,
) -> list[str]:
    """Trace one sparse search outside the timing loop and record CUDA event names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as trace:
        search_topk(blocks, queries[: cfg["query_batch_size"]], cfg["top_n"])
        sync(queries.device)
    trace.export_chrome_trace(str(path))
    return sorted({event.name for event in trace.events() if event.device_type.name == "CUDA"})


@torch.inference_mode()
def run_search(
    cfg: dict,
    vectors: EmbeddingsByModelAndRole,
    device: torch.device,
    trace_path: Path,
    corpus_indices: np.ndarray | None = None,
) -> tuple[list[dict], dict | None]:
    """Run one dense baseline and paired MRL/CSRv2 configurations."""
    spec = cfg["csrv2"]
    sae = (
        SparseAutoEncoder.load(
            spec["model"],
            subfolder="3_SparseAutoEncoder",
            revision=spec["revision"],
        )
        .to(device=device, dtype=getattr(torch, cfg["dtype"]))
        .eval()
    )
    if any(k > min(cfg["embedding_dim"], sae.hidden_dim) for k in cfg["ks"]):
        raise ValueError("Every k must fit both the dense embedding and sparse feature space.")
    cases = [("qwen_dense", "dense", None)]
    for k in cfg["ks"]:
        cases.append(("mrl", "dense", k))
        cases.append(("csrv2", "csrv2", k))
    results, trace_info = [], None
    for method, name, k in cases:
        queries, blocks = representation(
            vectors,
            name,
            k,
            sae if name == "csrv2" else None,
            cfg["corpus_block_size"],
            device,
            corpus_indices,
        )
        if name == "csrv2":
            verify_sparse(blocks, queries, cfg["top_n"])
        result = dict(method=method, k=k, **benchmark(blocks, queries, cfg))
        if name == "csrv2":
            result["corpus_nnz"] = sum(block.values().numel() for block in blocks)
        if cfg.get("search_phase_rounds", 0) and device.type == "cuda":
            result["cuda_search_phases"] = profile_search_phases(blocks, queries, cfg)
        results.append(result)
        logger.info("%s k=%s: %.3f ms/batch", method, k, result["ms_per_batch_mean"])
        if name == "csrv2" and trace_info is None and cfg["profile_cuda"] and device.type == "cuda":
            trace_info = dict(
                path=str(trace_path),
                cuda_events=profile_sparse(
                    blocks,
                    queries,
                    cfg,
                    trace_path,
                ),
            )
        del queries, blocks
    return results, trace_info


if __name__ == "__main__":
    main()
