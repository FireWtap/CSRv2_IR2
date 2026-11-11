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
EmbeddingsByModelAndRole = dict[tuple[str, str], np.ndarray]
EncodingOptions = tuple[str | None, int, int]


TextRow = dict[str, str]


def main() -> None:
    """Parse E7 options and run one independent profiling stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("encode", "search"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    commands.choices["search"].add_argument(
        "--source", choices=("cache", "encoded"), default="cache"
    )
    commands.choices["search"].add_argument(
        "--cache-root",
        type=Path,
        help="Directory containing emb_cache/ (overrides E7_CACHE_ROOT and SCRATCH_ROOT)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    for key in (
        "encoding_batch_size",
        "encoding_k",
        "cache_corpus_chunk_size",
        "query_batch_size",
        "search_rounds",
        "corpus_block_size",
        "top_n",
    ):
        if cfg[key] <= 0:
            parser.error(f"{key} must be positive")
    if not cfg["ks"] or min(cfg["ks"]) <= 0:
        parser.error("ks must contain positive values")
    if cfg["prompt_mode"] not in ("both", "query"):
        parser.error("prompt_mode must be both or query")
    for key, default in (("encoding_profile_stride", 1), ("encoding_calibration_pairs", 0)):
        value = cfg.get(key, default)
        if type(value) is not int or value < 0:
            parser.error(f"{key} must be a nonnegative integer")
    root = Path(os.environ.get("SCRATCH_ROOT", OUTPUTS))
    run_dir = root / "e7" / cfg["run_name"]
    out = RESULTS / "E7" / cfg["run_name"]
    if args.command == "search" and args.source == "cache":
        cache_root = args.cache_root or Path(os.environ.get("E7_CACHE_ROOT", root))
        logger.info("Reading cached embeddings under %s", cache_root / "emb_cache")
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
            load_cached(cfg, rows, prompts, cache_root)
            if args.source == "cache"
            else load_encoded(cfg, rows, prompts, run_dir)
        )
        trace_path = root / "e7" / "traces" / cfg["run_name"] / f"{args.source}.json"
        results, trace = run_search(cfg, vectors, device, trace_path)
        write_json(
            out / f"retrieval_{args.source}.json",
            dict(
                measurements=results,
                source=args.source,
                input_files=paths,
                trace=trace,
                provenance=provenance(cfg),
            ),
        )


# Inputs and saved embeddings


def load_rows(cfg: dict) -> list[TextRow]:
    """Read an ordered manifest or reconstruct the pinned MTEB retrieval input."""
    if "manifest" in cfg:
        with Path(cfg["manifest"]).open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        import mteb

        task = mteb.get_task(cfg["task"])
        split = cfg["split"]
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
    required_fields = {"id", "text", "role", "task"}
    if any(not isinstance(row, dict) or row.keys() != required_fields for row in rows):
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


def load_cached(
    cfg: dict,
    rows: list[TextRow],
    prompts: dict[str, str],
    root: Path,
) -> tuple[EmbeddingsByModelAndRole, list[str]]:
    """Return four pre-head matrices plus the file paths used to read them.

    Each model has a query matrix [Q, D] and a document matrix [N, D].
    The paths are recorded in the results JSON for provenance.
    """
    if "manifest" in cfg:
        raise ValueError("Use encode and --source encoded for a custom manifest.")
    embeddings = {}
    cache_paths = []
    for model_name in MODELS:
        model = cfg[model_name]
        for role in ROLES:
            texts = [row["text"] for row in rows if row["role"] == role]
            prompt = (
                prompts[cfg["task"]] if cfg["prompt_mode"] == "both" or role == "query" else None
            )
            chunk_size = len(texts) if role == "query" else cfg["cache_corpus_chunk_size"]
            embedding_chunks = []
            for start in range(0, len(texts), chunk_size):
                chunk_texts = texts[start : start + chunk_size]
                path = embedding_cache_path(
                    root / "emb_cache" / model["cache_run"],
                    model_tag(model, cfg),
                    chunk_texts,
                    prompt,
                    cfg["task"],
                    role,
                )
                # Each cache file has shape [texts in this chunk, D].
                embedding_chunks.append(read_vectors(path, len(chunk_texts), cfg["embedding_dim"]))
                cache_paths.append(str(path))

            # Join along the text axis in MTEB order. Keep a single file memory-mapped.
            if len(embedding_chunks) == 1:
                embeddings[model_name, role] = embedding_chunks[0]
            else:
                embeddings[model_name, role] = np.concatenate(embedding_chunks, axis=0)
    return embeddings, cache_paths


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


def calibrate_encoding_markers(
    backbone: SentenceTransformer,
    sae: SparseAutoEncoder,
    texts: list[str],
    options: EncodingOptions,
    device: torch.device,
    markers: tuple[torch.cuda.Event, ...],
    pairs: int,
) -> dict:
    """Compare marked and unmarked forwards on the same warmed-up CSRv2 batch."""
    deltas_ms = []
    unmarked_ms = []
    for pair in range(pairs):
        timings = {}
        for marked in ((False, True) if pair % 2 == 0 else (True, False)):
            _, seconds = time_encode_batch(
                backbone, sae, texts, options, device, markers if marked else ()
            )
            timings[marked] = seconds * 1000
        unmarked_ms.append(timings[False])
        deltas_ms.append(timings[True] - timings[False])
    median_unmarked = float(np.median(unmarked_ms))
    median_delta = float(np.median(deltas_ms))
    delta_pct = 100 * median_delta / median_unmarked
    return dict(
        pairs=pairs,
        texts_per_batch=len(texts),
        unmarked_ms_per_batch_median=median_unmarked,
        marked_minus_unmarked_ms_median=median_delta,
        marked_minus_unmarked_ms_iqr=np.percentile(deltas_ms, [25, 75]).tolist(),
        overhead_pct=delta_pct,
        recommended_stride=16 if abs(delta_pct) > 1 else 1,
    )


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
    calibration_pairs = cfg.get("encoding_calibration_pairs", 0)
    with TemporaryDirectory(prefix=f".{run_dir.name}-", dir=run_dir.parent) as tmp:
        staging = Path(tmp) / "run"
        (staging / "embeddings").mkdir(parents=True)
        for name in MODELS:
            backbone, sae = load_backbone(cfg, name, device)
            markers = ()
            if device.type == "cuda" and (profile_stride or (sae is not None and calibration_pairs)):
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
                calibration = None
                for task, items in rows_by_task.items():
                    prompt = (
                        prompts[task] if cfg["prompt_mode"] == "both" or role == "query" else None
                    )

                    options = (prompt, batch_size, cfg["encoding_k"])
                    warmup_texts = [text for _, text in items[:batch_size]]
                    for _ in range(cfg["encoding_warmup"]):
                        encode_batch(backbone, sae, warmup_texts, *options)
                    if (
                        sae is not None
                        and role == "document"
                        and markers
                        and calibration_pairs
                        and calibration is None
                    ):
                        # Warm the marked path on the exact batch used for calibration.
                        time_encode_batch(backbone, sae, warmup_texts, options, device, markers)
                        calibration = calibrate_encoding_markers(
                            backbone, sae, warmup_texts, options, device, markers, calibration_pairs
                        )
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
                if calibration is not None:
                    timings[name][role]["marker_calibration"] = calibration
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
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return dense queries [Q, F] and device-resident corpus blocks [C, F]."""
    queries = prepare_vectors(vectors[name, "query"], device, k, sae)
    corpus = vectors[name, "document"]
    blocks = []
    for start in range(0, len(corpus), block_size):
        block = prepare_vectors(corpus[start : start + block_size], device, k, sae)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scores, corpus row indices), each of shape [up to top_n, B]."""
    block_top_scores = []
    block_top_indices = []
    corpus_offset = 0
    for block in blocks:
        # Scores are [C, B]. Rank along the document axis to get [min(top_n, C), B].
        top_scores, local_indices = score_block(block, queries).topk(min(top_n, len(block)), dim=0)
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
        )
        if name == "csrv2":
            verify_sparse(blocks, queries, cfg["top_n"])
        result = dict(method=method, k=k, **benchmark(blocks, queries, cfg))
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
