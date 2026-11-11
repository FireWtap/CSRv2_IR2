"""mteb-compatible encoder that caches backbone embeddings on disk and applies a post-transform.

E1 fills the cache with dense Qwen3-Embedding-4B embeddings. Later experiments (MRL truncation,
CSR/CSRv2 heads on frozen embeddings) reuse the same cache with a different `transform`, so the
expensive backbone pass runs once and the scoring stays inside mteb (identical metric code).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

logger = logging.getLogger(__name__)

PROMPT_MODES = ("both", "query", "none")


class CachedEncoder:
    """Wraps a SentenceTransformer for mteb.MTEB.run.

    prompt_mode:
      "both"  - task instruction prepended to queries AND documents. This is what upstream's
                all_step_pipeline_mteb_evaluation.py does (model.prompts keyed by task name).
      "query" - Qwen3-Embedding's usage: instruction on queries, documents raw. Symmetric tasks
                (classification, clustering, STS, pair classification), where mteb passes no
                prompt_type, get the instruction on every text, as in Qwen3's own MTEB evaluation.
      "none"  - no instruction (sanity checks against plain mteb).

    require_cache: raise on a cache miss instead of encoding (scoring-only jobs, e.g. E2 on a MIG slice).
    """

    def __init__(
        self,
        model: SentenceTransformer,
        model_tag: str,
        prompts: dict[str, str],
        prompt_mode: str,
        cache_dir: Path,
        batch_size: int,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        meta_name: str | None = None,
        require_cache: bool = False,
    ):
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}, got {prompt_mode!r}")
        self.model = model
        self.model_tag = model_tag
        self.prompts = prompts
        self.prompt_mode = prompt_mode
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.transform = transform
        self.require_cache = require_cache
        self.mteb_model_meta = self._meta(meta_name or model_tag)

    def _meta(self, name: str):
        from mteb.models import model_meta_from_sentence_transformers

        meta = model_meta_from_sentence_transformers(self.model)
        return meta.model_copy(update={"name": name, "revision": "local"})

    def _prompt(self, task_name: str, prompt_type) -> str | None:
        is_document = prompt_type is not None and prompt_type.value != "query"
        if self.prompt_mode == "none" or (self.prompt_mode == "query" and is_document):
            return None
        if task_name not in self.prompts:
            raise KeyError(f"No upstream prompt for task {task_name}")
        return self.prompts[task_name]

    def _cache_path(
        self, sentences: list[str], prompt: str | None, task_name: str, prompt_type
    ) -> Path:
        h = hashlib.sha1()
        for part in (self.model_tag, prompt or "", *sentences):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        side = prompt_type.value if prompt_type is not None else "none"
        return self.cache_dir / task_name / f"{side}-{len(sentences)}-{h.hexdigest()[:16]}.npy"

    def embed(self, sentences, task_name: str, prompt_type=None) -> np.ndarray:
        """Backbone embeddings (cached), before the transform."""
        sentences = list(sentences)
        prompt = self._prompt(task_name, prompt_type)
        path = self._cache_path(sentences, prompt, task_name, prompt_type)
        if path.exists():
            logger.info("cache hit %s", path)
            return np.load(path)
        if self.require_cache:
            raise FileNotFoundError(f"embedding cache miss with require_cache=True: {path}")
        emb = self.model.encode(
            sentences,
            prompt=prompt,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, emb)
        tmp.rename(path)
        return emb

    def encode(self, sentences, *, task_name: str, prompt_type=None, **kwargs) -> np.ndarray:
        emb = self.embed(sentences, task_name, prompt_type)
        return self.transform(emb) if self.transform is not None else emb

    def similarity(self, a, b) -> torch.Tensor:
        # Paper App. B.1: cosine similarity (all task types).
        return cos_sim(torch.as_tensor(a), torch.as_tensor(b))


def sae_topk_transform(sae, k: int, batch_size: int = 4096) -> Callable[[np.ndarray], np.ndarray]:
    """Apply a sentence-transformers SparseAutoEncoder at sparsity k to cached (normalized) embeddings.

    Same computation as SparseEncoder.encode in inference mode: preprocess -> encoder -> TopK -> ReLU.
    """
    param = next(sae.parameters())

    def transform(x: np.ndarray) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for i in range(0, len(x), batch_size):
                feats = {"sentence_embedding": torch.from_numpy(x[i : i + batch_size]).to(param)}
                out.append(
                    sae(feats, max_active_dims=k)["sentence_embedding"].float().cpu().numpy()
                )
        return np.concatenate(out)

    return transform
