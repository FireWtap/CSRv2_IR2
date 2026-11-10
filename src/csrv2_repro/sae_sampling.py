"""D-013: preserve positive-pair row frequencies while packing distinct-ID batches."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math

import numpy as np


@dataclass(frozen=True)
class PackedBatch:
    indices: tuple[int, ...]
    positive_pairs: int


class PairFrequencySampler:
    """Each available training positive contributes one query and one document row.

    Every item absent from all such pairs appears once, including unlabelled corpus
    documents and missing-positive queries. Pair units stay together. Frequent IDs
    are placed first in the least-filled compatible batch, then singletons fill
    remaining slots. Ordering and equal-priority choices are seeded. Infeasible
    capacity/identity constraints produce extra partial batches, never dropped rows.
    """

    def __init__(self, dataset, batch_size: int = 256, seed: int = 1227):
        if dataset.partition != "train":
            raise ValueError("the training sampler requires the training partition")
        if batch_size < 2:
            raise ValueError("batch size must fit a query/document pair")
        self.batch_size, self.seed = batch_size, seed
        rows, self.task_ranges = {}, {}
        offset = 0
        for task in dataset.tasks:
            start = offset
            for kind, identifiers in (
                ("query", dataset.query_ids(task)),
                ("document", dataset.document_ids(task)),
            ):
                for identifier in identifiers:
                    rows[(task, kind, identifier)] = offset
                    offset += 1
            self.task_ranges[task] = (start, offset)
        if offset != len(dataset) or len(rows) != offset:
            raise ValueError("dataset row ordering or identity mismatch")
        self.expected_counts = np.zeros(offset, dtype=np.int64)
        self.units = []
        for task, qid, did, grade in dataset.positive_pairs():
            if grade <= 0:
                raise ValueError("positive_pairs must contain only positive grades")
            pair = (rows[(task, "query", qid)], rows[(task, "document", did)])
            self.units.append(pair)
            self.expected_counts[list(pair)] += 1
        self.pair_count = len(self.units)
        for i in np.flatnonzero(self.expected_counts == 0):
            self.units.append((int(i),))
            self.expected_counts[i] = 1

    def batches(self, epoch: int = 0):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        total = int(self.expected_counts.sum())
        count = max(math.ceil(total / self.batch_size), int(self.expected_counts.max()))
        buckets, seen = [[] for _ in range(count)], [set() for _ in range(count)]
        capacity = [self.batch_size] * count
        if count == math.ceil(total / self.batch_size):
            capacity[-1] = total - (count - 1) * self.batch_size
        tie = generator.random(count).tolist()
        heap = [(0, tie[i], i) for i in range(count)]
        heapq.heapify(heap)
        order = generator.permutation(self.pair_count).tolist()
        order.sort(key=lambda i: -max(self.expected_counts[list(self.units[i])]))
        order.extend(
            self.pair_count + int(i)
            for i in generator.permutation(len(self.units) - self.pair_count)
        )
        for unit_index in order:
            unit = self.units[unit_index]
            deferred = []
            while heap:
                load, priority, index = heapq.heappop(heap)
                if load + len(unit) <= capacity[index] and not seen[index].intersection(unit):
                    break
                deferred.append((load, priority, index))
            else:
                index, load, priority = len(buckets), 0, float(generator.random())
                buckets.append([])
                seen.append(set())
                capacity.append(self.batch_size)
            buckets[index].append(unit)
            seen[index].update(unit)
            if load + len(unit) < capacity[index]:
                heapq.heappush(heap, (load + len(unit), priority, index))
            for rejected in deferred:
                heapq.heappush(heap, rejected)
        packed = []
        for bucket in buckets:
            if not bucket:
                continue
            generator.shuffle(bucket)
            packed.append(
                PackedBatch(
                    tuple(i for unit in bucket for i in unit),
                    sum(len(unit) == 2 for unit in bucket),
                )
            )
        generator.shuffle(packed)
        # Match an ordinary loader's full batches followed by its residual batch(es).
        yield from (b for b in packed if len(b.indices) == self.batch_size)
        yield from (b for b in packed if len(b.indices) < self.batch_size)

    def audit(self, batches: list[PackedBatch]) -> dict:
        """Check the full epoch without loading or collating its embeddings."""
        observed = np.zeros_like(self.expected_counts)
        for batch in batches:
            if not batch.indices or len(batch.indices) > self.batch_size:
                raise ValueError("invalid batch size")
            if len(set(batch.indices)) != len(batch.indices):
                raise ValueError("duplicate item within batch")
            observed[list(batch.indices)] += 1
        if not np.array_equal(observed, self.expected_counts):
            raise ValueError("epoch frequencies changed")
        if sum(b.positive_pairs for b in batches) != self.pair_count:
            raise ValueError("positive-pair coverage changed")
        return {
            "positive_pairs": self.pair_count,
            "singleton_items": len(self.units) - self.pair_count,
            "row_appearances": int(observed.sum()),
            "unique_items": len(observed),
            "max_item_frequency": int(observed.max()),
            "batch_count": len(batches),
            "batch_size_counts": dict(sorted(Counter(len(b.indices) for b in batches).items())),
            "batches_without_pair_units": sum(b.positive_pairs == 0 for b in batches),
            "task_row_appearances": {
                t: int(observed[start:end].sum()) for t, (start, end) in self.task_ranges.items()
            },
            "exact_epoch_frequency_check": True,
            "unique_ids_within_batches": True,
        }
