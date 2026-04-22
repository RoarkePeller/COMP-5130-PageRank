from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

@dataclass
class PageRankResult:
    scores: Dict[int, float]
    iterations: int
    converged: bool
    final_error: float


def load_edge_list(
    file_path: str | Path,
    *,
    bidirectional: bool = False,
) -> Dict[int, List[int]]:
    adjacency: Dict[int, List[int]] = {}
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            src_text, dst_text, *_ = line.split()
            src = int(src_text)
            dst = int(dst_text)

            adjacency.setdefault(src, []).append(dst)
            adjacency.setdefault(dst, [])

            if bidirectional:
                adjacency[dst].append(src)

    return adjacency


def pagerank_power_iteration(
    adjacency: Dict[int, Sequence[int] | Mapping[int, float]],
    *,
    alpha: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 100,
    personalization: Mapping[int, float] | None = None,
    nstart: Mapping[int, float] | None = None,
    dangling: Mapping[int, float] | None = None,
) -> PageRankResult:
    """Compute PageRank"""
    if not adjacency:
        return PageRankResult(scores={}, iterations=0, converged=True, final_error=0.0)

    nodes = sorted(adjacency.keys())
    num_nodes = len(nodes)

    def normalize_distribution(
        values: Mapping[int, float] | None,
        *,
        default_uniform: bool,
    ) -> Dict[int, float]:
        if values is None:
            if default_uniform:
                return {node: 1.0 / num_nodes for node in nodes}
            return {node: 0.0 for node in nodes}

        dist = {node: 0.0 for node in nodes}
        for node, value in values.items():
            if node in dist:
                dist[node] = float(value)

        total = sum(dist.values())
        if total == 0.0:
            raise ZeroDivisionError()
        return {node: value / total for node, value in dist.items()}

    transition: Dict[int, Dict[int, float]] = {node: {} for node in nodes}
    for src in nodes:
        neighbors = adjacency.get(src, [])
        weighted_counts: Dict[int, float] = {}

        if isinstance(neighbors, Mapping):
            for dst, weight in neighbors.items():
                if dst not in transition:
                    continue
                weight_value = float(weight)
                if weight_value <= 0.0:
                    continue
                weighted_counts[dst] = weighted_counts.get(dst, 0.0) + weight_value
        else:
            for dst in neighbors:
                if dst not in transition:
                    continue
                weighted_counts[dst] = weighted_counts.get(dst, 0.0) + 1.0

        if weighted_counts:
            total = sum(weighted_counts.values())
            transition[src] = {dst: wt / total for dst, wt in weighted_counts.items()}

    if nstart is None:
        ranks = {node: 1.0 / num_nodes for node in nodes}
    else:
        ranks = normalize_distribution(nstart, default_uniform=False)

    personalization_dist = normalize_distribution(personalization, default_uniform=True)
    dangling_dist = (
        personalization_dist
        if dangling is None
        else normalize_distribution(dangling, default_uniform=False)
    )
    dangling_nodes = [node for node in nodes if not transition[node]]

    converged = False
    final_error = 0.0

    for iteration in range(1, max_iter + 1):
        previous = ranks
        ranks = {node: 0.0 for node in nodes}
        danglesum = alpha * sum(previous[node] for node in dangling_nodes)

        for src in nodes:
            for dst, wt in transition[src].items():
                ranks[dst] += alpha * previous[src] * wt

        for node in nodes:
            ranks[node] += danglesum * dangling_dist[node] + (1.0 - alpha) * personalization_dist[node]

        final_error = sum(abs(ranks[node] - previous[node]) for node in nodes)
        if final_error < num_nodes * tol:
            converged = True
            break

    total_rank = sum(ranks.values())
    if total_rank > 0.0:
        ranks = {node: value / total_rank for node, value in ranks.items()}

    scores = {node: float(ranks[node]) for node in nodes}
    return PageRankResult(
        scores=scores,
        iterations=iteration,
        converged=converged,
        final_error=final_error,
    )


def top_k(scores: Dict[int, float], k: int = 20) -> List[Tuple[int, float]]:
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:k]


def rank_order(scores: Dict[int, float]) -> List[int]:
    return [node for node, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]
