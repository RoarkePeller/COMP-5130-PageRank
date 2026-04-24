from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Mapping, Sequence, Tuple, TypeVar

NodeT = TypeVar("NodeT", bound=Hashable)

@dataclass
class PageRankResult:
    scores: Dict[NodeT, float]
    iterations: int
    converged: bool
    final_error: float


def load_edge_list(
    file_path: str | Path,
    *,
    node_parser: Callable[[str], NodeT] = int,
    bidirectional: bool = False,
    delimiter: str | None = None,
    source_index: int = 0,
    target_index: int = 1,
    skip_header: bool = False,
    weighted: bool = False,
    weight_index: int | None = 2,
) -> Dict[NodeT, List[NodeT] | Dict[NodeT, float]]:
    adjacency: Dict[NodeT, List[NodeT] | Dict[NodeT, float]] = {}
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(delimiter) if delimiter is not None else line.split()
            if len(parts) <= max(source_index, target_index):
                continue

            src_text, dst_text = parts[source_index], parts[target_index]
            if skip_header and src_text == "SOURCE_SUBREDDIT" and dst_text == "TARGET_SUBREDDIT":
                continue

            try:
                src = node_parser(src_text)
                dst = node_parser(dst_text)
            except (TypeError, ValueError):
                # Skip header lines or malformed records.
                continue

            if weighted:
                weight = 1.0
                if weight_index is not None and len(parts) > weight_index:
                    try:
                        weight = float(parts[weight_index])
                    except (TypeError, ValueError):
                        weight = 1.0

                src_neighbors = adjacency.setdefault(src, {})
                assert isinstance(src_neighbors, dict)
                src_neighbors[dst] = src_neighbors.get(dst, 0.0) + weight
                adjacency.setdefault(dst, {})

                if bidirectional:
                    dst_neighbors = adjacency.setdefault(dst, {})
                    assert isinstance(dst_neighbors, dict)
                    dst_neighbors[src] = dst_neighbors.get(src, 0.0) + weight
                continue

            src_neighbors = adjacency.setdefault(src, [])
            assert isinstance(src_neighbors, list)
            src_neighbors.append(dst)
            adjacency.setdefault(dst, [])

            if bidirectional:
                dst_neighbors = adjacency.setdefault(dst, [])
                assert isinstance(dst_neighbors, list)
                dst_neighbors.append(src)

    return adjacency


def load_edge_list_with_sentiment_filter(
    file_path: str | Path,
    *,
    target_sentiments: List[int] | int,
    sentiment_index: int,
    node_parser: Callable[[str], NodeT] = int,
    bidirectional: bool = False,
    delimiter: str | None = None,
    source_index: int = 0,
    target_index: int = 1,
    skip_header: bool = False,
    weighted: bool = False,
    weight_index: int | None = 2,
) -> Dict[NodeT, List[NodeT] | Dict[NodeT, float]]:
    """Load edge list with sentiment filtering. Only edges matching target_sentiments are included."""
    if isinstance(target_sentiments, int):
        target_sentiments = [target_sentiments]
    target_sentiments_set = set(target_sentiments)
    
    adjacency: Dict[NodeT, List[NodeT] | Dict[NodeT, float]] = {}
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(delimiter) if delimiter is not None else line.split()
            if len(parts) <= max(source_index, target_index, sentiment_index):
                continue

            src_text, dst_text = parts[source_index], parts[target_index]
            if skip_header and src_text == "SOURCE_SUBREDDIT" and dst_text == "TARGET_SUBREDDIT":
                continue

            try:
                src = node_parser(src_text)
                dst = node_parser(dst_text)
                sentiment = int(parts[sentiment_index])
            except (TypeError, ValueError):
                # Skip header lines or malformed records.
                continue

            # Filter by sentiment
            if sentiment not in target_sentiments_set:
                continue

            if weighted:
                weight = 1.0
                if weight_index is not None and len(parts) > weight_index:
                    try:
                        weight = float(parts[weight_index])
                    except (TypeError, ValueError):
                        weight = 1.0

                src_neighbors = adjacency.setdefault(src, {})
                assert isinstance(src_neighbors, dict)
                src_neighbors[dst] = src_neighbors.get(dst, 0.0) + weight
                adjacency.setdefault(dst, {})

                if bidirectional:
                    dst_neighbors = adjacency.setdefault(dst, {})
                    assert isinstance(dst_neighbors, dict)
                    dst_neighbors[src] = dst_neighbors.get(src, 0.0) + weight
                continue

            src_neighbors = adjacency.setdefault(src, [])
            assert isinstance(src_neighbors, list)
            src_neighbors.append(dst)
            adjacency.setdefault(dst, [])

            if bidirectional:
                dst_neighbors = adjacency.setdefault(dst, [])
                assert isinstance(dst_neighbors, list)
                dst_neighbors.append(src)

    return adjacency


def pagerank_power_iteration(
    adjacency: Dict[NodeT, Sequence[NodeT] | Mapping[NodeT, float]],
    *,
    alpha: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 100,
    personalization: Mapping[NodeT, float] | None = None,
    nstart: Mapping[NodeT, float] | None = None,
    dangling: Mapping[NodeT, float] | None = None,
) -> PageRankResult:
    """Compute PageRank"""
    if not adjacency:
        return PageRankResult(scores={}, iterations=0, converged=True, final_error=0.0)

    nodes = sorted(adjacency.keys())
    num_nodes = len(nodes)

    def normalize_distribution(
        values: Mapping[NodeT, float] | None,
        *,
        default_uniform: bool,
    ) -> Dict[NodeT, float]:
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

    transition: Dict[NodeT, Dict[NodeT, float]] = {node: {} for node in nodes}
    for src in nodes:
        neighbors = adjacency.get(src, [])
        weighted_counts: Dict[NodeT, float] = {}

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

def top_k(scores: Dict[NodeT, float], k: int = 20) -> List[Tuple[NodeT, float]]:
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:k]


def rank_order(scores: Dict[NodeT, float]) -> List[NodeT]:
    return [node for node, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]
