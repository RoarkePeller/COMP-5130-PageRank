from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import networkx as nx

from pagerank_impl import PageRankResult, load_edge_list, pagerank_power_iteration, top_k


SMALL_DATASET = Path("CA-GrQc.txt")
LARGE_DATASET = Path("com-dblp.ungraph.txt")
RESULTS_DIR = Path("results")

ALPHA = 0.85
TOL = 1e-6
MAX_ITER = 100
TOP_K = 20


def save_topk_csv(path: Path, rows: List[Tuple[int, float]], score_name: str) -> None:
    """Write top-ranked nodes and scores to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", score_name])
        for node, score in rows:
            writer.writerow([node, f"{score:.12f}"])


def rank_map(scores: Dict[int, float]) -> Dict[int, float]:
    """Convert score values into descending ranks"""
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    result: Dict[int, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for idx in range(i, j):
            result[ordered[idx][0]] = avg_rank
        i = j
    return result


def spearman_correlation(scores_a: Dict[int, float], scores_b: Dict[int, float]) -> float:
    """Compute Spearman rank on shared nodes between score maps."""
    common_nodes = list(set(scores_a).intersection(scores_b))
    if len(common_nodes) < 2:
        return 0.0

    ranks_a = rank_map({node: scores_a[node] for node in common_nodes})
    ranks_b = rank_map({node: scores_b[node] for node in common_nodes})

    vec_a = [ranks_a[node] for node in common_nodes]
    vec_b = [ranks_b[node] for node in common_nodes]

    mean_a = sum(vec_a) / len(vec_a)
    mean_b = sum(vec_b) / len(vec_b)
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(vec_a, vec_b))
    var_a = sum((a - mean_a) ** 2 for a in vec_a)
    var_b = sum((b - mean_b) ** 2 for b in vec_b)

    if var_a == 0.0 or var_b == 0.0:
        return 0.0
    return cov / (var_a ** 0.5 * var_b ** 0.5)


def topk_overlap(
    scores_a: Dict[int, float],
    scores_b: Dict[int, float],
    *,
    k: int = TOP_K,
) -> float:
    """Return overlap ratio between two top-k node sets."""
    top_a = {node for node, _ in top_k(scores_a, k=k)}
    top_b = {node for node, _ in top_k(scores_b, k=k)}
    return len(top_a.intersection(top_b)) / float(k)


def timed_pagerank(
    edge_file: Path,
    *,
    bidirectional: bool,
    alpha: float,
    tol: float,
    max_iter: int,
) -> Tuple[PageRankResult, float]:
    """Run PageRank and return result with runtime"""
    adjacency = load_edge_list(edge_file, bidirectional=bidirectional)
    start = perf_counter()
    result = pagerank_power_iteration(adjacency, alpha=alpha, tol=tol, max_iter=max_iter)
    runtime_s = perf_counter() - start
    return result, runtime_s


def timed_networkx_algo(func) -> Tuple[Dict[int, float], float]:
    """Run NetworkX pagerank and measure runtime."""
    start = perf_counter()
    scores = func()
    runtime_s = perf_counter() - start
    return scores, runtime_s


def timed_hits_authority(graph: nx.Graph | nx.DiGraph) -> Tuple[Dict[int, float], float]:
    """Run HITS authority scores"""
    attempts = (200, 1000)
    last_error: Exception | None = None
    for max_iter in attempts:
        try:
            return timed_networkx_algo(
                lambda: nx.hits(graph, max_iter=max_iter, normalized=True)[1]
            )
        except nx.PowerIterationFailedConvergence as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected HITS failure.")


def load_dataset_graph(dataset_path: Path, *, bidirectional: bool) -> nx.DiGraph:
    """Load edge list"""
    if bidirectional:
        undirected = nx.read_edgelist(
            dataset_path,
            comments="#",
            nodetype=int,
            create_using=nx.Graph(),
        )
        return nx.to_directed(undirected)
    return nx.read_edgelist(
        dataset_path,
        comments="#",
        nodetype=int,
        create_using=nx.DiGraph(),
    )


def run_dataset_analysis(
    *,
    dataset_name: str,
    dataset_path: Path,
    bidirectional: bool,
    runtime_rows: List[List[object]],
    metrics: Dict[str, Dict[str, object]],
) -> None:
    graph = load_dataset_graph(dataset_path, bidirectional=bidirectional)

    pagerank_result, pagerank_runtime = timed_pagerank(
        dataset_path,
        bidirectional=bidirectional,
        alpha=ALPHA,
        tol=TOL,
        max_iter=MAX_ITER,
    )
    runtime_rows.append([dataset_name, "pagerank", f"{pagerank_runtime:.6f}"])

    metrics[dataset_name] = {
        "pagerank": {
            "runtime_seconds": pagerank_runtime,
            "iterations": pagerank_result.iterations,
            "converged": pagerank_result.converged,
            "final_error": pagerank_result.final_error,
        }
    }

    hits_scores, hits_runtime = timed_hits_authority(graph)
    runtime_rows.append([dataset_name, "hits_authority", f"{hits_runtime:.6f}"])

    degree_scores, degree_runtime = timed_networkx_algo(lambda: nx.degree_centrality(graph))
    runtime_rows.append([dataset_name, "degree_centrality", f"{degree_runtime:.6f}"])

    metrics[dataset_name]["comparison"] = {
        "hits_authority": {
            "runtime_seconds": hits_runtime,
            "spearman_vs_pagerank": spearman_correlation(pagerank_result.scores, hits_scores),
            "topk_overlap_vs_pagerank": topk_overlap(
                pagerank_result.scores,
                hits_scores,
                k=TOP_K,
            ),
        },
        "degree_centrality": {
            "runtime_seconds": degree_runtime,
            "spearman_vs_pagerank": spearman_correlation(pagerank_result.scores, degree_scores),
            "topk_overlap_vs_pagerank": topk_overlap(
                pagerank_result.scores,
                degree_scores,
                k=TOP_K,
            ),
        },
    }

    save_topk_csv(
        RESULTS_DIR / f"{dataset_name}_pagerank_top20.csv",
        top_k(pagerank_result.scores, k=TOP_K),
        "pagerank",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, object]] = {}
    runtime_rows: List[List[object]] = [["dataset", "algorithm", "runtime_seconds"]]

    run_dataset_analysis(
        dataset_name="small",
        dataset_path=SMALL_DATASET,
        bidirectional=False,
        runtime_rows=runtime_rows,
        metrics=metrics,
    )
    run_dataset_analysis(
        dataset_name="large",
        dataset_path=LARGE_DATASET,
        bidirectional=True,
        runtime_rows=runtime_rows,
        metrics=metrics,
    )

    with (RESULTS_DIR / "runtime_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(runtime_rows)

    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {RESULTS_DIR / 'metrics.json'}")
    print(f"Saved runtimes to {RESULTS_DIR / 'runtime_summary.csv'}")


if __name__ == "__main__":
    main()
