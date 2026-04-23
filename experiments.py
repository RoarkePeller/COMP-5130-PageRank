from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Hashable, List, Mapping, Tuple

import networkx as nx

from pagerank_impl import PageRankResult, load_edge_list, load_edge_list_with_sentiment_filter, pagerank_power_iteration, top_k


SMALL_DATASET = Path("CA-GrQc.txt")
LARGE_DATASET = Path("com-dblp.ungraph.txt")
RESULTS_DIR = Path("results")

ALPHA = 0.85
TOL = 1e-6
MAX_ITER = 100
TOP_K = 20


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    bidirectional: bool
    node_parser: Callable[[str], Hashable]
    delimiter: str | None = None
    source_index: int = 0
    target_index: int = 1
    skip_header: bool = False
    weighted: bool = False
    weight_index: int | None = 2
    sentiment_index: int | None = None


def reddit_dataset_config(name: str, candidates: List[Path]) -> DatasetConfig | None:
    for path in candidates:
        if path.exists():
            return DatasetConfig(
                name=name,
                path=path,
                bidirectional=False,
                node_parser=str,
                delimiter="\t",
                source_index=0,
                target_index=1,
                skip_header=True,
                weighted=True,
                weight_index=None,
                sentiment_index=4,
            )
    return None


def default_dataset_configs() -> List[DatasetConfig]:
    configs = [
        DatasetConfig(
            name="small",
            path=SMALL_DATASET,
            bidirectional=False,
            node_parser=int,
        ),
        DatasetConfig(
            name="large",
            path=LARGE_DATASET,
            bidirectional=True,
            node_parser=int,
        ),
    ]
    configs.extend(optional_reddit_configs())
    return configs


def optional_reddit_configs() -> List[DatasetConfig]:
    configs: List[DatasetConfig] = []

    title_config = reddit_dataset_config(
        "reddit_title",
        [
            Path("soc-redditHyperlinks-title.tsv"),
            Path("data/soc-redditHyperlinks-title.tsv"),
            Path("reddit_hyperlinks_title.tsv"),
            Path("data/reddit_hyperlinks_title.tsv"),
        ],
    )
    if title_config is not None:
        configs.append(title_config)

    body_config = reddit_dataset_config(
        "reddit_body",
        [
            Path("soc-redditHyperlinks-body.tsv"),
            Path("data/soc-redditHyperlinks-body.tsv"),
            Path("reddit_hyperlinks_body.tsv"),
            Path("data/reddit_hyperlinks_body.tsv"),
        ],
    )
    if body_config is not None:
        configs.append(body_config)

    return configs


def save_topk_csv(path: Path, rows: List[Tuple[Hashable, float]], score_name: str) -> None:
    """Write top-ranked nodes and scores to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", score_name])
        for node, score in rows:
            writer.writerow([node, f"{score:.12f}"])


def rank_map(scores: Dict[Hashable, float]) -> Dict[Hashable, float]:
    """Convert score values into descending ranks"""
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    result: Dict[Hashable, float] = {}
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


def spearman_correlation(scores_a: Dict[Hashable, float], scores_b: Dict[Hashable, float]) -> float:
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
    scores_a: Dict[Hashable, float],
    scores_b: Dict[Hashable, float],
    *,
    k: int = TOP_K,
) -> float:
    """Return overlap ratio between two top-k node sets."""
    top_a = {node for node, _ in top_k(scores_a, k=k)}
    top_b = {node for node, _ in top_k(scores_b, k=k)}
    return len(top_a.intersection(top_b)) / float(k)


def timed_pagerank(
    config: DatasetConfig,
    *,
    alpha: float,
    tol: float,
    max_iter: int,
) -> Tuple[PageRankResult, float]:
    """Run PageRank and return result with runtime"""
    adjacency = load_edge_list(
        config.path,
        node_parser=config.node_parser,
        bidirectional=config.bidirectional,
        delimiter=config.delimiter,
        source_index=config.source_index,
        target_index=config.target_index,
        skip_header=config.skip_header,
        weighted=config.weighted,
        weight_index=config.weight_index,
    )
    start = perf_counter()
    result = pagerank_power_iteration(adjacency, alpha=alpha, tol=tol, max_iter=max_iter)
    runtime_s = perf_counter() - start
    return result, runtime_s


def timed_pagerank_sentiment_filtered(
    config: DatasetConfig,
    sentiment_value: int,
    *,
    alpha: float,
    tol: float,
    max_iter: int,
) -> Tuple[PageRankResult, float]:
    """Run PageRank on sentiment-filtered edges and return result with runtime"""
    if config.sentiment_index is None:
        raise ValueError(f"Dataset {config.name} does not support sentiment filtering")
    
    adjacency = load_edge_list_with_sentiment_filter(
        config.path,
        target_sentiments=[sentiment_value],
        sentiment_index=config.sentiment_index,
        node_parser=config.node_parser,
        bidirectional=config.bidirectional,
        delimiter=config.delimiter,
        source_index=config.source_index,
        target_index=config.target_index,
        skip_header=config.skip_header,
        weighted=config.weighted,
        weight_index=config.weight_index,
    )
    start = perf_counter()
    result = pagerank_power_iteration(adjacency, alpha=alpha, tol=tol, max_iter=max_iter)
    runtime_s = perf_counter() - start
    return result, runtime_s


def timed_networkx_algo(func) -> Tuple[Dict[Hashable, float], float]:
    """Run NetworkX pagerank and measure runtime."""
    start = perf_counter()
    scores = func()
    runtime_s = perf_counter() - start
    return scores, runtime_s


def timed_hits_authority(graph: nx.Graph | nx.DiGraph) -> Tuple[Dict[Hashable, float], float]:
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


def load_dataset_graph(config: DatasetConfig) -> nx.DiGraph:
    """Load edge list using the same parsing settings as custom PageRank."""
    adjacency = load_edge_list(
        config.path,
        node_parser=config.node_parser,
        bidirectional=config.bidirectional,
        delimiter=config.delimiter,
        source_index=config.source_index,
        target_index=config.target_index,
        skip_header=config.skip_header,
        weighted=config.weighted,
        weight_index=config.weight_index,
    )
    graph = nx.DiGraph()
    for src, neighbors in adjacency.items():
        if isinstance(neighbors, Mapping):
            for dst, weight in neighbors.items():
                graph.add_edge(src, dst, weight=float(weight))
        else:
            for dst in neighbors:
                graph.add_edge(src, dst)
        if src not in graph:
            graph.add_node(src)
    return graph


def run_dataset_analysis(
    *,
    config: DatasetConfig,
    runtime_rows: List[List[object]],
    metrics: Dict[str, Dict[str, object]],
) -> None:
    dataset_name = config.name
    graph = load_dataset_graph(config)

    pagerank_result, pagerank_runtime = timed_pagerank(
        config,
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

    degree_scores, degree_runtime = timed_networkx_algo(lambda: nx.degree_centrality(graph))
    runtime_rows.append([dataset_name, "degree_centrality", f"{degree_runtime:.6f}"])

    comparison_metrics: Dict[str, Dict[str, object]] = {
        "degree_centrality": {
            "runtime_seconds": degree_runtime,
            "spearman_vs_pagerank": spearman_correlation(pagerank_result.scores, degree_scores),
            "topk_overlap_vs_pagerank": topk_overlap(
                pagerank_result.scores,
                degree_scores,
                k=TOP_K,
            ),
        }
    }

    try:
        hits_scores, hits_runtime = timed_hits_authority(graph)
        runtime_rows.append([dataset_name, "hits_authority", f"{hits_runtime:.6f}"])
        comparison_metrics["hits_authority"] = {
            "runtime_seconds": hits_runtime,
            "spearman_vs_pagerank": spearman_correlation(pagerank_result.scores, hits_scores),
            "topk_overlap_vs_pagerank": topk_overlap(
                pagerank_result.scores,
                hits_scores,
                k=TOP_K,
            ),
        }
    except Exception as exc:
        comparison_metrics["hits_authority"] = {
            "runtime_seconds": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    metrics[dataset_name]["comparison"] = comparison_metrics

    save_topk_csv(
        RESULTS_DIR / f"{dataset_name}_pagerank_top20.csv",
        top_k(pagerank_result.scores, k=TOP_K),
        "pagerank",
    )


def add_reddit_cross_dataset_metrics(metrics: Dict[str, Dict[str, object]]) -> None:
    title_metrics = metrics.get("reddit_title")
    body_metrics = metrics.get("reddit_body")
    if title_metrics is None or body_metrics is None:
        return

    title_path = RESULTS_DIR / "reddit_title_pagerank_top20.csv"
    body_path = RESULTS_DIR / "reddit_body_pagerank_top20.csv"
    if not title_path.exists() or not body_path.exists():
        return

    def load_topk_scores(path: Path) -> Dict[str, float]:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            score_col = next(col for col in (reader.fieldnames or []) if col != "node")
            return {str(row["node"]): float(row[score_col]) for row in reader}

    title_scores = load_topk_scores(title_path)
    body_scores = load_topk_scores(body_path)

    comparison = {
        "spearman_vs_reddit_body_top20": spearman_correlation(title_scores, body_scores),
        "topk_overlap_vs_reddit_body": topk_overlap(title_scores, body_scores, k=TOP_K),
    }
    title_metrics["cross_dataset_comparison"] = comparison
    body_metrics["cross_dataset_comparison"] = {
        "spearman_vs_reddit_title_top20": comparison["spearman_vs_reddit_body_top20"],
        "topk_overlap_vs_reddit_title": comparison["topk_overlap_vs_reddit_body"],
    }


def run_reddit_sentiment_analysis(
    *,
    config: DatasetConfig,
    metrics: Dict[str, Dict[str, object]],
) -> None:
    """Run sentiment-split PageRank analysis for Reddit datasets."""
    if config.sentiment_index is None:
        return
    
    dataset_name = config.name
    
    # Run PageRank for positive sentiment (1)
    positive_result, positive_runtime = timed_pagerank_sentiment_filtered(
        config,
        sentiment_value=1,
        alpha=ALPHA,
        tol=TOL,
        max_iter=MAX_ITER,
    )
    
    # Run PageRank for negative sentiment (-1)
    negative_result, negative_runtime = timed_pagerank_sentiment_filtered(
        config,
        sentiment_value=-1,
        alpha=ALPHA,
        tol=TOL,
        max_iter=MAX_ITER,
    )
    
    # Store results in metrics
    if dataset_name not in metrics:
        metrics[dataset_name] = {}
    
    metrics[dataset_name]["pagerank_positive"] = {
        "runtime_seconds": positive_runtime,
        "iterations": positive_result.iterations,
        "converged": positive_result.converged,
        "final_error": positive_result.final_error,
    }
    
    metrics[dataset_name]["pagerank_negative"] = {
        "runtime_seconds": negative_runtime,
        "iterations": negative_result.iterations,
        "converged": negative_result.converged,
        "final_error": negative_result.final_error,
    }
    
    # Save top-k results to CSV
    save_topk_csv(
        RESULTS_DIR / f"{dataset_name}_positive_top20.csv",
        top_k(positive_result.scores, k=TOP_K),
        "pagerank",
    )
    
    save_topk_csv(
        RESULTS_DIR / f"{dataset_name}_negative_top20.csv",
        top_k(negative_result.scores, k=TOP_K),
        "pagerank",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, object]] = {}
    runtime_rows: List[List[object]] = [["dataset", "algorithm", "runtime_seconds"]]

    for config in default_dataset_configs():
        run_dataset_analysis(
            config=config,
            runtime_rows=runtime_rows,
            metrics=metrics,
        )

    # Run sentiment-split analysis for Reddit datasets
    for config in optional_reddit_configs():
        run_reddit_sentiment_analysis(
            config=config,
            metrics=metrics,
        )

    add_reddit_cross_dataset_metrics(metrics)

    with (RESULTS_DIR / "runtime_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(runtime_rows)

    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {RESULTS_DIR / 'metrics.json'}")
    print(f"Saved runtimes to {RESULTS_DIR / 'runtime_summary.csv'}")


if __name__ == "__main__":
    main()
