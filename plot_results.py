from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Mapping, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from pagerank_impl import load_edge_list


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
SMALL_DATASET = Path("CA-GrQc.txt")
LARGE_DATASET = Path("com-dblp.ungraph.txt")


@dataclass(frozen=True)
class DatasetPlotConfig:
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
    subgraph_modes: Tuple[str, ...] = ("induced_top20", "neighbors")


def reddit_plot_config(name: str, candidates: List[Path]) -> DatasetPlotConfig | None:
    for path in candidates:
        if path.exists():
            return DatasetPlotConfig(
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
                subgraph_modes=("connectivity",),
            )
    return None


def default_plot_configs() -> List[DatasetPlotConfig]:
    configs = [
        DatasetPlotConfig(
            name="small",
            path=SMALL_DATASET,
            bidirectional=False,
            node_parser=int,
        ),
        DatasetPlotConfig(
            name="large",
            path=LARGE_DATASET,
            bidirectional=True,
            node_parser=int,
        ),
    ]
    configs.extend(optional_reddit_plot_configs())
    return configs


def optional_reddit_plot_configs() -> List[DatasetPlotConfig]:
    configs: List[DatasetPlotConfig] = []

    title_config = reddit_plot_config(
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

    body_config = reddit_plot_config(
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


def parse_node_value(text: str) -> Hashable:
    try:
        return int(text)
    except ValueError:
        return text


"""Reads CSV and returns (node, score) rows."""
def read_topk(path: Path) -> List[Tuple[Hashable, float]]:
    rows: List[Tuple[Hashable, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        score_col = [c for c in reader.fieldnames or [] if c != "node"][0]
        for row in reader:
            rows.append((parse_node_value(row["node"]), float(row[score_col])))
    return rows

"""Reads runtime summary CSV and returns tuples."""
def read_runtime(path: Path) -> List[Tuple[str, str, float]]:
    rows: List[Tuple[str, str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["dataset"], row["algorithm"], float(row["runtime_seconds"])))
    return rows

"""Loads edge list"""
def load_dataset_graph(config: DatasetPlotConfig) -> nx.DiGraph:
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

"""Renders results as a table"""
def plot_topk_table(rows: List[Tuple[Hashable, float]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.axis("off")
    table_data = [[node, f"{score:.8f}"] for node, score in rows]
    table = ax.table(
        cellText=table_data,
        colLabels=["Node", "PageRank"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.2)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


"""Bar chart comparing runtime across datasets andalgorithms."""
def plot_runtime_chart(rows: List[Tuple[str, str, float]], out_path: Path) -> None:
    algo_order = {"pagerank": 0, "hits_authority": 1, "degree_centrality": 2}
    ordered_rows = sorted(rows, key=lambda row: (row[0], algo_order.get(row[1], 99), row[1]))
    labels = [f"{dataset}-{algorithm}" for dataset, algorithm, _ in ordered_rows]
    values = [runtime for _, _, runtime in ordered_rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values)
    ax.set_title("Runtime Comparison Across Datasets and Algorithms")
    ax.set_ylabel("Runtime (seconds)")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_reddit_top10_comparison(out_path: Path) -> None:
    title_path = RESULTS_DIR / "reddit_title_pagerank_top20.csv"
    body_path = RESULTS_DIR / "reddit_body_pagerank_top20.csv"
    if not title_path.exists() or not body_path.exists():
        return

    title_rows = read_topk(title_path)[:10]
    body_rows = read_topk(body_path)[:10]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False)
    datasets = [
        (axes[0], title_rows, "Reddit Title Top 10"),
        (axes[1], body_rows, "Reddit Body Top 10"),
    ]

    for ax, rows, title in datasets:
        labels = [str(node) for node, _ in rows][::-1]
        values = [score for _, score in rows][::-1]
        ax.barh(labels, values, color="#4c78a8")
        ax.set_title(title)
        ax.set_xlabel("PageRank score")

    fig.suptitle("Reddit PageRank Comparison: Title vs Body")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


"""Selects nodes for display"""
def choose_subgraph_nodes(
    graph: nx.DiGraph,
    ranked_nodes: List[Hashable],
    *,
    mode: str,
) -> set[Hashable]:
    if mode == "neighbors":
        core_nodes = ranked_nodes[:5]
        selected = set(core_nodes)
        for node in core_nodes:
            selected.update(graph.predecessors(node))
            selected.update(graph.successors(node))
        return selected
    if mode == "connectivity":
        return set(ranked_nodes[:15])
    return set(ranked_nodes[:20])


def mode_filename_suffix(mode: str) -> str:
    return {
        "induced_top20": "weighted_top_nodes_induced_top20",
        "neighbors": "weighted_top_nodes_neighbors",
        "connectivity": "weighted_top_nodes_connectivity",
    }.get(mode, f"weighted_top_nodes_{mode}")


"""Draws a weighted subgraph using top-ranked nodes."""
def plot_weighted_subgraph(
    graph: nx.DiGraph,
    score_map: Dict[Hashable, float],
    out_path: Path,
    *,
    mode: str,
    dataset_label: str,
) -> None:
    ranked_nodes = [node for node, _ in sorted(score_map.items(), key=lambda p: p[1], reverse=True)]
    selected = choose_subgraph_nodes(graph, ranked_nodes, mode=mode)
    subgraph = graph.subgraph(selected).copy()

    if subgraph.number_of_nodes() == 0:
        raise ValueError()

    node_sizes = [1200 + 25000 * score_map.get(node, 0.0) for node in subgraph.nodes()]
    edge_widths = []
    for src, dst in subgraph.edges():
        edge_widths.append(0.4 + 30.0 * (score_map.get(src, 0.0) + score_map.get(dst, 0.0)))

    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(subgraph, seed=42, k=0.4)
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(
        subgraph,
        pos,
        width=edge_widths,
        alpha=0.45,
        arrows=subgraph.is_directed(),
        ax=ax,
    )
    nx.draw_networkx_labels(subgraph, pos, font_size=8, ax=ax)
    ax.set_title(f"Weighted Top-Node Connectivity Graph ({dataset_label}, {mode})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


"""Generates all plots from stored results."""
def plot_dataset_outputs(config: DatasetPlotConfig) -> None:
    dataset_name = config.name
    top20_path = RESULTS_DIR / f"{dataset_name}_pagerank_top20.csv"
    if not top20_path.exists():
        return

    top20_rows = read_topk(top20_path)
    score_map = {node: score for node, score in top20_rows}
    dataset_label = config.path.name

    plot_topk_table(
        top20_rows,
        FIGURES_DIR / f"{dataset_name}_top20_table.png",
        title=f"Top 20 PageRank Nodes ({dataset_label})",
    )

    if not config.subgraph_modes:
        return

    graph = load_dataset_graph(config)
    for mode in config.subgraph_modes:
        plot_weighted_subgraph(
            graph,
            score_map,
            FIGURES_DIR / f"{dataset_name}_{mode_filename_suffix(mode)}.png",
            mode=mode,
            dataset_label=dataset_label,
        )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    runtime_rows = read_runtime(RESULTS_DIR / "runtime_summary.csv")
    plot_runtime_chart(runtime_rows, FIGURES_DIR / "runtime_comparison.png")
    plot_reddit_top10_comparison(FIGURES_DIR / "reddit_title_body_top10_comparison.png")

    for config in default_plot_configs():
        plot_dataset_outputs(config)

    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
