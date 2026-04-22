from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
SMALL_DATASET = Path("CA-GrQc.txt")
LARGE_DATASET = Path("com-dblp.ungraph.txt")


"""Reads CSV and returns (node, score) rows."""
def read_topk(path: Path) -> List[Tuple[int, float]]:
    rows: List[Tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        score_col = [c for c in reader.fieldnames or [] if c != "node"][0]
        for row in reader:
            rows.append((int(row["node"]), float(row[score_col])))
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
def load_dataset_graph(dataset_path: Path, *, bidirectional: bool) -> nx.DiGraph:
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

"""Renders results as a table"""
def plot_topk_table(rows: List[Tuple[int, float]], out_path: Path, title: str) -> None:
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
    runtime_map = {(dataset, algo): runtime for dataset, algo, runtime in rows}
    ordered_keys = [
        ("small", "pagerank"),
        ("large", "pagerank"),
        ("small", "hits_authority"),
        ("large", "hits_authority"),
        ("small", "degree_centrality"),
        ("large", "degree_centrality"),
    ]
    label_map = {
        ("small", "pagerank"): "small-pagerank",
        ("large", "pagerank"): "large-pagerank",
        ("small", "hits_authority"): "small-hits_authority",
        ("large", "hits_authority"): "large-hits_authority",
        ("small", "degree_centrality"): "small-degree_centrality",
        ("large", "degree_centrality"): "large-degree_centrality",
    }

    labels: List[str] = []
    values: List[float] = []
    for key in ordered_keys:
        if key in runtime_map:
            labels.append(label_map[key])
            values.append(runtime_map[key])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values)
    ax.set_title("Runtime Comparison Across Datasets and Algorithms")
    ax.set_ylabel("Runtime (seconds)")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


"""Selects nodes for display"""
def choose_subgraph_nodes(
    graph: nx.DiGraph,
    ranked_nodes: List[int],
    *,
    mode: str,
) -> set[int]:
    if mode == "neighbors":
        core_nodes = ranked_nodes[:5]
        selected = set(core_nodes)
        for node in core_nodes:
            selected.update(graph.predecessors(node))
            selected.update(graph.successors(node))
        return selected
    return set(ranked_nodes[:20])


"""Draws a weighted subgraph using top-ranked nodes."""
def plot_weighted_subgraph(
    graph: nx.DiGraph,
    score_map: Dict[int, float],
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
def plot_dataset_outputs(
    *,
    dataset_name: str,
    dataset_path: Path,
    bidirectional: bool,
) -> None:
    top20_rows = read_topk(RESULTS_DIR / f"{dataset_name}_pagerank_top20.csv")
    score_map = {node: score for node, score in top20_rows}
    dataset_label = dataset_path.name

    plot_topk_table(
        top20_rows,
        FIGURES_DIR / f"{dataset_name}_top20_table.png",
        title=f"Top 20 PageRank Nodes ({dataset_label})",
    )

    graph = load_dataset_graph(dataset_path, bidirectional=bidirectional)
    plot_weighted_subgraph(
        graph,
        score_map,
        FIGURES_DIR / f"{dataset_name}_weighted_top_nodes_induced_top20.png",
        mode="induced_top20",
        dataset_label=dataset_label,
    )
    plot_weighted_subgraph(
        graph,
        score_map,
        FIGURES_DIR / f"{dataset_name}_weighted_top_nodes_neighbors.png",
        mode="neighbors",
        dataset_label=dataset_label,
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    runtime_rows = read_runtime(RESULTS_DIR / "runtime_summary.csv")
    plot_runtime_chart(runtime_rows, FIGURES_DIR / "runtime_comparison.png")

    plot_dataset_outputs(
        dataset_name="small",
        dataset_path=SMALL_DATASET,
        bidirectional=False,
    )
    plot_dataset_outputs(
        dataset_name="large",
        dataset_path=LARGE_DATASET,
        bidirectional=True,
    )

    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
