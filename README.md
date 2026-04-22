# PageRank Experiment Pipeline

This project runs a full PageRank ranking on:
- Small dataset: `CA-GrQc.txt`
- Large dataset: `com-dblp.ungraph.txt`


## Files

- `pagerank_impl.py`: PageRank implementation and utility helpers.
- `experiments.py`: runs experiments, timings, and metric comparisons.
- `plot_results.py`: generates required tables/figures in `figures/`.
- `results/`: generated metrics and top-k outputs.
- `figures/`: generated plots/tables as image files.

## Setup

Install Python dependencies:

```bash
pip install networkx matplotlib numpy
```

## Run Experiments

From project root:

```bash
python experiments.py
```

This generates:
- `results/metrics.json`
- `results/runtime_summary.csv`
- `results/small_pagerank_top20.csv`
- `results/large_pagerank_top20.csv`
The script runs a full analysis for both datasets in one pass (PageRank, HITS authority, and degree centrality).

## Generate Figures/Tables

After `experiments.py` finishes:

```bash
python plot_results.py
```

This generates:
- `figures/small_top20_table.png`
- `figures/large_top20_table.png`
- `figures/runtime_comparison.png`
- `figures/small_weighted_top_nodes_induced_top20.png`
- `figures/small_weighted_top_nodes_neighbors.png`
- `figures/large_weighted_top_nodes_induced_top20.png`
- `figures/large_weighted_top_nodes_neighbors.png`