# PageRank

This project runs PageRank experiments on:
- `CA-GrQc.txt` (small graph)
- `com-dblp.ungraph.txt` (large graph)
- `soc-redditHyperlinks-title.tsv` (optional Reddit title links)
- `soc-redditHyperlinks-body.tsv` (optional Reddit body links)

## Files

- `pagerank_impl.py`: PageRank implementation and graph loading helpers.
- `experiments.py`: Runs experiments, timings, and metric comparisons.
- `plot_results.py`: Generates figures and tables in `figures/`.
- `results/`: Generated metrics and top-k outputs.
- `figures/`: Generated plot and table image files.

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
- `results/reddit_title_pagerank_top20.csv`
- `results/reddit_body_pagerank_top20.csv`

`experiments.py` parses subreddit labels as string node IDs and keeps edge direction as `SOURCE_SUBREDDIT -> TARGET_SUBREDDIT`.
Repeated subreddit-to-subreddit links are aggregated as edge weights, so duplicate edges increase that transition's weight in PageRank rather than being discarded.

## Generate Figures/Tables

After `experiments.py` finishes:

```bash
python plot_results.py
```

This generates:
- `figures/small_top20_table.png`
- `figures/large_top20_table.png`
- `figures/reddit_title_top20_table.png`
- `figures/reddit_body_top20_table.png`
- `figures/reddit_title_body_top10_comparison.png`
- `figures/runtime_comparison.png`
- `figures/small_weighted_top_nodes_induced_top20.png`
- `figures/small_weighted_top_nodes_neighbors.png`
- `figures/large_weighted_top_nodes_induced_top20.png`
- `figures/large_weighted_top_nodes_neighbors.png`
- `figures/reddit_title_weighted_top_nodes_connectivity.png`
- `figures/reddit_body_weighted_top_nodes_connectivity.png`

For Reddit title/body data, the plotting script generates:
- Top-20 PageRank tables
- Weighted top-node connectivity graphs
- A side-by-side title-vs-body top-10 comparison (when both files are present)