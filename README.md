# PageRank Experiment Pipeline

This project runs a full PageRank ranking on:
- Small dataset: `CA-GrQc.txt`
- Large dataset: `com-dblp.ungraph.txt`
- Optional labeled dataset: Reddit title hyperlinks (`soc-redditHyperlinks-title.tsv`)
- Optional labeled dataset: Reddit body hyperlinks (`soc-redditHyperlinks-body.tsv`)


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
- `results/reddit_title_pagerank_top20.csv` (only if a supported Reddit title TSV is present)
- `results/reddit_body_pagerank_top20.csv` (only if a supported Reddit body TSV is present)
The script runs a full analysis for both datasets in one pass (PageRank, HITS authority, and degree centrality).

## Optional Reddit Dataset

To include a directed, human-readable Reddit hyperlink graph, place one of these files in the project root or `data/`:

- `soc-redditHyperlinks-title.tsv`
- `soc-redditHyperlinks-body.tsv`
- `reddit_hyperlinks_title.tsv`
- `reddit_hyperlinks_body.tsv`

When present, `experiments.py` automatically adds `reddit_title` and/or `reddit_body` runs.
It parses subreddit labels as string node IDs and keeps edge direction as `SOURCE_SUBREDDIT -> TARGET_SUBREDDIT`.
Repeated subreddit-to-subreddit links are aggregated as edge weights, so duplicate edges increase that transition's weight in PageRank rather than being discarded.

## Generate Figures/Tables

After `experiments.py` finishes:

```bash
python plot_results.py
```

This generates:
- `figures/small_top20_table.png`
- `figures/large_top20_table.png`
- `figures/reddit_title_top20_table.png` (if Reddit title results exist)
- `figures/reddit_body_top20_table.png` (if Reddit body results exist)
- `figures/reddit_title_body_top10_comparison.png` (if both Reddit title and body results exist)
- `figures/runtime_comparison.png`
- `figures/small_weighted_top_nodes_induced_top20.png`
- `figures/small_weighted_top_nodes_neighbors.png`
- `figures/large_weighted_top_nodes_induced_top20.png`
- `figures/large_weighted_top_nodes_neighbors.png`

For Reddit title/body data, the plotting script generates the top-20 table, runtime chart, and a side-by-side title-vs-body top-10 comparison when both files are present. Subgraph image generation is disabled by default to avoid unreadable label overlap with long subreddit names.