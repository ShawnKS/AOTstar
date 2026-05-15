<h1 align="center">
  <img src="docs/media/aotstar-icon.png" width="180" /><br>
  <b><code>AOT*</code>: Efficient Synthesis Planning via LLM-Empowered AND-OR Tree Search</b><br>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" />
  <a href="https://arxiv.org/abs/2509.20988"><img src="https://img.shields.io/badge/paper-arXiv%3A2509.20988-B31B1B.svg" /></a>
</p>

`AOT*` is a retrosynthesis planning framework that integrates LLM-generated
multi-step synthesis pathways with systematic AND-OR tree search.

Instead of treating each LLM proposal as an isolated route, AOT* maps valid
generated pathways into a shared AND-OR tree. The search can then reuse
intermediate molecules, revisit promising unsolved nodes, and guide later LLM
calls with the structure discovered so far.

**Paper** (accepted as ACL2026 findings): [AOT*: Efficient Synthesis Planning via LLM-Empowered AND-OR Tree
Search](https://arxiv.org/abs/2509.20988)

## Method Overview

Retrosynthetic planning requires exploring a large combinatorial space of candidate reactions and intermediates.
AOT* addresses this challenge by integrating three components into a single search framework:

- LLM-based generation of multi-step retrosynthetic routes
- template-based validation and pathway-to-tree mapping
- systematic AND-OR tree search with backpropagation over accepted branches

## Repository Layout

```text
.
├── unified_search.py              # Main search runner
├── aotcore/
│   ├── llm_tree_optimizer.py      # AOT* tree search loop
│   ├── optimizer.py               # Reaction validation and template search
│   ├── oracle_rerank.py           # Template candidate reranking utilities
│   ├── prompts.py                 # LLM prompt construction
│   ├── tree_nodes.py              # AND/OR node data structures
│   ├── utils.py                   # Chemistry helpers
│   ├── data/                      # Dataset, inventory, and cache loaders
│   └── tools/                     # Data-preparation utilities
├── oracle/                        # Runtime scoring resources
└── scscore/                       # SCScore dependency used by the search
```

## Installation

```bash
git clone https://github.com/ShawnKS/AOTstar.git
cd AOTstar
conda create -n aotstar python=3.10
conda activate aotstar
```

Install the chemistry, search, and LLM-client packages used by the runner:

```bash
pip install rdkit openai httpx rdchiral
pip install numpy pandas psutil PyYAML tqdm
pip install PyTDC selfies
pip install "syntheseus[all]"
```

AOT* uses an OpenAI-compatible chat-completion endpoint for LLM calls. The
endpoint URL, model name, and API key are supplied when launching the search.

## Data Preparation

Full runtime assets are not shipped with this repository. For the benchmark
bundle, download it
[here](https://www.dropbox.com/scl/fi/dmmypid2ooohp3freiox8/dataset.zip?rlkey=fmrhvds6fmxck2cp8h94albpc&st=8fmtxls4&dl=0).

After downloading, unzip the archive and place the resulting `dataset/`
directory at the repository root:

```bash
unzip /path/to/dataset.zip -d .
```

## Running AOT*

After preparing the dataset, run:

```bash
python -u unified_search.py \
  --dataset uspto_190 \
  --log-dir runs/search \
  --llm-base-url "https://your-openai-compatible-endpoint/v1" \
  --llm-model "your-model-name" \
  --llm-api-key "your-api-key"
```

By default, the runner searches all targets in the selected dataset with the
main AOT* search settings. To run a subset, select a contiguous range:

```bash
python -u unified_search.py \
  --dataset uspto_190 \
  --start-idx 0 \
  --n-targets 10 \
  --log-dir runs/search \
  --llm-base-url "https://your-openai-compatible-endpoint/v1" \
  --llm-model "your-model-name" \
  --llm-api-key "your-api-key"
```

For a custom target set, pass `--targets-file path/to/targets.json`.

## Outputs

Each run writes structured JSON files under the selected log directory:

- `result_*.json`: per-target search result and solution tree.
- `search_log_*.json`: per-target search trace.
- `summary.json`: aggregate success rate and timing summary.
- `all_results.json`: collected per-target results for the run.

## Citation

```bibtex
@article{song2025aotstar,
  title={AOT*: Efficient Synthesis Planning via LLM-Empowered AND-OR Tree Search},
  author={Song, Xiaozhuang and Pan, Xuanhao and Zhao, Xinjian and
          Ye, Hangting and Zhang, Shufei and Tang, Jian and Yu, Tianshu},
  journal={arXiv preprint arXiv:2509.20988},
  year={2025}
}
```
