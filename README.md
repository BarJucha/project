# Decentralized Graph Navigation

> **Decentralized Graph Navigation: How to Prevent Action Cycles and Community Trapping Under Partial Observability?**

An empirical and theoretical investigation into memory mechanisms for decentralized, local-view graph routing. This project builds upon the foundational **GARDEN** baseline (*Pisacane et al., LoG 2024*) and investigates **Sequential Memory (GRU)** vs. **Set-Based Memory (Deep Sets)** to eliminate action cycles, ping-pong oscillations, and community trapping under 1-hop partial observability.

---

## 📁 Repository Structure

```
.
├── configs/                  # Experiment configurations (hyperparameters, splits, models)
├── data/
│   ├── raw/                  # Downloaded SNAP Facebook ego-networks
│   └── processed/            # Pre-processed graph structures and feature matrices
├── docs/                     # Project documentation, research proposals, and literature
│   ├── concept.txt           # Initial brainstorming notes
│   ├── literature_matrix.md  # Detailed comparison matrix of referenced papers
│   ├── literature_matrix_pl.md # Polish version of literature matrix
│   ├── project_plan.md       # 3-Phase project plan with milestone gates
│   ├── topic.md              # Complete research proposal & mathematical formulation
│   ├── papers/               # Analyzed literature
│   │   ├── pdf/              # Foundational papers (Pisacane 2024, Shen 2018, Pándy 2022)
│   │   ├── summary/          # In-depth English paper summaries
│   │   └── summary_pl/       # In-depth Polish paper summaries
│   └── presentations/        # Topic defense and progress presentations
├── notebooks/                # Exploratory data analysis and visualization notebooks
├── src/                      # Source code
│   ├── baselines/            # Oracles (BFS, Dijkstra) & Heuristics (Greedy, Softmax, Hub, Random, Tabu)
│   ├── data/                 # Dataset loaders (SNAP Facebook) & Synthetic graph generators
│   ├── env/                  # Dec-POMDP 1-hop graph routing simulation environment
│   ├── models/               # GAT encoders, GRU memory, Deep Sets memory, Actor-Critic networks
│   ├── training/             # CTDE Recurrent Advantage Actor-Critic (R-A2C) trainer
│   └── utils/                # Metrics (R_trunc, R_oracle, R_win, loop counts), logging, seeding
├── tests/                    # Pytest test suite for environment, models, and baselines
├── pyproject.toml            # Package specifications and dependencies
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🛠️ Environment Setup

### 1. Prerequisites
- Python 3.12 (or >= 3.10)
- NVIDIA GPU with CUDA support (tested with RTX 4050 Laptop, CUDA 12.4)
- `uv` (recommended) or standard `python -m venv`

### 2. Quick Setup with `uv`

```powershell
# Create virtual environment
uv venv .venv --python 3.12

# Activate virtual environment
.venv\Scripts\activate

# Install PyTorch with CUDA 12.4 support
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install all project dependencies in editable mode
uv pip install -e ".[dev]"
```

Alternatively, with standard `pip`:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## 🔬 Research Hypotheses & Evaluated Models

1. **Memory-Less Baseline (GARDEN - Pisacane et al., LoG 2024):**
   - Decisions based solely on current 1-hop ego-graph embedding and target embedding: $[x_{u'}^{\text{GAT}} \parallel x_{u_{\text{tgt}}}^{\text{GAT}}]$.
2. **Sequential Memory (GRU - Shen et al., NeurIPS 2018):**
   - Chronological update $z_{t+1} = \text{GRU}(z_t, x_{v_{t+1}}^{\text{GAT}})$. Preserves trajectory sequence and movement momentum to eliminate immediate 2-hop oscillations.
3. **Set-Based Memory (Deep Sets - Pándy et al., LoG 2022):**
   - Permutation-invariant aggregation $z_t = \text{MLP}\left( \frac{1}{|V_t|} \sum_{v \in V_{\text{visited}}} \text{MLP}(x_v^{\text{GAT}}) \right)$. Captures topological coverage to prevent re-entering exhausted community clusters.
4. **Action-Masked Tabu Baseline:**
   - Heuristic baseline retaining a FIFO buffer of the last $k$ visited nodes, masking backward hops.

---

## 📊 Core Evaluation Metrics

- **$R_{\text{trunc}}$ (%):** Truncation rate — percentage of episodes exceeding $T_{\max} = 100$ steps without reaching the target.
- **$\bar{R}_{\text{oracle}}$:** Average ratio of discovered path length to the shortest path (Dijkstra/BFS oracle).
- **$R_{\text{win}}$ (%):** Percentage of test episodes where the model achieves the shortest path among all competing methods.
- **Revisit Rate / Loop Metrics:** Number of repeated nodes and cyclic sub-paths traversed per episode.

---

## 📅 Roadmap & Milestones

See [docs/project_plan.md](docs/project_plan.md) for full details:
- **Phase 1: Baseline & Data Check:** SNAP Facebook ego-network downloader & auditor, leak-free 80/10/10 target splits, Dec-POMDP gym-like environment, Dijkstra oracle, heuristic walkers, and GARDEN memory-less baseline.
- **Phase 2: First Research Results:** Recurrent Actor-Critic with Sequential (GRU) vs. Set-Based (Deep Sets) memory, failure analysis of community entrapment.
- **Phase 3: Internal Review & Ablations:** Synthetic graph stress tests over homophily $\beta$, 5-seed uncertainty validation, final presentation.
