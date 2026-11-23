# One-Page Project Plan
**Project:** *Decentralized Graph Navigation: How to Prevent Action Cycles and Community Trapping Under Partial Observability?*

---

## Phase 1: Baseline & Data Check
* **Data Audit:** Verify graph connectivity, degree distributions, clustering coefficients, and feature completeness across the 5 SNAP Facebook ego-networks (148–532 nodes) and the synthetic homophilic graph generator. Construct leak-free **80/10/10 target-node splits** ($V_{\text{train}}, V_{\text{val}}, V_{\text{test}}$) and separate unseen synthetic graphs for inductive testing.
* **Working Baseline:** Implement the 1-hop Dec-POMDP simulation environment ($T_{\max} = 100$), centralized shortest-path oracle (Dijkstra/BFS), classical heuristic walkers (*Greedy, Softmax Distance, Degree Hub, Random*), and the **memory-less Graph Attention (GAT) Actor-Critic baseline** (Pisacane et al., 2024).
* **First Reproducible Evaluation:** Lock random seeds, training schedules, and hyperparameters to record reproducible baseline metrics ($R_{\text{trunc}}$, $\bar{R}_{\text{oracle}}$, $R_{\text{win}}$, Revisit Rate) across all 5 Facebook graphs, ensuring alignment with published benchmarks.

## Phase 2: First Research Results
* **First Hypothesis Test:** Implement and train the memory-based routing models using Recurrent Actor-Critic: **Sequential Memory (GRU)**, **Set-Based Memory (Deep Sets)**, and the **Action-Masked Tabu** baseline. Test them on the Facebook graphs to evaluate **Hypothesis 1** (fewer loops than the memory-less model) and **Hypothesis 2** (sequence order vs. visited coverage).
* **Failures:** Inspect failed episodes (reaching the 100-step limit) and overly long paths to identify the main error causes: 2-step back-and-forth loops, getting stuck inside dense clusters, unstable recurrent RL training, or information loss when averaging embeddings in set memory.
* **Next Controlled Experiment:** Based on these failure patterns, run targeted experiments changing one setting at a time on the same data splits—such as testing a sliding memory window ($W$), GRU hidden size, tabu list length ($k$), or adding a small step penalty ($-0.01$) / frozen GAT weights.

## Phase 3: Internal Research Review
* **Ablation / Stress Test:** Test Sequential vs. Set-Based memory on synthetic graphs by changing edge density ($\beta$) and cluster density to see where each approach works best or breaks. Also run component tests by swapping or turning off parts of the model (e.g., GAT vs. MLP, adding node degrees, or freezing GAT weights).
* **Uncertainty:** Repeat all main experiments and stress tests across **5 independent random seeds**, reporting **mean $\pm$ standard deviation** and paired significance bounds to verify that performance gains are statistically significant rather than random training noise.
* **Feedback:** Present preliminary result tables, failure examples, and trajectory plots to peers and the supervisor to collect **feedback** on baseline fairness, parameter settings, and chart clarity.
* **Draft Final Presentation:** Summarize all experimental findings, failure patterns, and conclusions comparing sequential vs. set-based memory into a final presentation and code repository.

---

## Summary Milestones

| Phase | Key Components | Verification Gate |
| :--- | :--- | :--- |
| **1. Baseline & Data Check** | **Data Audit** • **Working Baseline** • **First Reproducible Evaluation** | Zero data leaks across 80/10/10 splits; verified baseline benchmarks |
| **2. First Research Results** | **First Hypothesis Test** • **Failures** • **Next Controlled Experiment** | Memory models outperform baseline; root causes of loops isolated |
| **3. Internal Research Review** | **Ablation / Stress Test** • **Uncertainty** • **Feedback** • **Draft Final Presentation** | Significant gains across 5 seeds; feedback integrated into final deck |
