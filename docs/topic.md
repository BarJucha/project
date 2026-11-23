# Project Description

## Title
**Decentralized Graph Navigation: How to Prevent Action Cycles and Community Trapping Under Partial Observability?**

---

## 1. Motivation
Classical graph search algorithms (e.g., $A^*$, Dijkstra) rely on a **centralized, global view** of the network. In real-world environments—such as peer-to-peer (P2P) systems, mobile ad-hoc networks (IoT), and privacy-sensitive social graphs—maintaining a global routing directory is infeasible due to privacy restrictions, dynamic topology changes, and memory bottlenecks. 

Decentralized routing forwards messages hop-by-hop using only **local, partial observations** (1-hop ego-neighborhoods) by exploiting network homophily (node attribute similarities) and structural hubs. However, navigating complex networks without global visibility poses a fundamental challenge: agents easily become entrapped in dense local communities or fall into repetitive action cycles.

---

## 2. Foundational Baseline & The Literature Gap

### Foundational Paper: *Reinforcement Learning Discovers Efficient Decentralized Graph Path Search Strategies* (Pisacane et al., LoG 2024)
* **Core Contribution:** Formulates decentralized routing as a collaborative Decentralized Partially Observable Markov Decision Process (**Dec-POMDP**). Autonomous nodes share parameters under the Centralized Training with Decentralized Execution (**CTDE**) framework. Each node uses **Graph Attention Networks (GAT)** over its 1-hop ego-neighborhood to embed topological structure and attribute homophily, coupled with an **Advantage Actor-Critic (A2C)** policy to forward the message to immediate neighbors.
* **The Critical Limitation (The Gap):** The navigation policy is strictly **memory-less**. The roaming message carries only the target representation, meaning agents have no awareness of prior trajectory steps. Consequently, agents cannot detect failed explorations, frequently resulting in **ping-pong loops, action cycles, and entrapment within dense community clusters** until reaching the truncation threshold ($T_{\max} = 100$).
* **Open Research Need:** As highlighted by the authors in their concluding remarks, extending the transmitted message to incorporate a historical representation of visited nodes is the most critical open direction to overcome action cycles.

---

## 3. Primary Research Question

Based on the critical gap identified above, this project addresses the overarching problem:

> **How can decentralized graph routing agents prevent action cycles and community trapping under partial local visibility?**

To answer this question, we turn to the broader literature on graph navigation and search to investigate how history and exploration states can be represented.

---

## 4. Literature Insights: Two Paradigms of History Representation

In adjacent graph learning domains, tracking search history has been addressed through two fundamentally different representations:

```
                    +------------------------------------------+
                    |   How Can History Be Represented?        |
                    +------------------------------------------+
                                  /            \
                                 /              \
     [ Order-Based: Sequential Trajectory ]    [ Coverage-Based: Explored Subgraph Set ]
                    |                                          |
        Inspiration: Shen et al. (2018)             Inspiration: Pándy et al. (2022)
                 (M-Walk)                                        (PHIL)
                    |                                          |
       Chronological Recurrence (GRU)              Permutation-Invariant Pooling (Deep Sets)
       (v_1 -> v_2 -> ... -> v_t)                  {v_i in V_visited}
```

### 1. Sequential Trajectory Modeling: *Learning to Walk over Graphs using Monte Carlo Tree Search* (Shen et al., NeurIPS 2018)
* **General Contribution:** Investigates how an RL agent can learn multi-hop walks over knowledge graphs under sparse, delayed rewards.
* **Key History Mechanism:** To resolve history-dependent state ambiguity, the authors employ a **recurrent neural network (GRU-RNN)** that sequentially encodes the chronological trajectory of traversed edges, nodes, and local candidate actions into a hidden state vector $h_{S,t}$.
* **Extracted Memory Concept:** **Sequential / Order-Based Memory.** History is viewed as an ordered, directional sequence $(v_1 \to v_2 \to \dots \to v_t)$, preserving temporal recency and movement momentum.

### 2. Explored Subgraph Aggregation: *Learning Graph Search Heuristics* (Pándy et al., LoG 2022)
* **General Contribution:** Investigates how neural networks can learn search heuristics ($A^*$ / Greedy Best-First Search) that avoid expanding unnecessary nodes.
* **Key History Mechanism:** To avoid re-exploring dead ends and escape local minima, the model maintains an **explored graph memory** that updates when new nodes are expanded, using **permutation-invariant pooling (mean aggregation)** across the explored set.
* **Extracted Memory Concept:** **Coverage / Set-Based Memory.** History is viewed as an unordered topological coverage set $\{v_i \in V_{\text{visited}}\}$, emphasizing which modular communities have already been explored rather than the exact sequence of steps.

---

## 5. Comparative Research Question & Hypotheses

By translating these two paradigms into our decentralized message-passing framework, we formulate the specific comparative research question:

> **Which memory representation—sequential trajectory encoding or set-based topological coverage—better prevents action cycles and community trapping in decentralized graph navigation?**

### Hypotheses
1. **Loop Elimination (General Memory Benefit):** Augmenting the message packet with a historical trajectory representation will significantly reduce navigation failures caused by endless cyclic exploration and trapped wandering compared to memory-less routing.
2. **Order vs. Coverage Inductive Bias:**
   * *Sequential Memory (Order-based)* will be more effective at resolving immediate back-and-forth oscillations ("ping-pong" loops) and maintaining directional momentum across bridge ties.
   * *Set-based Memory (Coverage-based)* will be superior at navigating out of dense, highly-clustered topological communities by penalizing re-entry into already-exhausted clusters regardless of the order in which they were visited.

---

## 6. Proposed Methodology

### 6.1 Architecture Overview
The message packet passed from node $u$ to neighbor $u'$ carries both the target goal and the updated history: $m_t = [x_{u_{\text{tgt}}}^{\text{GAT}} \parallel z_t]$, where $z_t \in \mathbb{R}^d$ is a fixed-size latent memory vector. By relying on a fixed-dimensional continuous embedding, the packet overhead remains constant ($O(1)$) across arbitrarily long search horizons.

```
+---------------------------+     +-------------------------------+     +--------------------------------+
| Local 1-Hop Ego-Graph G_u | --> | Memory Update Unit            | --> | Recurrent Actor-Critic         |
| Graph Attention Net (GAT) |     | - Sequential (GRU)            |     | - Policy: pi(a | G_u, m_t)     |
| Continuous Embedding x_u  |     | - Set-based (Deep Sets Pool)  |     | - Critic: v(x_u, m_t)          |
+---------------------------+     +-------------------------------+     +--------------------------------+
                                                                                        |
                                                                                        v
                                                                             Next Hop: u' in N(u)
```

### 6.2 Evaluated Memory Variants
1. **Memory-Less Baseline:** Reactive routing where decisions depend solely on $[x_{u'}^{\text{GAT}} \parallel x_{u_{\text{tgt}}}^{\text{GAT}}]$.
2. **Sequential Memory (GRU):** Chronological update $z_{t+1} = \text{GRU}(z_t, x_{v_{t+1}}^{\text{GAT}})$, preserving trajectory order and temporal recency.
3. **Set-Based Memory (Deep Sets):** Permutation-invariant aggregation $z_t = \text{MLP}\left( \frac{1}{|V_t|} \sum_{v \in V_{\text{visited}}} \text{MLP}(x_v^{\text{GAT}}) \right)$, capturing explored topological coverage.
4. **Action-Masked Tabu Baseline:** A symbolic heuristic baseline maintaining a FIFO buffer of the last $k$ visited node IDs in the packet header, setting logits of immediate backward hops to $-\infty$.

### 6.3 Optimization
Trained under **Centralized Training with Decentralized Execution (CTDE)** using **Recurrent Advantage Actor-Critic (R-A2C)** with entropy regularization and Backpropagation Through Time (BPTT).

---

## 7. Experimental Setup & Evaluation

### Datasets
* **Real-World Social Networks:** Facebook ego-networks (SNAP) with anonymized attribute vectors (148–532 nodes) to evaluate real community clustering and structural bridges.
* **Synthetic Modular Networks:** Homophilic spatial graphs with tunable modularity and density ($\beta$) to stress-test community trapping under controlled conditions.

### Data Splits
To prevent path memorization and ensure strict generalization to unseen destinations and topologies:
* **Target-Node Split (Real-World Graphs):** Following standard evaluation protocols, the node set $V$ of each Facebook ego-network is partitioned into **80% training ($V_{\text{train}}$)**, **10% validation ($V_{\text{val}}$)**, and **10% test ($V_{\text{test}}$)**. During testing, target destinations $u_{\text{tgt}}$ are sampled strictly from the unseen $V_{\text{test}}$ subset.
* **Inductive Graph Split (Synthetic Networks):** Models are trained on a set of generated graphs and tested on completely unseen synthetic graph instances generated with independent random seeds.

### Baselines
* **Memory-less Graph Attention Policy (Pisacane et al., 2024 / GARDEN):** The primary decentralized baseline, operating without historical memory.
* **Action-Masked Tabu Walker:** A symbolic heuristic baseline enforcing a hard constraint against immediate $k$-step backtracking.
* **Multi-Layer Perceptron Baselines (MLP-A2C and MLP-A2C-WD):** Architectures utilizing raw attributes and node degrees without graph attention message passing.
* **Decentralized Heuristic Walkers:** Greedy Attribute Walker, Softmax Distance Walker, Degree Hub Walker, Random Walk.
* **Centralized Shortest-Path Oracle (Dijkstra / BFS):** Theoretical upper bound with full global network visibility ($R_{\text{oracle}} = 1.0$).

### Evaluation Metrics
1. **Truncation Rate ($R_{\text{trunc}}$ %):** Percentage of episodes exceeding $T_{\max} = 100$ steps (primary metric for trapping/loop failure).
2. **Mean Oracle Ratio ($\bar{R}_{\text{oracle}}$):** Ratio of traversed path length to true shortest path.
3. **Win Rate ($R_{\text{win}}$ %):** Percentage of episodes where a model achieves the shortest path among decentralized baselines.
4. **Revisit Rate & Cycle Count:** Average number of repeated node visits per trajectory.

### Comparison Plan
All experiments will be executed across multiple random seeds to report mean performance and standard deviations across three structured evaluations:
1. **Real-World Benchmark (Verifying Hypothesis 1):** Compare the memory-augmented policies (Sequential GRU, Set-Based Deep Sets) against the Memory-less baseline, Action-Masked Tabu, MLP variants, and classical heuristics across all 5 Facebook ego-networks using $R_{\text{trunc}}$, $\bar{R}_{\text{oracle}}$, and $R_{\text{win}}$.
2. **Controlled Structural Stress-Testing (Verifying Hypothesis 2):** Sweep edge density ($\beta$) and cluster modularity in synthetic networks to isolate when Sequential Memory outperforms Set-Based Memory (and vice versa) under intense community trapping.
3. **Loop & Oscillation Analysis:** Categorize trajectory failures into immediate 2-step backtracks vs. extended cluster wandering using Revisit Rate and Cycle Count to explain the structural behavior of each memory mechanism.

---

## 8. Potential Risks & Fallback Strategy

### 1. Training Instability in Recurrent Reinforcement Learning
* **Risk:** Training recurrent neural networks (GRU) with reinforcement learning can be unstable or slow to converge, especially because rewards are sparse (granted only after reaching the destination) and search paths can reach up to 100 steps.
* **Fallback Plan:** Freeze the pre-trained Graph Attention Network (GAT) weights and train only the recurrent Actor-Critic decision layers. If the agent struggles to find the target, introduce a small step penalty (e.g., $-0.01$ per hop) to encourage faster, forward-moving exploration.

### 2. Information Loss in Set-Based Memory
* **Risk:** In set-based memory, averaging or summing the embeddings of all visited nodes over long walks can wash out important details, making it difficult for the agent to distinguish recently visited areas from older ones.
* **Fallback Plan:** Limit the set-based memory to a sliding window of the last $W$ visited nodes, or apply weighted pooling that places higher importance on structural hubs and bridge nodes.

### 3. Overfitting to Training Paths
* **Risk:** The memory model might memorize specific training routes rather than learning general strategies for avoiding loops on new, unseen test graphs.
* **Fallback Plan:** Use the Action-Masked Tabu baseline (which directly forbids returning to the last $k$ visited nodes) as a simple, rule-based fallback that works reliably without requiring neural generalization.

