# Learning Graph Search Heuristics: Paper Breakdown & Analysis (Pándy et al., 2022)

**Authors:** Michal Pándy, Weikang Qiu, Gabriele Corso, Petar Veličković, Rex Ying, Jure Leskovec, Pietro Liò  
**Conference:** *Proceedings of the First Learning on Graphs Conference (LoG 2022)*, PMLR vol. 198  
**Original Title:** *Learning Graph Search Heuristics* (PHIL architecture — *Path Heuristic with Imitation Learning*)

---

## Abstract

Finding a path between two nodes in a graph is one of the most fundamental and extensively studied problems in computer science. Across numerous fields such as robotics, artificial intelligence, and biology, practitioners manually engineer search heuristics to accelerate pathfinding algorithms. However, this is a time-consuming and complex process. This paper presents **PHIL** (*Path Heuristic with Imitation Learning*) — a novel neural architecture and learning algorithm designed to automatically discover search and navigation heuristics on graphs from data, leveraging recent advances in imitation learning and graph representation learning.

During training, datasets containing search trajectories and ground-truth shortest path distances are aggregated. These data are used to train a specialized heuristic function based on Graph Neural Networks (GNNs) using truncated backpropagation through time. The heuristic function learns graph embeddings useful for reasoning about node distances, operates in constant time $O(1)$ regardless of graph size, and can be seamlessly integrated with algorithms such as A* or Greedy Best-First Search at test time. Experiments demonstrate that PHIL reduces the number of explored nodes by an average of **58.5%** compared to the state-of-the-art (SOTA), generalizes across diverse graph domains (from biological to road networks), and enables fast planning in time-critical robotic applications.

---

## 1. Introduction

Search heuristics are essential in many domains. In robotics, complex robot geometry leads to slow collision checking, and motion planning algorithms are constrained by computational onboard budgets, necessitating heuristics that minimize the number of visited nodes. In artificial intelligence, domain-specific heuristics accelerate inference in knowledge bases. They are also applied to reduce computational workload in protein-protein interaction networks and chemical reaction planning.

The search problem can be formulated as navigating a graph from a start node $v_s$ to a goal node $v_g$ while minimizing the number of visited nodes. The simplest approach, such as breadth-first search (BFS), expands the graph uniformly from $v_s$ without utilizing any priority knowledge, resulting in the exploration of a vast number of unnecessary nodes. To prevent this, a heuristic function $h$, defined over the graph nodes, is used to prune less promising directions. However, manual design of such heuristics requires substantial domain expertise and trial-and-error.

Although classical approaches rely on handcrafted rules and machine learning algorithms seek to automate this process, existing heuristic learning methods face three primary challenges:
1. **Non-i.i.d. Data Distribution**: Newly encountered nodes depend on heuristic values in previous steps, causing traditional supervised learning to fail.
2. **Constant Time Complexity $O(1)$ Requirement**: Computing heuristic values must be extremely fast. If the heuristic function scales with the graph size, the asymptotic complexity of the entire search increases.
3. **Reinforcement Learning (RL) Scalability**: As environment (graph) size grows, reinforcement learning methods suffer from poor sample efficiency.

PHIL resolves these issues by combining imitation learning with a cost-to-go oracle and an explored graph memory recurrent module, guaranteeing $O(1)$ complexity with respect to graph size.

---

## 2. Preliminaries

### Graph Search Formulation
Let $G = (V, E)$ be an undirected, unweighted, and connected graph where each node $i \in V$ has a feature vector $x_i \in \mathbb{R}^{D_v}$ and each edge $(i, j) \in E$ has features $e_{ij} \in \mathbb{R}^{D_e}$. The objective is to navigate from a source node $v_s \in V$ to a target node $v_g \in V$.

At any point during the search, the set of graph nodes is partitioned into three disjoint subsets:
$$\text{V} = \text{CLOSE} \cup \text{OPEN} \cup \text{REST}$$
- **CLOSE**: Nodes already explored/visited.
- **OPEN**: Candidate nodes for exploration (connected to nodes in CLOSE, but not yet belonging to it — the frontier/fringe).
- **REST**: The remaining, undiscovered part of the graph.

An expansion consists of moving a node from OPEN to CLOSE and adding its neighbors in REST to OPEN. The set of newly added frontier nodes at a given step is denoted as $V_{new}$. At the beginning of the search, $\text{CLOSE} = \{v_s\}$, and the process continues until $v_g \in \text{CLOSE}$.

### Greedy Best-First Search
In this regime, at each step the agent expands the node $v \in \text{OPEN}$ that minimizes a tailored heuristic function $h(v, v_g)$. The goal of PHIL is to learn a function $h$ that accurately predicts the shortest path distance to the goal, thereby minimizing the size of $|\text{CLOSE}|$.

### Imitating an Ideal Heuristic
The heuristic learning problem can be formulated within a Partially Observable Markov Decision Process (POMDP), where state is a tuple $s = (\text{CLOSE}, \text{OPEN}, \text{REST})$, action $a$ is moving a node from OPEN to CLOSE, and observations $o$ are the features of newly incorporated nodes. An oracle $h^*$ is employed, which possesses full state information $s$ and provides the true shortest path distances $h^*(s, v, v_g)$ for $v \in \text{OPEN}$.

---

## 3. Related Work

The literature on graph search and navigation learning can be categorized into several main paradigms:

| Method Category | Description & Representative Approaches | Limitations & Relation to PHIL |
| :--- | :--- | :--- |
| **General Heuristic Design** | Approaches such as BFWS (*Best-First Width Search*) or landmark-based heuristics. | Effective at escaping local minima, but fail to fully exploit specific graph data structures. |
| **Search Heuristic Learning** | Methods learning node evaluations (e.g., SAIL, Retro*, Neural A*). | Methods like SAIL require manual feature engineering and have linear computational complexity with respect to the number of nodes, failing to guarantee $O(1)$ time. |
| **General Search Learning** | Learning navigation policies using MCTS (e.g., AlphaGo/MCTSNet) or integrated planning (Value Iteration Networks). | Treat the problem as a general MDP/POMDP, incurring excessive computational overhead that prevents them from competing with fast greedy search. |
| **Imitation Learning (IL)** | Algorithms with a cost-to-go oracle, such as AggreVaTe, AggreVaTeD, LOLS, DART, THOR. | PHIL extends the SAIL/AggreVaTe algorithm by learning a recurrent value function with graph structure memory while preserving constant $O(1)$ time. |

---

## 4. Architecture and Method: PHIL

### Objective Function and State Representation
The primary objective is to train a parameterized heuristic function $h_\theta(\psi, v, v_g)$ that predicts true distances $h^*$, minimizing the expected squared error:

$$L(\theta) = \mathbb{E}_{\substack{G \sim P_G, (v_s, v_g) \sim P_{vsg} \\ t \sim U(0, \dots, T), s, \psi \sim P_s}} \left[ \frac{1}{|\text{OPEN}|} \sum_{v \in \text{OPEN}} \left( h^*(s, v, v_g) - h_\theta(\psi, v, v_g) \right)^2 \right]$$

To reduce computational complexity, $h_\theta$ takes an encoded memory vector $z_t \in \mathbb{R}^d$ representing the search history $\psi$ at step $t$. Additionally, to avoid constantly recalculating heuristic values for all nodes in OPEN (which would incur $O(|\text{OPEN}|)$ complexity), **PHIL computes heuristic values exclusively for newly added frontier nodes $V_{new}$**.

### Imitation Learning Algorithm
Training proceeds according to the following procedure:

1. **Roll-in:** For a sampled graph $G$ and pair $(v_s, v_g)$, the current greedy policy $\pi_\theta$ executes $t$ exploration steps, generating the initial state $s_t = (\text{CLOSE}_0, \text{OPEN}_0, \text{REST}_0)$ and the initial memory state $z_t$.
2. **Roll-out:** Next, a trajectory is executed for $t_\tau$ steps using a mixed policy $\pi_{mix} = (1 - \beta) \cdot \pi_\theta + \beta \cdot \pi^*$, where $\pi^*$ is the oracle policy. During the roll-out, sets of newly added frontier nodes $V_{new}$ and their ground-truth distances $h^*$ are collected.
3. **Optimization (TBTT):** Truncated Backpropagation Through Time (TBTT) is applied to the collected trajectories to update parameters $\theta$.

### Recurrent Graph Neural Network (GNN) Architecture
At each step, the heuristic function $h_\theta$ processes the new frontier nodes $V_{new}$, the goal node $v_g$, and the current memory state $z_t$. The computational pipeline consists of the following stages:

1. **Feature Projection:** Node features $x_i$ and neighbor features $x_j$ are concatenated with target features $x_g$ as well as their Euclidean distance $D_{\text{EUC}}$ and cosine distance $D_{\text{COS}}$:
   $$x_i \leftarrow f(x_i, x_g, D_{\text{EUC}}(x_i, x_g), D_{\text{COS}}(x_i, x_g))$$
2. **Graph Convolution (GNN):** For each node $i \in V_{new}$, a subset of neighbors from its 1-hop neighborhood $N_i$ (where $|N_i| \le n$) is sampled. A single-layer message aggregation is performed:
   $$g_i \leftarrow \phi\left(x_i, \bigoplus_{j \in N_i} \gamma(x_i, x_j, e_{ij})\right)$$
3. **Memory Update (GRU):** The representation vector $g_i$ is updated through a GRU cell along with the previous memory state:
   $$g'_i, z_{i, t+1} \leftarrow \text{GRU}(g_i, z_{i, t})$$
4. **Permutation-Invariant Aggregation:** The new memory state $z_{t+1}$ is computed as the sample mean of states $z_{i, t+1}$ across all $i \in V_{new}$. This ensures permutation invariance with respect to node ordering in the batch and enables efficient parallel processing on GPUs.
5. **Distance Prediction:** Distance $\hat{h}_i$ is predicted by a multi-layer perceptron: $\hat{h}_i \leftarrow \text{MLP}(g'_i, x_g)$.

### Constant Time Complexity $O(1)$
Because the sampled neighborhood size is bounded ($|N_i| \le n$) for each node $i \in V_{new}$, the total number of operations per single node is at most $n c_1 + (n+1)c_2$, where $c_1$ represents node evaluation cost (e.g., collision check) and $c_2$ is model operation cost. Consequently, the execution time of the heuristic function is **independent of total graph size** $O(1)$.

---

## 5. Experiments & Results

Experiments were conducted across three primary test environments: 2D grids, diverse real-world graphs, and 3D drone flight planning simulations.

### 5.1 Grid Search (200 $\times$ 200 8-connected grids)
Evaluations were carried out on 8 benchmark datasets featuring complex obstacle configurations and trap topologies (e.g., *Single Bugtrap*, *Forest*, *Mazes*).

The table below presents the ratio of explored nodes by PHIL and other methods normalized to the leading baseline (**SAIL**):

| Dataset | SAIL | SL (Supervised) | CEM | QL | $h_{euc}$ | $h_{man}$ | A* | MHA* | BFWS | Neural A* | **PHIL (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Alternating gaps** | 0.039 | 0.432 | 0.042 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.340 | 0.546 | **0.024** |
| **Single Bugtrap** | 0.158 | 0.214 | 0.057 | 1.000 | 0.184 | 0.192 | 1.000 | 0.286 | 0.099 | 0.394 | **0.077** |
| **Shifting gaps** | 0.104 | 0.464 | 1.000 | 1.000 | 0.506 | 0.589 | 1.000 | 0.804 | 0.206 | 0.563 | **0.027** |
| **Forest** | 0.036 | 0.043 | 0.048 | 0.121 | 0.041 | 0.043 | 1.000 | 0.075 | 0.039 | 0.399 | **0.027** |
| **Bugtrap+Forest** | 0.147 | 0.384 | 0.182 | 1.000 | 0.410 | 0.337 | 1.000 | 3.177 | 0.149 | 0.651 | **0.135** |
| **Gaps+Forest** | 0.221 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.401 | 0.580 | **0.039** |
| **Mazes** | 0.103 | 0.238 | 0.479 | 0.399 | 0.185 | 0.171 | 1.000 | 0.279 | 0.095 | 1.000 | **0.069** |
| **Multiple Bugtraps** | 0.479 | 0.480 | 1.000 | 0.835 | 0.648 | 0.617 | 1.000 | 0.876 | 0.169 | 0.331 | **0.136** |

**Grid Evaluation Key Findings:**
- PHIL outperformed SAIL across all datasets, reducing the number of explored nodes by an average of **58.5%** (with a peak reduction of **82.3%** on *Gaps+Forest*).
- PHIL demonstrates superior capability in escaping local minima thanks to its explored graph memory.
- Convergence rate for PHIL was 5 times faster than SAIL (reaching convergence in approximately 36 iterations).

### 5.2 Real-World Graph Search
Evaluation encompassed 4 graph domain categories:
1. **Citation Networks:** Cora, PubMed, CiteSeer, Coauthor (cs, physics).
2. **Biological Networks:** OGBG-Molhiv, PPI, Proteins (Full), Enzymes.
3. **AST Trees:** OGBG-Code2.
4. **Road Networks:** OSMnx (Modena, New York).

Results show that PHIL outperformed the best baselines by an average of **13.4%** across all graphs. When excluding OGB datasets (where scaffold/project splits artificially impose structural distribution shifts between training and test sets), PHIL's advantage rose to **19.5%** over SOTA and **20.4%** over Euclidean heuristic ($h_{euc}$).

### 5.3 Drone Flight Planning (Robotics)
In the CoppeliaSim / Ivy simulation environment, collision-free 3D drone trajectory planning was tested (discretization: $50 \times 50 \times 25$).

| Environment / Setting | SL | A* | $h_{euc}$ | BFS | SAIL | BFWS | **PHIL** | Shortest Path (Oracle) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Room simple** | 1.124 | 76.052 | 1.000 | 291.888 | 0.973 | 1.286 | **0.785** | 0.782 |
| **Room adversarial** | 2.022 | 67.215 | 1.000 | 238.768 | 0.944 | 1.583 | **0.895** | 0.853 |

Results indicate that PHIL operates near-optimally — exploring merely **0.3%** more nodes than the theoretical minimum (oracle) in the simple environment, and **4.9%** more in the obstacle-laden environment (*room adversarial*). In comparison, the traditional Euclidean heuristic $h_{euc}$ expanded 27.8% and 17.2% more nodes, respectively.

### 5.4 Runtime Analysis
During latency benchmarks, PHIL demonstrated high operational efficiency:
- Executed **57.9% faster** than BFWS.
- Executed **32.2% faster** than SAIL.
- Computational overhead relative to classical A* was only 34.7%, which is heavily compensated for by the dramatic reduction in collision checks and expanded nodes.

---

## 6. Conclusions

This paper introduced **PHIL**, a comprehensive framework for learning graph search heuristics. The method effectively unites imitation learning with a cost-to-go oracle, a recurrent memory of the explored subgraph, and graph neural networks. The principal contributions and findings are:
- Guaranteed constant time complexity $O(1)$ of the heuristic function regardless of graph size.
- Significant reduction in the number of explored nodes (averaging 58.5% on grids and up to 19.5% on real-world graphs).
- Near-optimal performance in practical robotics applications (3D drone trajectory planning).
- Seamless integration of the learned heuristic into traditional search algorithms such as A*.
