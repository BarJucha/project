# Monte Carlo Tree Search for Graph Walking: M-Walk Architecture and Analysis

## Abstract

This document provides a comprehensive structural and technical breakdown of the publication **"M-Walk: Learning to Walk over Graphs using Monte Carlo Tree Search"** authored by Yelong Shen, Jianshu Chen, Po-Sen Huang, Yuqing Guo, and Jianfeng Gao (*32nd Conference on Neural Information Processing Systems / NeurIPS 2018*). 

The paper addresses the fundamental problem of navigating a graph from a source entity $n_S$ to an unknown target entity $n_T$ under a given relation query $q$, a scenario central to Knowledge Base Completion (KBC). To overcome the twin challenges of **history-dependent state representations** and **extremely sparse delayed rewards**, the authors introduce **M-Walk**, a novel reinforcement learning (RL) framework that integrates a Deep Recurrent Neural Network (RNN) state encoder with Monte Carlo Tree Search (MCTS). By exploiting deterministic environment transitions, M-Walk utilizes MCTS to generate high-reward off-policy trajectories and trains shared policy and Q-networks via Q-learning.

---

## 1. Introduction & Problem Formulation

In Knowledge Base Completion (KBC), a knowledge graph is represented as $G = (N, E)$ with entities $N = \{n_i\}$ and directed relations $E = \{e_{ij}\}$. Given an input query $q$ (e.g., `CITIZENSHIP`) and a source node $n_S$ (e.g., `Obama`), the goal is to infer an unobserved target node $n_T$ (e.g., `USA`). Unlike classical pathfinding algorithms such as $A^*$, which require a known target node to evaluate heuristic functions, a graph-walking agent must learn its routing policy from training triplets $(n_S, q, n_T)$ without intermediate path supervision.

The authors highlight two major technical hurdles in graph navigation:
1. **History-Dependent State Space**: Current local information (e.g., current entity `Hawaii`) is insufficient to decide the next step (`USA`). The agent requires full historical context, including previous relations, visited entities, and the initial query, to make valid inference steps.
2. **Sparse Delayed Rewards**: Evaluative feedback is only provided upon reaching a destination entity (+1 for correct target, 0 otherwise). Standard policy gradient methods (e.g., REINFORCE) struggle due to low sample efficiency and catastrophic failure in exploring high-dimensional graph topologies during early training epochs.

---

## 2. Graph Walking as a Markov Decision Process (MDP)

The authors formalize decentralized graph path search as a finite Markov Decision Process $(S, A, R, P)$:

* **State Space ($S$)**: The state $s_t$ at time step $t$ is recursively defined to incorporate the complete sequence of observations:
  $$s_t = s_{t-1} \cup \{a_{t-1}, n_t, E_{n_t}, N_{n_t}\}, \quad s_0 = \{q, n_S, E_{n_S}, N_{n_S}\}$$
  where $n_t$ is the current entity, $E_{n_t}$ denotes connected edges, $N_{n_t}$ represents 1-hop neighboring nodes, and $a_{t-1}$ is the previously selected action.
* **Action Space ($A$)**: The set of feasible actions at step $t$ is dynamic:
  $$A_t = E_{n_t} \cup \{\text{STOP}\}$$
  Selecting an edge transitions the agent to the corresponding adjacent node $n_{t+1}$, while choosing `STOP` terminates the path search, designating current node $\hat{n}_T = n_t$ as the predicted target.
* **Deterministic Transitions ($P$)**: Because the underlying knowledge graph structure $G$ is fully known and static, the state transition $p(s_{t+1} | s_t, a_t)$ is **known and 100% deterministic**.
* **Reward Function ($R$)**: Terminal evaluation yields $R = +1$ if $\hat{n}_T = n_T$, and $R = 0$ otherwise. Intermediate steps carry zero reward.

---

## 3. The M-Walk Agent Architecture & Algorithms

M-Walk unifies a deep neural representation network with model-based MCTS trajectory generation and model-free Q-learning.

### 3.1 Joint Neural Architecture for Policy $\pi_\theta$ and Q-Function $Q_\theta$

Rather than maintaining isolated networks, M-Walk shares model parameters $\theta$ between the policy $\pi_\theta(a_t|s_t)$ and the action-value function $Q_\theta(s_t, a_t)$.

State vector $h_t = \text{ENC}_{\theta_e}(s_t)$ is factored into component vectors of dimension $M$:
1. $h_{S,t}$: Encodes historical trajectory up to step $t$.
2. $h_{n',t}$: Encodes candidate action $n' \in N_{n_t}$ along with edge $e_{n_t, n'}$.
3. $h_{A,t}$: Summarizes available outgoing edges $E_{n_t}$ and neighbors $N_{n_t}$.

Action scores are computed via inner products and network mappings:
$$u_0 = f_{\theta_\pi}(h_{S,t}, h_{A,t}), \quad u_{n'} = \langle h_{S,t}, h_{n',t} \rangle \quad \forall n' \in N_{n_t}$$

The Q-value and Policy distributions share these pre-softmax logits:
$$Q_\theta(s_t, \cdot) = \sigma(u_0, u_{n_1'}, \dots, u_{n_k'})$$
$$\pi_\theta(\cdot | s_t) = \varphi_\tau(u_0, u_{n_1'}, \dots, u_{n_k'})$$
where $\sigma(\cdot)$ is element-wise Sigmoidal activation and $\varphi_\tau(\cdot)$ is Softmax with temperature $\tau$.

### 3.2 Training Algorithm: MCTS & Off-Policy Q-Learning

During training, M-Walk runs $E$ simulations of MCTS per episode. Node expansion follows a modified **PUCT (Predictor Upper Confidence Bounds for Trees)** formula:
$$a_t = \arg\max_a \left[ c \cdot \pi_\theta(a|s_t) \frac{\sqrt{\sum_{a'} N(s_t, a')}}{1 + N(s_t, a)} + \frac{W(s_t, a)}{N(s_t, a)} \right]$$
where $N(s, a)$ is the edge visit count and $W(s, a)$ is total accumulated action reward.

* **Trajectory Generation**: MCTS generates exploratory paths rich in positive terminal rewards by balancing policy priors $\pi_\theta$ with value estimates.
* **Off-Policy Q-Learning**: Generated trajectories are off-policy relative to $\pi_\theta$. M-Walk updates parameters $\theta$ via standard Q-learning:
  $$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta Q_\theta(s_t, a_t) \times \left( r(s_t, a_t) + \gamma \max_{a'} Q_\theta(s_{t+1}, a') - Q_\theta(s_t, a_t) \right)$$
  Due to parameter sharing, updating $Q_\theta$ directly improves the policy network $\pi_\theta$.

### 3.3 Prediction & Target Node Inference

At test time, M-Walk constructs an MCTS search tree for unseen query pairs $(n_S, q)$. Multiple path trajectories may arrive at the same terminal entity $n \in G$. To aggregate predictions, candidate node scores are computed as a weighted average over all leaf states $s_T \to n$:
$$\text{Score}(n) = \sum_{s_T \to n} \frac{N(s_T, a_T)}{N} \times Q_\theta(s_T, \text{STOP})$$
The predicted target node is chosen as $\hat{n}_T = \arg\max_n \text{Score}(n)$.

### 3.4 GRU-RNN State Encoder Details

The encoder maps history variable $q_t = s_{t-1} \cup \{a_{t-1}, n_t\}$ into vector $h_{S,t}$:
1. **Candidate Action Encoding**: Fully-connected network $f_{\theta_A}$ processes edge-node pairs $(e_{n_t, n'}, n')$ to produce $h_{n',t}$.
2. **Neighborhood Aggregation**: Coordinate-wise max-pooling over $\{h_{n',t}\}$ yields fixed-length summary $h_{A,t}$.
3. **Sequential Recurrence**: GRU network $f_{\theta_q}$ updates state vector $q_t$:
   $$q_{t+1} = f_{\theta_q}(q_t, [h_{A,t} \parallel h_{a_t,t} \parallel n_{t+1}])$$
4. **History Representation**: Vector $q_t$ is projected via FCN $f_{\theta_S}$ into final trajectory vector $h_{S,t}$.

---

## 4. Experiments & Empirical Analysis

The authors evaluate M-Walk across a synthetic mathematical puzzle and two standard Knowledge Base Completion benchmarks.

### 4.1 Synthetic Task: Three Glass Puzzle

The **Three Glass Puzzle** involves measuring a target volume $q$ using three ungraduated liquid containers $(A, B, C)$. Actions include filling, emptying, or pouring between containers.
* **Dataset**: 600 unique puzzles (500 train, 100 test). Maximum path horizon $T_{\max} = 12$.
* **Results**: M-Walk achieves **$(99.0 \pm 1.0)\%$ test accuracy**, drastically outperforming REINFORCE policy gradients, which achieve only **$(49.0 \pm 2.6)\%$**.
* **Traversal Efficiency**: M-Walk reaches targets in an average of **94.9 steps** (max 897), compared to **264.7 steps** for BFS and **192.2 steps** for DFS under full target visibility.

### 4.2 Knowledge Base Completion Benchmarks

M-Walk was evaluated on **NELL-995** (10 relation tasks) and **WN18RR** (WordNet relation prediction without test leakage), alongside additional tests on **FB15k-237**.

#### NELL-995 Performance (Mean Average Precision - MAP %)

| Task | M-Walk | PG-Walk | Q-Walk | MINERVA | DeepPath | PRA | TransE | TransR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AthletePlaysForTeam** | **84.7** | 80.8 | 82.6 | 82.7 | 72.1 | 54.7 | 62.7 | 67.3 |
| **AthletePlaysInLeague** | **97.8** | 96.0 | 96.2 | 95.2 | 92.7 | 84.1 | 77.3 | 91.2 |
| **AthleteHomeStadium** | 91.9 | 91.9 | 91.1 | **92.8** | 84.6 | 85.9 | 71.8 | 72.2 |
| **AthletePlaysSport** | 98.3 | 98.0 | 97.0 | **98.6** | 91.7 | 47.4 | 87.6 | 96.3 |
| **TeamPlaySports** | **88.4** | 87.4 | 78.5 | 87.5 | 69.6 | 79.1 | 76.1 | 81.4 |
| **OrgHeadquaterCity** | **95.0** | 94.0 | 94.0 | 94.5 | 79.0 | 81.1 | 62.0 | 65.7 |
| **WorksFor** | **84.2** | 84.0 | 82.7 | 82.7 | 69.9 | 68.1 | 67.7 | 69.2 |
| **BornLocation** | 81.2 | **82.3** | 81.4 | 78.2 | 75.5 | 66.8 | 71.2 | 81.2 |
| **PersonLeadsOrg** | **88.8** | 87.2 | 86.9 | 83.0 | 79.0 | 70.0 | 75.1 | 77.2 |
| **OrgHiredPerson** | **88.8** | 87.2 | 87.8 | 87.0 | 73.8 | 59.9 | 71.9 | 73.7 |
| **Overall MAP** | **89.9** | 88.9 | 87.8 | 87.6 | 78.8 | 69.7 | 72.3 | 77.5 |

#### WN18RR Dataset Results (%)

| Metric | M-Walk | PG-Walk | Q-Walk | MINERVA | ComplEx | ConvE | DistMult | NeuralLP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HITS@1** | **41.4** | 39.3 | 38.2 | 35.1 | 38.5 | 39.6 | 38.4 | 37.2 |
| **HITS@3** | 44.5 | 41.9 | 40.8 | 44.5 | 43.9 | **44.7** | 42.4 | 43.4 |
| **MRR** | **43.7** | 41.3 | 40.1 | 40.9 | 42.2 | 43.3 | 41.3 | 43.5 |

### 4.3 Analytical Insights & Ablation Studies

1. **Ablation Models**:
   * **PG-Walk**: Uses M-Walk's neural encoder but trains with REINFORCE and tests via beam search (similar to MINERVA). M-Walk outperforms PG-Walk by 1.0% MAP on NELL-995 and 2.4% MRR on WN18RR, confirming MCTS benefits.
   * **Q-Walk**: Trains via pure Q-learning without MCTS trajectory rollouts. Performance drops by 2.1% MAP on NELL-995 and 3.6% MRR on WN18RR, proving MCTS trajectory generation is essential for overcoming sparse rewards.
2. **Mitigating Sparse Rewards**: M-Walk maintains a consistently higher positive reward rate during training than PG-Walk and Q-Walk across all benchmark tasks.
3. **Multi-Step Path Reasoning**: When evaluating performance across path lengths, M-Walk outperforms ConvE by **4.6–10.9% in HITS@1** on paths requiring 2 or 3 reasoning steps, demonstrating superior compositional reasoning over longer paths.
4. **Error Analysis**: M-Walk evaluates candidate nodes generated by MCTS (typically 20–60 unique candidates). If ground-truth targets are present in candidate sets, HITS@1 reaches **99.5%**. The primary error source is **out-of-candidate-set error** (77.9–93.3% of total failures), indicating that expanding candidate set coverage is the primary bottleneck for future gains.
5. **Execution Efficiency**: Test-time inference takes only **$1.6 \times 10^{-3}$ to $6.0 \times 10^{-3}$ seconds per sample**, making M-Walk substantially faster than MINERVA ($2.0 \times 10^{-2}$ s/sample).

---

## 5. Related Work & Comparative Taxonomy

The paper contextualizes M-Walk within two primary research fields:

1. **Reinforcement Learning & Tree Search**:
   * Inspired by AlphaGo / AlphaZero (Silver et al.), M-Walk integrates neural networks with MCTS.
   * *Key Difference*: AlphaZero updates policy networks using MCTS root node visit counts. M-Walk updates policy networks indirectly via Q-learning across **entire trajectory paths**, exploiting global search tree information.
2. **Knowledge Base Completion**:
   * **Embedding Models** (TransE, DistMult, ComplEx, ConvE): Map entities/relations into continuous vector spaces. Effective for 1-hop links but prone to cascading errors on multi-step reasoning.
   * **Logic & Rule-Based Models** (NeuralLP): Learn soft logical rules via memory networks but struggle with computational complexity over entire graphs.
   * **RL Pathfinding** (DeepPath, MINERVA): DeepPath requires known target entities in its state, preventing use when targets are unknown. MINERVA relies on REINFORCE, suffering from sample inefficiency under sparse rewards. M-Walk resolves both issues.

---

## 6. Conclusions and Future Research Directions

M-Walk establishes a powerful framework for graph-walking by combining RNN state encoders, MCTS trajectory generation, and parameter-shared Q-learning. By leveraging deterministic graph transitions, M-Walk effectively overcomes sparse rewards and history-dependent state representation challenges.

**Future Directions**:
* Mitigating out-of-candidate-set errors by incorporating broader candidate generation heuristics or dynamic rollouts.
* Combining embedding-based entity representations (e.g., ConvE embeddings) with M-Walk's RL pathfinding to enhance link prediction on dense multi-relational graphs like FB15k-237.
