# Monte Carlo Tree Search w nawigacji po grafach: Architektura i analiza M-Walk

## Streszczenie (Abstract)

Niniejszy dokument przedstawia kompleksową analizę strukturalną i techniczną publikacji **"M-Walk: Learning to Walk over Graphs using Monte Carlo Tree Search"** autorstwa Yelong Shen, Jianshu Chen, Po-Sen Huang, Yuqing Guo oraz Jianfeng Gao (*32nd Conference on Neural Information Processing Systems / NeurIPS 2018*). 

Artykuł podejmuje fundamentalny problem nawigacji po grafie od encji źródłowej $n_S$ do nieznanej encji docelowej $n_T$ na podstawie zadanego zapytania relacyjnego $q$ – scenariusz kluczowy dla uzupełniania baz wiedzy (ang. *Knowledge Base Completion*, KBC). Aby przezwyciężyć podwójne wyzwanie w postaci **reprezentacji stanów zależnych od historii** oraz **skrajnie rzadkich nagród z odroczeniem** (*extremely sparse delayed rewards*), autorzy wprowadzają **M-Walk** – nowatorską architekturę uczenia ze wzmocnieniem (RL), łączącą rekurencyjny enkoder stanu oparty na głębokiej sieci neuronowej (RNN) z algorytmem Monte Carlo Tree Search (MCTS). Wykorzystując determinizm przejść w środowisku grafowym, M-Walk stosuje MCTS do generowania trajektorii *off-policy* o wysokich nagrodach oraz trenuje współdzielone sieci polityki i wartości Q za pomocą metody Q-learning.

---

## 1. Wprowadzenie i sformułowanie problemu (Introduction & Problem Formulation)

W zadaniu uzupełniania baz wiedzy (KBC) graf wiedzy reprezentowany jest jako $G = (N, E)$, gdzie encje stanowią zbiór $N = \{n_i\}$, a relacje skierowane zbiór $E = \{e_{ij}\}$. Dla zadanego zapytania wejściowego $q$ (np. `CITIZENSHIP` – obywatelstwo) oraz węzła źródłowego $n_S$ (np. `Obama`), celem jest wywnioskowanie nieobserwowanego węzła docelowego $n_T$ (np. `USA`). W przeciwieństwie do klasycznych algorytmów wyszukiwania ścieżek, takich jak $A^*$, które wymagają znajomości węzła docelowego do ewaluacji funkcji heurystycznych, agent poruszający się po grafie musi wyuczyć się strategii routingu na podstawie trójek treningowych $(n_S, q, n_T)$ bez bezpośredniego nadzoru nad ścieżkami pośrednimi.

Autorzy wskazują na dwie główne bariery techniczne w nawigacji po grafach:
1. **Przestrzeń stanów zależna od historii (History-Dependent State Space)**: Bieżąca informacja lokalna (np. aktualna encja `Hawaii`) jest niewystarczająca do podjęcia decyzji o kolejnym kroku (`USA`). Agent wymaga pełnego kontekstu historycznego, w tym poprzednich relacji, odwiedzonych encji oraz pierwotnego zapytania, aby podejmować poprawne kroki wnioskowania.
2. **Rzadkie nagrody z odroczeniem (Sparse Delayed Rewards)**: Informacja zwrotna (nagroda) jest przyznawana wyłącznie po dotarciu do encji docelowej (+1 za poprawny cel, 0 w przeciwnym razie). Standardowe metody gradientu polityki (np. REINFORCE) radzą sobie słabo ze względu na niską efektywność próbkowania (*sample efficiency*) oraz drastyczne trudności w eksploracji wielowymiarowych topologii grafowych we wczesnych fazach uczenia.

---

## 2. Spacer po grafie jako proces decyzyjny Markowa (MDP)

Autorzy formalizują zdecentralizowane przeszukiwanie ścieżek w grafie jako skończony proces decyzyjny Markowa (ang. *Markov Decision Process*, MDP) $(S, A, R, P)$:

* **Przestrzeń stanów ($S$)**: Stan $s_t$ w kroku czasowym $t$ jest zdefiniowany rekurencyjnie tak, aby uwzględniać pełną sekwencję dotychczasowych obserwacji:
  $$s_t = s_{t-1} \cup \{a_{t-1}, n_t, E_{n_t}, N_{n_t}\}, \quad s_0 = \{q, n_S, E_{n_S}, N_{n_S}\}$$
  gdzie $n_t$ to bieżąca encja, $E_{n_t}$ oznacza wychodzące krawędzie, $N_{n_t}$ reprezentuje węzły w 1-krokowym (1-hop) sąsiedztwie, a $a_{t-1}$ to akcja wybrana w poprzednim kroku.
* **Przestrzeń akcji ($A$)**: Zbiór dopuszczalnych akcji w kroku $t$ ma charakter dynamiczny:
  $$A_t = E_{n_t} \cup \{\text{STOP}\}$$
  Wybór krawędzi powoduje przejście agenta do odpowiedniego sąsiedniego węzła $n_{t+1}$, natomiast wybór akcji `STOP` kończy poszukiwanie ścieżki, wskazując bieżący węzeł $\hat{n}_T = n_t$ jako przewidywany cel.
* **Deterministyczne przejścia ($P$)**: Ponieważ bazowa struktura grafu wiedzy $G$ jest w pełni znana i statyczna, funkcja przejścia między stanami $p(s_{t+1} | s_t, a_t)$ jest **w pełni znana i w 100% deterministyczna**.
* **Funkcja nagrody ($R$)**: Ocena końcowa zwraca $R = +1$, jeśli $\hat{n}_T = n_T$, oraz $R = 0$ w przeciwnym razie. Kroki pośrednie nie generują żadnej nagrody ($R = 0$).

---

## 3. Architektura agenta M-Walk i algorytmy (The M-Walk Agent Architecture & Algorithms)

M-Walk łączy głęboką reprezentację neuronową z modelowym generowaniem trajektorii za pomocą MCTS oraz bezmodelowym uczeniem Q-learning.

### 3.1 Wspólna architektura neuronowa dla polityki $\pi_\theta$ oraz funkcji wartości $Q_\theta$

Zamiast utrzymywać odrębne sieci neuronowe, M-Walk współdzieli parametry modelu $\theta$ pomiędzy polityką $\pi_\theta(a_t|s_t)$ a funkcją wartości akcji $Q_\theta(s_t, a_t)$.

Wektor stanu $h_t = \text{ENC}_{\theta_e}(s_t)$ jest rozkładany na wektory składowe o wymiarze $M$:
1. $h_{S,t}$: Koduje historyczną trajektorię do kroku $t$.
2. $h_{n',t}$: Koduje kandydującą akcję przejścia do $n' \in N_{n_t}$ wraz z krawędzią $e_{n_t, n'}$.
3. $h_{A,t}$: Podsumowuje dostępne krawędzie wychodzące $E_{n_t}$ oraz sąsiadów $N_{n_t}$.

Oceny (ang. *action scores* / logity) dla akcji obliczane są za pomocą iloczynów skalarnych oraz przekształceń sieciowych:
$$u_0 = f_{\theta_\pi}(h_{S,t}, h_{A,t}), \quad u_{n'} = \langle h_{S,t}, h_{n',t} \rangle \quad \forall n' \in N_{n_t}$$

Rozkłady wartości Q oraz prawdopodobieństwa polityki współdzielą te same logity przed normalizacją:
$$Q_\theta(s_t, \cdot) = \sigma(u_0, u_{n_1'}, \dots, u_{n_k'})$$
$$\pi_\theta(\cdot | s_t) = \varphi_\tau(u_0, u_{n_1'}, \dots, u_{n_k'})$$
gdzie $\sigma(\cdot)$ oznacza aktywację sigmoidalną (element-wise), a $\varphi_\tau(\cdot)$ to funkcja Softmax z parametrem temperatury $\tau$.

### 3.2 Algorytm uczenia: MCTS i off-policy Q-Learning

Podczas uczenia M-Walk wykonuje $E$ symulacji MCTS w każdym epizodzie. Rozwijanie węzłów drzewa (ang. *expansion*) odbywa się według zmodyfikowanej reguły **PUCT (Predictor Upper Confidence Bounds for Trees)**:
$$a_t = \arg\max_a \left[ c \cdot \pi_\theta(a|s_t) \frac{\sqrt{\sum_{a'} N(s_t, a')}}{1 + N(s_t, a)} + \frac{W(s_t, a)}{N(s_t, a)} \right]$$
gdzie $N(s, a)$ oznacza liczbę odwiedzin danej krawędzi (akcji), a $W(s, a)$ to skumulowana nagroda uzyskana z tej akcji.

* **Generowanie trajektorii**: Algorytm MCTS generuje ścieżki eksploracyjne bogate w pozytywne nagrody końcowe poprzez równoważenie apriorycznych wskazań polityki $\pi_\theta$ z szacunkami wartości nagród.
* **Uczenie off-policy metodą Q-Learning**: Wygenerowane trajektorie mają charakter *off-policy* względem bieżącej polityki $\pi_\theta$. M-Walk aktualizuje wagi parametrów $\theta$ za pomocą klasycznej reguły Q-learning:
  $$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta Q_\theta(s_t, a_t) \times \left( r(s_t, a_t) + \gamma \max_{a'} Q_\theta(s_{t+1}, a') - Q_\theta(s_t, a_t) \right)$$
  Dzięki współdzieleniu parametrów, aktualizacja $Q_\theta$ bezpośrednio optymalizuje również sieć polityki $\pi_\theta$.

### 3.3 Predykcja i wnioskowanie o węźle docelowym

W fazie testowej M-Walk buduje drzewo przeszukiwań MCTS dla niewidzianych dotąd par zapytania $(n_S, q)$. Wiele wygenerowanych trajektorii może prowadzić do tego samego węzła końcowego $n \in G$. W celu agregacji predykcji, zagregowany wynik kandydata obliczany jest jako ważona średnia po wszystkich stanach liści $s_T \to n$:
$$\text{Score}(n) = \sum_{s_T \to n} \frac{N(s_T, a_T)}{N} \times Q_\theta(s_T, \text{STOP})$$
Ostatecznie przewidywany węzeł docelowy wybierany jest zachłannie: $\hat{n}_T = \arg\max_n \text{Score}(n)$.

### 3.4 Szczegóły enkodera stanu GRU-RNN

Enkoder rzutuje zmienną historii $q_t = s_{t-1} \cup \{a_{t-1}, n_t\}$ w wektor reprezentacji $h_{S,t}$:
1. **Kodowanie akcji kandydujących**: W pełni połączona sieć neuronowa (FCN) $f_{\theta_A}$ przetwarza pary krawędź-węzeł $(e_{n_t, n'}, n')$, generując wektory $h_{n',t}$.
2. **Agregacja sąsiedztwa**: Operacja max-poolingu po współrzędnych na zbiorze $\{h_{n',t}\}$ daje podsumowujący wektor o stałej długości $h_{A,t}$.
3. **Rekurencja sekwencyjna**: Komórka GRU $f_{\theta_q}$ aktualizuje wektor stanu $q_t$:
   $$q_{t+1} = f_{\theta_q}(q_t, [h_{A,t} \parallel h_{a_t,t} \parallel n_{t+1}])$$
4. **Reprezentacja historii**: Wektor $q_t$ jest rzutowany przez sieć $f_{\theta_S}$ do ostatecznego wektora trajektorii $h_{S,t}$.

---

## 4. Eksperymenty i analiza empiryczna (Experiments & Empirical Analysis)

Autorzy ewaluują M-Walk na syntetycznej łamigłówce matematycznej oraz na dwóch standardowych zbiorach testowych z dziedziny uzupełniania baz wiedzy.

### 4.1 Zadanie syntetyczne: Łamigłówka trzech naczyń (Three Glass Puzzle)

**Łamigłówka trzech naczyń (Three Glass Puzzle)** polega na odmierzeniu zadanej objętości płynu $q$ za pomocą trzech naczyń o określonych pojemnościach bez podziałki $(A, B, C)$. Dostępne akcje obejmują napełnianie, opróżnianie oraz przelewanie płynu pomiędzy naczyniami.
* **Zbiór danych**: 600 unikalnych zagadek (500 treningowych, 100 testowych). Maksymalny horyzont ścieżki $T_{\max} = 12$.
* **Wyniki**: M-Walk osiąga dokładność testową na poziomie **$(99,0 \pm 1,0)\%$**, drastycznie deklasując algorytm gradientu polityki REINFORCE, który uzyskuje zaledwie **$(49,0 \pm 2,6)\%$**.
* **Wydajność przeszukiwania**: M-Walk dociera do celu średnio w **94,9 krokach** (maksymalnie 897), podczas gdy BFS wymaga **264,7 kroków**, a DFS **192,2 kroków** – nawet przy pełnej widoczności celu.

### 4.2 Benchmarki uzupełniania baz wiedzy (Knowledge Base Completion)

Ewaluację M-Walk przeprowadzono na zbiorze **NELL-995** (10 zadań relacyjnych) oraz **WN18RR** (predykcja relacji w WordNet bez wycieku danych testowych), wraz z dodatkowymi testami na **FB15k-237**.

#### Wyniki na zbiorze NELL-995 (Średnia precyzja – MAP w %)

| Zadanie | M-Walk | PG-Walk | Q-Walk | MINERVA | DeepPath | PRA | TransE | TransR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AthletePlaysForTeam** | **84,7** | 80,8 | 82,6 | 82,7 | 72,1 | 54,7 | 62,7 | 67,3 |
| **AthletePlaysInLeague** | **97,8** | 96,0 | 96,2 | 95,2 | 92,7 | 84,1 | 77,3 | 91,2 |
| **AthleteHomeStadium** | 91,9 | 91,9 | 91,1 | **92,8** | 84,6 | 85,9 | 71,8 | 72,2 |
| **AthletePlaysSport** | 98,3 | 98,0 | 97,0 | **98,6** | 91,7 | 47,4 | 87,6 | 96,3 |
| **TeamPlaySports** | **88,4** | 87,4 | 78,5 | 87,5 | 69,6 | 79,1 | 76,1 | 81,4 |
| **OrgHeadquaterCity** | **95,0** | 94,0 | 94,0 | 94,5 | 79,0 | 81,1 | 62,0 | 65,7 |
| **WorksFor** | **84,2** | 84,0 | 82,7 | 82,7 | 69,9 | 68,1 | 67,7 | 69,2 |
| **BornLocation** | 81,2 | **82,3** | 81,4 | 78,2 | 75,5 | 66,8 | 71,2 | 81,2 |
| **PersonLeadsOrg** | **88,8** | 87,2 | 86,9 | 83,0 | 79,0 | 70,0 | 75,1 | 77,2 |
| **OrgHiredPerson** | **88,8** | 87,2 | 87,8 | 87,0 | 73,8 | 59,9 | 71,9 | 73,7 |
| **Średnia całkowita MAP** | **89,9** | 88,9 | 87,8 | 87,6 | 78,8 | 69,7 | 72,3 | 77,5 |

#### Wyniki na zbiorze WN18RR (%)

| Metryka | M-Walk | PG-Walk | Q-Walk | MINERVA | ComplEx | ConvE | DistMult | NeuralLP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HITS@1** | **41,4** | 39,3 | 38,2 | 35,1 | 38,5 | 39,6 | 38,4 | 37,2 |
| **HITS@3** | 44,5 | 41,9 | 40,8 | 44,5 | 43,9 | **44,7** | 42,4 | 43,4 |
| **MRR** | **43,7** | 41,3 | 40,1 | 40,9 | 42,2 | 43,3 | 41,3 | 43,5 |

### 4.3 Analiza i badania ablacyjne (Analytical Insights & Ablation Studies)

1. **Warianty ablacyjne**:
   * **PG-Walk**: Wykorzystuje enkoder neuronowy z M-Walk, lecz trenuje go za pomocą REINFORCE i testuje z użyciem przeszukiwania wiązkowego (ang. *beam search*, analogicznie jak MINERVA). M-Walk osiąga wynik o 1,0% MAP wyższy na NELL-995 oraz o 2,4% wyższy MRR na WN18RR w stosunku do PG-Walk, co potwierdza bezpośrednie korzyści z zastosowania MCTS.
   * **Q-Walk**: Model trenowany wyłącznie metodą Q-learning bez generowania ścieżek przez MCTS. Wydajność spada o 2,1% MAP na NELL-995 oraz o 3,6% MRR na WN18RR, co dowodzi, że generowanie trajektorii za pomocą MCTS jest kluczowe w warunkach rzadkich nagród.
2. **Łagodzenie problemu rzadkich nagród**: W trakcie treningu M-Walk utrzymuje wyraźnie wyższy odsetek prób zakończonych pozytywną nagrodą niż PG-Walk i Q-Walk we wszystkich zadaniach benchmarkowych.
3. **Wnioskowanie wielokrokowe (Multi-Step Path Reasoning)**: Przy analizie skuteczności w zależności od długości ścieżki, M-Walk przewyższa ConvE o **4,6–10,9% w HITS@1** na ścieżkach wymagających 2 lub 3 kroków wnioskowania, wykazując wyraźną przewagę w kompozycyjnym rozumowaniu na dłuższych dystansach.
4. **Analiza błędów**: M-Walk ewaluuje węzły kandydujące wygenerowane przez MCTS (zwykle 20–60 unikalnych kandydatów). Jeśli rzeczywisty cel znajduje się w zbiorze kandydatów, wynik HITS@1 sięga aż **99,5%**. Głównym źródłem błędów jest **brak węzła docelowego w zbiorze kandydatów** (*out-of-candidate-set error*, odpowiadający za 77,9–93,3% wszystkich niepowodzeń), co wskazuje, że poszerzenie pokrycia zbioru kandydatów stanowi najważniejsze wyzwanie dla dalszego rozwoju metody.
5. **Wydajność wykonania (czas inferencji)**: Wnioskowanie w fazie testowej zajmuje zaledwie **od $1,6 \times 10^{-3}$ do $6,0 \times 10^{-3}$ sekundy na próbkę**, dzięki czemu M-Walk jest znacznie szybszy niż MINERVA ($2,0 \times 10^{-2}$ s/próbkę).

---

## 5. Prace powiązane i taksonomia porównawcza (Related Work & Comparative Taxonomy)

W publikacji autorzy umiejscawiają M-Walk na styku dwóch głównych nurtów badawczych:

1. **Uczenie ze wzmocnieniem i przeszukiwanie drzewiaste (Reinforcement Learning & Tree Search)**:
   * Zainspirowany sukcesami AlphaGo / AlphaZero (Silver et al.), M-Walk integruje głębokie sieci neuronowe z MCTS.
   * *Kluczowa różnica*: AlphaZero aktualizuje sieć polityki na podstawie liczby odwiedzin węzłów w korzeniu drzewa MCTS. M-Walk aktualizuje sieć polityki pośrednio za pomocą Q-learningu wzdłuż **całych trajektorii ścieżek**, w pełni wykorzystując globalne informacje zgromadzone w drzewie przeszukiwań.
2. **Uzupełnianie baz wiedzy (Knowledge Base Completion)**:
   * **Modele zanurzeń / embeddingowe** (TransE, DistMult, ComplEx, ConvE): Rzutują encje i relacje w ciągłe przestrzenie wektorowe. Są wysoce efektywne dla połączeń 1-krokowych (1-hop), lecz podatne na kaskadowe propagowanie błędów w wielokrokowym wnioskowaniu.
   * **Modele regułowe i logiczne** (NeuralLP): Uczą się miękkich reguł logicznych z użyciem sieci pamięciowych, lecz napotykają bariery złożoności obliczeniowej na dużych grafach.
   * **Wyszukiwanie ścieżek oparte na RL** (DeepPath, MINERVA): DeepPath wymaga obecności docelowej encji w definicji stanu, co uniemożliwia jego stosowanie, gdy cel jest nieznany. MINERVA opiera się na algorytmie REINFORCE, zmagając się z niską efektywnością próbkowania przy rzadkich nagrodach. M-Walk skutecznie eliminuje oba te ograniczenia.

---

## 6. Wnioski i przyszłe kierunki badań (Conclusions and Future Research Directions)

M-Walk stanowi efektywny i solidny fundament dla nawigacji po grafach, łącząc rekurencyjne enkodery stanu (RNN), generowanie trajektorii z wykorzystaniem MCTS oraz optymalizację Q-learning ze współdzieleniem parametrów. Dzięki wykorzystaniu deterministycznego charakteru przejść w grafie, M-Walk skutecznie pokonuje wyzwania związane z rzadkimi nagrodami oraz zależną od historii reprezentacją stanów.

**Kierunki dalszych badań**:
* Ograniczenie błędów wynikających z braku celu w zbiorze kandydatów (*out-of-candidate-set*) poprzez zastosowanie szerszych heurystyk generowania kandydatów lub dynamicznych rozwinięć (*rollouts*).
* Połączenie reprezentacji wektorowych encji (np. zanurzeń z ConvE) z podejściem przeszukiwania ścieżek za pomocą RL z M-Walk w celu poprawy predykcji powiązań na gęstych grafach wielorelacyjnych, takich jak FB15k-237.
