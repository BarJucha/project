# Learning Graph Search Heuristics: Opracowanie artykułu (Pándy et al., 2022)

**Autorzy:** Michal Pándy, Weikang Qiu, Gabriele Corso, Petar Veličković, Rex Ying, Jure Leskovec, Pietro Liò  
**Konferencja:** *Proceedings of the First Learning on Graphs Conference (LoG 2022)*, PMLR vol. 198  
**Tytuł oryginału:** *Learning Graph Search Heuristics* (opis architektury PHIL — *Path Heuristic with Imitation Learning*)

---

## Streszczenie (Abstract)

Wyszukiwanie ścieżki między dwoma węzłami w grafie jest jednym z najbardziej fundamentalnych i najszerzej badanych problemów w informatyce. W licznych dziedzinach, takich jak robotyka, sztuczna inteligencja czy biologia, praktycy ręcznie projektują heurystyki wyszukiwania, aby przyspieszyć algorytmy znajdowania ścieżek. Jest to jednak proces czasochłonny i skomplikowany. W pracy zaprezentowano **PHIL** (*Path Heuristic with Imitation Learning*) — nowatorską architekturę neuronową oraz algorytm uczenia służący do automatycznego odkrywania heurystyk wyszukiwania i nawigacji na grafach z danych, wykorzystując najnowsze osiągnięcia uczenia przez imitację (imitation learning) oraz uczenia reprezentacji grafowych (graph representation learning).

Podczas treningu agregowane są zbiory danych zawierające trajektorie wyszukiwania oraz prawdziwe odległości najkrótszych ścieżek (*ground-truth shortest path distances*). Dane te służą do trenowania wyspecjalizowanej funkcji heurystycznej opartej na Grafowych Sieciach Neuronowych (GNN) za pomocą propagacji wstecznej w czasie (*truncated backpropagation through time*). Funkcja heurystyczna uczy się zanurzeń grafowych (graph embeddings) przydatnych do wnioskowania o odległościach węzłów, działa w czasie stałym $O(1)$ niezależnie od rozmiaru grafu i może być łatwo zintegrowana z algorytmami takimi jak A* lub Greedy Best-First Search w fazie testowej. Eksperymenty wykazały, że PHIL redukuje liczbę eksplorowanych węzłów średnio o **58,5%** w porównaniu ze stanem wiedzy (SOTA), daje się bezpośrednio stosować w zróżnicowanych grafach (od sieci biologicznych po drogowe) oraz umożliwia szybkie planowanie w krytycznych czasowo zastosowaniach robotycznych.

---

## 1. Wprowadzenie (Introduction)

Heurystyki wyszukiwania są kluczowe w wielu dziedzinach. W robotyce skomplikowana geometria robotów prowadzi do powolnych testów kolizji, a algorytmy planowania ruchu są ograniczone zasobami obliczeniowymi robotów, co wymaga heurystyk minimalizujących liczbę odwiedzonych węzłów. W obszarze sztucznej inteligencji heurystyki dziedzinowe przyspieszają wnioskowanie w bazach wiedzy. Stosuje się je również do ograniczenia nakładu pracy w sieciach oddziaływań białko-białko oraz w planowaniu reakcji chemicznych.

Problem wyszukiwania można sformułować jako nawigację w grafie od węzła początkowego $v_s$ do węzła docelowego $v_g$ przy minimalnej liczbie odwiedzonych węzłów. Najprostsze podejście, takie jak przeszukiwanie wszerz (BFS), rozszerza graf równomiernie od $v_s$, nie wykorzystując żadnej wiedzy priorytetowej, co skutkuje eksploracją ogromnej liczby zbędnych węzłów. Aby temu zapobiec, stosuje się funkcję heurystyczną $h$, zdefiniowaną na węzłach grafu, która pozwala przycinać mniej obiecujące kierunki. Ręczne projektowanie takich heurystyk wymaga jednak ogromnej wiedzy eksperckiej i metody prób i błędów.

Choć klasyczne podejścia bazują na ręcznie tworzonych regułach, a algorytmy uczenia maszynowego próbują automatyzować ten proces, dotychczasowe metody uczenia się heurystyk napotykają trzy główne wyzwania:
1. **Rozkład danych nie jest i.i.d.:** Nowo napotkane węzły zależą od wartości heurystyki w poprzednich krokach, co sprawia, że tradycyjne uczenie nadzorowane (supervised learning) zawodzi.
2. **Wymóg stałej złożoności czasowej $O(1)$:** Obliczanie wartości heurystyki musi być niezwykle szybkie. Jeśli funkcja heurystyczna skaluje się z rozmiarem grafu, rośnie asymptotyczna złożoność całego wyszukiwania.
3. **Problem skalowalności uczenia ze wzmocnieniem (RL):** W miarę wzrostu rozmiaru środowiska (grafu) metody uczenia ze wzmocnieniem wykazują słabą wydajność próbkowania.

Metoda PHIL rozwiązuje te problemy poprzez połączenie uczenia przez imitację z wyrocznią odległości (*cost-to-go oracle*) oraz rekurencyjnego modułu pamięci eksplorowanego grafu (*explored graph memory*), zapewniając złożoność $O(1)$ względem rozmiaru grafu.

---

## 2. Wstęp teoretyczny (Preliminaries)

### Sformułowanie wyszukiwania w grafie
Dany jest nieskierowany, nieważony i spójny graf $G = (V, E)$, gdzie każdy węzeł $i \in V$ posiada wektor cech $x_i \in \mathbb{R}^{D_v}$, a każda krawędź $(i,j) \in E$ posiada cechy $e_{ij} \in \mathbb{R}^{D_e}$. Zadaniem jest nawigacja od węzła początkowego $v_s \in V$ do węzła docelowego $v_g \in V$.

W dowolnym momencie wyszukiwania zbiór węzłów grafu można podzielić na trzy rozłączne podzbiory:
$$\text{V} = \text{CLOSE} \cup \text{OPEN} \cup \text{REST}$$
- **CLOSE:** Węzły już zbadane (wyeksplorowane).
- **OPEN:** Węzły kandydujące do eksploracji (połączone z węzłami w CLOSE, ale jeszcze do nich nienależące — tzw. front/fringe).
- **REST:** Pozostała, nieznana część grafu.

Rozszerzenie (*expansion*) polega na przeniesieniu węzła z OPEN do CLOSE oraz dodaniu jego sąsiadów z REST do OPEN. Zbiór nowo dodanych węzłów frontu w danym kroku oznacza się jako $V_{new}$. Na początku wyszukiwania $\text{CLOSE} = \{v_s\}$, a proces trwa do momentu napotkania $v_g \in \text{CLOSE}$.

### Zachłanne wyszukiwanie po najlepszych (Greedy Best-First Search)
W tym reżimie w każdym kroku rozszerza się węzeł $v \in \text{OPEN}$, który minimalizuje skrojoną na miarę funkcję heurystyczną $h(v, v_g)$. Celem PHIL jest wyuczenie funkcji $h$, która precyzyjnie przewiduje długość najkrótszej ścieżki do celu, minimalizując tym samym rozmiar zbioru $|\text{CLOSE}|$.

### Imitacja idealnej heurystyki
Problem uczenia heurystyk można opisać w ramach częściowo obserwowalnych procesów decyzyjnych Markowa (POMDP), gdzie stan stanowi krotka $s = (\text{CLOSE}, \text{OPEN}, \text{REST})$, akcja $a$ to przeniesienie węzła z OPEN do CLOSE, a obserwacje $o$ stanowią cechy nowo włączonych węzłów. Stosuje się wyrocznię $h^*$, która posiada pełną informację o stanie $s$ i podaje rzeczywiste odległości najkrótszych ścieżek $h^*(s, v, v_g)$ dla $v \in \text{OPEN}$.

---

## 3. Prace powiązane (Related Work)

W literaturze problem uczenia nawigacji i wyszukiwania analizowany jest w kilku głównych nurtach:

| Kategoria metod | Opis i reprezentatywne podejścia | Ograniczenia i stosunek do PHIL |
| :--- | :--- | :--- |
| **Projektowanie heurystyk ogólnych** | Metody takie jak BFWS (*Best-First Width Search*) lub heurystyki oparte na punktach orientacyjnych (*landmarks*). | Dobrze zapobiegają lokalnym minimom, ale nie potrafią w pełni wykorzystać specyficznej struktury danych grafowych. |
| **Uczenie heurystyk wyszukiwania** | Metody uczenia się ocen węzłów (np. SAIL, Retro*, Neural A*). | Metody takie jak SAIL wymagają ręcznie projektowanych cech (feature engineering) i wykazują liniową złożoność obliczeniową względem liczby węzłów, przez co nie gwarantują czasu $O(1)$. |
| **Uczenie ogólnego wyszukiwania** | Uczenie się polityk nawigacji z użyciem MCTS (np. AlphaGo/MCTSNet) lub planowania zintegrowanego (Value Iteration Networks). | Traktują problem jako ogólne MDP/POMDP, co wiąże się ze zbyt dużą złożonością obliczeniową, aby konkurować z szybkim przeszukiwaniem zachłannym. |
| **Uczenie przez imitację (IL)** | Algorytmy z wyrocznią *cost-to-go*, takie jak AggreVaTe, AggreVaTeD, LOLS, DART, THOR. | PHIL rozszerza algorytm SAIL/AggreVaTe o uczenie rekurencyjnej funkcji wartości z pamięcią struktury grafu przy zachowaniu stałego czasu $O(1)$. |

---

## 4. Architektura i Metoda: PHIL

### Funkcja celu i reprezentacja stanu
Głównym celem jest wyszkolenie parametryzowanej funkcji heurystycznej $h_\theta(\psi, v, v_g)$, która przewiduje prawdziwe odległości $h^*$, minimalizując oczekiwany błąd kwadratowy:

$$L(\theta) = \mathbb{E}_{\substack{G \sim P_G, (v_s, v_g) \sim P_{vsg} \\ t \sim U(0, \dots, T), s, \psi \sim P_s}} \left[ \frac{1}{|\text{OPEN}|} \sum_{v \in \text{OPEN}} \left( h^*(s, v, v_g) - h_\theta(\psi, v, v_g) \right)^2 \right]$$

Aby zredukować złożoność obliczeniową, $h_\theta$ przyjmuje zakodowany wektor pamięci $z_t \in \mathbb{R}^d$, reprezentujący historię przeszukiwania $\psi$ w kroku $t$. Dodatkowo, aby uniknąć konieczności ciągłego przeliczania wartości heurystyki dla wszystkich węzłów w OPEN (co wiązałoby się ze złożonością $O(|\text{OPEN}|)$), **PHIL oblicza wartości heurystyczne wyłącznie dla nowo dodanych węzłów frontu $V_{new}$**.

### Algorytm uczenia przez imitację
Trening przebiega według procedury przedstawionej w poniższym algorytmie:

1. **Roll-in:** Dla próbkowanego grafu $G$ oraz pary $(v_s, v_g)$, aktualna polityka zachłanna $\pi_\theta$ wykonuje $t$ kroków eksploracji, generując stan początkowy $s_t = (\text{CLOSE}_0, \text{OPEN}_0, \text{REST}_0)$ oraz początkowy stan pamięci $z_t$.
2. **Roll-out:** Następnie przez $t_\tau$ kroków realizowana jest trajektoria przy użyciu polityki mieszanej $\pi_{mix} = (1 - \beta) \cdot \pi_\theta + \beta \cdot \pi^*$, gdzie $\pi^*$ jest polityką wyroczni. Podczas roll-outu zbierane są zbiory nowych węzłów frontu $V_{new}$ oraz ich prawdziwe odległości $h^*$.
3. **Optymalizacja (TBTT):** Na zgromadzonych trajektoriach stosuje się obciętą propagację wsteczną w czasie (*Truncated Backpropagation Through Time*), aktualizując parametry $\theta$.

### Architektura rekurencyjnej sieci grafowej (GNN)
W każdym kroku funkcja heurystyczna $h_\theta$ przetwarza nowe węzły frontu $V_{new}$, węzeł docelowy $v_g$ oraz dotychczasowy stan pamięci $z_t$. Procedura obliczeniowa przebiega w następujących etapach:

1. **Rzutowanie cech:** Cechy węzła $x_i$ oraz cechy sąsiadów $x_j$ łączone są z cechami celu $x_g$ oraz ich odległością euklidesową $D_{\text{EUC}}$ i kosinusową $D_{\text{COS}}$:
   $$x_i \leftarrow f(x_i, x_g, D_{\text{EUC}}(x_i, x_g), D_{\text{COS}}(x_i, x_g))$$
2. **Konwolucja grafowa (GNN):** Dla każdego węzła $i \in V_{new}$ próbkujemy zbiór sąsiadów z jego 1-hopowego otoczenia $N_i$ (gdzie $|N_i| \le n$). Wykonujemy jednawarstwową agregację wiadomości:
   $$g_i \leftarrow \phi\left(x_i, \bigoplus_{j \in N_i} \gamma(x_i, x_j, e_{ij})\right)$$
3. **Aktualizacja pamięci (GRU):** Wektor reprezentacji $g_i$ poddawany jest transformacji przez komórkę GRU wraz z dotychczasowym stanem pamięci:
   $$g'_i, z_{i, t+1} \leftarrow \text{GRU}(g_i, z_{i, t})$$
4. **Agregacja niezmiennicza względem permutacji:** Nowy stan pamięci $z_{t+1}$ oblicza się jako średnią próbkowa ze stanów $z_{i, t+1}$ dla wszystkich $i \in V_{new}$. Zapewnia to niezmienniczość pamięci względem kolejności węzłów w paczce i umożliwia równoległe przetwarzanie na GPU.
5. **Predykcja odległości:** Odległość $\hat{h}_i$ przewidywana jest przez wielowarstwowy perceptron: $\hat{h}_i \leftarrow \text{MLP}(g'_i, x_g)$.

### Złożoność czasowa $O(1)$
Ponieważ dla każdego węzła $i \in V_{new}$ wielkość próbowanego otoczenia jest ograniczona ($|N_i| \le n$), całkowita liczba operacji przypadająca na pojedynczy węzeł wynosi co najwyżej $n c_1 + (n+1)c_2$, gdzie $c_1$ reprezentuje koszt oceny węzła (np. test kolizji), a $c_2$ koszt operacji modelowych. Oznacza to, że czas wykonania funkcji heurystycznej jest **niezależny od całkowitego rozmiaru grafu** $O(1)$.

---

## 5. Eksperymenty i Wyniki (Experiments)

Eksperymenty przeprowadzono w trzech głównych środowiskach testowych: siatkach dwuwymiarowych, zróżnicowanych grafach świata rzeczywistego oraz w symulacji planowania lotu drona.

### 5.1 Wyszukiwanie na siatkach (200 $\times$ 200 8-connected grids)
Testy wykonano na 8 benchmarkowych zbiorach danych zawierających trudne konfiguracje przeszkód i pułapek (np. *Single Bugtrap*, *Forest*, *Mazes*).

Poniższa tabela przedstawia stosunek liczby wyeksplorowanych węzłów przez PHIL i inne metody w porównaniu do wiodącej metody baseline (**SAIL**):

| Zbiór danych | SAIL | SL (Supervised) | CEM | QL | $h_{euc}$ | $h_{man}$ | A* | MHA* | BFWS | Neural A* | **PHIL (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Alternating gaps** | 0.039 | 0.432 | 0.042 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.340 | 0.546 | **0.024** |
| **Single Bugtrap** | 0.158 | 0.214 | 0.057 | 1.000 | 0.184 | 0.192 | 1.000 | 0.286 | 0.099 | 0.394 | **0.077** |
| **Shifting gaps** | 0.104 | 0.464 | 1.000 | 1.000 | 0.506 | 0.589 | 1.000 | 0.804 | 0.206 | 0.563 | **0.027** |
| **Forest** | 0.036 | 0.043 | 0.048 | 0.121 | 0.041 | 0.043 | 1.000 | 0.075 | 0.039 | 0.399 | **0.027** |
| **Bugtrap+Forest** | 0.147 | 0.384 | 0.182 | 1.000 | 0.410 | 0.337 | 1.000 | 3.177 | 0.149 | 0.651 | **0.135** |
| **Gaps+Forest** | 0.221 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.401 | 0.580 | **0.039** |
| **Mazes** | 0.103 | 0.238 | 0.479 | 0.399 | 0.185 | 0.171 | 1.000 | 0.279 | 0.095 | 1.000 | **0.069** |
| **Multiple Bugtraps** | 0.479 | 0.480 | 1.000 | 0.835 | 0.648 | 0.617 | 1.000 | 0.876 | 0.169 | 0.331 | **0.136** |

**Wnioski z testów siatkowych:**
- PHIL przewyższył SAIL we wszystkich zbiorach danych, redukując liczbę eksplorowanych węzłów średnio o **58,5%** (z maksymalną redukcją o **82,3%** w zbiorze *Gaps+Forest*).
- PHIL wykazuje zdolność do ucieczki z lokalnych minimów dzięki pamięci eksplorowanego grafu.
- Szybkość zbieżności PHIL była 5-krotnie wyższa niż SAIL (zbieżność następuje już po około 36 iteracjach).

### 5.2 Wyszukiwanie w grafach rzeczywistych
Weryfikacja objęła 4 grupy struktur grafowych:
1. **Sieci cytowań:** Cora, PubMed, CiteSeer, Coauthor (cs, physics).
2. **Sieci biologiczne:** OGBG-Molhiv, PPI, Proteins (Full), Enzymes.
3. **Drzewa AST:** OGBG-Code2.
4. **Sieci drogowe:** OSMnx (Modena, New York).

Wyniki pokazały, że PHIL przewyższył najlepsze baseline'y o średnio **13,4%** we wszystkich grafach. Po wykluczeniu zbiorów OGB (gdzie podziały *scaffold/project split* sztucznie wymuszają niezgodność strukturalną między zbiorami treningowymi i testowymi), przewaga PHIL wzrosła do **19,5%** względem SOTA oraz o **20,4%** względem heurystyki odległości euklidesowej ($h_{euc}$).

### 5.3 Planowanie lotu drona (Robotyka)
W środowisku symulacyjnym CoppeliaSim / Ivy przetestowano planowanie bezkolizyjnych tras drona w przestrzeni 3D (dyskretyzacja $50 \times 50 \times 25$).

| Zbiór / Środowisko | SL | A* | $h_{euc}$ | BFS | SAIL | BFWS | **PHIL** | Najkrótsza ścieżka (Wyrocznia) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Room simple** | 1.124 | 76.052 | 1.000 | 291.888 | 0.973 | 1.286 | **0.785** | 0.782 |
| **Room adversarial** | 2.022 | 67.215 | 1.000 | 238.768 | 0.944 | 1.583 | **0.895** | 0.853 |

Wyniki wskazują, że PHIL działa niemal optymalnie — w prostym środowisku eksploruje zaledwie o **0,3%** więcej węzłów niż teoretyczne minimum (wyrocznia), a w środowisku z przeszkodami (*room adversarial*) o **4,9%** więcej. Dla porównania, tradycyjna heurystyka $h_{euc}$ ekspandowała odpowiednio o 27,8% i 17,2% więcej węzłów.

### 5.4 Czas wykonania (Runtime Analysis)
Podczas testów czasowych PHIL wykazał bardzo wysoką wydajność:
- Działał o **57,9% szybciej** niż BFWS.
- Działał o **32,2% szybciej** niż SAIL.
- Narzut obliczeniowy w stosunku do klasycznego algorytmu A* wynosił zaledwie 34,7%, co jest z nawiązką rekompensowane ogromną redukcją liczby testowanych kolizji / ekspandowanych węzłów.

---

## 6. Podsumowanie (Conclusion)

W artykule przedstawiono kompleksowy system uczenia heurystyk nawigacji grafowej **PHIL**. Metoda skutecznie łączy uczenie przez imitację z wyrocznią odległości, rekurencyjną pamięć wyeksplorowanego podgrafu oraz grafowe sieci neuronowe. Najważniejsze osiągnięcia pracy to:
- Zapewnienie stałej złożoności czasowej $O(1)$ funkcji heurystycznej niezależnie od rozmiaru sieci.
- Znaczna redukcja liczby wyeksplorowanych węzłów (średnio o 58,5% na siatkach oraz do 19,5% w realnych grafach).
- Bliska optymalnej skuteczność w praktycznych zastosowaniach robotycznych (planowanie lotu dronów w 3D).
- Możliwość łatwej integracji wyuczonej heurystyki z tradycyjnymi algorytmami przeszukiwania, takimi jak A*.
