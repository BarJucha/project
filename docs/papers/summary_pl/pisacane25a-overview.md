# Reinforcement Learning Discovers Efficient Decentralized Graph Path Search Strategies: Analiza i Synteza Publikacji

## 1. Wstęp (Introduction)

Poszukiwanie optymalnych ścieżek w grafach jest jednym z fundamentalnych problemów informatyki, mającym kluczowe znaczenie w takich dziedzinach jak bazodanowe systemy wiedzy, robotyka autonomiczna oraz analiza sieci społecznościowych. Klasyczne sformułowania tego problemu opierają się na perspektywie scentralizowanej, zakładając pełną widoczność globalnej struktury grafu. W wielu praktycznych zastosowaniach – takich jak wielkoskalowe sieci P2P, dynamiczne sieci bezprzewodowe czy wrażliwe na prywatność sieci społecznościowe – założenie o dostępności globalnej mapy połączeń jest jednak niemożliwe do spełnienia ze względu na ograniczenia pamięciowe, narzut komunikacyjny oraz wymogi ochrony prywatności.

Niniejsza publikacja autorstwa Alexeia Pisacane, Victora-Alexandru Darvariu oraz Mirco Musolesiego (zaprezentowana na konferencji *Learning on Graphs Conference - LoG 2024*) podejmuje wyzwanie w pełni zdecentralizowanego poszukiwania ścieżki przy ograniczonej, lokalnej obserwowalności otoczenia. Inspiracją naukową dla przedstawionego podejścia jest słynny eksperyment „małego świata” przeprowadzony przez Stanleya Milgrama w 1969 roku. Milgram wykazał, że w ludzkich sieciach społecznościowych krótkie ścieżki pomiędzy odległymi jednostkami mogą być skutecznie odkrywane wyłącznie na podstawie lokalnych informacji o sąsiadach oraz wysuwanych na podstawie atrybutów węzłów ocen podobieństwa.

Zdecentralizowane nawigowanie w sieciach jest możliwe dzięki dwóm kluczowym właściwościom strukturalnym:
* **Homofilii (*Homophily*):** tendencji węzłów o podobnych atrybutach do tworzenia bezpośrednich połączeń.
* **Heterogeniczności strukturalnej (*Structural Heterogeneity*):** organizacji sieci w gęsto połączone lokalne społeczności, skomunikowane ze sobą za pośrednictwem nielicznych słabych powiązań (*weak ties*) lub węzłów centralnych (*hubs*).

Autorzy artykułu proponują wieloagentowe sformułowanie problemu w ramach zdecentralizowanego, częściowo obserwowalnego procesu decyzyjnego Markowa (**Dec-POMDP**). Wprowadzają autorską metodę **GARDEN** (*Graph Attention-guided Routing for DEcentralised Networks*), która wykorzystuje paradygmat **CTDE** (*Centralized Training with Decentralized Execution*) oraz sieci uwagi na grafach (**GAT**) do automatycznego wyłaniania ciągłych reprezentacji węzłów służących do skutecznej nawigacji.

---

## 2. Prace powiązane (Related Work)

Przegląd literatury przedstawiony w publikacji został podzielony na dwa główne obszary: naukę o sieciach (*Network Science*) oraz uczenie ze wzmocnieniem w trasowaniu i poszukiwaniu na grafach (*RL for Graph Routing and Search*).

### 2.1 Nauka o sieciach (Network Science)
Klasyczne algorytmy wyszukiwania ścieżek wymagają utrzymywania globalnej wiedzy o topologii grafu. W dużych lub dynamicznych sieciach prowadzi to do wąskich gardeł skalowalności i zagrożeń prywatności. Eksperyment Milgrama oraz późniejsze badania na dużą skalę w sieciach cyfrowych potwierdziły istnienie tzw. efektu małego świata – średnie długości ścieżek są niewielkie, a efektywne strategie trasowania mogą bazować na lokalnych atrybutach.

Jednak wyszukiwanie oparte wyłącznie na homofilii lub stopniu węzłów bywa nieskuteczne. W sieciach podzielonych na gęste klastry, agenci bez odpowiedniej reprezentacji otoczenia mogą uwięznąć wewnątrz lokalnej społeczności. Kluczowe dla sprawnej nawigacji jest odnajdywanie tzw. słabych powiązań (*weak ties*) łączących odrębne społeczności, co wymaga uwzględnienia struktury sąsiedztwa. Badania nad tzw. **ukrytą przestrzenią metryczną** (*hidden metric space*) sugerują, że jeśli atrybuty węzłów odzwierciedlają ich położenie w ukrytej przestrzeni cech, efektywna zdecentralizowana nawigacja polega na poruszaniu się w stronę węzłów minimalizujących odległość metryczną. Najbliższe podejściu autorów prace Simseka i Jensena wymagały znajomości statystyk globalnych całej sieci; metoda GARDEN eliminuje to ograniczenie, nie wymagając żadnych informacji globalnych.

### 2.2 Uczenie ze wzmocnieniem w trasowaniu i poszukiwaniu grafowym
Uczenie ze wzmocnieniem zdobywa popularność jako narzędzie do odkrywania algorytmów przewyższających tradycyjne heurystyki. W obszarze trasowania sieciowego wcześniejsze prace (np. Q-routing) cierpiały na problemy ze skalowalnością tabularną, co rozwiązywano w nowszych pracach poprzez zastosowanie aparatów aproksymacji funkcji i Grafowych Sieci Neuronowych (GNN).

W kontekście poszukiwania na grafach i przeszukiwania grafów wiedzy (*Knowledge Graph Completion*), algorytmy takie jak M-Walk czy metody bazujące na optymalizacji A* (np. PHIL autorstwa Pándy'ego i in.) skupiały się na klasycznym wyszukiwaniu lub uzupełnianiu relacji. Żadne z dotychczasowych podejść nie rozwiązywało jednak wprost problemu zdecentralizowanego poszukiwania ścieżek, w którym każdy agent dysponuje wyłącznie lokalną widocznością otoczenia 1-hopowego.

---

## 3. Metody (Methods)

Przedstawiona w artykule metodologia obejmuje formalne sformułowanie problemu w konwencji Dec-POMDP, architekturę uczenia wieloagentowego oraz szczegółową budowę modelu GARDEN.

### 3.1 Sformułowanie Dec-POMDP (MDP Formulation)
Zdecentralizowane wyszukiwanie ścieżki odbywa się na niedokierowanym, nieewaluowanym grafie z cechami węzłów $G = (V, E)$, gdzie $|V| = n$ oraz $|E| = m$. Na każdym węźle $u_i \in V$ umieszczony jest autonomiczny agent. Każdy agent $u_i$ posiada wektor atrybutów $x_{u_i} \in \mathbb{R}^d$. Celem systemu agentów jest przekazanie pojedynczej globalnej wiadomości od węzła źródłowego $u_{\text{src}}$ do węzła docelowego $u_{\text{tgt}}$.

Formalne komponenty procesu Dec-POMDP zdefiniowano następująco:

* **Stany ($S_t$):** Stan globalny w chwili $t$ reprezentowany jest przez wektor stanów indywidualnych $S_t = \langle S_t^{(1)}, S_t^{(2)}, \dots, S_t^{(n)} \rangle$, gdzie stan agenta $S_t^{(i)} = (M_t^{(i)}, u_{\text{tgt}})$. Zmienna wskaźnikowa $M_t^{(i)} \in \{0, 1\}$ określa obecność wiadomości w węźle $u_i$ w kroku $t$. Identyfikator celu $u_{\text{tgt}}$ jest częścią stanu środowiska (zapewniając stacjonarność MDP), lecz nie jest ujawniany agentowi w bezpośredniej obserwacji.
* **Akcje ($A_t$):** Przestrzeń akcji jest iloczynem kartezjańskim akcji poszczególnych agentów. W wybranym kroku $t$ agent $u_i$ posiadający wiadomość wybiera spośród swoich bezpośrednich sąsiadów $N(u_i)$ węzeł $u_j$, do którego przekazuje wiadomość ($a_{u_i \to u_j}$). Pozostali agenci wykonują akcję pustą (*no-op*).
* **Obserwacje ($O_t$):** Środowisko generuje obserwację $O_t = \langle O_t^{(1)}, \dots, O_t^{(n)} \rangle$. Agent $u_i$ obserwuje podgrafy 1-hopowe $G_{u_j}$ wycentrowane na swoich sąsiadach $u_j \in N(u_i)$, włączając w to krawędzie pomiędzy sąsiadami. Jeśli agent dysponuje wiadomością, obserwuje również podgraf $G_{u_{\text{tgt}}}$ wycentrowany na celu. Podgraf $G_{u_j} = (V_{u_j}, E_{u_j})$ zdefiniowano jako $V_{u_j} = N(u_j) \cup \{u_j\}$ oraz $E_{u_j} = \{(u_k, u_l) \in E \mid u_k, u_l \in V_{u_j}\}$.
* **Przejścia (Transitions):** Wiadomość przemieszcza się deterministycznie do wybranego węzła, aktualizując zmienne $M_t^{(i)}$.
* **Nagrody i Trunkacja (Rewards & Truncation):** Epizod kończy się sukcesem w momencie dotarcia wiadomości do celu $u_{\text{tgt}}$, przyznając wszystkim agentom wspólną nagrodę $+1$. Aby zapobiec nieskończonemu krążeniu po grafie i powstawaniu pętli akcji, wprowadzono sztuczną trunkację po $T_{\text{max}} = 100$ krokach.

### 3.2 Architektura uczenia i paradygmat CTDE
Uczenie odbywa się w reżimie **CTDE** (*Centralized Training with Decentralized Execution*). Wszystkie sieci agentów dzielą ze sobą parametry (*parameter-sharing*). Podczas treningu centralna sieć odbiera lokalne obserwacje i uczy się optymalnej polityki oraz funkcji wartości przy użyciu algorytmu **Advantage Actor-Critic (A2C)**.

Dla węzła $u_i$ posiadającego wiadomość $m$, stochastyczna polityka wyboru sąsiada $u_j \in N(u_i)$ sparametryzowana przez perceptron wielowarstwowy $f_{\Theta_1}^\pi$ z funkcją SoftMax zdefiniowana jest wzorem:

$$\pi_{u_i}(a_{u_i \to u_j} \mid O^{(i)}) = \frac{\exp\left(f_{\Theta_1}^\pi ([x_{u_j} \parallel x_{u_{\text{tgt}}}]\right)}{\sum_{u_k \in N(u_i)} \exp\left(f_{\Theta_1}^\pi ([x_{u_k} \parallel x_{u_{\text{tgt}}}]\right)}$$

gdzie $[ \cdot \parallel \cdot ]$ oznacza operację konkatenacji wektorów. Szacowanie funkcji wartości realizuje sieć $f_{\Theta_2}^v$:

$$v(O^{(i)}) = f_{\Theta_2}^v ([x_{u_i} \parallel x_{u_{\text{tgt}}}])$$

Autorzy wyróżniają dwa warianty bazowe wykorzystujące surowe cechy:
1. **MLPA2C:** wykorzystujący wyłącznie surowe atrybuty węzłów $x_{u_i}$.
2. **MLPA2CWD:** wzbogacający cechy węzłów o ich stopień: $x_{u_i}^{\text{WD}} = [x_{u_i} \parallel \text{deg}(u_i)]$.

### 3.3 Architektura GARDEN
Metoda **GARDEN** zastępuje surowe atrybuty ciągłymi reprezentacjami $x_{u_i}^{\text{GAT}}$ wygenerowanymi przez 3-warstwową sieć Graph Attention Network ($f_{\Theta_3}^{\text{rep}}$). Reprezentacje te wyliczane są poprzez procedurę przekazywania wiadomości (*message-passing*) na podstawie lokalnej topologii podgrafu oraz cech wejściowych.

Procedurę uczenia modelu GARDEN przedstawia poniższy algorytm:

```
Algorytm 1: GARDEN (Graph Attention-guided Routing for DEcentralised Networks)
--------------------------------------------------------------------------------
Wymagania: Sieć polityki f_Theta1_pi, Sieć wartości f_Theta2_v, Sieć reprezentacji f_Theta3_rep,
           Podgrafy 1-hop {G_u | u in V}, współczynnik entropii lambda, dyskonto gamma.
Wyjście: Wytrenowane parametry Theta = {Theta_1, Theta_2, Theta_3}.

1. Inicjalizuj losowo parametry Theta.
2. For epizod = 1 do N_episodes do:
3.     Inicjalizuj bufor epizodu B.
4.     Wylosuj węzeł początkowy u oraz cel u_tgt.
5.     For każdy węzeł w in V do:
6.         Oblicz reprezentację x_w^GAT = f_Theta3_rep(G_w).
7.     m = x_u_tgt^GAT, t = 0.
8.     While u != u_tgt i t < T_max do:
9.         Wylosuj akcję a_{u -> u'} ~ pi_u(. | m) z rozkładu polityki.
10.        Przekaż wiadomość do u', odbierz nagrodę r.
11.        Zapisz przejście (u, u', r) w B.
12.        u = u', t = t + 1.
13.    Inicjalizuj stratę epizodu L = 0.
14.    For każde przejście (u, u', r) w B do:
15.        A_hat = r + gamma * stop_gradient(v([x_u'^GAT || m])) - v([x_u^GAT || m])
16.        L(pi) = - A_hat * log(pi_u(a_{u -> u'} | m)) - lambda * H(pi(u | m))
17.        L(v) = (A_hat)^2
18.        L = L + L(pi) + L(v)
19.    Wykonaj krok spadku gradientu na całkowitej stracie L względem Theta.
```

Wszystkie parametry $\Theta = \{\Theta_1, \Theta_2, \Theta_3\}$ uczą się wspólnie w sposób end-to-end z wykorzystaniem regularyzacji entropijnej $H(\pi)$ z współczynnikiem $\lambda = 1 \times 10^{-3}$, co zapobiega zbyt wczesnej zbieżności i promuje eksplorację.

---

## 4. Konfiguracja eksperymentalna (Experimental Setup)

Środowisko testowe zostało zaprojektowane z myślą o rzetelnej ewaluacji zdolności uogólniania modeli na nieznane wcześniej cele.

### 4.1 Zbiory danych
* **Rzeczywiste grafy społecznościowe:** Wykorzystano 5 podgrafów ego z sieci Facebook pochodzących z repozytorium SNAP (od 148 do 532 węzłów). Węzły posiadają anonimizowane, binarne atrybuty z ankiet użytkowników.
* **Syntetyczne grafy homofilne:** Wygenerowane z użyciem modelu przestrzennego wzrostu Kaisera i Hilgetaga. Węzłom przypisuje się atrybuty jednostajnie z kostki $[0, 1]^d$, a prawdopodobieństwo utworzenia krawędzi wynosi $p((u, u')) = \max(1, \beta e^{-\alpha \|x_u - x_{u'}\|^2})$, gdzie $\alpha$ i $\beta$ kontrolują gęstość i stopień homofilii.

| Graf | Liczba węzłów ($n$) | Średnia najkrótsza ścieżka ($l_G$) | Gęstość krawędzi ($\rho$) | Wymiar atrybutów ($d$) |
| :--- | :---: | :---: | :---: | :---: |
| **Facebook Graph 1** | 148 | 2.69 | 0.16 | 105 |
| **Facebook Graph 2** | 168 | 2.43 | 0.12 | 63 |
| **Facebook Graph 3** | 224 | 2.52 | 0.13 | 161 |
| **Facebook Graph 4** | 324 | 3.75 | 0.05 | 224 |
| **Facebook Graph 5** | 532 | 3.45 | 0.03 | 262 |

### 4.2 Modele odniesienia (Baselines)
Metodę GARDEN porównano z następującymi wariantami:
1. **GreedyWalker:** Zachłanny agent wybierający sąsiada o najmniejszej odległości euklidesowej atrybutów od celu.
2. **DistanceWalker:** Stochastyczny agent oparty na SoftMax z odległości euklidesowej atrybutów z dostrojonym parametrem temperatury $\tau$.
3. **ConnectionWalker:** Stochastyczny agent preferujący węzły o wysokim stopniu (huby) z dostrojonym $\tau$.
4. **RandomWalker:** Agent wybierający sąsiada jednostajnie losowo.
5. **MLPA2C / MLPA2CWD:** Warianty RL bez komponentu GAT.

### 4.3 Ewaluacja i podział danych
Dla uniknięcia memoryzacji tras, zbiór węzłów podzielono w proporcjach 80% trening ($V_{\text{train}}$), 10% walidacja ($V_{\text{val}}$) i 10% test ($V_{\text{test}}$). Cel $u_{\text{tgt}}$ w fazie testowej wywodzono wyłącznie z podzbioru $V_{\text{test}}$.

Skuteczność oceniano za pomocą trzech metryk:
* **Średni stosunek do wyroczni ($\bar{R}_{\text{oracle}}$):** stosunek długości odnalezionej ścieżki do najkrótszej możliwej trasie w grafie.
* **Wskaźnik trunkacji ($R_{\text{trunc}}$):** odsetek epizodów, które nie dotarły do celu przed przekroczeniem limitu $T_{\text{max}} = 100$ kroków.
* **Wskaźnik wygranych ($R_{\text{win}}$):** odsetek epizodów, w których dany model uzyskał najkrótszą trasę spośród wszystkich badanych podejść.

---

## 5. Wyniki eksperymentalne (Experimental Results)

### 5.1 Wyniki na grafach Facebooka
Model GARDEN uzyskał zdecydowaną przewagę nad wszystkimi baseline'ami we wszystkich 5 rzeczywistych sieciach Facebooka, osiągając najniższy wskaźnik $\bar{R}_{\text{oracle}}$, najmniej przerwanych epizodów $R_{\text{trunc}}$ oraz dominujący wskaźnik wygranych $R_{\text{win}}$.

| Graf | Metryka | GreedyWalker | DistanceWalker | ConnectionWalker | RandomWalker | **GARDEN (Ours)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Graph 1** | $\bar{R}_{\text{oracle}} \downarrow$ | $27.17 \pm 1.09$ | $15.98 \pm 0.79$ | $33.39 \pm 1.15$ | $33.17 \pm 1.16$ | **$10.95 \pm 0.76$** |
| | $R_{\text{trunc}} (\%) \downarrow$ | 79.00 | 39.70 | 76.90 | 76.70 | **14.90** |
| | $R_{\text{win}} (\%) \uparrow$ | 7.90 | 26.20 | 2.70 | 2.50 | **60.70** |
| **Graph 2** | $\bar{R}_{\text{oracle}} \downarrow$ | $31.78 \pm 1.22$ | $13.01 \pm 0.90$ | $34.62 \pm 1.17$ | $34.27 \pm 1.18$ | **$9.89 \pm 0.78$** |
| | $R_{\text{trunc}} (\%) \downarrow$ | 78.00 | 19.50 | 68.50 | 64.10 | **12.60** |
| | $R_{\text{win}} (\%) \uparrow$ | 10.90 | 29.90 | 4.30 | 3.90 | **51.00** |
| **Graph 3** | $\bar{R}_{\text{oracle}} \downarrow$ | $27.35 \pm 1.19$ | $9.99 \pm 0.78$ | $38.84 \pm 1.42$ | $38.88 \pm 1.43$ | **$8.74 \pm 0.76$** |
| | $R_{\text{trunc}} (\%) \downarrow$ | 70.00 | 13.90 | 69.80 | 67.20 | **13.50** |
| | $R_{\text{win}} (\%) \uparrow$ | 11.90 | 33.80 | 1.50 | 2.00 | **50.80** |
| **Graph 4** | $\bar{R}_{\text{oracle}} \downarrow$ | $25.69 \pm 0.88$ | $14.33 \pm 0.65$ | $28.25 \pm 1.00$ | $28.01 \pm 1.01$ | **$12.08 \pm 0.71$** |
| | $R_{\text{trunc}} (\%) \downarrow$ | 88.00 | 44.00 | 81.50 | 79.40 | **40.10** |
| | $R_{\text{win}} (\%) \uparrow$ | 6.10 | 32.10 | 5.30 | 8.20 | **48.30** |
| **Graph 5** | $\bar{R}_{\text{oracle}} \downarrow$ | $28.29 \pm 0.66$ | $22.71 \pm 0.70$ | $28.11 \pm 0.69$ | $28.22 \pm 0.67$ | **$16.97 \pm 0.74$** |
| | $R_{\text{trunc}} (\%) \downarrow$ | 96.00 | 72.70 | 89.50 | 90.00 | **48.20** |
| | $R_{\text{win}} (\%) \uparrow$ | 2.20 | 24.90 | 7.60 | 10.50 | **54.80** |

Analiza czasu wykonania (*runtime analysis*) wykazała, że podejście GARDEN cechuje się bardzo wysoką wydajnością – średni czas wyznaczania akcji wynosi zaledwie **0.15–0.18 ms na krok** na procesorze CPU, przy narzucie na obliczenie reprezentacji ego-grafu rzędu **0.21–0.50 ms**.

### 5.2 Analiza wrażliwości parametru gęstości grafu ($\beta$)
Eksperymenty na grafach syntetycznych wykazują, że GARDEN utrzymuje przewagę dla większości wartości parametru gęstości $\beta$. W rzadkich grafach ($\beta \le 0.1$) GARDEN przewyższa inne metody w redukcji wskaźnika przerwanych tras. Przy bardzo wysokiej gęstości ($\beta \ge 0.5$) DistanceWalker osiąga niższy wskaźnik $R_{\text{trunc}}$, ponieważ korzysta z bezpośredniego dostępu do prawdziwych odległości w znanej przestrzeni generowania atrybutów, co nie występuje w grafach rzeczywistych.

### 5.3 Badanie ablacjne reprezentacji węzłów
Wyniki badania ablacjnego przeprowadzonego na grafach syntetycznych jednoznacznie wskazują na kluczową rolę modułu GAT w budowaniu efektywnych reprezentacji:

| Model | $\bar{R}_{\text{oracle}} \downarrow$ | $R_{\text{trunc}} (\%) \downarrow$ | $R_{\text{win}} (\%) \uparrow$ |
| :--- | :---: | :---: | :---: |
| **GARDEN (GAT Embeddings)** | **$1.95 \pm 0.09$** | **1.68** | 34.44 |
| **MLPA2CWD (Cechy + Stopień)** | $2.23 \pm 0.13$ | 2.94 | 30.78 |
| **MLPA2C (Surowe atrybuty)** | $2.31 \pm 0.12$ | 4.32 | **34.78** |

Zastosowanie reprezentacji z GAT pozwoliło obniżyć wskaźnik przerwanych epizodów z 4.32% w podstawowej sieci MLP do 1.68% w GARDEN oraz znacząco skrócić średnią długość ścieżki w porównaniu z wyrocznią.

---

## 6. Wnioski i przyszłe kierunki badań (Conclusions and Future Work)

Artykuł przynosi istotne wnioski teoretyczne i praktyczne:
1. **Potwierdzenie hipotezy ukrytej metryki:** Uczenie ze wzmocnieniem połączone z przekazywaniem wiadomości w GAT potrafi odtworzyć reprezentację metryczną otoczenia, łącząc atrybuty węzłów z lokalną strukturą połączeń.
2. **Organiczna wyłanialność nawigacji:** Zdecentralizowana nawigacja zbliżona do zachowań obserwowanych w eksperymencie Milgrama może wyłonić się w sposób organiczny dzięki optymalizacji opartej na nagrodzie środowiskowej.
3. **Interpretowalność funkcji wartości:** Wyuczona sieć Krytyka przypisuje najwyższe wartości węzłom znajdującym się w klastrze celu lub węzłom będącym mostami strukturalnymi.

**Ograniczenia i kierunki rozwoju:**
Głównym ograniczeniem przedstawionego modelu GARDEN jest jego **bezpamięciowy charakter** (*memory-less search*). Agent podejmuje decyzję wyłącznie na podstawie aktualnego stanu i celu, co sprawia, że jest podatny na wpadanie w pętle akcji i krążenie wewnątrz gęstych społeczności. Autorzy wskazują jako najważniejszy kierunek dalszych badań włączenie historii odwiedzonych węzłów do przesyłanej wiadomości i zastosowanie mechanizmów rekurencyjnych (sieci RNN/GRU) do kodowania trajektorii.

---

## 7. Dodatek i Szczegóły Implementacyjne (Appendix)

### 7.1 Hiperparametry modelu GARDEN
Poniższa tabela przedstawia wspólną konfigurację hiperparametrów wykorzystaną we wszystkich eksperymentach uczenia ze wzmocnieniem:

| Parametr | Wartość |
| :--- | :--- |
| Warstwy sieci Aktora (Actor Layers) | 3 |
| Wymiar ukryty Aktora (Actor Hidden Dim) | 64 |
| Warstwy sieci Krytyka (Critic Layers) | 3 |
| Wymiar ukryty Krytyka (Critic Hidden Dim) | 64 |
| Warstwy GAT (Graph Attention Layers) | 3 |
| Wymiar ukryty GAT (GAT Hidden Dim) | 64 |
| Głowice uwagi GAT (GAT Attention Heads) | 1 |
| Współczynnik regularyzacji entropijnej ($\lambda$) | $1 \times 10^{-3}$ |
| Współczynnik dyskonta ($\gamma$) | 0.99 |
| Maksymalna długość epizodu ($T_{\text{max}}$) | 100 |
| Optymalizator | Adam (learning rate $1 \times 10^{-4}$) |
