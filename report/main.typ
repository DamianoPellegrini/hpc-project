#import "@local/unimib-templates:0.1.0": report-footer, unimib
#import "data.typ": *
#import "figures.typ": (
  measured-vs-theoretical-speedup-chart, reference-breakdown-stacked-chart, reference-total-chart,
  theoretical-efficiency-chart, theoretical-speedup-chart, total-vs-density-chart,
)

#set text(lang: "it")
#set math.equation(numbering: "(1)")

#show: unimib.with(
  title: [Implementazione parallela dell'algoritmo di Boruvka per Minimum Spanning Tree],
  authors: (
    (name: "Damiano Pellegrini", matr: "886261"),
  ),
  extras: (),
  keywords: (
    "Boruvka",
    "minimum spanning tree",
    "parallel computing",
    "OpenMP",
    "MPI",
    "CUDA",
    "HPC",
  ),
  area: [Scuola di Scienze],
  department: [Dipartimento di Informatica, Sistemi e Comunicazione],
  course: [Sistemi di Calcolo Parallelo],
  front-page-footer: report-footer,
  dark: false,
  abstract: [
    Il lavoro presenta tre implementazioni indipendenti dell'algoritmo di Boruvka per il calcolo del Minimum Spanning Tree -- MPI, OpenMP e CUDA -- ciascuna in un singolo file autosufficiente. Dopo una descrizione della struttura del progetto e del workflow di misura sul cluster, il Capitolo 2 deriva dal modello di Kumar lavoro, overhead, speedup ed efficienza per i tre backend. Il Capitolo 3 confronta queste previsioni con una carrellata di run a seed fisso eseguita sul cluster. Il Capitolo 4 discute quando la parallelizzazione conviene alla luce dei risultati.
  ],
)

#let mpi-ref = run-at-density("mpi", reference-density)
#let openmp-ref = run-at-density("openmp", reference-density)
#let cuda-ref = run-at-density("cuda", reference-density)

= Progetto e contesto sperimentale

La relazione confronta tre implementazioni dello stesso algoritmo -- Boruvka per il Minimum Spanning Tree -- su tre modelli di esecuzione: MPI per memoria distribuita, OpenMP per memoria condivisa e CUDA per GPU.

== L'algoritmo di Boruvka

Boruvka calcola l'MST mantenendo una partizione dei vertici in componenti connesse, inizialmente una per vertice. Ogni round procede in tre fasi:

1. *scan*: per ogni componente si cerca l'arco uscente di peso minimo (un arco è "uscente" se collega due vertici di componenti diverse);
2. *riduzione*: tra i candidati trovati nello scan si sceglie, per ciascuna componente, un unico arco minimo;
3. *contrazione*: gli archi scelti vengono aggiunti all'MST e le componenti che collegano vengono fuse in un'unica componente.

Il round si ripete finché resta più di una componente. Poiché ogni round almeno dimezza il numero di componenti -- ogni componente si fonde con almeno un'altra -- il numero di round è $O(log |cal(V)|)$. La scelta dell'arco minimo per componente, con un criterio di pareggio coerente (peso, poi indice dell'arco), garantisce che l'insieme di archi scelti non formi mai un ciclo: è questa proprietà a rendere Boruvka parallelizzabile fase per fase, perché lo scan può essere eseguito su tutti gli archi contemporaneamente senza coordinamento.

== Struttura del progetto

Ciascun backend -- `src/openmp.cpp`, `src/mpi.cpp`, `src/cuda.cu` -- è implementato in un singolo file autosufficiente, che contiene la rappresentazione del grafo, il generatore di grafi casuali, l'algoritmo di Boruvka parallelo, il riferimento sequenziale (Kruskal) per la verifica e il `main` con la riga di comando. Accanto a questi, `src/sequential.cpp` ha la stessa struttura ma esegue Boruvka in versione seriale (stessa sequenza di round snapshot/scan/merge degli altri tre, senza `parallel for` né atomici): è il programma usato per misurare $T_s$, la baseline sequenziale dello speedup empirico del Capitolo 3.

Per un confronto tra modelli di esecuzione, dove l'obiettivo è mostrare come la stessa idea algoritmica si traduce in stili di parallelismo diversi (e nella sua assenza), avere ciascuno stile per intero in un solo file permette di leggere l'implementazione di un backend senza dover saltare tra moduli condivisi, isolando le scelte specifiche di MPI, OpenMP, CUDA e della versione seriale.

I quattro programmi condividono la stessa interfaccia a riga di comando, `<vertici> <archi> <seed>`: `vertici` è $|cal(V)|$, `archi` è $|cal(E)|$ (il generatore garantisce prima un albero di copertura casuale per la connessione, poi aggiunge archi casuali fino al totale richiesto) e `seed` inizializza pesi e topologia. La generazione è identica byte per byte a parità di parametri: lo stesso grafo è processato dalla versione seriale e da ciascun backend, rendendo $T_s$ e $T_p$ confrontabili -- ed è anche ciò che permette agli script Slurm di restare identici a parte binario e risorse.

Ogni programma stampa, oltre all'esito della verifica sequenziale, tre tempi: l'overhead (generazione del grafo, inizializzazione di MPI/CUDA, allocazioni, copie host$arrow.r$device), il tempo di esecuzione (solo il loop dell'algoritmo di Boruvka, seriale per `sequential_app`) e il totale, la loro somma. Un quarto tempo misura a parte la verifica con Kruskal e non entra né nell'overhead né nel totale: serve solo a controllare la correttezza.

== Infrastruttura di build

Il percorso locale usa CMake con il preset `default`, che configura una build con generatore *Ninja* e rileva OpenMP, MPI e CUDA quando disponibili: `add_executable` collega ciascun target eseguibile direttamente a uno dei quattro file in `src/`, senza sotto-directory né librerie intermedie. Il target seriale (`sequential_app`) non dipende da OpenMP/MPI/CUDA e viene quindi sempre compilato.

Il percorso cluster usa un `Makefile` scritto a mano, indipendente da CMake: i nodi di calcolo non hanno `cmake`, e un Makefile generato da CMake con generatore "Unix Makefiles" richiamerebbe comunque `cmake` per controllare lo stato della build. Il Makefile espone target separati (`make sequential`, `make openmp`, `make mpi`, `make cuda`) con compilatori configurabili da variabile d'ambiente (`CXX`, `MPICXX`, `NVCC`), e va tenuto sincronizzato a mano con `CMakeLists.txt` quando cambiano sorgenti o flag.

== Workflow sul cluster

Gli script in `scripts/slurm/` (`sequential.sh`, `openmp.sh`, `mpi.sh`, `cuda.sh`) eseguono ciascuno un job Slurm per backend. Ogni script compila il proprio target con il Makefile, poi esegue una *carrellata di run* a seed fisso: lo stesso grafo di base ($|cal(V)| = 32768$ vertici, seed $#mpi-ref.seed$) viene rigenerato con un numero crescente di archi, in modo da coprire le densità $|cal(E)| slash |cal(V)| in {1, 2, 4, 6, 12, 24, 48, 96, 192, 384}$ -- dieci punti per backend, una run ciascuno (nessuna ripetizione). Ogni run produce una riga con backend, dimensioni del grafo, seed, risorse allocate, i tre tempi misurati e l'esito della verifica.

Le risorse allocate riflettono la scelta di usare il massimo disponibile per backend (vincolo del progetto): MPI usa #workers(mpi-ref) (partizione `ulow`, un rank per CPU), OpenMP usa #workers(openmp-ref) (stessa partizione, `OMP_NUM_THREADS` pari alle CPU allocate) e CUDA usa #workers(cuda-ref) (partizione `only-one-gpu`, scelta di restare su una singola GPU per limitare la complessità implementativa). Per CUDA la dimensione di blocco non è fissata a mano: `cudaOccupancyMaxPotentialBlockSize` la sceglie a runtime sul kernel di scan, adattandosi alla GPU effettivamente assegnata dallo scheduler. Il job sequenziale (`sequential.sh`) gira invece su un solo core della partizione `ulow`.

I quattro CSV (uno per backend, uno per job Slurm) sono la fonte dati dei capitoli successivi: il Capitolo 2 deriva dal modello teorico le previsioni di overhead, speedup ed efficienza; il Capitolo 3 le confronta con questi dati misurati sulla carrellata di run, incluso lo speedup empirico $S_p = T_s slash T_p$ con $T_s$ misurato dal job sequenziale.

= Analisi teorica di Boruvka parallelo

Questo capitolo deriva il modello dalle tre implementazioni. I numeri misurati nel Capitolo 3 servono a verificare il modello, non a definirlo: lavoro, overhead, speedup e soglie sono ricavati qui solo dalla struttura del codice.

== Dinamica sequenziale

Boruvka mantiene una partizione dei vertici in componenti connesse. Ogni round sceglie, per ciascuna componente, l'arco uscente di peso minimo e poi contrae le componenti collegate dagli archi scelti -- lo schema scan/riduzione/contrazione descritto nel Capitolo 1.

Il costo di un round sequenziale è $Theta(|cal(E)| + |cal(V)|)$: la scan visita $|cal(E)|$ archi, la gestione dei candidati e la fusione delle componenti toccano al più $|cal(V)|$ elementi. Il numero di round è $r = O(log |cal(V)|)$, perché il numero di componenti si riduce almeno di un fattore costante a ogni round.

$
  W = T_s = Theta(r dot (|cal(E)| + |cal(V)|)) = Theta(|cal(E)| dot log |cal(V)| + |cal(V)| dot log |cal(V)|)
$ <eq:seq-work>

La @eq:seq-work è il *lavoro* $W$ nel senso di Kumar: il tempo della migliore esecuzione sequenziale dello stesso algoritmo (Boruvka), non quello di un algoritmo diverso. `src/sequential.cpp` (Capitolo 1) implementa esattamente questa versione -- stessa struttura a round, senza parallelismo -- e il suo tempo di esecuzione misurato è il $T_s$ usato come baseline per lo speedup empirico nel Capitolo 3. Poiché ripete la stessa sequenza snapshot/scan/fusione dei tre backend, lo speedup confronta implementazioni a parità di algoritmo e struttura, isolando il solo effetto della parallelizzazione. Kruskal (ordinamento degli archi più union-find, $Theta(|cal(E)| log |cal(E)|)$), presente in tutti e quattro i programmi, resta invece solo un controllo di correttezza indipendente, non la baseline di questo modello.

Per $|cal(E)| = Omega(|cal(V)|)$ -- vero per tutte le densità della carrellata, dove $|cal(E)| slash |cal(V)| >= 1$ -- il termine $|cal(E)| dot log |cal(V)|$ domina, quindi $W = Theta(|cal(E)| log |cal(V)|)$. Useremo questa forma per confrontare $W$ con l'overhead nelle sezioni seguenti.

== Modello di costo

L'overhead parallelo misura la differenza tra il costo processore-tempo e il lavoro sequenziale:

$
  T_o = p dot T_p - T_s
$ <eq:overhead-def>

Speedup ed efficienza seguono dall'identità $W = T_s$ (@eq:seq-work) e da @eq:overhead-def:

$
  S_p = frac(T_s, T_p)
$ <eq:speedup-def>

$
  E_p = frac(S_p, p) = frac(T_s, p dot T_p) = frac(W, W + T_o)
$ <eq:efficiency-def>

Un algoritmo è cost-optimal quando il costo processore-tempo ha lo stesso ordine del lavoro sequenziale:

$
  p dot T_p = Theta(W) <=> T_o = O(W)
$ <eq:cost-optimality>

L'isoefficienza fissa un'efficienza minima e ricava quanto deve crescere il lavoro perché valga:

$
  W = K dot T_o quad "con" quad K = frac(E_("min"), 1 - E_("min"))
$ <eq:isoefficiency>

La soglia di efficienza scelta è $E_("min") = 1 / 2$, da cui $K=1$:

$
  E_("min") = frac(1, 2) => W = T_o
$ <eq:half-efficiency>

Nelle tre sezioni seguenti $T_p$ è il tempo di un round; il fattore $r$ compare identico a numeratore e denominatore di $S_p$ e si elide, quindi gli speedup di @eq:mpi-speedup, @eq:omp-speedup e @eq:cuda-sm-speedup sono espressi direttamente nei termini per round. Per overhead e isoefficienza, dove $T_o$ e $W$ sono cumulati su $r$ round, il fattore $r$ compare su entrambi i lati di @eq:half-efficiency e si elide allo stesso modo: anche le soglie sono quindi condizioni *per round*. L'elisione è esatta solo se le grandezze per round sono costanti; in realtà la contrazione riduce $|cal(E)|$ e $|cal(V)|$ a ogni round e le formule sono valutate sui valori iniziali, quindi vanno lette come approssimazioni dominate dal round iniziale -- quello in cui la scan su $|cal(E)|$ archi pesa di più.

== Modello per backend

=== MPI

`src/mpi.cpp` distribuisce gli $|cal(E)|$ archi in blocchi contigui tra $p$ rank (`counts`/`displs`, righe 224-236): ogni rank scansiona i propri $|cal(E)| slash p$ archi e produce, in `best`, un candidato locale per ciascuna delle $|cal(V)|$ componenti che tocca (struct `CandEdge`, righe 254-264). I candidati locali vengono combinati con `MPI_Allreduce` e un operatore `MPI_Op` definito ad-hoc (`cand_min`, righe 117-123) che sceglie, chiave per chiave, il candidato di peso minore con l'id più piccolo come tie-break. L'Allreduce restituisce a ogni rank lo stesso vettore di $|cal(V)|$ candidati globali.

Da qui ogni rank esegue la *stessa* fusione union-find su una copia locale (righe 273-304): per ognuna delle al più $|cal(V)|$ componenti applica `unite`, poi rietichetta tutti i $|cal(V)|$ vertici con `comp[v] = find(comp[v])`. Questo passo è $Theta(|cal(V)|)$ per rank -- ridondante ($p$ rank lo eseguono tutti) ma non distribuito, perché la coerenza tra le copie locali dipende dal fatto che ognuna applichi le stesse fusioni nello stesso ordine, partendo dallo stesso risultato dell'Allreduce.

La scan locale costa $Theta(|cal(E)| slash p)$. Per l'Allreduce su $|cal(V)|$ chiavi, il modello $alpha$-$beta$ di una riduzione gerarchica dà $Theta(alpha log p + beta |cal(V)| log p)$; trascurando il termine di latenza $alpha log p$ per $|cal(V)|$ grande, resta $Theta(|cal(V)| log p)$. Il termine $log p$ assume una riduzione ad albero, quella indotta dall'`MPI_Op` custom: una collettiva ottimizzata per vettori grandi avvicinerebbe questo costo a $Theta(|cal(V)|)$, abbassando la soglia di @eq:mpi-threshold verso quella di OpenMP. La fusione locale è qui modellata $Theta(|cal(V)|)$ ed è dominata dal termine dell'Allreduce per $p > 1$; trascura il costo dell'ordinamento dei candidati scelti (righe 273-304), $O(|cal(V)| log |cal(V)|)$ nel caso peggiore, che il modello assorbe nel termine di riduzione. Il tempo per round è quindi:

$
  T_p^("MPI") = Theta(frac(|cal(E)|, p) + |cal(V)| log p)
$ <eq:mpi-time>

Lo speedup per round, dalla @eq:seq-work per round ($W_("round") = Theta(|cal(E)|+|cal(V)|)$):

$
  S_p^("MPI") = frac(|cal(E)|+|cal(V)|, frac(|cal(E)|, p) + |cal(V)| log p)
$ <eq:mpi-speedup>

L'overhead, da @eq:overhead-def e @eq:mpi-time:

$
  T_o^("MPI") = Theta(p dot |cal(V)| log p)
$ <eq:mpi-overhead>

Cost-optimality (@eq:cost-optimality):

$
  |cal(E)| = Omega(p dot |cal(V)| log p)
$ <eq:mpi-cost-optimality>

L'isoefficienza a $E_("min")=1/2$ (@eq:half-efficiency, $W=T_o$) dà $|cal(E)|+|cal(V)| = Theta(p |cal(V)| log p)$, e per $|cal(E)|=Omega(|cal(V)|)$:

$
  frac(|cal(E)|, |cal(V)|) >= p log p
$ <eq:mpi-threshold>

=== OpenMP

`src/openmp.cpp` non usa buffer per-thread: tutti i thread scrivono nello stesso array `cheapest[V]` di interi a 64 bit, con un atomic-min lock-free (`atomic_min_u64`, righe 59-64) che impacchetta peso e indice dell'arco -- lo stesso schema dell'atomica CUDA, ma su CPU. Per ogni round: uno `#pragma omp parallel for` su $|cal(V)|$ vertici fa lo snapshot delle radici (`comp[v] = dsu.find(v)`, righe 84-86), uno su $|cal(V)|$ resetta `cheapest[]`, e uno su $|cal(E)|$ archi esegue lo scan con gli atomic-min (righe 99-108).

La fusione, però, è *seriale*: un singolo `for` su $|cal(V)|$ slot (righe 115-127) legge `cheapest[c]`, e quando contiene un arco valido chiama `dsu.unite` (union-by-rank, senza path compression -- `find` è di sola lettura per restare sicuro durante lo scan parallelo, righe 37-41). Questo passo non è diviso tra i thread.

Le tre fasi parallele (snapshot, reset, scan) costano $Theta(|cal(V)| slash p)$, $Theta(|cal(V)| slash p)$ e $Theta(|cal(E)| slash p)$. La fusione seriale costa $Theta(|cal(V)|)$ ed è la sola fase non divisa per $p$: per $p>1$ domina le altre due. Il tempo per round è:

$
  T_p^("OMP") = Theta(frac(|cal(E)|, p) + |cal(V)|)
$ <eq:omp-time>

Lo speedup per round:

$
  S_p^("OMP") = frac(|cal(E)|+|cal(V)|, frac(|cal(E)|, p) + |cal(V)|)
$ <eq:omp-speedup>

L'overhead, da @eq:overhead-def e @eq:omp-time -- qui interamente dovuto alla fusione seriale, che gli altri $p-1$ thread attendono:

$
  T_o^("OMP") = Theta((p-1) |cal(V)|) = Theta(p |cal(V)|)
$ <eq:omp-overhead>

Cost-optimality:

$
  |cal(E)| = Omega(|cal(V)| dot p)
$ <eq:omp-cost-optimality>

Isoefficienza a $E_("min")=1/2$: $|cal(E)|+|cal(V)| = Theta(|cal(V)| p)$, da cui per $|cal(E)|=Omega(|cal(V)|)$:

$
  frac(|cal(E)|, |cal(V)|) >= p
$ <eq:omp-threshold>

A parità di $p$, @eq:omp-threshold è strutturalmente più favorevole di @eq:mpi-threshold ($p$ contro $p log p$): la fusione seriale di OpenMP costa $Theta(|cal(V)|)$ una volta sola, mentre la collettiva MPI paga $Theta(|cal(V)| log p)$ per la topologia gerarchica della riduzione. Il confronto sperimentale del Capitolo 3 verifica se questo vantaggio asintotico si traduce in tempi misurati migliori.

=== CUDA

`src/cuda.cu` non usa una struttura union-find: ogni componente mantiene un puntatore "successore" verso la componente con cui si fonde. Per round, cinque kernel assegnano un thread logico a ogni arco o a ogni componente, distribuiti dall'hardware su $q$ Streaming Multiprocessor:

- `k_reset_min` e `k_iota`: $Theta(|cal(V)| slash q)$, inizializzazione;
- `k_find_min_edges` (righe 89-101): $Theta(|cal(E)| slash q)$, scan degli archi con `atomicMin` su chiave impacchettata (peso, indice) -- stesso schema di OpenMP;
- `k_build_successor` e `k_mark_and_break` (righe 105-140): $Theta(|cal(V)| slash q)$ ciascuno; il secondo rompe gli unici cicli possibili (lunghezza 2, per il tie-break sull'indice) e marca gli archi MST;
- `k_jump` (righe 143-152): un passo di *pointer jumping* raddoppia a ogni iterazione la distanza coperta da ciascun puntatore verso la radice del proprio albero di fusione; si ripete finché nessun puntatore cambia più, poi `k_relabel` applica le nuove radici a tutti i $|cal(V)|$ vertici.

Il numero di iterazioni di pointer jumping è $O(log D)$, dove $D$ è la profondità massima degli alberi di fusione formati in un round dai link successore. Nel caso peggiore $D = O(|cal(V)|)$, ma il pointer jumping resta comunque limitato da $O(log |cal(V)|)$ iterazioni indipendentemente da $D$. Usando questo limite superiore, ogni iterazione costa $Theta(|cal(V)| slash q)$, quindi il contributo di `k_jump` è $O(|cal(V)| log |cal(V)| slash q)$. Sommando le fasi:

$
  T_p^("CUDA") = O(frac(|cal(E)| + |cal(V)| log |cal(V)|, q))
$ <eq:cuda-time>

Lo speedup per round, da @eq:seq-work per round:

$
  S_p^("CUDA") = Omega(frac(q (|cal(E)|+|cal(V)|), |cal(E)| + |cal(V)| log |cal(V)|))
$ <eq:cuda-sm-speedup>

Il termine $|cal(V)| log |cal(V)|$ introdotto dal pointer jumping *non* si cancella con $|cal(E)|+|cal(V)|$ in @eq:cuda-sm-speedup. Per $|cal(E)| >> |cal(V)| log |cal(V)|$ (grafi densi) @eq:cuda-sm-speedup tende a $q$, costante; per $|cal(E)| = Theta(|cal(V)|)$ (grafi sparsi) tende a $q slash log |cal(V)|$, più piccolo. CUDA ha quindi, in questo modello, una soglia di densità asintotica, conseguenza diretta dello schema successore + pointer jumping.

L'overhead, da @eq:overhead-def e @eq:cuda-time:

$
  T_o^("CUDA") = O(|cal(V)| log |cal(V)|)
$ <eq:cuda-overhead>

Cost-optimality:

$
  |cal(V)| log |cal(V)| = O(|cal(E)| + |cal(V)|)
$ <eq:cuda-cost-optimality>

che per $|cal(E)|=Omega(|cal(V)|)$ diventa $|cal(V)| log |cal(V)| = O(|cal(E)|)$, cioè:

$
  frac(|cal(E)|, |cal(V)|) >= log |cal(V)|
$ <eq:cuda-threshold>

A differenza di @eq:mpi-threshold e @eq:omp-threshold, @eq:cuda-threshold non dipende da $q$: il numero di SM scala il tempo assoluto ma non sposta il punto in cui $|cal(V)| log |cal(V)|$ smette di essere trascurabile rispetto a $|cal(E)|$. Questo è anche il motivo per cui l'efficienza teorica $E_p^("CUDA") = S_p^("CUDA") slash q = (d+1) slash (d + log_2 |cal(V)|)$, con $d=|cal(E)| slash |cal(V)|$, è calcolabile dai CSV senza conoscere $q$ (Capitolo 3): $q$ si cancella nel rapporto. La GPU usata per le run è una NVIDIA L40S, $q=142$ SM: questo valore permette di tradurre $E_p^("CUDA")$ in uno speedup assoluto $S_p^("CUDA") = q dot E_p^("CUDA")$ (Capitolo 3). Si conta un SM come unità analoga a un core CPU -- entrambi unità di scheduling indipendenti -- mentre i CUDA core sono corsie SIMD interne all'SM e non vengono contati, esattamente come non si contano le corsie vettoriali (AVX) interne ai core CPU. L'efficienza per-SM non va quindi confrontata uno-a-uno con quella per-core di MPI e OpenMP, e l'efficienza misurata più bassa (Capitolo 3) riflette il sottoutilizzo *dentro* ogni SM.

== Confronto dei modelli

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    table.header([*Backend*], [*$T_p$ per round*], [*$T_o$ per round*], [*Soglia di isoefficienza* ($E_("min")=1/2$)]),
    [MPI], [$Theta(|cal(E)| slash p + |cal(V)| log p)$], [$Theta(p |cal(V)| log p)$], [$frac(|cal(E)|, |cal(V)|) >= p log p$],
    [OpenMP], [$Theta(|cal(E)| slash p + |cal(V)|)$], [$Theta(p |cal(V)|)$], [$frac(|cal(E)|, |cal(V)|) >= p$],
    [CUDA], [$O((|cal(E)| + |cal(V)| log |cal(V)|) slash q)$], [$O(|cal(V)| log |cal(V)|)$], [$frac(|cal(E)|, |cal(V)|) >= log |cal(V)|$],
  ),
  caption: [Modelli teorici dei tre backend nella stessa base $|cal(E)|$, $|cal(V)|$, $p$ (processi/thread) e $q$ (SM); soglie valutate alla densità di pareggio costo-lavoro per $E_("min")=1/2$ (@eq:half-efficiency).],
) <tab:theory-summary>

Per $|cal(V)| = 32768$ e $p=8$, le tre soglie valgono $p log_2 p = 24$ per MPI, $p = 8$ per OpenMP e $log_2 |cal(V)| = 15$ per CUDA: tutte cadono nell'intervallo della carrellata ($|cal(E)| slash |cal(V)| in {1,...,384}$), e $24$ è anche uno dei punti misurati. Ciascuna soglia nasce da come il backend riporta i $|cal(V)|$ candidati a un risultato unico per componente: la collettiva MPI paga $|cal(V)| log p$ per round, la fusione seriale OpenMP $|cal(V)|$ (un fattore $log p$ in meno), il pointer jumping CUDA $|cal(V)| log |cal(V)|$ indipendentemente da $p$ o $q$. La scan $Theta(|cal(E)| slash dot)$ resta il lavoro comune inevitabile.

= Misurazione sperimentale

Le misure di questo capitolo vengono dai quattro CSV prodotti dagli script Slurm (Capitolo 1): una carrellata di run con densità $|cal(E)| slash |cal(V)| in {1,2,4,6,12,24,48,96,192,384}$ a $|cal(V)|=32768$ e seed $#mpi-ref.seed$ fisso, una run per punto, per ciascun backend (incluso il riferimento sequenziale). Ogni riga riporta l'overhead (generazione del grafo, init, allocazioni, copie), il tempo di esecuzione (solo il loop di Boruvka) e il totale, somma dei due. Trattandosi di una sola run per punto, le variazioni entro pochi punti percentuali vanno lette come variabilità di misura, non come segnale.

== Tempi totali sulla carrellata

#total-vs-density-chart

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*Densità $|cal(E)| slash |cal(V)|$*], [*MPI*], [*OpenMP*], [*CUDA*]),
    ..swept-densities
      .map(density => (
        [#calc.round(density, digits: 0)],
        [#duration(run-at-density("mpi", density).total)],
        [#duration(run-at-density("openmp", density).total)],
        [#duration(run-at-density("cuda", density).total)],
      ))
      .flatten(),
  ),
  caption: [Tempo totale (overhead + esecuzione) misurato per ciascun punto della carrellata di run.],
) <tab:total-by-density>

La @tab:total-by-density e il grafico precedente mostrano tre andamenti diversi. MPI è il più rapido per quasi tutta la carrellata, da poche decine di millisecondi alle densità basse a circa #duration(run-at-density("mpi", 384.0).total) a $d=384$. OpenMP è il più lento a ogni densità, stabilmente tra #duration(run-at-density("openmp", 1.0).total) e #duration(run-at-density("openmp", 384.0).total), senza un trend monotono netto. CUDA parte più lento di MPI alle densità basse (#duration(run-at-density("cuda", 1.0).total) a $d=1$, per l'overhead di allocazione e copia su device), ma resta comunque sempre più rapido di OpenMP, e ai due punti più densi della carrellata ($d=192$ e $d=384$) raggiunge totali sostanzialmente equivalenti a MPI (dove le differenze cadono entro il rumore di run singole). Va però notato che ad alta densità il tempo totale è dominato dall'overhead, in gran parte la generazione del grafo (a $d=384$ circa il #percent(run-at-density("mpi", 384.0).overhead, run-at-density("mpi", 384.0).total) del totale MPI), un costo di benchmark amortizzabile in un workload reale: il confronto algoritmicamente rilevante resta quello sull'esecuzione pura.

== Overhead ed esecuzione pura

#reference-total-chart

#reference-breakdown-stacked-chart <fig:reference-breakdown>

Alla densità di riferimento $d=#calc.round(reference-density, digits: 0)$ ($|cal(E)|=#mpi-ref.edges$), la quota di overhead sul totale è #percent(mpi-ref.overhead, mpi-ref.total) per MPI, #percent(openmp-ref.overhead, openmp-ref.total) per OpenMP e #percent(cuda-ref.overhead, cuda-ref.total) per CUDA. Per MPI e OpenMP il loop di Boruvka domina il tempo totale; per CUDA accade l'opposto -- l'esecuzione pura dura #duration(cuda-ref.exec), contro #duration(cuda-ref.overhead) di overhead.

L'overhead misurato include la generazione del grafo, comune ai tre programmi e $Theta(|cal(E)|+|cal(V)|)$ indipendentemente dal backend, più l'inizializzazione specifica (MPI_Init/broadcast, allocazioni e copie host$arrow.r$device per CUDA). Per CUDA questo secondo termine è significativo: anche al punto più sparso della carrellata ($d=1$) l'overhead è #duration(run-at-density("cuda", 1.0).overhead), quasi interamente dovuto all'allocazione e alla copia iniziale dei buffer su device, costi assenti negli altri due backend.

== Esecuzione pura in funzione della densità

L'esecuzione pura di OpenMP resta nell'ordine delle centinaia di millisecondi su tutta la carrellata (da #duration(run-at-density("openmp", 1.0).exec) a $d=1$ fino a un picco di #duration(run-at-density("openmp", 192.0).exec) a $d=192$), un ordine di grandezza sopra MPI (tra #duration(run-at-density("mpi", 1.0).exec) e #duration(run-at-density("mpi", 384.0).exec)) e CUDA (da #duration(run-at-density("cuda", 1.0).exec) a #duration(run-at-density("cuda", 384.0).exec)).

Per MPI, @eq:mpi-time prevede $T_p = Theta(|cal(E)| slash p + |cal(V)| log p)$: con $p=8$, $|cal(V)| log p = 98304$ è confrontabile con $|cal(E)| slash p$ solo per $|cal(E)| approx 786432$ ($d approx 24$, la soglia @eq:mpi-threshold). Sotto questa densità l'esecuzione dovrebbe restare piatta, sopra crescere con $|cal(E)|$. La misura resta invece piatta su *tutto* l'intervallo (a meno di un punto isolato a $d=4$, riconducibile al rumore di una run singola): la collettiva `MPI_Allreduce` pesa più del previsto rispetto alla scan locale anche alle densità alte, oppure la scan su $|cal(E)| slash p$ archi è più economica per arco di quanto il modello -- che conta operazioni, non tempo -- assuma.

Per OpenMP, @eq:omp-time prevede lo stesso tipo di pareggio a $|cal(E)| slash p = |cal(V)|$, cioè $d=p=8$ (@eq:omp-threshold). L'esecuzione misurata cresce in modo non monotono con la densità, ma resta nello stesso ordine di grandezza ($Theta(0.1"-"1)$ s) su tutta la carrellata: la fusione seriale $Theta(|cal(V)|)$ è quindi un costo fisso rilevante anche quando $|cal(E)| slash p$ lo supera.

Per CUDA, @eq:cuda-time prevede $T_p = O((|cal(E)|+|cal(V)| log |cal(V)|) slash q)$, con $|cal(V)| log_2 |cal(V)| = 491520$ confrontabile con $|cal(E)|$ attorno a $d=15$ (@eq:cuda-threshold). L'esecuzione misurata cresce di più di un ordine di grandezza, da #duration(run-at-density("cuda", 1.0).exec) a #duration(run-at-density("cuda", 384.0).exec): a differenza di MPI e OpenMP, CUDA è l'unico backend la cui esecuzione pura segue chiaramente un trend di crescita con la densità, coerente con un termine $Theta(|cal(E)| slash q)$ che non è ancora nascosto da un costo fisso comparabile.

== Speedup ed efficienza teorici

#theoretical-speedup-chart

#theoretical-efficiency-chart

I due grafici precedenti non usano dati misurati: sono @eq:mpi-speedup, @eq:omp-speedup e @eq:cuda-sm-speedup valutate alle densità della carrellata, con $|cal(V)|=32768$, $p=8$ (CUDA: $q=142$ SM, NVIDIA L40S). A $d=24$ lo speedup teorico MPI è $S_p approx #calc.round(theoretical-speedup(run-at-density("mpi", 24.0)), digits: 2)$ (efficienza $approx$ #calc.round(theoretical-efficiency(run-at-density("mpi", 24.0)) * 100, digits: 0)%, vicino a $E_("min")=1/2$); a $d=8$ OpenMP raggiunge $S_p approx #calc.round(theoretical-speedup((vertices: 32768, edges: 8 * 32768, density: 8.0, backend: "openmp", resources: 8)), digits: 2)$. CUDA va da $S_p approx #calc.round(theoretical-speedup(run-at-density("cuda", 1.0)), digits: 1)$ a $d=1$ a $S_p approx #calc.round(theoretical-speedup(run-at-density("cuda", 384.0)), digits: 1)$ a $d=384$, saturando verso $q$ perché $E_p^("CUDA")$ tende a $1$ per $d arrow infinity$.

La distanza tra queste curve e i tempi misurati nelle sezioni precedenti è il punto centrale del capitolo: il modello prevede *quando* la scan inizia a dominare il round, ma le costanti -- una `MPI_Allreduce`, una fusione seriale OpenMP, i lanci di kernel e la contesa sugli atomici per CUDA -- restano fuori dall'analisi asintotica e determinano se quel pareggio si traduce in un tempo assoluto migliore. La sezione seguente confronta queste curve con uno speedup misurato.

== Speedup misurato vs teorico

#measured-vs-theoretical-speedup-chart <fig:measured-vs-theoretical-speedup>

A differenza dei due grafici precedenti, qui $S_p = T_s slash T_p$ usa un $T_s$ misurato: `sequential_app` (Capitolo 1) esegue la stessa Boruvka in modo seriale sullo stesso grafo, stesso seed, stessa densità di ciascun punto della carrellata. Per MPI e OpenMP $T_s$ e $T_p$ sono misurati sulla stessa partizione CPU (`ulow`), quindi il rapporto isola il solo effetto della parallelizzazione; per CUDA, invece, $T_s$ resta il tempo su un core di `ulow` mentre $T_p$ è il tempo su GPU, quindi lo speedup CUDA va letto come "quanto conviene la GPU rispetto a un core CPU", non come accelerazione a parità di hardware.

Per *MPI*, lo speedup misurato cresce con la densità seguendo la curva teorica, ma a circa metà: a $d=384$, $S_p approx #ratio(measured-speedup(run-at-density("mpi", 384.0)))$ misurato contro $S_p approx #ratio(theoretical-speedup(run-at-density("mpi", 384.0)))$ teorico (~54%). Più sorprendente è la bassa densità: per $d <= 48$, $S_p < 1$ -- MPI è più *lento* del seriale, perché l'overhead di `MPI_Allreduce` su $|cal(V)|$ candidati supera la scan locale quando questa è piccola. Il pareggio $S_p=1$ cade tra $d=48$ ($S_p approx #ratio(measured-speedup(run-at-density("mpi", 48.0)))$) e $d=96$ ($S_p approx #ratio(measured-speedup(run-at-density("mpi", 96.0)))$). Il modello idealizzato prevede il break-even molto prima -- $S_p=1$ a $d approx (log_2 p - 1) p slash (p-1) approx 2.3$ -- e già alla soglia di mezza-efficienza $d approx 24$ (@eq:mpi-threshold), dove la teoria attende $S_p approx 4$, la misura non ha ancora raggiunto il pareggio: sotto queste densità l'overhead di sincronizzazione domina.

Per *OpenMP*, lo speedup misurato resta sotto $1$ su *tutta* la carrellata: da $S_p approx #ratio(measured-speedup(run-at-density("openmp", 1.0)))$ a $d=1$ a $S_p approx #ratio(measured-speedup(run-at-density("openmp", 384.0)))$ a $d=384$, contro uno speedup teorico sempre maggiore di $1$ ($S_p in [#ratio(theoretical-speedup(run-at-density("openmp", 1.0))), #ratio(theoretical-speedup(run-at-density("openmp", 384.0)))]$, @eq:omp-speedup). In questa implementazione, parallelizzare la sola scan con OpenMP non ripaga mai il costo dei thread e della fusione seriale $Theta(|cal(V)|)$ -- per nessuna densità testata.

Per *CUDA*, lo speedup misurato è il più alto in assoluto ($S_p$ tra #ratio(measured-speedup(run-at-density("cuda", 1.0))) e #ratio(measured-speedup(run-at-density("cuda", 192.0)))), ma resta una piccola frazione di quello teorico ($S_p approx #ratio(theoretical-speedup(run-at-density("cuda", 192.0)))$ a $d=192$): la scansione $Theta(|cal(E)| slash q)$ con $q=142$ batte sempre il seriale, ma kernel launch e contesa sugli atomici impediscono di avvicinarsi al limite asintotico. Lo speedup misurato *non è monotono*: cresce fino a $d=192$ ($S_p approx #ratio(measured-speedup(run-at-density("cuda", 192.0)))$) e poi *cala* a $d=384$ ($S_p approx #ratio(measured-speedup(run-at-density("cuda", 384.0)))$), mentre il teorico continua a crescere verso $q$ -- l'unico punto in cui una curva misurata si allontana dal modello in direzione opposta a quella prevista.

= Conclusioni

I dati del Capitolo 3 e i modelli del Capitolo 2 raccontano due storie diverse, ed è proprio questo scarto a essere la conclusione principale di questo lavoro.

*MPI è l'unico backend il cui speedup misurato attraversa $S_p=1$*, ma non al punto previsto dalla teoria: come si vede in @fig:measured-vs-theoretical-speedup, il pareggio reale (tra $d=48$ e $d=96$) cade molto oltre il break-even teorico ($d approx 2.3$), e sotto MPI è più *lento* del seriale. Il costo della `MPI_Allreduce` su $|cal(V)|$ candidati domina la scan locale ben oltre il punto in cui l'analisi asintotica li dichiara comparabili -- probabilmente perché il modello conta operazioni, non byte, e la latenza fissa della collettiva pesa più di $|cal(V)| log p$. Oltre quel pareggio MPI continua ad allontanarsi da $S_p=1$, raggiungendo $S_p approx 4.07$ al punto più denso -- circa metà del valore teorico: alle densità alte MPI risulta conveniente, con margine crescente.

*OpenMP non risulta conveniente in nessun punto testato*: come mostra @fig:measured-vs-theoretical-speedup, lo speedup misurato resta sotto $1$ ovunque (massimo $S_p approx 0.45$), contro un teorico sempre $> 1$ (@eq:omp-speedup). L'esecuzione pura resta un ordine di grandezza sopra il seriale su tutta la carrellata, senza il trend di crescita previsto da @eq:omp-time oltre la soglia $d=p=8$ (@eq:omp-threshold). La fusione seriale $Theta(|cal(V)|)$ -- l'unica fase non parallelizzata -- è un costo fisso che da solo spiega gran parte del tempo, indipendentemente dagli archi processati dalla scan. Da sola, però, spiega l'assenza di accelerazione, non il rallentamento rispetto al seriale: per la sola frazione seriale Amdahl manterrebbe $S_p >= 1$, e per scendere sotto $1$ pesano i costi aggiunti dalle fasi parallele -- la contesa sull'atomic-min nell'array condiviso `cheapest[]`, la barriera di sincronizzazione a ogni round e l'overhead di gestione dei thread. Parallelizzare solo la scan non basta dunque a rendere OpenMP competitivo, e il guadagno previsto da @eq:omp-speedup per $d > 8$ resta sulla carta finché anche la fusione non viene parallelizzata.

*CUDA è l'unico backend più rapido del seriale a tutte le densità testate*, con lo speedup misurato più alto di tutti (picco $S_p approx 22.1$) -- primato da intendere a wall-clock: per efficienza il quadro si rovescia, perché quei $22.1$ sono distribuiti su $q=142$ SM (~15.5%) contro il ~51% di MPI ($4.07$ su $8$ core). CUDA è anche -- come mostra @fig:measured-vs-theoretical-speedup -- il più distante dal proprio modello, restando una piccola frazione dello speedup teorico. Alle densità basse l'overhead di allocazione e copia host$arrow.r$device (@fig:reference-breakdown) domina un'esecuzione pura ancora minima, penalizzando CUDA sul tempo *totale* -- più lento di MPI, benché mai di OpenMP -- pur avendo già lo speedup di *esecuzione* più alto: la distinzione overhead/esecuzione del Capitolo 3 è essenziale qui.

La soglia $d = log_2 |cal(V)| approx 15$ (@eq:cuda-threshold) resta un buon indicatore di dove l'esecuzione pura inizia a crescere, ma alle densità più alte lo speedup misurato *cala* mentre il teorico sale verso $q=142$: @eq:cuda-sm-speedup, modellando la scan come $Theta(|cal(E)| slash q)$, non cattura la contesa sugli atomici in `k_find_min_edges`. Ad alta densità centinaia di migliaia di archi collegano le stesse coppie di componenti, e altrettanti thread eseguono `atomicMin` sulla stessa cella di `min_edge[]`, costringendo l'hardware a serializzare gran parte degli aggiornamenti -- un collo di bottiglia crescente con la densità.

Borůvka è parallelizzabile nella pratica, ma "conviene" dipende da backend e densità, e la teoria del Capitolo 2 predice la *direzione* del pareggio ma non la *posizione*. OpenMP non risulta mai conveniente (parallelizzazione parziale insufficiente); MPI conviene solo a densità sufficientemente alte, ben oltre la soglia teorica; CUDA conviene sempre, ma usa solo una piccola frazione del proprio potenziale, e in modo non monotono alle densità alte. Lo scarto è sistematico e a senso unico, con il misurato sempre minore del teorico. Questo segno non è una proprietà garantita del modello -- un'idealizzazione che ignora le costanti non è di per sé un limite superiore allo speedup (basti pensare allo speedup superlineare indotto dalla cache) -- ma il risultato empirico osservato qui, dove le costanti reali (collettiva, fusione seriale, lanci di kernel, contesa atomica) penalizzano il parallelo più del seriale a tutte le densità testate. Il contributo dell'analisi non è quindi il segno dello scarto ma la sua entità e lo spostamento dei crossover. È coerente con modelli che colgono *quali* termini contano (scan, collettiva, fusione seriale, pointer jumping) ma non le costanti reali di questa implementazione: per quantificarlo servirebbe profilare separatamente ciascuna fase.

Resta inoltre aperta la possibilità che parte di questo scarto non sia dovuta solo ai limiti dei modelli, ma a scelte implementative non ottimali nei tre backend: lo schema ad atomici condivisi di OpenMP e CUDA, l'assenza di buffer per-thread, o pattern di comunicazione MPI non ottimizzati sono tutte aree in cui un'implementazione più matura potrebbe ridurre le costanti moltiplicative osservate -- senza cambiare le conclusioni qualitative sulla *direzione* delle soglie, ma potenzialmente spostandone la *posizione*.
