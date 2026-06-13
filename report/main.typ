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
    Il lavoro presenta tre implementazioni indipendenti dell'algoritmo di Boruvka per il calcolo del Minimum Spanning Tree -- MPI, OpenMP e CUDA -- ciascuna in un singolo file autosufficiente. Dopo una descrizione della struttura del progetto e del workflow di misura sul cluster, il Capitolo 2 deriva dal modello di Kumar lavoro, overhead, speedup ed efficienza per i tre backend. Il Capitolo 3 confronta queste previsioni con una carrellata di densità a seed fisso eseguita sul cluster. Il Capitolo 4 discute quando la parallelizzazione conviene alla luce dei risultati.
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

Il round si ripete finché resta più di una componente. Poiché ogni round dimezza (in media) il numero di componenti, il numero di round è $O(log |V|)$. La scelta dell'arco minimo per componente, con un criterio di pareggio coerente (peso, poi indice dell'arco), garantisce che l'insieme di archi scelti non formi mai un ciclo: è questa proprietà a rendere Boruvka parallelizzabile fase per fase, perché lo scan può essere eseguito su tutti gli archi contemporaneamente senza coordinamento.

== Struttura del progetto

Ciascun backend -- `src/openmp.cpp`, `src/mpi.cpp`, `src/cuda.cu` -- è implementato in un singolo file autosufficiente, che contiene la rappresentazione del grafo, il generatore di grafi casuali, l'algoritmo di Boruvka parallelo, il riferimento sequenziale (Kruskal) per la verifica e il `main` con la riga di comando. Accanto a questi, `src/sequential.cpp` ha la stessa struttura ma esegue Boruvka in versione seriale (stessa sequenza di round snapshot/scan/merge degli altri tre, senza `parallel for` né atomici): è il programma usato per misurare $T_s$, la baseline sequenziale dello speedup empirico del Capitolo 3.

Per un confronto tra modelli di esecuzione, dove l'obiettivo è mostrare come la stessa idea algoritmica si traduce in stili di parallelismo diversi (e nella sua assenza), avere ciascuno stile per intero in un solo file permette di leggere l'implementazione di un backend senza dover saltare tra moduli condivisi, isolando le scelte specifiche di MPI, OpenMP, CUDA e della versione seriale.

I quattro programmi condividono la stessa interfaccia a riga di comando, `<vertici> <archi> <seed>`: `vertici` è $|V|$, `archi` è $|E|$ (il generatore garantisce prima un albero di copertura casuale per la connessione, poi aggiunge archi casuali fino al totale richiesto) e `seed` inizializza pesi e topologia. La generazione è identica byte per byte a parità di parametri: lo stesso grafo è processato dalla versione seriale e da ciascun backend, rendendo $T_s$ e $T_p$ confrontabili -- ed è anche ciò che permette agli script Slurm di restare identici a parte binario e risorse.

Ogni programma stampa, oltre all'esito della verifica sequenziale, tre tempi: `overhead_seconds` (generazione del grafo, inizializzazione di MPI/CUDA, allocazioni, copie host$arrow.r$device), `exec_seconds` (solo il loop dell'algoritmo di Boruvka, seriale per `sequential_app`) e `total_seconds`, la loro somma. Un quarto tempo, `verify_seconds`, misura a parte la verifica con Kruskal e non entra né nell'overhead né nel totale: serve solo a controllare la correttezza.

== Infrastruttura di build

Il percorso locale usa CMake con il preset `default`, che configura una build con generatore *Ninja* e rileva OpenMP, MPI e CUDA quando disponibili: `add_executable` collega ciascun target eseguibile direttamente a uno dei quattro file in `src/`, senza sotto-directory né librerie intermedie. Il target seriale (`sequential_app`) non dipende da OpenMP/MPI/CUDA e viene quindi sempre compilato.

Il percorso cluster usa un `Makefile` scritto a mano, indipendente da CMake: i nodi di calcolo non hanno `cmake`, e un Makefile generato da CMake con generatore "Unix Makefiles" richiamerebbe comunque `cmake` per controllare lo stato della build. Il Makefile espone target separati (`make sequential`, `make openmp`, `make mpi`, `make cuda`) con compilatori configurabili da variabile d'ambiente (`CXX`, `MPICXX`, `NVCC`), e va tenuto sincronizzato a mano con `CMakeLists.txt` quando cambiano sorgenti o flag.

== Workflow sul cluster

Gli script in `scripts/slurm/` (`sequential.sh`, `openmp.sh`, `mpi.sh`, `cuda.sh`) eseguono ciascuno un job Slurm per backend. Ogni script compila il proprio target con il Makefile, poi esegue una *carrellata di densità* a seed fisso: lo stesso grafo di base ($|V| = 32768$ vertici, seed $#mpi-ref.seed$) viene rigenerato con un numero crescente di archi, in modo da coprire le densità $|E| slash |V| in {1, 2, 4, 6, 12, 24, 48, 96, 192, 384}$ -- dieci punti per backend, una run ciascuno (nessuna ripetizione). Ogni run produce una riga di un CSV con schema `backend,vertici,archi,densità,seed,risorse,overhead_seconds,exec_seconds,total_seconds,verified`.

Le risorse allocate riflettono la scelta di usare il massimo disponibile per backend (vincolo del progetto): MPI usa #workers(mpi-ref) (partizione `ulow`, un rank per CPU), OpenMP usa #workers(openmp-ref) (stessa partizione, `OMP_NUM_THREADS` pari alle CPU allocate) e CUDA usa #workers(cuda-ref) (partizione `only-one-gpu`, scelta di restare su una singola GPU per limitare la complessità implementativa). Per CUDA la dimensione di blocco non è fissata a mano: `cudaOccupancyMaxPotentialBlockSize` la sceglie a runtime sul kernel di scan, adattandosi alla GPU effettivamente assegnata dallo scheduler. Il job sequenziale (`sequential.sh`) gira invece su un solo core della partizione `ulow`.

I quattro CSV (uno per backend, uno per job Slurm) sono la fonte dati dei capitoli successivi: il Capitolo 2 deriva dal modello teorico le previsioni di overhead, speedup ed efficienza; il Capitolo 3 le confronta con questi dati misurati sulla carrellata `random`, incluso lo speedup empirico $S_p = T_s slash T_p$ con $T_s$ misurato dal job sequenziale.

= Analisi teorica di Boruvka parallelo

Questo capitolo deriva il modello dalle tre implementazioni. I numeri misurati nel Capitolo 3 servono a verificare il modello, non a definirlo: lavoro, overhead, speedup e soglie sono ricavati qui solo dalla struttura del codice.

== Dinamica sequenziale

Boruvka mantiene una partizione dei vertici in componenti connesse. Ogni round sceglie, per ciascuna componente, l'arco uscente di peso minimo e poi contrae le componenti collegate dagli archi scelti -- lo schema scan/riduzione/contrazione descritto nel Capitolo 1.

Il costo di un round sequenziale è $Theta(|E| + |V|)$: la scan visita $|E|$ archi, la gestione dei candidati e la fusione delle componenti toccano al più $|V|$ elementi. Il numero di round è $r = O(log |V|)$, perché il numero di componenti si riduce almeno di un fattore costante a ogni round.

$
  W = T_s = Theta(r dot (|E| + |V|)) = Theta(|E| dot log |V| + |V| dot log |V|)
$ <eq:seq-work>

La @eq:seq-work è il *lavoro* $W$ nel senso di Kumar: il tempo della migliore esecuzione sequenziale dello stesso algoritmo (Boruvka), non quello di un algoritmo diverso. `src/sequential.cpp` (Capitolo 1) implementa esattamente questa versione -- stessa struttura a round, senza parallelismo -- e il suo `exec_seconds` è il $T_s$ misurato usato come baseline per lo speedup empirico nel Capitolo 3. Kruskal (ordinamento degli archi più union-find, $Theta(|E| log |E|)$), presente in tutti e quattro i programmi, resta invece solo un controllo di correttezza indipendente, non la baseline di questo modello.

$
  W = T_s
$ <eq:work-def>

Per $|E| = Omega(|V|)$ -- vero per tutte le densità della carrellata, dove $|E| slash |V| >= 1$ -- il termine $|E| dot log |V|$ domina, quindi $W = Theta(|E| log |V|)$. Useremo questa forma per confrontare $W$ con l'overhead nelle sezioni seguenti.

== Modello di costo

L'overhead parallelo misura la differenza tra il costo processore-tempo e il lavoro sequenziale:

$
  T_o = p dot T_p - T_s
$ <eq:overhead-def>

Speedup ed efficienza seguono da @eq:work-def e @eq:overhead-def:

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

La soglia operativa scelta è $E_("min") = 1 / 2$, da cui $K=1$:

$
  E_("min") = frac(1, 2) => W = T_o
$ <eq:half-efficiency>

Nelle tre sezioni seguenti $T_p$ è il tempo di un round; il fattore $r$ compare identico a numeratore e denominatore di $S_p$ e si elide, quindi gli speedup di @eq:mpi-speedup, @eq:omp-speedup e @eq:cuda-sm-speedup sono espressi direttamente nei termini per round. Per overhead e isoefficienza, dove $T_o$ e $W$ sono cumulati su $r$ round, il fattore $r$ compare su entrambi i lati di @eq:half-efficiency e si elide allo stesso modo: anche le soglie sono quindi condizioni *per round*.

== Modello per backend

=== MPI

`src/mpi.cpp` distribuisce gli $|E|$ archi in blocchi contigui tra $p$ rank (`counts`/`displs`, righe 224-236): ogni rank scansiona i propri $|E| slash p$ archi e produce, in `best`, un candidato locale per ciascuna delle $|V|$ componenti che tocca (struct `CandEdge`, righe 254-264). I candidati locali vengono combinati con `MPI_Allreduce` e un operatore `MPI_Op` definito ad-hoc (`cand_min`, righe 117-123) che sceglie, chiave per chiave, il candidato di peso minore con l'id più piccolo come tie-break. L'Allreduce restituisce a ogni rank lo stesso vettore di $|V|$ candidati globali.

Da qui ogni rank esegue la *stessa* fusione union-find su una copia locale (righe 273-304): per ognuna delle al più $|V|$ componenti applica `unite`, poi rietichetta tutti i $|V|$ vertici con `comp[v] = find(comp[v])`. Questo passo è $Theta(|V|)$ per rank -- ridondante ($p$ rank lo eseguono tutti -- ma non distribuito, perché la coerenza tra le copie locali dipende dal fatto che ognuna applichi le stesse fusioni nello stesso ordine, partendo dallo stesso risultato dell'Allreduce.

La scan locale costa $Theta(|E| slash p)$. Per l'Allreduce su $|V|$ chiavi, il modello $alpha$-$beta$ di una riduzione gerarchica dà $Theta(alpha log p + beta |V| log p)$; trascurando il termine di latenza $alpha log p$ per $|V|$ grande, resta $Theta(|V| log p)$. La fusione locale, $Theta(|V|)$, è dominata da questo termine per $p > 1$. Il tempo per round è quindi:

$
  T_p^("MPI") = Theta(frac(|E|, p) + |V| log p)
$ <eq:mpi-time>

Lo speedup per round, dalla @eq:seq-work per round ($W_("round") = Theta(|E|+|V|)$):

$
  S_p^("MPI") = frac(|E|+|V|, frac(|E|, p) + |V| log p)
$ <eq:mpi-speedup>

L'overhead, da @eq:overhead-def e @eq:mpi-time:

$
  T_o^("MPI") = Theta(p dot |V| log p)
$ <eq:mpi-overhead>

Cost-optimality (@eq:cost-optimality):

$
  |E| = Omega(p dot |V| log p)
$ <eq:mpi-cost-optimality>

L'isoefficienza a $E_("min")=1/2$ (@eq:half-efficiency, $W=T_o$) dà $|E|+|V| = Theta(p |V| log p)$, e per $|E|=Omega(|V|)$:

$
  frac(|E|, |V|) >= p log p
$ <eq:mpi-threshold>

=== OpenMP

`src/openmp.cpp` non usa buffer per-thread: tutti i thread scrivono nello stesso array `cheapest[V]` di interi a 64 bit, con un atomic-min lock-free (`atomic_min_u64`, righe 59-64) che impacchetta peso e indice dell'arco -- lo stesso schema dell'atomica CUDA, ma su CPU. Per ogni round: uno `#pragma omp parallel for` su $|V|$ vertici fa lo snapshot delle radici (`comp[v] = dsu.find(v)`, righe 84-86), uno su $|V|$ resetta `cheapest[]`, e uno su $|E|$ archi esegue lo scan con gli atomic-min (righe 99-108).

La fusione, però, è *seriale*: un singolo `for` su $|V|$ slot (righe 115-127) legge `cheapest[c]`, e quando contiene un arco valido chiama `dsu.unite` (union-by-rank, senza path compression -- `find` è di sola lettura per restare sicuro durante lo scan parallelo, righe 37-41). Questo passo non è diviso tra i thread.

Le tre fasi parallele (snapshot, reset, scan) costano $Theta(|V| slash p)$, $Theta(|V| slash p)$ e $Theta(|E| slash p)$. La fusione seriale costa $Theta(|V|)$ ed è la sola fase non divisa per $p$: per $p>1$ domina le altre due. Il tempo per round è:

$
  T_p^("OMP") = Theta(frac(|E|, p) + |V|)
$ <eq:omp-time>

Lo speedup per round:

$
  S_p^("OMP") = frac(|E|+|V|, frac(|E|, p) + |V|)
$ <eq:omp-speedup>

L'overhead, da @eq:overhead-def e @eq:omp-time -- qui interamente dovuto alla fusione seriale, che gli altri $p-1$ thread attendono:

$
  T_o^("OMP") = Theta((p-1) |V|) = Theta(p |V|)
$ <eq:omp-overhead>

Cost-optimality:

$
  |E| = Omega(|V| dot p)
$ <eq:omp-cost-optimality>

Isoefficienza a $E_("min")=1/2$: $|E|+|V| = Theta(|V| p)$, da cui per $|E|=Omega(|V|)$:

$
  frac(|E|, |V|) >= p
$ <eq:omp-threshold>

A parità di $p$, @eq:omp-threshold è strutturalmente più favorevole di @eq:mpi-threshold ($p$ contro $p log p$): la fusione seriale di OpenMP costa $Theta(|V|)$ una volta sola, mentre la collettiva MPI paga $Theta(|V| log p)$ per la topologia gerarchica della riduzione. Il confronto sperimentale del Capitolo 3 verifica se questo vantaggio asintotico si traduce in tempi misurati migliori.

=== CUDA

`src/cuda.cu` non usa una struttura union-find: ogni componente mantiene un puntatore "successore" verso la componente con cui si fonde. Per round, cinque kernel assegnano un thread logico a ogni arco o a ogni componente, distribuiti dall'hardware su $q$ Streaming Multiprocessor:

- `k_reset_min` e `k_iota`: $Theta(|V| slash q)$, inizializzazione;
- `k_find_min_edges` (righe 89-101): $Theta(|E| slash q)$, scan degli archi con `atomicMin` su chiave impacchettata (peso, indice) -- stesso schema di OpenMP;
- `k_build_successor` e `k_mark_and_break` (righe 105-140): $Theta(|V| slash q)$ ciascuno; il secondo rompe gli unici cicli possibili (lunghezza 2, per il tie-break sull'indice) e marca gli archi MST;
- `k_jump` (righe 143-152): un passo di *pointer jumping* raddoppia a ogni iterazione la distanza coperta da ciascun puntatore verso la radice del proprio albero di fusione; si ripete finché nessun puntatore cambia più, poi `k_relabel` applica le nuove radici a tutti i $|V|$ vertici.

Il numero di iterazioni di pointer jumping è $O(log D)$, dove $D$ è la profondità massima degli alberi di fusione formati in un round dai link successore. Nel caso peggiore $D = O(|V|)$, ma il pointer jumping resta comunque limitato da $O(log |V|)$ iterazioni indipendentemente da $D$. Usando questo limite superiore, ogni iterazione costa $Theta(|V| slash q)$, quindi il contributo di `k_jump` è $O(|V| log |V| slash q)$. Sommando le fasi:

$
  T_p^("CUDA") = O(frac(|E| + |V| log |V|, q))
$ <eq:cuda-time>

Lo speedup per round, da @eq:seq-work per round:

$
  S_p^("CUDA") = Omega(frac(q (|E|+|V|), |E| + |V| log |V|))
$ <eq:cuda-sm-speedup>

Il termine $|V| log |V|$ introdotto dal pointer jumping *non* si cancella con $|E|+|V|$ in @eq:cuda-sm-speedup. Per $|E| >> |V| log |V|$ (grafi densi) @eq:cuda-sm-speedup tende a $q$, costante; per $|E| = Theta(|V|)$ (grafi sparsi) tende a $q slash log |V|$, più piccolo. CUDA ha quindi, in questo modello, una soglia di densità asintotica, conseguenza diretta dello schema successore + pointer jumping.

L'overhead, da @eq:overhead-def e @eq:cuda-time:

$
  T_o^("CUDA") = O(|V| log |V|)
$ <eq:cuda-overhead>

Cost-optimality:

$
  |V| log |V| = O(|E| + |V|)
$ <eq:cuda-cost-optimality>

che per $|E|=Omega(|V|)$ diventa $|V| log |V| = O(|E|)$, cioè:

$
  frac(|E|, |V|) >= log |V|
$ <eq:cuda-threshold>

A differenza di @eq:mpi-threshold e @eq:omp-threshold, @eq:cuda-threshold non dipende da $q$: il numero di SM scala il tempo assoluto ma non sposta il punto in cui $|V| log |V|$ smette di essere trascurabile rispetto a $|E|$. Questo è anche il motivo per cui l'efficienza teorica $E_p^("CUDA") = S_p^("CUDA") slash q = (d+1) slash (d + log_2 |V|)$, con $d=|E| slash |V|$, è calcolabile dai CSV senza conoscere $q$ (Capitolo 3): $q$ si cancella nel rapporto. La GPU usata per le run è una NVIDIA L40S, $q=142$ SM: questo valore permette di tradurre $E_p^("CUDA")$ in uno speedup assoluto $S_p^("CUDA") = q dot E_p^("CUDA")$ (Capitolo 3).

== Confronto dei modelli

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    table.header([*Backend*], [*$T_p$ per round*], [*$T_o$ per round*], [*Soglia operativa* ($E_("min")=1/2$)]),
    [MPI], [$Theta(|E| slash p + |V| log p)$], [$Theta(p |V| log p)$], [$frac(|E|, |V|) >= p log p$],
    [OpenMP], [$Theta(|E| slash p + |V|)$], [$Theta(p |V|)$], [$frac(|E|, |V|) >= p$],
    [CUDA], [$O((|E| + |V| log |V|) slash q)$], [$O(|V| log |V|)$], [$frac(|E|, |V|) >= log |V|$],
  ),
  caption: [Modelli teorici dei tre backend nella stessa base $|E|$, $|V|$, $p$ (processi/thread) e $q$ (SM); soglie valutate alla densità di pareggio costo-lavoro per $E_("min")=1/2$ (@eq:half-efficiency).],
) <tab:theory-summary>

Per $|V| = 32768$ e $p=8$, le tre soglie valgono $p log_2 p = 24$ per MPI, $p = 8$ per OpenMP e $log_2 |V| = 15$ per CUDA: tutte cadono nell'intervallo della carrellata ($|E| slash |V| in {1,...,384}$), e $24$ è anche uno dei punti misurati. Ciascuna soglia nasce da come il backend riporta i $|V|$ candidati a un risultato unico per componente: la collettiva MPI paga $|V| log p$ per round, la fusione seriale OpenMP $|V|$ (un fattore $log p$ in meno), il pointer jumping CUDA $|V| log |V|$ indipendentemente da $p$ o $q$. La scan $Theta(|E| slash dot)$ resta il lavoro comune inevitabile.

= Misurazione sperimentale

Le misure di questo capitolo vengono dai quattro CSV prodotti dagli script Slurm (Capitolo 1): una carrellata di densità $|E| slash |V| in {1,2,4,6,12,24,48,96,192,384}$ a $|V|=32768$ e seed $#mpi-ref.seed$ fisso, una run per punto, per ciascun backend (incluso il riferimento sequenziale). Ogni riga riporta `overhead_seconds` (generazione del grafo, init, allocazioni, copie), `exec_seconds` (solo il loop di Boruvka) e `total_seconds` = overhead + esecuzione.

== Tempi totali sulla carrellata

#total-vs-density-chart

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*Densità $|E| slash |V|$*], [*MPI*], [*OpenMP*], [*CUDA*]),
    ..swept-densities
      .map(density => (
        [#calc.round(density, digits: 0)],
        [#duration(run-at-density("mpi", density).total)],
        [#duration(run-at-density("openmp", density).total)],
        [#duration(run-at-density("cuda", density).total)],
      ))
      .flatten(),
  ),
  caption: [Tempo totale (overhead + esecuzione) misurato per ciascun punto della carrellata `random`.],
) <tab:total-by-density>

La @tab:total-by-density e il grafico precedente mostrano tre andamenti diversi. MPI è il più rapido per quasi tutta la carrellata, da poche decine di millisecondi alle densità basse a circa #duration(run-at-density("mpi", 384.0).total) a $d=384$. OpenMP è il più lento, stabilmente tra #duration(run-at-density("openmp", 1.0).total) e #duration(run-at-density("openmp", 384.0).total), senza un trend monotono netto. CUDA parte come il più lento alle densità basse (#duration(run-at-density("cuda", 1.0).total) a $d=1$) ma cresce più lentamente di OpenMP, superando leggermente MPI alle due densità più alte della carrellata ($d=192$ e $d=384$).

== Overhead ed esecuzione pura

#reference-total-chart

#reference-breakdown-stacked-chart <fig:reference-breakdown>

Alla densità di riferimento $d=#calc.round(reference-density, digits: 0)$ ($|E|=#mpi-ref.edges$), la quota di overhead sul totale è #percent(mpi-ref.overhead, mpi-ref.total) per MPI, #percent(openmp-ref.overhead, openmp-ref.total) per OpenMP e #percent(cuda-ref.overhead, cuda-ref.total) per CUDA. Per MPI e OpenMP il loop di Boruvka domina il tempo totale; per CUDA accade l'opposto -- l'esecuzione pura dura #duration(cuda-ref.exec), contro #duration(cuda-ref.overhead) di overhead.

L'overhead misurato include la generazione del grafo, comune ai tre programmi e $Theta(|E|+|V|)$ indipendentemente dal backend, più l'inizializzazione specifica (MPI_Init/broadcast, allocazioni e copie host$arrow.r$device per CUDA). Per CUDA questo secondo termine è significativo: anche al punto più sparso della carrellata ($d=1$) l'overhead è #duration(run-at-density("cuda", 1.0).overhead), quasi interamente dovuto all'allocazione e alla copia iniziale dei buffer su device, costi assenti negli altri due backend.

== Esecuzione pura in funzione della densità

L'esecuzione pura di OpenMP varia tra #duration(run-at-density("openmp", 1.0).exec) e #duration(run-at-density("openmp", 384.0).exec), un ordine di grandezza sopra MPI (tra #duration(run-at-density("mpi", 1.0).exec) e #duration(run-at-density("mpi", 384.0).exec)) e CUDA (tra #duration(run-at-density("cuda", 1.0).exec) e #duration(run-at-density("cuda", 384.0).exec)).

Per MPI, @eq:mpi-time prevede $T_p = Theta(|E| slash p + |V| log p)$: con $p=8$, $|V| log p = 98304$ è confrontabile con $|E| slash p$ solo per $|E| approx 786432$ ($d approx 24$, la soglia @eq:mpi-threshold). Sotto questa densità l'esecuzione dovrebbe restare piatta, sopra crescere con $|E|$. La misura resta invece piatta su *tutto* l'intervallo: la collettiva `MPI_Allreduce` pesa più del previsto rispetto alla scan locale anche alle densità alte, oppure la scan su $|E| slash p$ archi è più economica per arco di quanto il modello -- che conta operazioni, non tempo -- assuma.

Per OpenMP, @eq:omp-time prevede lo stesso tipo di pareggio a $|E| slash p = |V|$, cioè $d=p=8$ (@eq:omp-threshold). L'esecuzione misurata cresce in modo non monotono con la densità, ma resta nello stesso ordine di grandezza ($Theta(0.1"-"1)$ s) su tutta la carrellata: la fusione seriale $Theta(|V|)$ è quindi un costo fisso rilevante anche quando $|E| slash p$ lo supera.

Per CUDA, @eq:cuda-time prevede $T_p = O((|E|+|V| log |V|) slash q)$, con $|V| log_2 |V| = 491520$ confrontabile con $|E|$ attorno a $d=15$ (@eq:cuda-threshold). L'esecuzione misurata cresce di più di un ordine di grandezza, da #duration(run-at-density("cuda", 1.0).exec) a #duration(run-at-density("cuda", 384.0).exec): a differenza di MPI e OpenMP, CUDA è l'unico backend la cui esecuzione pura segue chiaramente un trend di crescita con la densità, coerente con un termine $Theta(|E| slash q)$ che non è ancora nascosto da un costo fisso comparabile.

== Speedup ed efficienza teorici

#theoretical-speedup-chart

#theoretical-efficiency-chart

I due grafici precedenti non usano dati misurati: sono @eq:mpi-speedup, @eq:omp-speedup e @eq:cuda-sm-speedup valutate alle densità della carrellata, con $|V|=32768$, $p=8$ (CUDA: $q=142$ SM, NVIDIA L40S). A $d=24$ lo speedup teorico MPI è $S_p approx #calc.round(theoretical-speedup(run-at-density("mpi", 24.0)), digits: 2)$ (efficienza $approx$ #calc.round(theoretical-efficiency(run-at-density("mpi", 24.0)) * 100, digits: 0)%, vicino a $E_("min")=1/2$); a $d=8$ OpenMP raggiunge $S_p approx #calc.round(theoretical-speedup((vertices: 32768, edges: 8 * 32768, density: 8.0, backend: "openmp", resources: 8)), digits: 2)$. CUDA va da $S_p approx #calc.round(theoretical-speedup(run-at-density("cuda", 1.0)), digits: 1)$ a $d=1$ a $S_p approx #calc.round(theoretical-speedup(run-at-density("cuda", 384.0)), digits: 1)$ a $d=384$, saturando verso $q$ perché $E_p^("CUDA")$ tende a $1$ per $d arrow infinity$.

La distanza tra queste curve e i tempi misurati nelle sezioni precedenti è il punto centrale del capitolo: il modello prevede *quando* la scan inizia a dominare il round, ma le costanti -- una `MPI_Allreduce`, una fusione seriale OpenMP, i lanci di kernel e la contesa sugli atomici per CUDA -- restano fuori dall'analisi asintotica e determinano se quel pareggio si traduce in un tempo assoluto migliore. La sezione seguente confronta queste curve con uno speedup misurato.

== Speedup misurato vs teorico

#measured-vs-theoretical-speedup-chart <fig:measured-vs-theoretical-speedup>

A differenza dei due grafici precedenti, qui $S_p = T_s slash T_p$ usa un $T_s$ misurato: `sequential_app` (Capitolo 1) esegue la stessa Boruvka in modo seriale sullo stesso grafo, stesso seed, stessa densità di ciascun punto della carrellata.

Per *MPI*, lo speedup misurato cresce con la densità seguendo la curva teorica, ma a circa metà: a $d=384$, $S_p approx #ratio(measured-speedup(run-at-density("mpi", 384.0)))$ misurato contro $S_p approx #ratio(theoretical-speedup(run-at-density("mpi", 384.0)))$ teorico (~54%). Più sorprendente è la bassa densità: per $d <= 48$, $S_p < 1$ -- MPI è più *lento* del seriale, perché l'overhead di `MPI_Allreduce` su $|V|$ candidati supera la scan locale quando questa è piccola. Il pareggio $S_p=1$ cade tra $d=48$ ($S_p approx #ratio(measured-speedup(run-at-density("mpi", 48.0)))$) e $d=96$ ($S_p approx #ratio(measured-speedup(run-at-density("mpi", 96.0)))$), a densità ben più alte della soglia teorica @eq:mpi-threshold ($d approx 24$): sotto, l'overhead di sincronizzazione domina.

Per *OpenMP*, lo speedup misurato resta sotto $1$ su *tutta* la carrellata: da $S_p approx #ratio(measured-speedup(run-at-density("openmp", 1.0)))$ a $d=1$ a $S_p approx #ratio(measured-speedup(run-at-density("openmp", 384.0)))$ a $d=384$, contro uno speedup teorico sempre maggiore di $1$ ($S_p in [#ratio(theoretical-speedup(run-at-density("openmp", 1.0))), #ratio(theoretical-speedup(run-at-density("openmp", 384.0)))]$, @eq:omp-speedup). In questa implementazione, parallelizzare la sola scan con OpenMP non ripaga mai il costo dei thread e della fusione seriale $Theta(|V|)$ -- per nessuna densità testata.

Per *CUDA*, lo speedup misurato è il più alto in assoluto ($S_p$ tra #ratio(measured-speedup(run-at-density("cuda", 1.0))) e #ratio(measured-speedup(run-at-density("cuda", 192.0)))), ma resta una piccola frazione di quello teorico ($S_p approx #ratio(theoretical-speedup(run-at-density("cuda", 192.0)))$ a $d=192$): la scansione $Theta(|E| slash q)$ con $q=142$ batte sempre il seriale, ma kernel launch e contesa sugli atomici impediscono di avvicinarsi al limite asintotico. Lo speedup misurato *non è monotono*: cresce fino a $d=192$ ($S_p approx #ratio(measured-speedup(run-at-density("cuda", 192.0)))$) e poi *cala* a $d=384$ ($S_p approx #ratio(measured-speedup(run-at-density("cuda", 384.0)))$), mentre il teorico continua a crescere verso $q$ -- l'unico punto in cui una curva misurata si allontana dal modello in direzione opposta a quella prevista.

= Conclusioni

I dati del Capitolo 3 e i modelli del Capitolo 2 raccontano due storie diverse, ed è proprio questo scarto a essere la conclusione principale di questo lavoro.

*MPI è l'unico backend il cui speedup misurato attraversa $S_p=1$*, ma non al punto previsto dalla teoria: come si vede in @fig:measured-vs-theoretical-speedup, il pareggio reale cade a densità ben più alte della soglia @eq:mpi-threshold ($d approx 24$), e sotto MPI è più *lento* del seriale. Il costo della `MPI_Allreduce` su $|V|$ candidati domina la scan locale ben oltre il punto in cui l'analisi asintotica li dichiara comparabili -- probabilmente perché il modello conta operazioni, non byte, e la latenza fissa della collettiva pesa più di $|V| log p$. Oltre quel pareggio MPI continua ad allontanarsi da $S_p=1$, raggiungendo $S_p approx 4.07$ al punto più denso -- circa metà del valore teorico: alle densità alte MPI risulta conveniente, con margine crescente.

*OpenMP non risulta conveniente in nessun punto testato*: come mostra @fig:measured-vs-theoretical-speedup, lo speedup misurato resta sotto $1$ ovunque (massimo $S_p approx 0.45$), contro un teorico sempre $> 1$ (@eq:omp-speedup). L'esecuzione pura resta un ordine di grandezza sopra il seriale su tutta la carrellata, senza il trend di crescita previsto da @eq:omp-time oltre la soglia $d=p=8$ (@eq:omp-threshold). La fusione seriale $Theta(|V|)$ -- l'unica fase non parallelizzata -- è un costo fisso che da solo spiega gran parte del tempo, indipendentemente dagli archi processati dalla scan: parallelizzare solo la scan non basta a rendere OpenMP competitivo, e il guadagno previsto da @eq:omp-speedup per $d > 8$ resta sulla carta finché anche la fusione non viene parallelizzata.

*CUDA è l'unico backend più rapido del seriale a tutte le densità testate*, con lo speedup misurato più alto di tutti (picco $S_p approx 22.1$), ma anche -- come mostra @fig:measured-vs-theoretical-speedup -- il più distante dal proprio modello, restando una piccola frazione dello speedup teorico. Alle densità basse l'overhead di allocazione e copia host$arrow.r$device (@fig:reference-breakdown) domina un'esecuzione pura ancora minima, rendendo CUDA il più lento sul tempo *totale* pur avendo già lo speedup di *esecuzione* più alto -- la distinzione overhead/esecuzione del Capitolo 3 è essenziale qui.

La soglia $d = log_2 |V| approx 15$ (@eq:cuda-threshold) resta un buon indicatore di dove l'esecuzione pura inizia a crescere, ma alle densità più alte lo speedup misurato *cala* mentre il teorico sale verso $q=142$: @eq:cuda-sm-speedup, modellando la scan come $Theta(|E| slash q)$, non cattura la contesa sugli atomici in `k_find_min_edges`. Ad alta densità centinaia di migliaia di archi collegano le stesse coppie di componenti, e altrettanti thread eseguono `atomicMin` sulla stessa cella di `min_edge[]`, costringendo l'hardware a serializzare gran parte degli aggiornamenti -- un collo di bottiglia crescente con la densità.

In sintesi: Borůvka è parallelizzabile nella pratica, ma "conviene" dipende da backend e densità, e la teoria del Capitolo 2 predice la *direzione* del pareggio ma non la *posizione*. OpenMP non risulta mai conveniente (parallelizzazione parziale insufficiente); MPI conviene solo a densità sufficientemente alte, ben oltre la soglia teorica; CUDA conviene sempre, ma usa solo una piccola frazione del proprio potenziale, e in modo non monotono alle densità alte. Lo scarto sistematico -- sempre nella stessa direzione, misurato sempre minore del teorico -- è coerente con modelli che colgono *quali* termini contano (scan, collettiva, fusione seriale, pointer jumping) ma non le costanti reali di questa implementazione: per verificarlo servirebbe profilare separatamente ciascuna fase.

Resta inoltre aperta la possibilità che parte di questo scarto non sia dovuta solo ai limiti dei modelli, ma a scelte implementative non ottimali nei tre backend: lo schema ad atomici condivisi di OpenMP e CUDA, l'assenza di buffer per-thread, o pattern di comunicazione MPI non ottimizzati sono tutte aree in cui un'implementazione più matura potrebbe ridurre le costanti moltiplicative osservate -- senza cambiare le conclusioni qualitative sulla *direzione* delle soglie, ma potenzialmente spostandone la *posizione*.
