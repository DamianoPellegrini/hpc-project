#import "@local/unimib-templates:0.1.0": report-footer, unimib
#import "data.typ": *
#import "figures.typ": backend-breakdown-chart, cuda-sweep-speedup-chart, cuda-sweep-time-chart, random-total-chart

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
    Il lavoro presenta un'implementazione parallela dell'algoritmo di Boruvka per il calcolo del Minimum Spanning Tree, sviluppata e confrontata su tre backend: MPI, OpenMP e CUDA. Dopo una descrizione della struttura del progetto, vengono introdotti il funzionamento dell'algoritmo, i punti in cui il parallelismo può essere sfruttato e le scelte adottate nei tre modelli di esecuzione. La parte finale discute i risultati ottenuti sul cluster a partire dai report prodotti dalle run Slurm.
  ],
)

// #set heading(numbering: none)

#let mpi-random = run("mpi", "random")
#let openmp-random = run("openmp", "random")
#let cuda-random = run("cuda", "random")

= Capitolo 1 - Progetto e contesto sperimentale

La relazione confronta tre modelli di esecuzione applicati alla stessa idea algoritmica: l'algoritmo di Boruvka per il calcolo del Minimum Spanning Tree. I tre backend sono MPI per memoria distribuita, OpenMP per memoria condivisa e CUDA per GPU.

Il confronto usa la stessa rappresentazione del grafo, la stessa semantica degli archi candidati e lo stesso verificatore sequenziale CPU. La differenza tra i backend riguarda solo il modo in cui vengono eseguite scan degli archi, riduzione dei candidati e contrazione della DSU.

== Struttura del codice

Il modulo comune risiede in `include/mst`. Il sotto-modulo `core` definisce grafi validati, vertici, archi e pesi; `dsu` contiene le strutture Union-Find; `boruvka` contiene il verificatore sequenziale e i contratti dell'algoritmo; `app` seleziona il grafo dalle variabili d'ambiente; `reporting` produce i report JSON.

I backend sono separati in `mpi/main.cpp`, `openmp/main.cpp` e `cuda/main.cu`. Questa separazione mantiene stabile il dominio del problema e concentra le scelte parallele nei file di backend.

I concept C++20 descrivono Boruvka come contratto statico. Il concept `boruvka_round_engine` richiede un dominio di esecuzione, uno spazio di memoria, una politica di riduzione, una politica di contrazione e le operazioni del round. Il compilatore controlla quindi che un backend esponga le responsabilità richieste senza imporre una gerarchia dinamica.

== Infrastruttura di build

Il percorso locale principale usa CMake. Il preset `default` configura una build Debug con Ninja e rileva OpenMP, MPI e CUDA quando sono disponibili.

Il percorso remoto usa il `Makefile` perché gli script Slurm devono funzionare anche quando il preset CMake non è la via più stabile sul nodo assegnato. Il Makefile espone target separati per i backend e accetta compilatori espliciti tramite variabili d'ambiente.

== Workflow sul cluster

Gli script in `scripts/slurm` preparano l'ambiente, caricano i moduli, compilano il backend e scrivono un report JSON per ogni run. Il report contiene backend, grafo, tempi, risorse, risultato dell'MST e verifica sequenziale.

Le run di riferimento usano un task con quattro CPU per OpenMP, due processi per MPI e una GPU NVIDIA L40S per CUDA. Il report CUDA rileva 142 SM e 1024 thread per blocco.

Il grafo `random` usato per il confronto principale ha $n = 32768$ vertici, $m = 229375$ archi, seed 886261 e peso massimo 10000. La densità del grafo è $m / n approx 7$.

= Capitolo 2 - Analisi teorica di Boruvka parallelo

Questo capitolo deriva il modello dall'algoritmo. I risultati sperimentali non vengono usati per definire lavoro, overhead, speedup o soglie.

== Dinamica sequenziale

Boruvka mantiene una partizione dei vertici in componenti connesse. Ogni round sceglie, per ciascuna componente, l'arco uscente di peso minimo e poi contrae le componenti collegate dagli archi scelti.

Il round contiene tre fasi. La scan visita gli archi e calcola i candidati. La reduce seleziona un candidato per componente. La contract aggiorna la DSU e ammette gli archi nell'MST.

```text
while component_count > 1:
  best[component] = none
  for edge in edges:
    ru = find(edge.u)
    rv = find(edge.v)
    if ru != rv:
      best[ru] = min(best[ru], edge)
      best[rv] = min(best[rv], edge)
  for candidate in best:
    if candidate exists and unite(candidate):
      add candidate to MST
```

Il costo di un round sequenziale è $Theta(m + n)$, perché la scan visita $m$ archi e la gestione dei candidati visita al più $n$ componenti. Per un grafo connesso vale $m >= n - 1$, quindi il termine dominante è $m$.

$
  W = T_s = Theta(r dot m)
$ <eq:seq-work>

La formula @eq:seq-work usa $r$ come numero di round. Per Boruvka il numero di componenti diminuisce di un fattore costante per round, quindi $r = O(log n)$.

== Parallelismo disponibile

La scan è data-parallel sugli archi. Dato uno snapshot dei rappresentanti, la valutazione di un arco non dipende dalla valutazione degli altri archi.

La reduce ha parallelismo per componente. Ogni worker può produrre una tabella locale di candidati di dimensione $n$, ma le $p$ tabelle devono essere combinate per ottenere un candidato globale per componente.

La contract ha parallelismo più debole. Le fusioni modificano la DSU, quindi due candidati possono competere sugli stessi rappresentanti e richiedere sincronizzazione.

Le tre fasi hanno qualità di parallelismo diversa. La scan scala con $m$, la reduce scala con $n$ e $p$, la contract scala con la contesa sulla DSU.

== Modello di costo

Il modello segue il formalismo di Kumar, capitolo 5. Il lavoro è il tempo della migliore esecuzione sequenziale dello stesso algoritmo.

$
  W = T_s
$ <eq:work-def>

L'overhead parallelo misura la differenza tra il costo processore-tempo e il lavoro sequenziale.

$
  T_o = p dot T_p - T_s
$ <eq:overhead-def>

Lo speedup e l'efficienza si ottengono da @eq:work-def e @eq:overhead-def.

$
  S_p = frac(T_s, T_p)
$ <eq:speedup-def>

$
  E_p = frac(S_p, p) = frac(T_s, p dot T_p) = frac(W, W + T_o)
$ <eq:efficiency-def>

Un algoritmo parallelo è cost-optimal quando il costo processore-tempo è dello stesso ordine del lavoro sequenziale.

$
  p dot T_p = Theta(W) <=> T_o = O(W)
$ <eq:cost-optimality>

L'isoefficienza fissa un valore minimo di efficienza e ricava quanto deve crescere il lavoro per mantenere quel valore.

$
  W = K dot T_o quad "con" quad K = frac(E_("min"), 1 - E_("min"))
$ <eq:isoefficiency>

La soglia operativa scelta è $E_("min") = 1 / 2$. Da @eq:isoefficiency segue $K = 1$, quindi il lavoro richiesto coincide con l'overhead.

$
  E_("min") = frac(1, 2) => W = T_o
$ <eq:half-efficiency>

== Modello per backend

=== MPI

MPI distribuisce staticamente gli archi tra $p$ processi. Ogni processo esegue la scan su $m / p$ archi.

La riduzione usa una `MPI_Allreduce` su $n$ chiavi. Nel modello alpha-beta il costo è $Theta(alpha dot log p + beta dot n dot log p)$, dove $alpha$ è la latenza e $beta$ è il costo per parola.

La contract viene applicata da tutti i processi sulla stessa lista ridotta. Il costo conservativo della fase è $Theta(n)$, perché la lista dei candidati ha una posizione per componente.

Trascurando $alpha dot log p$ per $n$ grande, il tempo parallelo per round è:

$
  T_p^("MPI") = Theta(frac(m, p) + n dot log p)
$ <eq:mpi-time>

L'overhead totale su $r$ round deriva da @eq:overhead-def e @eq:mpi-time.

$
  T_o^("MPI") = Theta(r dot p dot n dot log p)
$ <eq:mpi-overhead>

La cost-optimality richiede che @eq:mpi-overhead sia asintoticamente dominato da @eq:seq-work.

$
  m = Omega(p dot n dot log p)
$ <eq:mpi-cost-optimality>

L'isoefficienza corrispondente è:

$
  W = Theta(p dot n dot log p)
$ <eq:mpi-isoefficiency>

Con la soglia operativa @eq:half-efficiency, la densità richiesta per MPI è:

$
  frac(m, n) >= p dot log p
$ <eq:mpi-threshold>

=== OpenMP

OpenMP usa memoria condivisa e divide la scan tra $p$ thread. Ogni thread valuta $m / p$ archi e scrive il candidato in uno slot locale di dimensione $n$.

La riduzione ottimale dei candidati può essere organizzata come albero sui $p$ thread. Il tempo della reduce è $Theta(n dot log p / p)$, mentre il lavoro totale introdotto dai buffer locali è $Theta(p dot n)$.

La contract usa una DSU condivisa con path compression e operazioni atomiche. Il costo conservativo per round è $Theta(n)$, perché la fase visita i candidati delle componenti.

Il tempo parallelo per round è:

$
  T_p^("OMP") = Theta(frac(m, p) + n)
$ <eq:omp-time>

L'overhead totale su $r$ round è:

$
  T_o^("OMP") = Theta(r dot p dot n)
$ <eq:omp-overhead>

La cost-optimality richiede:

$
  m = Omega(p dot n)
$ <eq:omp-cost-optimality>

L'isoefficienza di OpenMP è:

$
  W = Theta(p dot n)
$ <eq:omp-isoefficiency>

Con la soglia @eq:half-efficiency, la densità richiesta diventa:

$
  frac(m, n) >= p
$ <eq:omp-threshold>

OpenMP elimina il fattore $log p$ della collettiva MPI. Il confronto tra @eq:mpi-threshold e @eq:omp-threshold prevede quindi una soglia asintotica migliore per la memoria condivisa.

=== CUDA

CUDA associa un thread logico a ogni arco nella fase di scan, quindi per questa fase vale $p = m$. Il kernel `scan_edges_kernel` valuta gli archi in parallelo, calcola i rappresentanti delle estremità e aggiorna la tabella globale `best` con una `atomicMin` diretta su una chiave `uint64_t` impacchettata, con peso nei 32 bit alti e indice dell'arco nei 32 bit bassi. L'implementazione non usa un loop CAS e non introduce retry espliciti.

Su una distribuzione random degli archi tra componenti, l'aggiornamento atomico ha costo atteso $O(1)$ ammortizzato per thread. La scan ha quindi tempo parallelo:

$
  T_("scan")^("CUDA") = Theta(1)
$ <eq:cuda-scan-time>

La funzione `find_root_device` usa path splitting con CAS. Il costo ammortizzato della ricerca è $O(log^* n)$ e viene trattato come fattore quasi costante nel modello asintotico delle fasi aggregate.

La contract è eseguita da `contract_candidates_kernel` con un thread per componente, non con un thread per arco. Se $q = min(m, "thread fisici GPU")$, il costo parallelo della contract è:

$
  T_("contract")^("CUDA") = Theta(frac(n, q))
$ <eq:cuda-contract-time>

La forma conservativa di @eq:cuda-contract-time è $Theta(n)$, perché il kernel visita al più una posizione per componente. La probabilità di race su `unite_device` resta bassa nel modello random, dato che la contract opera sui candidati per componente e non su tutti gli archi.

Combinando @eq:cuda-scan-time e @eq:cuda-contract-time con $p = m$, il tempo parallelo per round è:

$
  T_p^("CUDA") = Theta(frac(n, m) + 1)
$ <eq:cuda-time>

Da @eq:cuda-time segue $T_p^("CUDA") = Theta(1)$ per $m >= n$. Per $m < n$, il termine dominante diventa $Theta(n / m)$.

L'overhead della scan si cancella rispetto al lavoro sequenziale sugli archi. Il termine residuo è la contract sulle componenti:

$
  T_o^("CUDA") = (Theta(m dot 1) - Theta(m)) + Theta(n) = Theta(n)
$ <eq:cuda-overhead>

La cost-optimality segue da @eq:cuda-overhead e dal fatto che, per un grafo connesso, $n = O(m)$:

$
  T_o^("CUDA") = Theta(n) = O(frac(W, r)) = O(m)
$ <eq:cuda-cost-optimality>

L'isoefficienza CUDA coincide con il lower bound teorico delle slide basate su Kumar, capitolo 5:

$
  W = Theta(p)
$ <eq:cuda-isoefficiency>

Su grafi con componenti ad altissimo grado, per esempio una stella, molti thread possono aggiornare la stessa cella di `best` e la contesa su `atomicMin` può degradare la scan. Il modello di @eq:cuda-scan-time descrive il comportamento atteso su grafi random, dove la distribuzione degli archi tra componenti mantiene il costo ammortizzato per thread pari a $O(1)$.

I costi fissi di setup sono $Theta(n + m)$ una tantum, mentre i lanci dei kernel sono $Theta(1)$ per round. Questi termini non cambiano @eq:cuda-isoefficiency, ma introducono una costante moltiplicativa rilevante.

Il modello CUDA è il più favorevole per grafi densi, cioè per $m >> n$. In quel regime la scan espone abbastanza parallelismo da ammortizzare la contract su $n$ componenti.

== Confronto dei modelli

#figure(
  table(
    columns: (1fr, 1.35fr, 1.35fr, 1.25fr),
    align: (left, left, left, left),
    table.header([*Backend*], [*$T_o$*], [*Isoefficienza*], [*Soglia operativa*]),
    [MPI], [$Theta(r dot p dot n dot log p)$], [$W = Theta(p dot n dot log p)$], [$m / n >= p dot log p$],
    [OpenMP], [$Theta(r dot p dot n)$], [$W = Theta(p dot n)$], [$m / n >= p$],
    [CUDA], [$Theta(n)$], [$W = Theta(p)$], [lower bound teorico],
  ),
  caption: [Modelli teorici dei tre backend.],
) <tab:theory-summary>

Nota. Per CUDA vale $p = m$, quindi la soglia non è confrontabile direttamente con MPI e OpenMP. Il modello usa l'implementazione concreta con `atomicMin` diretta su chiavi `uint64_t`, non un'ipotesi ottimistica priva di contesa.

La tabella @tab:theory-summary riassume il ruolo delle fasi non-scan. MPI paga una riduzione collettiva su $n$ chiavi, OpenMP paga strutture locali in memoria condivisa, CUDA paga atomiche e contract su $n$ componenti.

La scan è il collo di bottiglia comune perché visita tutti gli archi. Reduce e contract determinano la scalabilità perché hanno struttura diversa nei tre modelli.

= Capitolo 3 - Implementazioni

Il codice implementa lo stesso round astratto nei tre backend. Ogni backend realizza scan, reduce e contract con primitive diverse e si discosta dal modello ideale in punti diversi.

== MPI

Il backend MPI usa una distribuzione statica degli archi. Il processo di rango $i$ riceve l'intervallo $[m i / p, m (i + 1) / p)$ e calcola i candidati locali per le componenti incontrate.

La reduce impacchetta ogni candidato in una chiave ordinabile. La chiave contiene peso e indice dell'arco, quindi una `MPI_Allreduce` con operazione `MIN` seleziona lo stesso candidato globale su tutti i processi.

La contract viene eseguita da ogni processo sulla lista ridotta. Tutti i processi applicano le stesse fusioni alla DSU locale e mantengono lo stesso stato logico dopo il round.

Lo scostamento dall'ottimo teorico nasce dalla collettiva. Anche con pochi processi, la `MPI_Allreduce` comunica $n$ chiavi per round e introduce il termine $n dot log p$ di @eq:mpi-time.

== OpenMP

Il backend OpenMP usa memoria condivisa. I thread dividono la scan con un ciclo parallelo sugli archi e leggono uno snapshot dei rappresentanti delle componenti.

La versione ottimale del backend mantiene un buffer per-thread di dimensione $n$ fuori dal loop MST. A ogni round il buffer viene azzerato e poi riusato, evitando una riallocazione completa nel percorso critico.

La reduce combina i candidati locali per componente. Una riduzione parallela ad albero raggiunge il costo teorico di @eq:omp-time, mentre una scansione seriale dei buffer aumenterebbe le costanti senza modificare l'ordine asintotico del lavoro totale.

La contract usa una DSU parallela con operazioni atomiche di tipo compare-and-swap. Le atomiche impediscono fusioni incoerenti, ma trasformano la contract nella fase meno regolare del backend.

OpenMP si discosta dall'ideale quando la contract incontra contesa sulla DSU o quando il buffer locale viene gestito nel corpo del round. Il modello @eq:omp-overhead resta valido se il costo dominante non diventa la gestione ripetuta della memoria.

== CUDA

Il backend CUDA mantiene su device gli archi, i parent della DSU, la tabella `best` dei candidati e i contatori del round. L'allocazione device viene trattata come costo di setup esterno alla parte ripetuta del loop MST.

La fase di init prepara parent e strutture ausiliarie. Un kernel successivo resetta lo stato del round e inizializza `best`.

La scan assegna un thread a ogni arco. Ogni thread trova i rappresentanti delle due estremità e aggiorna la tabella globale dei candidati con `atomicMin`.

La contract usa un kernel separato sui candidati. Il kernel tenta le fusioni nella DSU device e registra gli archi ammessi.

La compress esegue un ulteriore kernel sui vertici. La compressione riduce il costo dei `find` nei round successivi, ma richiede un lancio kernel aggiuntivo.

CUDA si discosta dall'ideale per tre motivi. Le atomiche concentrano contesa su componenti ad alto grado, i kernel hanno costi di lancio non nulli e il numero di thread fisici simultanei è inferiore al numero logico $m$ quando il grafo è grande.

= Capitolo 4 - Misure sperimentali

Le misure sperimentali verificano il modello del Capitolo 2 sui report prodotti dalle run Slurm. Ogni report registra backend, grafo, tempi, risorse, peso MST e verifica rispetto al verificatore sequenziale CPU.

Il grafo `random` è il punto principale del confronto tra backend. La configurazione ha $n = #mpi-random.vertices$ vertici, $m = #mpi-random.edges$ archi e densità $m / n = #calc.round(edge-density(mpi-random), digits: 2)$.

== Ambiente e configurazione

#figure(
  table(
    columns: (1.1fr, 1.2fr, 1.3fr, 1.4fr),
    align: (left, left, right, left),
    table.header([*Backend*], [*Nodo/device*], [*Risorse*], [*Job Slurm*]),
    ..backends
      .map(backend => {
        let item = run(backend, "random")
        (
          [#backend-label(backend)],
          [#platform(item)],
          [#workers(item)],
          [#item.raw.slurm_job_id],
        )
      })
      .flatten(),
  ),
  caption: [Risorse rilevate nei report per il grafo `random`.],
) <tab:run-config>

La tabella @tab:run-config conserva la configurazione di esecuzione del documento originale. OpenMP usa #workers(openmp-random), MPI usa #workers(mpi-random) e CUDA usa una #platform(cuda-random) con #workers(cuda-random).

== MPI - profilo temporale

La run MPI usa #workers(mpi-random). Sul grafo `random` il loop MST dura #duration(mpi-random.loop), mentre la baseline sequenziale CPU misurata nello stesso report dura #duration(sequential-cpu-seconds(mpi-random)).

Il compute locale massimo vale #duration(mpi-random.raw.timings.max_local_compute_seconds), cioè il #percent(mpi-random.raw.timings.max_local_compute_seconds, mpi-random.loop) del loop. La riduzione massima vale #duration(mpi-random.raw.timings.max_reduce_seconds), cioè il #percent(mpi-random.raw.timings.max_reduce_seconds, mpi-random.loop) del loop.

Il profilo MPI è coerente con @eq:mpi-overhead. La riduzione collettiva su $n$ chiavi domina il tempo del loop e rappresenta il termine $n dot log p$ di @eq:mpi-time.

La soglia teorica di @eq:mpi-threshold vale $p dot log p = 2 dot 1 = 2$ assumendo logaritmo in base due. La densità misurata è $m / n = #calc.round(edge-density(mpi-random), digits: 2)$, quindi il grafo è sopra la soglia asintotica, ma la costante della `MPI_Allreduce` mantiene il loop MPI sopra la baseline sequenziale.

#backend-breakdown-chart(mpi-random)

== OpenMP - profilo temporale

La run OpenMP usa #workers(openmp-random). Sul grafo `random` il loop MST dura #duration(openmp-random.loop), mentre la baseline sequenziale CPU misurata nello stesso report dura #duration(sequential-cpu-seconds(openmp-random)).

La scansione degli archi richiede #duration(openmp-random.raw.timings.scan_seconds), pari al #percent(openmp-random.raw.timings.scan_seconds, openmp-random.loop) del loop. Le fasi di riduzione, contrazione e compressione sommano #duration(openmp-random.raw.timings.reduce_seconds + openmp-random.raw.timings.contract_seconds + openmp-random.raw.timings.compress_seconds), pari al #percent(openmp-random.raw.timings.reduce_seconds + openmp-random.raw.timings.contract_seconds + openmp-random.raw.timings.compress_seconds, openmp-random.loop) del loop.

Il profilo OpenMP corrisponde alla previsione di @eq:omp-time. Il termine $m / p$ della scan resta la parte dominante, mentre l'overhead strutturale su $n$ candidati resta più contenuto della collettiva MPI.

La soglia teorica di @eq:omp-threshold vale $p = 4$. La densità misurata è $m / n = #calc.round(edge-density(openmp-random), digits: 2)$, quindi il grafo supera la soglia asintotica, ma lo speedup sperimentale è $T_s / T_p = #calc.round(empirical-speedup(openmp-random), digits: 2)$ e non raggiunge il pareggio.

OpenMP è il backend parallelo più vicino alla baseline sequenziale sul loop MST del grafo `random`. Il confronto con MPI segue @eq:mpi-threshold e @eq:omp-threshold, perché la memoria condivisa elimina il fattore collettivo $log p$.

#backend-breakdown-chart(openmp-random)

== CUDA - profilo temporale

La run CUDA usa una #platform(cuda-random) con #workers(cuda-random). Sul grafo `random` il loop MST dura #duration(cuda-random.loop), mentre la baseline sequenziale CPU misurata nello stesso report dura #duration(sequential-cpu-seconds(cuda-random)).

#figure(
  table(
    columns: (2.2fr, 1fr),
    align: (left, right),
    table.header([*Voce*], [*Tempo*]),
    [Tempo totale della run], [#duration(cuda-random.total)],
    [Loop MST complessivo], [#duration(cuda-random.loop)],
    [Verifica sequenziale CPU], [#duration(sequential-cpu-seconds(cuda-random))],
    [Sottosezioni CUDA strumentate], [#duration(profiled-seconds(cuda-random))],
    [Residuo del loop non attribuito], [#duration(unprofiled-mst-seconds(cuda-random))],
    [Tempo fuori dal loop MST], [#duration(setup-before-loop-seconds(cuda-random))],
  ),
  caption: [Scomposizione dei tempi CUDA sul grafo `random`.],
) <tab:cuda-timing-gap>

La tabella @tab:cuda-timing-gap separa il loop MST, la verifica sequenziale CPU e il residuo non attribuito dalla strumentazione. Le sottosezioni CUDA strumentate coprono il #percent(profiled-seconds(cuda-random), cuda-random.loop) del loop MST.

Il termine dominante del profilo CUDA è il setup device registrato nel breakdown. La scansione degli archi richiede #duration(timing-value(cuda-random, "scan_seconds")), quindi il limite osservato non è il costo marginale della scan ma l'ammortamento dei costi fissi descritti dopo @eq:cuda-isoefficiency.

Lo speedup sperimentale CUDA sul grafo `random` è $T_s / T_p = #calc.round(empirical-speedup(cuda-random), digits: 2)$. Il valore è inferiore a uno perché $m = #cuda-random.edges$ non è sufficiente ad ammortizzare setup, lanci kernel e copie.

#backend-breakdown-chart(cuda-random)

#figure(
  table(
    columns: (1fr, 0.8fr, 1fr, 1fr, 0.8fr),
    align: (right, right, right, right, right),
    table.header([*$m$*], [*$m / n$*], [*CUDA loop*], [*$T_s$ CPU*], [*$T_s / T_("CUDA")$*]),
    ..cuda-sweep-runs
      .map(item => (
        [#item.edges],
        [#calc.round(edge-density(item), digits: 2)],
        [#duration(item.loop)],
        [#duration(sequential-cpu-seconds(item))],
        [#calc.round(empirical-speedup(item), digits: 2)],
      ))
      .flatten(),
  ),
  caption: [Sweep CUDA sul grafo `random` con $n$ fissato. Il tempo CUDA è il loop MST, quindi include setup device, copie e kernel, ma non la generazione del grafo.],
) <tab:cuda-sweep>

La tabella @tab:cuda-sweep conserva la sweep CUDA con $n$ fissato e $m$ crescente. Il rapporto $T_s / T_("CUDA")$ cresce con la densità e supera il pareggio nel primo punto utile a $m = #cuda-sweep-first-crossover.edges$ archi, cioè densità #calc.round(edge-density(cuda-sweep-first-crossover), digits: 1).

La regressione lineare nella sweep non sostituisce il modello teorico del Capitolo 2. La regressione stima solo la soglia empirica di crossover, pari a circa #calc.round(cuda-threshold-edges / 1000000, digits: 2) milioni di archi e densità #calc.round(cuda-threshold-density, digits: 1) per $n = #cuda-sweep-runs.at(0).vertices$.

La soglia empirica supporta la previsione qualitativa di @eq:cuda-overhead. CUDA diventa competitivo quando $m >> n$, perché la scan espone abbastanza parallelismo da compensare i costi fissi della GPU.

#cuda-sweep-time-chart

#cuda-sweep-speedup-chart

== Confronto complessivo

#random-total-chart

#figure(
  table(
    columns: (0.9fr, 1fr, 1fr, 0.8fr),
    align: (left, right, right, right),
    table.header([*Backend*], [*Loop MST*], [*$T_s$ CPU*], [*$T_s / T_p$*]),
    ..random-runs
      .map(item => (
        [#backend-label(item.backend)],
        [#duration(item.loop)],
        [#duration(sequential-cpu-seconds(item))],
        [#calc.round(empirical-speedup(item), digits: 2)],
      ))
      .flatten(),
  ),
  caption: [Confronto sul grafo `random` tra il loop MST parallelo e la baseline sequenziale CPU misurata nello stesso report.],
) <tab:random-speedup>

#figure(
  table(
    columns: (0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.8fr, 0.8fr),
    align: (left, left, right, right, right, right),
    table.header([*Backend*], [*Grafo*], [*Vertici*], [*Archi*], [*Round*], [*Totale*]),
    ..reports
      .map(item => (
        [#backend-label(item.backend)],
        [#graph-label(item.graph)],
        [#item.vertices],
        [#item.edges],
        [#item.rounds],
        [#duration(item.total)],
      ))
      .flatten(),
  ),
  caption: [Tempi totali per tutte le combinazioni backend-grafo disponibili.],
) <tab:all-times>

#figure(
  table(
    columns: (0.9fr, 0.9fr, 1fr, 1fr, 1fr),
    align: (left, left, right, right, center),
    table.header([*Backend*], [*Grafo*], [*Peso MST*], [*Archi MST*], [*Verifica*]),
    ..reports
      .map(item => (
        [#backend-label(item.backend)],
        [#graph-label(item.graph)],
        [#item.weight],
        [#item.mst_edges],
        [#if item.verified [ok] else [fallita]],
      ))
      .flatten(),
  ),
  caption: [Verifica dei risultati rispetto al verificatore sequenziale CPU.],
) <tab:verification>

La tabella @tab:random-speedup confronta il loop MST dei tre backend con la baseline sequenziale CPU misurata nello stesso report. Nessun backend parallelo supera la baseline sul grafo `random`, perché tutti i rapporti $T_s / T_p$ sono inferiori a uno.

La tabella @tab:all-times conserva i tempi complessivi per tutte le combinazioni backend-grafo disponibili. I grafi piccoli favoriscono MPI e OpenMP rispetto a CUDA perché evitano i costi fissi di setup device.

La tabella @tab:verification conserva la verifica di correttezza. Tutte le run riportano esito `ok`, quindi il confronto temporale riguarda implementazioni che producono lo stesso MST del verificatore sequenziale CPU.

Il confronto complessivo segue il modello del Capitolo 2. MPI scala peggio di OpenMP per il termine collettivo di @eq:mpi-time, mentre CUDA si avvicina al regime favorevole solo nella sweep a densità crescente.

= Capitolo 5 - Conclusioni

Il modello teorico prevede che OpenMP abbia una soglia migliore di MPI per lo stesso numero di worker. Il confronto sul grafo `random` misura #duration(openmp-random.loop) per OpenMP e #duration(mpi-random.loop) per MPI, quindi OpenMP è circa #calc.round(mpi-random.loop / openmp-random.loop, digits: 1) volte più rapido sul loop MST.

Il modello teorico prevede che CUDA sia favorevole per $m >> n$. La sweep CUDA supera il pareggio a $m = #cuda-sweep-first-crossover.edges$ archi con $T_s / T_("CUDA") = #calc.round(empirical-speedup(cuda-sweep-first-crossover), digits: 2)$, mentre la soglia stimata dalla regressione è circa #calc.round(cuda-threshold-edges / 1000000, digits: 2) milioni di archi.

Nessun backend parallelo supera la baseline sequenziale sul grafo `random`. Questo risultato non contraddice @eq:mpi-threshold, @eq:omp-threshold o @eq:cuda-isoefficiency, perché le soglie asintotiche indicano quando l'overhead può essere ammortizzato in ordine di grandezza e non quando le costanti implementative producono speedup reale.

Per MPI, la densità $m / n = #calc.round(edge-density(mpi-random), digits: 2)$ è sopra la soglia $p dot log p = 2$, ma la `MPI_Allreduce` costa #duration(mpi-random.raw.timings.max_reduce_seconds) e rappresenta il #percent(mpi-random.raw.timings.max_reduce_seconds, mpi-random.loop) del loop. Per OpenMP, la stessa densità è sopra la soglia $p = 4$, ma lo speedup sul loop è #calc.round(empirical-speedup(openmp-random), digits: 2).

Il prossimo esperimento utile per CUDA deve ripetere i punti oltre il pareggio con più repliche. Le densità 80, 88, 96, 112 e 128 mostrano il regime di crossover, ma servono repliche per separare trend e rumore di scheduling.
