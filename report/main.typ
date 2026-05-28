#import "@local/unimib-templates:0.1.0": report-footer, unimib
#import "data.typ": *
#import "figures.typ": backend-breakdown-chart, random-total-chart

#set text(lang: "it")

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

#set heading(numbering: none)

#let mpi-random = run("mpi", "random")
#let openmp-random = run("openmp", "random")
#let cuda-random = run("cuda", "random")

= Il progetto e il contesto sperimentale

Il calcolo del Minimum Spanning Tree (MST) è un problema classico della teoria dei grafi, ma diventa interessante in un contesto di calcolo parallelo perché alterna fasi molto regolari, come la scansione degli archi, a fasi più delicate, come la fusione delle componenti. In questo lavoro viene utilizzato l'algoritmo di Boruvka proprio per osservare questo equilibrio: da un lato il problema espone parallelismo naturale, dall'altro richiede strutture dati coerenti e sincronizzazioni non banali.

L'obiettivo non è solamente produrre un MST corretto, ma confrontare tre modi diversi di eseguire la stessa idea algoritmica: memoria distribuita con MPI, memoria condivisa con OpenMP e parallelismo massivo su GPU con CUDA. Per rendere il confronto più controllabile, il dominio del problema è condiviso in C++20 e ogni backend viene verificato rispetto a una implementazione sequenziale CPU.

Durante le run sul cluster ogni esecuzione produce un report con tempi, risorse usate, peso dell'MST e risultato della verifica. Questo permette di collegare il comportamento osservato alla specifica configurazione di backend, grafo e risorse assegnate da Slurm.

== Organizzazione del report

Partiamo dalla struttura del progetto e dal workflow di esecuzione, così da chiarire quali parti sono condivise e quali invece appartengono ai singoli backend. Successivamente introduciamo l'algoritmo di Boruvka nella sua forma sequenziale e discutiamo quali passaggi possono essere parallelizzati. Infine analizziamo i risultati raccolti sul cluster, confrontando MPI, OpenMP e CUDA sullo stesso insieme di grafi.

== Struttura del codice

La parte comune del progetto vive in `include/mst`. Il modulo `core` contiene i tipi del dominio, come grafi validati, vertici, pesi e archi; `dsu` raccoglie le strutture Union-Find usate per rappresentare le componenti; `boruvka` contiene il verificatore sequenziale e i contratti dell'algoritmo; `app` seleziona il grafo a partire dalle variabili d'ambiente; `reporting` produce i report di esecuzione.

I tre backend sono invece mantenuti separati in `mpi/main.cpp`, `openmp/main.cpp` e `cuda/main.cu`. Questa scelta permette di cambiare il modo in cui vengono implementate scansione, riduzione e contrazione senza cambiare la semantica condivisa dell'algoritmo. Tutti i backend ricevono lo stesso grafo e confrontano il risultato con lo stesso verificatore sequenziale.

Un punto importante dell'architettura è l'uso dei concept di C++20 per descrivere Boruvka come contratto statico, non come implementazione unica. In `mst::boruvka` il concept `boruvka_round_engine` richiede a un backend di esporre un dominio di esecuzione, uno spazio di memoria, una politica di riduzione, una politica di contrazione e le operazioni fondamentali del round: inizializzazione, ricerca dei minimi locali, riduzione dei minimi per componente, contrazione, compressione dei parent e produzione del risultato. In questo modo l'algoritmo viene definito come sequenza di responsabilità, mentre MPI, OpenMP e CUDA rimangono liberi di implementare quelle responsabilità con meccanismi paralleli diversi.

== Scelte progettuali e infrastruttura di build

L'implementazione è costruita attorno a una scomposizione semplice del round di Boruvka: prima si cerca il miglior arco uscente per ogni componente, poi si contraggono le componenti collegate dagli archi scelti, infine si comprime la struttura DSU. Questa divisione rende esplicito dove si trova il lavoro parallelo e dove invece bisogna proteggere la coerenza dello stato.

La scansione degli archi è il punto più naturale da parallelizzare, perché ogni arco può essere valutato indipendentemente una volta fissato lo snapshot dei rappresentanti delle componenti. Le fasi successive sono meno immediate: ridurre i candidati e fondere componenti richiede una strategia diversa a seconda che il modello di esecuzione sia a memoria condivisa, distribuita o su GPU.

Il build system usa CMake come percorso locale principale e un `Makefile` compatibile con l'ambiente Slurm quando CMake o Ninja non sono disponibili. I target sono separati per backend e CMake rileva automaticamente OpenMP, MPI e CUDA quando presenti.

== Workflow di esecuzione sul cluster

Le run remote sono gestite dagli script in `scripts/slurm`. Ogni script prepara l'ambiente, compila il backend richiesto e poi esegue la stessa matrice di grafi, producendo un report per ogni combinazione backend-grafo. In questo modo il risultato sperimentale rimane vicino al codice che lo ha prodotto.

OpenMP viene eseguito con un task e quattro CPU, MPI con due processi, CUDA con una GPU nella partizione dedicata. La configurazione del grafo `random` usata nei report disponibili è: #run("mpi", "random").vertices vertici, #run("mpi", "random").edges archi, seed #run("mpi", "random").raw.graph.seed e peso massimo #run("mpi", "random").raw.graph.max_weight.

= Boruvka come algoritmo parallelo

== Dinamica della versione sequenziale

Boruvka mantiene una partizione dei vertici in componenti connesse. All'inizio ogni vertice è una componente isolata. A ogni round, per ogni componente viene scelto l'arco uscente di peso minimo; gli archi scelti diventano candidati per l'MST e le componenti collegate vengono fuse. Il processo continua finché resta una sola componente oppure finché non viene più ammessa alcuna contrazione.

In forma compatta:

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

Il costo di un round è dominato dalla scansione degli archi. Poiché ogni round riduce il numero di componenti di almeno un fattore costante nel caso ideale, il numero di round rimane in genere contenuto; nei report disponibili il grafo `random` termina in #run("mpi", "random").rounds round.

== Dove nasce il parallelismo

Il primo punto utile da parallelizzare è la ricerca degli archi migliori. Ogni worker può valutare un sottoinsieme degli archi e produrre candidati locali, perché la decisione su un arco dipende solo dai rappresentanti delle due estremità nello snapshot corrente. Dopo questa fase serve però una riduzione per componente, così da ottenere un unico candidato globale per ciascuna componente.

La contrazione e la compressione della DSU possono essere anch'esse accelerate, ma sono più sensibili alla sincronizzazione. Il parallelismo diventa conveniente quando il costo della scansione degli archi domina il costo di riduzione, coordinazione e aggiornamento della DSU. Su grafi piccoli l'overhead del backend può superare il lavoro utile; su grafi più grandi la scansione diventa abbastanza ampia da ammortizzare la coordinazione.

== Modello di costo e soglia di convenienza

Indichiamo con $n = |V|$ il numero di vertici, con $m = |E|$ il numero di archi, con $p$ il numero di worker e con $r$ il numero di round. Come baseline, consideriamo prima il costo sequenziale di un round. La parte dominante è la scansione degli archi, perché per ogni arco bisogna leggere i rappresentanti delle due estremità e aggiornare, se necessario, il candidato della componente:

$ T_1^("round") approx c_e m + c_u n $

dove $c_e$ rappresenta il costo medio di valutazione di un arco e $c_u n$ raccoglie la fase di contrazione dei candidati. Su $r$ round otteniamo quindi:

$ T_1 approx r dot (c_e m + c_u n) $

Nel caso parallelo generico conviene separare tempo parallelo e work totale. In un round, la scansione continua a visitare tutti gli archi, quindi il work utile resta $O(m)$, ma il tempo ideale della scansione diventa $O(m / p)$. Ogni worker mantiene poi una tabella locale dei migliori candidati per componente: inizializzarla richiede $O(n)$ per worker, quindi $O(p n)$ work complessivo. La riduzione finale guarda i candidati dei $p$ worker per ogni componente e aggiunge un altro $O(p n)$ work. Contrazione e compressione rimangono lineari nel numero di componenti attive.

Il work parallelo di un round è quindi:

$ W_p^("round") = O(m + 2 p n + n) $

Il tempo parallelo dipende da quanto bene il backend riesce a distribuire inizializzazione e riduzione dei candidati. Nel caso ideale, questi due passaggi lineari sono divisi tra i worker e il tempo per round è:

$ T_p^("round") approx c_e frac(m, p) + c_o n + L_p $

dove $L_p$ raccoglie sincronizzazioni, comunicazione e latenze specifiche del backend. La soglia pratica usata qui nasce però dal work: il lavoro utile sugli archi deve almeno ammortizzare il lavoro aggiuntivo sui candidati. Questo significa chiedere:

$ m >= 2 p n $

La stessa condizione si può leggere per worker dividendo per $p$:

$ m / p >= 2 n $

La lettura è semplice: ogni worker deve ricevere almeno circa due archi per ogni vertice, perché nello stesso round paga due passaggi lineari sui candidati, uno per prepararli e uno per ridurli. Se il grafo sta sotto questa soglia, il parallelismo tende a fare troppo lavoro di coordinazione rispetto alla scansione degli archi.

Se invece confrontiamo direttamente il tempo parallelo ideale con la baseline sequenziale e assumiamo $L_p approx 0$ e $c_o approx 2 c_e$, otteniamo:

$ frac(m, p) + 2 n < m $

cioè:

$ m > frac(2 p n, p - 1) $

Questa condizione è una stima di speedup temporale nel caso ideale. La soglia $2 p n$ è più utile per valutare se il parallelismo è sano dal punto di vista del work: se viene rispettata, il costo extra delle strutture parallele non domina il lavoro effettivo sugli archi.

#figure(
  table(
    columns: (0.9fr, 0.7fr, 1fr, 1fr, 1fr, 0.9fr),
    align: (left, right, right, right, right, right),
    table.header([*Backend*], [*$p$*], [*$m / p$*], [*$2 n$*], [*$2 p n$*], [*$m / (2 p n)$*]),
    [MPI], [#worker-count(mpi-random)], [#calc.round(per-worker-edge-count(mpi-random), digits: 0)], [#per-worker-candidate-threshold(mpi-random)], [#cpu-cost-threshold(mpi-random)], [#calc.round(cpu-cost-ratio(mpi-random), digits: 2)],
    [OpenMP], [#worker-count(openmp-random)], [#calc.round(per-worker-edge-count(openmp-random), digits: 0)], [#per-worker-candidate-threshold(openmp-random)], [#cpu-cost-threshold(openmp-random)], [#calc.round(cpu-cost-ratio(openmp-random), digits: 2)],
  ),
  caption: [Calcolo della soglia pratica $m >= 2 p n$ sul grafo `random` per i backend CPU.]
) <tab:cost-threshold>

Nel grafo `random` disponibile, MPI supera la soglia pratica con rapporto #calc.round(cpu-cost-ratio(mpi-random), digits: 2), mentre OpenMP rimane leggermente sotto con rapporto #calc.round(cpu-cost-ratio(openmp-random), digits: 2). Questo indica che, con quattro thread, il grafo non è ancora abbastanza denso da rendere il lavoro sugli archi chiaramente dominante rispetto alle strutture per-thread. Per CUDA la stessa idea resta qualitativamente valida, ma non è corretto sostituire direttamente $p$ con il numero di SM: il parallelismo è molto più fine e il costo pratico include anche lanci kernel, sincronizzazioni e gestione della memoria device.

Lo speedup e l'efficienza rimangono definiti nel modo classico:

$ S_p = T_1 / T_p $
$ E_p = S_p / p = T_1 / (p dot T_p) $

L'overhead parallelo è:

$ T_o = p dot T_p - T_1 $

Il work sequenziale è $W_1 = T_1$. Il work complessivo consumato dai worker paralleli, dal modello sopra, è:

$ W_p = O(r dot (m + 2 p n + n)) $

L'overhead è il lavoro in più introdotto dal backend rispetto alla baseline sequenziale:

$ T_o = W_p - W_1 = O(r dot p n) $

Per un numero fissato di worker, la soglia pratica resta quindi governata dal rapporto tra $m$ e $2 p n$: sotto quella soglia il costo di coordinazione può cancellare il beneficio della scansione parallela. In termini di isoefficienza, invece, se si vuole aumentare $p$ mantenendo efficienza costante, il lavoro utile deve crescere almeno quanto l'overhead; nel modello sopra questo significa far crescere il numero di archi almeno nell'ordine di $p n$.

Nelle misure attuali manca una baseline sequenziale temporizzata. Per questo il modello usa $T_1$ come riferimento teorico e il confronto sperimentale resta relativo ai backend disponibili.

== Strategie parallele adottate

=== Distribuzione degli archi con MPI

MPI distribuisce gli archi tra processi. Ogni processo calcola i migliori candidati locali sul proprio intervallo di archi; poi una `MPI_Allreduce` con minimo sulle chiavi dei candidati produce il candidato globale per ogni componente. La contrazione viene eseguita in modo coerente da tutti i processi sulla stessa lista ridotta.

Il costo caratteristico di MPI è la riduzione globale per round. Questo backend risulta adatto quando la scansione locale degli archi è abbastanza ampia da compensare il costo della comunicazione collettiva.

=== Memoria condivisa con OpenMP

OpenMP usa memoria condivisa e divide la scansione degli archi tra thread. Per evitare aggiornamenti concorrenti diretti sulla stessa tabella `best`, ogni thread produce un array locale di candidati; una fase di riduzione combina poi i risultati locali per componente. La contrazione usa strutture locali per gli archi ammessi e una DSU parallela.

Il costo caratteristico è la gestione delle strutture per-thread e delle sincronizzazioni in memoria condivisa. La granularità del lavoro è quindi importante: su input piccoli, il costo fisso della regione parallela può dominare.

=== Esecuzione massiva su GPU con CUDA

CUDA porta su device gli archi e la DSU, poi usa kernel separati per inizializzare i candidati, scansionare gli archi, contrarre e comprimere. Il backend sfrutta molti thread GPU, ma introduce costi di setup, sincronizzazione tra kernel e copie host-device/device-host.

Il modello è vantaggioso quando il lavoro per round è sufficiente a riempire la GPU. Su grafi piccoli, i tempi fissi del lancio kernel e della gestione device possono pesare più della scansione effettiva.

== Implicazioni sui tre modelli di esecuzione

I tre backend mettono in evidenza costi diversi. MPI paga comunicazione collettiva ma può scalare su più nodi; OpenMP evita la comunicazione di rete, ma resta vincolato alla memoria condivisa e al numero di core della macchina; CUDA espone un parallelismo molto più fine, ma richiede trasferimenti e sincronizzazioni esplicite. Il confronto reale dipende quindi da dimensione del grafo, densità, numero di round, granularità della scansione e costo fisso del backend.

= Misure sperimentali sul cluster

Le misure che seguono derivano dai report prodotti dalle run sul cluster. Ogni run registra backend, grafo, tempi, risorse, peso MST e verifica rispetto alla CPU sequenziale.

== Ambiente di misura e configurazione delle run

#figure(
  table(
    columns: (1.1fr, 1.2fr, 1.3fr, 1.4fr),
    align: (left, left, right, left),
    table.header([*Backend*], [*Nodo/device*], [*Risorse*], [*Job Slurm*]),
    ..backends.map(backend => {
      let item = run(backend, "random")
      (
        [#backend-label(backend)],
        [#platform(item)],
        [#workers(item)],
        [#item.raw.slurm_job_id],
      )
    }).flatten()
  ),
  caption: [Risorse rilevate nei report per il grafo `random`.]
) <tab:run-config>

== Profilo temporale del backend MPI

La run MPI usa #workers(mpi-random). Sul grafo `random` il tempo totale è #duration(mpi-random.total). Di questo tempo, #duration(mpi-random.raw.timings.max_local_compute_seconds) sono attribuiti al compute locale massimo e #duration(mpi-random.raw.timings.max_reduce_seconds) alla riduzione massima.

#backend-breakdown-chart(mpi-random)

== Profilo temporale del backend OpenMP

La run OpenMP usa #workers(openmp-random). Sul grafo `random` il tempo totale è #duration(openmp-random.total). La scansione degli archi richiede #duration(openmp-random.raw.timings.scan_seconds), mentre riduzione, contrazione e compressione rimangono nello stesso ordine di grandezza.

#backend-breakdown-chart(openmp-random)

== Profilo temporale del backend CUDA

La run CUDA usa una #platform(cuda-random) con #workers(cuda-random). Sul grafo `random` il tempo totale è #duration(cuda-random.total), ma questo valore non è la somma delle barre del breakdown: le barre rappresentano solo le regioni esplicitamente strumentate, cioè copie host-device, kernel principali e copia finale verso host.

#figure(
  table(
    columns: (2.2fr, 1fr),
    align: (left, right),
    table.header([*Voce*], [*Tempo*]),
    [Tempo totale della run], [#duration(cuda-random.total)],
    [Loop MST complessivo], [#duration(cuda-random.loop)],
    [Sottosezioni CUDA strumentate], [#duration(profiled-seconds(cuda-random))],
    [Setup/runtime nel loop non strumentato], [#duration(unprofiled-mst-seconds(cuda-random))],
    [Tempo prima del loop MST], [#duration(setup-before-loop-seconds(cuda-random))],
  ),
  caption: [Scomposizione dei tempi CUDA sul grafo `random`.]
) <tab:cuda-timing-gap>

La differenza principale è dovuta al fatto che il timer interno non include tutta la vita del backend CUDA. In particolare, allocazioni device, inizializzazione del contesto CUDA e setup del runtime avvengono dentro il loop MST complessivo ma fuori dai timer di fase. Inoltre il tempo totale include anche la preparazione del grafo prima dell'esecuzione vera e propria dell'algoritmo.

#backend-breakdown-chart(cuda-random)

== Confronto complessivo

#random-total-chart

#figure(
  table(
    columns: (0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.8fr, 0.8fr),
    align: (left, left, right, right, right, right),
    table.header([*Backend*], [*Grafo*], [*Vertici*], [*Archi*], [*Round*], [*Totale*]),
    ..reports.map(item => (
      [#backend-label(item.backend)],
      [#graph-label(item.graph)],
      [#item.vertices],
      [#item.edges],
      [#item.rounds],
      [#duration(item.total)],
    )).flatten()
  ),
  caption: [Tempi totali per tutte le combinazioni backend-grafo disponibili.]
) <tab:all-times>

#figure(
  table(
    columns: (0.9fr, 0.9fr, 1fr, 1fr, 1fr),
    align: (left, left, right, right, center),
    table.header([*Backend*], [*Grafo*], [*Peso MST*], [*Archi MST*], [*Verifica*]),
    ..reports.map(item => (
      [#backend-label(item.backend)],
      [#graph-label(item.graph)],
      [#item.weight],
      [#item.mst_edges],
      [#if item.verified [ok] else [fallita]],
    )).flatten()
  ),
  caption: [Verifica dei risultati rispetto al verificatore sequenziale CPU.]
) <tab:verification>

Sul grafo `random`, nei report disponibili MPI è il backend più veloce, seguito da CUDA e poi da OpenMP. Questo risultato non va interpretato come una proprietà generale dell'algoritmo: misura una specifica implementazione, una specifica macchina e una sola configurazione per backend. La differenza più evidente è che OpenMP e CUDA mostrano costi fissi visibili anche sui grafi piccoli, mentre MPI ottiene tempi molto bassi già con due processi sul nodo usato.

= Considerazioni conclusive

Il progetto mostra che Boruvka si presta naturalmente alla parallelizzazione della scansione degli archi e della riduzione dei candidati. La parte più delicata non è la correttezza teorica dell'algoritmo, ma la gestione dei costi di coordinazione introdotti dai backend. MPI mette in evidenza il costo della comunicazione collettiva, OpenMP quello delle sincronizzazioni e delle strutture locali per thread, CUDA quello dei lanci kernel e della gestione della memoria device.

Per rendere l'analisi sperimentale più completa servirebbero run su scala più ampia: più processi MPI, più thread OpenMP, dimensioni crescenti per il grafo `random` e una baseline sequenziale temporizzata nello stesso formato dei report prodotti dai backend. Con questi dati sarebbe possibile calcolare speedup ed efficienza sperimentali, oltre alla sola analisi teorica e al confronto relativo tra backend.
