#import "@local/unimib-templates:0.1.0": report-footer, unimib
#import "data.typ": *
#import "figures.typ": (
  backend-breakdown-chart, backend-speedup-theory-chart, cross-backend-speedup-theory-chart,
  efficiency-isoefficiency-chart, random-breakdown-stacked-chart, random-total-chart,
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
    Il lavoro presenta un'implementazione parallela dell'algoritmo di Boruvka per il calcolo del Minimum Spanning Tree, sviluppata e confrontata su tre backend: MPI, OpenMP e CUDA. Dopo una descrizione della struttura del progetto, vengono introdotti il funzionamento dell'algoritmo, i punti in cui il parallelismo può essere sfruttato e le scelte adottate nei tre modelli di esecuzione. La parte finale discute i risultati ottenuti sul cluster a partire dai report prodotti dalle run Slurm.
  ],
)

// #set heading(numbering: none)

#let mpi-random = run("mpi", "random")
#let openmp-random = run("openmp", "random")
#let cuda-random = run("cuda", "random")
#let mpi-random-scan = timing-value(mpi-random, "max_local_compute_seconds", fallback: timing-value(
  mpi-random,
  "scan_seconds",
))
#let mpi-random-reduce = timing-value(mpi-random, "max_reduce_seconds", fallback: timing-value(
  mpi-random,
  "reduce_seconds",
))

= Capitolo 1 - Progetto e contesto sperimentale

La relazione confronta tre modelli di esecuzione applicati alla stessa idea algoritmica: l'algoritmo di Boruvka per il calcolo del Minimum Spanning Tree. I tre backend sono MPI per memoria distribuita, OpenMP per memoria condivisa e CUDA per GPU.

Il confronto usa la stessa rappresentazione del grafo, la stessa semantica degli archi candidati e lo stesso verificatore sequenziale CPU. La differenza tra i backend riguarda solo il modo in cui vengono eseguite scan degli archi, riduzione dei candidati e contrazione della DSU.

== Struttura del codice

Il modulo comune risiede in `include/mst`. Il sotto-modulo `core` definisce grafi validati, vertici, archi e pesi; `dsu` contiene le strutture Union-Find; `boruvka` contiene il verificatore sequenziale e i contratti dell'algoritmo; `app` seleziona il grafo dalle variabili d'ambiente; `reporting` produce i report JSON.

La scelta di isolare una parte comune non è solo organizzativa. Senza un dominio condiviso, ogni backend avrebbe potuto introdurre piccole differenze nella rappresentazione degli archi, nella gestione dei casi di pareggio o nella validazione del grafo; il confronto sperimentale avrebbe quindi misurato insieme il modello di esecuzione e differenze semantiche accidentali. Mantenere in comune tipi, ordinamento dei candidati, grafi di esempio, configurazione e verificatore permette invece di confrontare MPI, OpenMP e CUDA sulla stessa specifica.

Questa struttura riduce anche il costo di controllo della correttezza. I tipi forti distinguono vertici, componenti, indici di arco e pesi, evitando conversioni implicite tra quantità che hanno lo stesso tipo primitivo ma significato diverso. Il grafo validato separa l'input grezzo dallo stato che gli algoritmi possono assumere corretto: una volta superata la validazione, i backend lavorano su un oggetto che espone solo dati coerenti.

I backend sono separati in `mpi/main.cpp`, `openmp/main.cpp` e `cuda/main.cu`. Questa separazione mantiene stabile il dominio del problema e concentra le scelte parallele nei file di backend.

I concept C++20 descrivono Boruvka come contratto statico. Il concept `boruvka_round_engine` richiede un dominio di esecuzione, uno spazio di memoria, una politica di riduzione, una politica di contrazione e le operazioni del round. Il compilatore controlla quindi che un backend esponga le responsabilità richieste senza imporre una gerarchia dinamica.

L'uso dei concept è stato scelto per rendere esplicita la forma attesa di un backend senza pagare il costo concettuale di una classe base astratta. Le fasi dell'algoritmo -- scan dei candidati, riduzione, contrazione e compressione dei parent -- restano verificabili a compile time: se una nuova implementazione non espone una fase o restituisce un risultato incompatibile, l'errore emerge durante la compilazione e non come comportamento divergente a runtime. In questo senso i concept non dimostrano la correttezza dell'algoritmo, ma impediscono una classe di integrazioni incomplete o incoerenti.

I test completano questo controllo statico con regressioni eseguibili. `tests/core_tests.cpp` verifica gli invarianti dei tipi tramite `static_assert`, la generazione deterministica dei grafi, l'ordinamento stabile dei candidati, la configurazione dell'applicazione, la serializzazione dei report e il verificatore sequenziale CPU. Ogni backend confronta poi il proprio risultato con `verify_against_sequential_cpu`: l'obiettivo è separare gli errori di parallelizzazione dagli errori nella specifica comune e far fallire l'applicazione quando il peso dell'MST non coincide con il riferimento sequenziale.

== Infrastruttura di build

Il percorso locale principale usa CMake. Il preset `default` configura una build Debug con Ninja e rileva OpenMP, MPI e CUDA quando sono disponibili.

Il percorso remoto usa il `Makefile` perché gli script Slurm devono funzionare anche quando il preset CMake non è la via più stabile sul nodo assegnato. Il Makefile espone target separati per i backend e accetta compilatori espliciti tramite variabili d'ambiente.

== Workflow sul cluster

Gli script in `scripts/slurm` preparano l'ambiente, caricano i moduli, compilano il backend e scrivono un report JSON per ogni run. Il report contiene backend, grafo, tempi, risorse, risultato dell'MST e verifica sequenziale.

Le run di riferimento usano #workers(openmp-random) per OpenMP, #workers(mpi-random) per MPI e una #platform(cuda-random) per CUDA. Il report CUDA rileva #workers(cuda-random).

Il grafo `random` usato per il confronto principale ha $|V| = #mpi-random.vertices$ vertici, $|E| = #mpi-random.edges$ archi, seed 886261 e peso massimo 10000. La densità del grafo è $frac(|E|, |V|) approx #calc.round(edge-density(mpi-random), digits: 0)$.

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

Il costo di un round sequenziale è $Theta(|E| + |V|)$, perché la scan visita $|E|$ archi e la gestione dei candidati visita al più $|V|$ componenti. In questa relazione non si sostituisce automaticamente $|E| + |V|$ con $|E|$: l'approssimazione $Theta(|E|)$ è corretta solo quando si assume esplicitamente una famiglia di grafi non minimale o densa, con $|E| >> |V|$.

$
  W = T_s = Theta(r dot (|E| + |V|))
$ <eq:seq-work>

La formula @eq:seq-work usa $r$ come numero di round. Per Boruvka il numero di componenti diminuisce di un fattore costante per round, quindi $r = O(log |V|)$.

Segue da @eq:seq-work che il lavoro di un singolo round è $W_("round") = Theta(|E| + |V|)$, con $W = r dot W_("round")$: la stessa quantità si guarda "per round" o cumulata su $r$ round a seconda di cosa serve confrontare. Nelle sezioni seguenti useremo entrambe le prospettive: quella cumulata per derivare le soglie di isoefficienza (dove il fattore $r$ compare su entrambi i lati di $W = T_o$ e si elide), quella per round dove la cost-optimality emerge direttamente dal confronto tra $W_("round")$ e il costo del singolo round in parallelo.

== Parallelismo disponibile

La scan è data-parallel sugli archi. Dato uno snapshot dei rappresentanti, la valutazione di un arco non dipende dalla valutazione degli altri archi.

La reduce ha parallelismo per componente. Ogni worker può produrre una tabella locale di candidati di dimensione $|V|$, ma le $p$ tabelle devono essere combinate per ottenere un candidato globale per componente.

La contract ha parallelismo più debole. Le fusioni modificano la DSU, quindi due candidati possono competere sugli stessi rappresentanti e richiedere sincronizzazione.

Le tre fasi hanno qualità di parallelismo diversa. La scan scala con $|E|$, la reduce scala con $|V|$ e $p$, la contract scala con la contesa sulla DSU.

== Modello di costo

Il modello segue il formalismo di Kumar. Il lavoro è il tempo della migliore esecuzione sequenziale dello stesso algoritmo.

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

Le tre fasi del round si appoggiano su strutture Union-Find con garanzie di costo diverse, ed è necessario fissare un'ipotesi su questo costo prima di derivare i modelli per backend. Il verificatore sequenziale e la contract MPI usano una DSU con union-by-size e compressione completa del cammino: questa combinazione garantisce $O(alpha(|V|))$ ammortizzato per `find`/`unite`, quasi costante. Le contract di OpenMP e CUDA usano invece una DSU lock-free con linking per indice numerico, non per rango o dimensione, e compressione one-shot tramite CAS. Per questa combinazione non si assume la stessa garanzia $O(alpha(|V|))$: nel seguito si assume solo che, su grafi random a bassa contesa, il costo per operazione resti vicino alla costante. È un'ipotesi di modello esplicita, falsificabile con i contatori di retry/collisione esposti dai report, non una garanzia strutturale della struttura dati. Ogni volta che una sezione successiva tratta la contract come $Theta(|V|)$, è questa l'ipotesi su cui poggia.

== Modello per backend

=== MPI

Una realizzazione a memoria distribuita può distribuire staticamente gli $|E|$ archi tra $p$ processi: ciascuno esegue la scan sulla propria porzione e produce candidati locali per le componenti incontrate. I candidati locali vanno poi combinati in un unico candidato globale per componente -- un'operazione di riduzione collettiva di tipo minimo su chiavi che impacchettano peso e indice dell'arco è la scelta naturale, perché restituisce a tutti i processi lo stesso risultato con un solo passo di comunicazione. Calcolato il candidato globale, ogni processo applica le stesse fusioni a una propria copia della struttura union-find: l'insieme delle fusioni e il loro ordine sono identici ovunque, quindi le copie restano logicamente equivalenti senza bisogno di sincronizzare esplicitamente lo stato tra i round.

La scan locale costa $Theta(|E| slash p)$. Per la riduzione collettiva su $|V|$ chiavi, il modello $alpha$-$beta$ $Theta(alpha dot log p + beta dot |V| dot log p)$ corrisponde a una riduzione gerarchica. Non è una proprietà generale delle collettive MPI: per messaggi grandi alcune librerie possono usare schemi con un termine $beta$ asintoticamente diverso. Sulla configurazione di riferimento ($p=2$) questi schemi collassano comunque sullo stesso ordine, e le run a disposizione non permettono di distinguerli. La contract visita una lista di al più $|V|$ candidati e applica altrettante fusioni: sotto l'ipotesi di operazioni union-find quasi costanti dichiarata in apertura di questa sezione, il suo costo è $Theta(|V|)$.

Trascurando $alpha dot log p$ per $|V|$ grande, il tempo parallelo per round è:

$
  T_p^("MPI") = Theta(frac(|E|, p) + |V| dot log p)
$ <eq:mpi-time>

Il tempo parallelo totale su $r$ round è $r dot T_p^("MPI")$; poiché $T_s = Theta(r(|E|+|V|))$ (@eq:seq-work), il fattore $r$ compare identico a numeratore e denominatore dello speedup e si elide:

$
  S_p^("MPI") = frac(T_s, r dot T_p^("MPI")) = Theta(frac(|E| + |V|, frac(|E|, p) + |V| dot log p))
$ <eq:mpi-speedup>

Questa forma chiusa dipende sia dalla densità $|E| slash |V|$ sia dal numero di processi $p$ -- un punto a cui torneremo nel confronto con CUDA. L'overhead totale su $r$ round deriva da @eq:overhead-def e @eq:mpi-time:

$
  T_o^("MPI") = Theta(r dot p dot |V| dot log p)
$ <eq:mpi-overhead>

La cost-optimality richiede che @eq:mpi-overhead sia asintoticamente dominato da @eq:seq-work.

$
  |E| = Omega(p dot |V| dot log p)
$ <eq:mpi-cost-optimality>

L'isoefficienza si ottiene imponendo @eq:isoefficiency con $K=1$ (soglia @eq:half-efficiency): $W = T_o$, cioè $r(|E|+|V|) = Theta(r dot p dot |V| dot log p)$. Il fattore $r$ compare identico su entrambi i lati e si elide -- l'equilibrio tra lavoro e overhead è una proprietà *per round*, indipendente dal numero di round eseguiti:

$
  |E| + |V| = Theta(p dot |V| dot log p)
$ <eq:mpi-isoefficiency>

Per $|E| = Omega(|V|)$ -- l'assunzione di grafo non-minimale dichiarata in @eq:seq-work -- questa condizione equivale a $|E| = Theta(p dot |V| dot log p)$, da cui la soglia operativa per MPI:

$
  frac(|E|, |V|) >= p dot log p
$ <eq:mpi-threshold>

Questo modello descrive il comportamento atteso del backend MPI implementato nel progetto; i dettagli di codice che realizzano scan, riduzione e contract sono raccolti nel Capitolo 3.

=== OpenMP

Una realizzazione a memoria condivisa può dividere la scan tra $p$ thread, ciascuno dei quali valuta $|E| slash p$ archi e scrive un candidato in un buffer locale di $|V|$ slot, uno per componente. I $p$ buffer locali vanno poi combinati in un unico candidato globale per componente prima della contract: due schemi tipici sono una riduzione gerarchica ad albero sui thread, che ai $log p$ livelli combina ripetutamente $|V|$ chiavi per un costo $Theta(|V| dot log p)$, oppure una scansione piatta che, per ciascuno degli $|V|$ componenti, confronta i $p$ contributi locali e ne tiene il minimo. Il lavoro totale è $Theta(|V| dot p)$ in entrambi i casi -- ciascuna delle $|V|$ chiavi deve comunque incontrare i $p$ contributi -- ma cambia l'asse su cui si distribuisce: nello schema piatto, parallelizzando sulle $|V|$ componenti anziché sui $p$ thread, il tempo per thread scende a $Theta(|V|)$, senza il fattore $log p$ che la gerarchia introdurrebbe. Segue una contract su una struttura union-find condivisa, sincronizzata con operazioni compare-and-swap, e una fase che comprime i cammini di tutti i vertici a fine round -- il meccanismo che mantiene quasi costante il costo dei `find` nello scan del round successivo, sotto l'ipotesi dichiarata in apertura di questa sezione di modello.

Sotto lo schema a scansione piatta, scan, contract e compressione dividono per $p$ il proprio lavoro totale ($Theta(|E| slash p)$, $Theta(|V| slash p)$, $Theta(|V| slash p)$ rispettivamente). La riduzione no: il suo lavoro totale $Theta(|V| dot p)$ cresce con $p$ tanto quanto si distribuisce tra i thread, lasciando un costo $Theta(|V|)$ indipendente da $p$ -- è lei la fase non-scan dominante. Il tempo parallelo per round è quindi:

$
  T_p^("OMP") = Theta(frac(|E|, p) + |V|)
$ <eq:omp-time>

Come per MPI, il fattore $r$ si elide nello speedup:

$
  S_p^("OMP") = frac(T_s, r dot T_p^("OMP")) = Theta(frac(|E|+|V|, frac(|E|, p) + |V|))
$ <eq:omp-speedup>

L'overhead totale su $r$ round -- essenzialmente la ridondanza computazionale introdotta dalla riduzione piatta, dato che le altre fasi dividono il proprio lavoro per $p$ -- deriva da @eq:overhead-def e @eq:omp-time:

$
  T_o^("OMP") = Theta(r dot |V| dot p)
$ <eq:omp-overhead>

La cost-optimality richiede:

$
  |E| = Omega(|V| dot p)
$ <eq:omp-cost-optimality>

L'isoefficienza segue lo stesso schema di @eq:mpi-isoefficiency: imponendo $W = T_o$ con $K=1$, il fattore $r$ si elide e resta una condizione per round,

$
  |E| + |V| = Theta(|V| dot p)
$ <eq:omp-isoefficiency>

da cui, per $|E| = Omega(|V|)$, la soglia operativa per OpenMP:

$
  frac(|E|, |V|) >= p
$ <eq:omp-threshold>

Il confronto strutturale con MPI riguarda dove ciascun modello colloca il fattore $log p$. La collettiva MPI lo introduce nel costo della comunicazione gerarchica; una riduzione ad albero su memoria condivisa lo introdurrebbe allo stesso modo nel costo della combinazione dei buffer. Lo schema a scansione piatta non eredita questa gerarchia -- il suo overhead è pura ridondanza computazionale, $p$ scansioni indipendenti dello stesso spazio di $|V|$ componenti -- e la sua soglia operativa cresce più lentamente in $p$ ($p$ invece di $p log p$). A parità di densità, il modello strutturale prevede dunque per OpenMP un punto di pareggio costo-lavoro a valori di $p$ più grandi, o equivalentemente a densità più basse, di quello di MPI; il confronto sperimentale del Capitolo 4 mette alla prova questa previsione.

Questo modello descrive lo schema a buffer locali e riduzione piatta usato dal backend OpenMP; i dettagli di allocazione dei buffer, contract lock-free e compressione dei cammini sono raccolti nel Capitolo 3.

=== CUDA

Un'esecuzione SIMT assegna un thread logico a ogni elemento dell'input -- un arco nella scan, una componente nella contract -- organizzati in una griglia che la GPU schedula a ondate sui multiprocessori fisici disponibili. Il parallelismo fisico effettivo non è quindi il numero di thread logici lanciati, ma il numero $q$ di Streaming Multiprocessor: è $q$ il grado di parallelismo rilevante per il modello hardware asintotico. La scan aggiorna concorrentemente una tabella globale di candidati con un'operazione atomica di minimo su chiavi che impacchettano peso e indice dell'arco; la contract applica le fusioni a una struttura union-find lock-free sincronizzata con compare-and-swap; round successivi sono scanditi da kernel separati (preparazione, scan, contract, compressione dei cammini).

Su una distribuzione random degli archi tra le componenti, l'aggiornamento atomico ha costo atteso $O(1)$ ammortizzato per thread logico, e la scan deve comunque processare $|E|$ archi su $q$ SM:

$
  T_("scan")^("CUDA") = Theta(frac(|E|, q))
$ <eq:cuda-scan-time>

La ricerca dei rappresentanti nello scan è read-only, senza compressione, e il suo costo ammortizzato è trattato come fattore quasi costante: è la fase di compressione a fine round -- si veda sotto -- a riportare l'albero a profondità 1 prima dell'inizio del round successivo, rendendo l'ipotesi una conseguenza del progetto e non un'assunzione gratuita.

La contract assegna invece un thread logico a ciascuna componente, non a ciascun arco:

$
  T_("contract")^("CUDA") = Theta(frac(|V|, q))
$ <eq:cuda-contract-time>

La forma conservativa di @eq:cuda-contract-time è $Theta(|V|)$, perché il kernel visita al più una posizione per componente -- sotto l'ipotesi di operazioni union-find quasi costanti dichiarata in apertura di questa sezione di modello. Una stima più conservativa darebbe $Theta(|V| log |V| slash q)$, dominata comunque dal termine di scan $Theta(|E| slash q)$ quando $|E| >> |V| log |V|$ -- una condizione più stringente di "$|E| >> |V|$", ma coerente con la lettura "CUDA favorisce i grafi densi" del resto della sezione.

Combinando @eq:cuda-scan-time e @eq:cuda-contract-time, il tempo parallelo per round è:

$
  T_p^("CUDA") = Theta(frac(|E| + |V|, q))
$ <eq:cuda-time>

Il fattore $r$ si elide nello speedup esattamente come per MPI e OpenMP (@eq:mpi-speedup, @eq:omp-speedup), ma qui il termine $(|E|+|V|)$ del lavoro per round si cancella anche con quello del tempo parallelo per round, lasciando una costante:

$
  S_p^("CUDA") = frac(T_s, r dot T_p^("CUDA")) = frac(|E| + |V|, frac(|E| + |V|, q)) = q
$ <eq:cuda-sm-speedup>

indipendente dalla densità $d = |E| slash |V|$. La differenza con @eq:mpi-speedup e @eq:omp-speedup, che restano funzioni di $|E|$, $|V|$ e $p$, è strutturale: scan e contract condividono lo stesso grado di parallelismo $q$, mentre in MPI e OpenMP la scan scala con $p$ e la fase non-scan con $|V|$ (eventualmente $|V| log p$) -- nessun rapporto si cancella, e lo speedup ideale resta legato alla forma del grafo.

La ragione per cui $q$ governa entrambe le fasi è architetturale, non incidentale. Scan e contract sono qui entrambe realizzate come kernel: griglie di thread logici che l'hardware distribuisce a ondate sugli stessi $q$ multiprocessori fisici, sincronizzati da operazioni atomiche direttamente sulla struttura condivisa (`best`, la DSU device) -- il grado di parallelismo fisico è una proprietà dell'hardware, indifferente alla fase in corso e alla taglia del suo dominio logico ($|E|$ per la scan, $|V|$ per la contract). In MPI e OpenMP, invece, $p$ ha un doppio ruolo che tira in direzioni opposte: per la scan è la fonte di parallelismo (gli $|E|$ archi si dividono tra $p$ worker), ma per le fasi non-scan è anche il numero di risultati parziali da fondere -- una collettiva che muove $|V|$ chiavi su una topologia a $log p$ livelli in MPI, una fusione esplicita di $p$ copie locali di dimensione $|V|$ in OpenMP. Più worker significano più parallelismo nella scan ma anche più lavoro di fusione altrove: è questa tensione -- assente nel modello SIMT, dove le atomiche sincronizzano a livello di singolo dato anziché richiedere un passo esplicito di combinazione tra copie -- a slegare il grado di parallelismo della scan da quello che governa il costo delle altre fasi, ed è la radice ultima per cui MPI e OpenMP hanno una soglia operativa e CUDA no.

Applicando Kumar al singolo round, il lavoro sequenziale del round è $W_("round") = Theta(|E| + |V|)$ (corollario di @eq:seq-work introdotto in apertura del capitolo) e il tempo parallelo è @eq:cuda-time, da cui segue direttamente

$
  q dot T_p^("CUDA") = Theta(|E| + |V|) = Theta(W_("round"))
$ <eq:cuda-cost-optimality>

cioè il costo processore-tempo del round ha lo stesso ordine del lavoro sequenziale del round: per la definizione di cost-optimality (@eq:cost-optimality), $T_o^("CUDA") = O(W_("round"))$ -- CUDA è cost-optimal in senso strutturale, per costruzione del modello. La derivazione si ferma all'ordine di grandezza perché $Theta$ nasconde le costanti: la differenza $q dot T_p^("CUDA") - W_("round")$ tra due quantità dello stesso ordine $Theta(|E|+|V|)$ può risultare positiva, negativa o nulla a seconda di costanti che il modello asintotico non fissa. Scrivere $T_o^("CUDA") = Theta(1)$, come farebbe un'applicazione meccanica di @eq:overhead-def a questo punto, affermerebbe un overhead residuo che non si annulla mai -- l'opposto della cost-optimality appena derivata.

Non esiste quindi una soglia asintotica di densità indotta da questo overhead strutturale: la soglia pratica nasce dai termini non asintotici -- lanci di kernel, latenze delle atomiche, copie host-device, setup. I costi fissi di setup sono $Theta(|V| + |E|)$ una tantum, mentre i lanci dei kernel sono $Theta(1)$ per round; nessuno dei due cambia la cost-optimality strutturale di @eq:cuda-cost-optimality, ma entrambi introducono costanti rilevanti nelle misure reali.

Su grafi con componenti ad altissimo grado, per esempio una stella, molti thread possono aggiornare la stessa cella di `best` e la contesa su `atomicMin` può degradare la scan: il modello di @eq:cuda-scan-time descrive il comportamento atteso su grafi random, dove la distribuzione degli archi tra componenti mantiene il costo ammortizzato per thread pari a $O(1)$. Il modello SM resta in ogni caso un limite superiore -- non contiene accessi globali, throughput effettivo delle atomiche o costi di lancio kernel -- e va letto come riferimento massimo, non come previsione puntuale: il mancato raggiungimento del limite teorico non contraddice il modello strutturale, ma indica che questi costi reali non sono rappresentati dallo speedup ideale $q$.

Il modello CUDA è il più favorevole per grafi densi, cioè per $|E| >> |V|$: in quel regime la scan espone abbastanza parallelismo da ammortizzare la contract su $|V|$ componenti.

Nelle run su L40S vale $q = #worker-count(cuda-random)$, cioè il numero di multiprocessori incluso nei capabilities del report. Il Capitolo 3 collega questo modello ai kernel CUDA effettivamente usati per scan, contract e compressione.

== Confronto dei modelli

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    table.header([*Backend*], [*$T_p$ per round*], [*$T_o$ su $r$ round*], [*Soglia operativa* ($E_("min")=1/2$)]),
    [MPI], [$Theta(|E| slash p + |V| log p)$], [$Theta(r p |V| log p)$], [$frac(|E|, |V|) >= p log p$],
    [OpenMP], [$Theta(|E| slash p + |V|)$], [$Theta(r |V| p)$], [$frac(|E|, |V|) >= p$],
    [CUDA], [$Theta((|E|+|V|) slash q)$], [$O(r(|E| + |V|))$ -- cost-optimal per costruzione], [nessuna soglia asintotica],
  ),
  caption: [Modelli teorici dei tre backend, nella stessa base $|E|$ (archi), $|V|$ (vertici/componenti), $p$ (processi/thread), $q$ (SM) e $r$ (round, $=Theta(log |V|)$): le tre righe si leggono fase per fase. La soglia operativa è la traduzione pratica della condizione di isoefficienza alla soglia $E_("min") = 1 slash 2$ (@eq:half-efficiency); per CUDA il modello strutturale non genera una condizione di questo tipo -- il suo $T_o$ è scritto nella stessa scala cumulata delle altre righe, cioè su $r$ round, espandendo $W = r dot W_("round") = Theta(r(|E|+|V|))$ per renderlo confrontabile con MPI e OpenMP (si veda la discussione dopo @eq:cuda-cost-optimality).],
) <tab:theory-summary>

Nota. Per CUDA la griglia contiene un thread logico per arco, ma il modello hardware usa $q$ come numero di SM. Il modello usa l'implementazione concreta con `atomicMin` diretta su chiavi `uint64_t` e tratta il limite SM come riferimento ideale, non come previsione dei tempi misurati.

La @tab:theory-summary riassume il ruolo delle fasi non-scan. MPI paga una riduzione collettiva gerarchica su $|V|$ chiavi, OpenMP paga la ridondanza $Theta(|V| p)$ della propria riduzione piatta su strutture locali in memoria condivisa, CUDA paga atomiche e contract su $|V|$ componenti.

La scan è il lavoro inevitabile comune perché visita tutti gli archi. Reduce e contract determinano la scalabilità perché hanno struttura diversa nei tre modelli.

= Capitolo 3 - Implementazioni

Il Capitolo 2 ha definito il modello teorico dei tre backend. Questo capitolo collega quel modello alle scelte concrete di codice e mette in evidenza i punti in cui l'implementazione si discosta dall'ideale: i costi che l'analisi asintotica non rappresenta, e che il Capitolo 4 misura.

== MPI

Il rank radice carica (o genera) il grafo una sola volta e lo trasmette a tutti gli altri con `broadcast_graph` (`mpi/main.cpp:103-130`): due `MPI_Bcast` collettivi -- prima le dimensioni, poi gli archi impacchettati come triplette di interi da `pack_edges`/`unpack_edges` (`mpi/main.cpp:78-101`) -- così ogni rank costruisce e valida la propria copia a partire dalla stessa sorgente, anziché generarla in modo indipendente. È un costo fisso $Theta(|V| + |E|)$ una tantum, fuori dal loop dei round e quindi dal termine $T_p^("MPI")$ di @eq:mpi-time, nello stesso spirito del setup descritto per CUDA. Il rango $i$ riceve poi l'intervallo $[ |E| i slash p, |E| (i+1) slash p)$ tramite `edge_begin_for_rank`/`edge_end_for_rank` (`mpi/main.cpp:46-53`). Ogni processo calcola i candidati locali sulla propria porzione di archi e poi chiama `MPI_Allreduce` con operazione `MIN` su esattamente $|V|$ chiavi `uint64_t` impacchettate (`reduce_minima`, `mpi/main.cpp:197-206`). La contract usa `mst::dsu::disjoint_set<uncompressed_parents>` (`mpi/main.cpp:276`), cioè la DSU sequenziale con union-by-size e compressione completa.

Lo scostamento dall'ottimo teorico nasce dalla collettiva: anche con pochi processi, la `MPI_Allreduce` comunica $|V|$ chiavi per round e introduce il termine $|V| dot log p$ di @eq:mpi-time indipendentemente da quanto la scan locale sia rapida -- un costo fisso per round che il modello imputa alla libreria, non al codice applicativo. Nessuna comunicazione esplicita dei parent del DSU è necessaria fra un round e l'altro: la coerenza tra le copie locali emerge dal replay deterministico delle stesse fusioni, applicate da ogni rank nello stesso ordine a partire dallo stesso `MPI_Allreduce` (`apply_contractions`, `mpi/main.cpp:176-199`).

== OpenMP

La combinazione dei buffer locali segue lo schema a scansione piatta, non una riduzione ad albero: `local_best_candidate_keys` (`openmp/main.cpp:90-150`) esegue la scan parallela sugli archi e poi, per ciascuno degli $|V|$ componenti, confronta i $p$ contributi locali in `local_keys_by_thread` (`openmp/main.cpp:135-145`). I buffer sono allocati una sola volta nel costruttore di `openmp_workspace` (`openmp/main.cpp:52-64`) e azzerati a ogni round da `reset_candidates` (`openmp/main.cpp:66-73`). La contract chiama `dsu.unite` su `mst::dsu::parallel_disjoint_set` (`apply_contractions_parallel`, `openmp/main.cpp:152-206`) e `compress_all_parallel` (`openmp/main.cpp:208-218`) comprime i cammini di tutti i vertici a fine round.

OpenMP si discosta dall'ideale quando la contract incontra contesa sulle compare-and-swap della DSU condivisa, o quando la gestione dei buffer locali e la riduzione piatta pesano nel corpo del round più di quanto @eq:omp-overhead -- che astrae quei costi in un termine asintotico -- riesca a rappresentare. Il backend espone sia il contatore `dsu_contention_retries` sia i tempi di reset, merge e overhead nel blocco `backend` dei report JSON, quindi questi scostamenti sono misurabili nella campagna sperimentale.

== CUDA

CUDA mantiene su device gli archi, i parent della DSU, la tabella `best` e i contatori di round; l'allocazione e la copia iniziale sono trattate come costo di setup esterno al loop ripetuto, coerentemente con $Theta(|V|+|E|)$ una tantum. `scan_edges_kernel` (`cuda/boruvka_kernels.cuh:45-73`) assegna un thread a ogni arco, calcola i rappresentanti con `find_root_device_read_only` (`cuda/device_dsu.cuh:18-28`) e aggiorna `best` con `atomicMin` su una chiave `uint64_t` impacchettata. `contract_candidates_kernel` (`cuda/boruvka_kernels.cuh:75-96`) assegna un thread a ogni componente e applica le fusioni con `unite_device` (`cuda/device_dsu.cuh:30-44`); `compress_all_kernel` (`cuda/boruvka_kernels.cuh:98-103`) chiude il round comprimendo i cammini di tutti i vertici. La sequenza preparazione, scan, contract, compressione corrisponde a quattro lanci per round nel ciclo di `run_boruvka_on_device` (`cuda/main.cu:414-482`), costante rispetto a $|V|$ e $|E|$.

Lo scostamento dall'ideale ha tre origini: le atomiche concentrano contesa sulle componenti ad alto grado (per esempio attorno al centro di un grafo a stella), i quattro lanci kernel per round hanno costi fissi non nulli, e il numero di thread fisici simultanei resta comunque inferiore al numero logico $|E|$ quando il grafo è grande -- tutti costi che il modello SM, per costruzione, non rappresenta.

= Capitolo 4 - Misure sperimentali

Le misure sperimentali verificano il modello del Capitolo 2 sui report prodotti dalle run Slurm. Ogni report registra backend, grafo, tempi, risorse, peso MST e verifica rispetto al verificatore sequenziale CPU.

Il grafo `random` è il punto principale del confronto tra backend. La configurazione ha $|V| = #mpi-random.vertices$ vertici, $|E| = #mpi-random.edges$ archi e densità $frac(|E|, |V|) = #calc.round(edge-density(mpi-random), digits: 2)$.

== Ambiente e configurazione

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, right),
    table.header([*Backend*], [*Nodo/device*], [*Risorse*]),
    ..backends
      .map(backend => {
        let item = run(backend, "random")
        (
          [#backend-label(backend)],
          [#platform(item)],
          [#workers(item)],
        )
      })
      .flatten(),
  ),
  caption: [Risorse rilevate nei report per il grafo `random`.],
) <tab:run-config>

La @tab:run-config riporta la configurazione di esecuzione della campagna. OpenMP usa #workers(openmp-random), MPI usa #workers(mpi-random) e CUDA usa una #platform(cuda-random) con #workers(cuda-random).

== MPI - profilo temporale

La run MPI usa #workers(mpi-random). Sul grafo `random` il loop MST dura #duration(mpi-random.loop), mentre la baseline sequenziale CPU misurata nello stesso report dura #duration(sequential-cpu-seconds(mpi-random)).

La scan locale massima vale #duration(mpi-random-scan), cioè il #percent(mpi-random-scan, mpi-random.loop) del loop. La riduzione massima vale #duration(mpi-random-reduce), cioè il #percent(mpi-random-reduce, mpi-random.loop) del loop.

Il profilo MPI è coerente con @eq:mpi-overhead. La riduzione collettiva su $|V|$ chiavi domina il tempo del loop e rappresenta il termine $|V| dot log p$ di @eq:mpi-time.

La soglia teorica di @eq:mpi-threshold vale $p dot log p = 2 dot 1 = 2$ assumendo logaritmo in base due. La densità misurata è $frac(|E|, |V|) = #calc.round(edge-density(mpi-random), digits: 2)$, quindi il grafo è sopra la soglia asintotica, ma la costante della `MPI_Allreduce` mantiene il loop MPI sopra la baseline sequenziale.

#backend-breakdown-chart(mpi-random)

== OpenMP - profilo temporale

La run OpenMP usa #workers(openmp-random). Sul grafo `random` il loop MST dura #duration(openmp-random.loop), mentre la baseline sequenziale CPU misurata nello stesso report dura #duration(sequential-cpu-seconds(openmp-random)).

La scansione degli archi richiede #duration(openmp-random.raw.timings.scan_seconds), pari al #percent(openmp-random.raw.timings.scan_seconds, openmp-random.loop) del loop. Le fasi di riduzione, contrazione e compressione sommano #duration(openmp-random.raw.timings.reduce_seconds + openmp-random.raw.timings.contract_seconds + openmp-random.raw.timings.compress_seconds), pari al #percent(openmp-random.raw.timings.reduce_seconds + openmp-random.raw.timings.contract_seconds + openmp-random.raw.timings.compress_seconds, openmp-random.loop) del loop.

Il profilo OpenMP mostra che il termine $frac(|E|, p)$ della scan non domina da solo: scan, reduce, contract e compressione hanno tutti peso rilevante nel loop. Questo è coerente con @eq:omp-time quando il termine $|V|$ della riduzione e la sincronizzazione della DSU non sono trascurabili rispetto alla scan.

La soglia teorica di @eq:omp-threshold vale $p = 4$. La densità misurata è $frac(|E|, |V|) = #calc.round(edge-density(openmp-random), digits: 2)$, quindi il grafo è *sopra* questa soglia operativa. Il modello strutturale indica che, in ordine di grandezza, OpenMP dovrebbe avere già superato il punto di pareggio costo-lavoro a questa densità; lo speedup sperimentale, però, è $T_s / T_p = #calc.round(empirical-speedup(openmp-random), digits: 2)$, quindi le costanti implementative impediscono alla previsione asintotica di tradursi in speedup reale.

Sul grafo `random` OpenMP è il backend peggiore tra quelli misurati: il loop MST dura #duration(openmp-random.loop), contro #duration(mpi-random.loop) di MPI e #duration(cuda-random.loop) del backend CUDA. La discrepanza rispetto al modello ideale è attribuibile al costo della riduzione piatta, della gestione dei buffer locali per-thread e della sincronizzazione della DSU condivisa nel percorso critico; la telemetria del report separa questi termini e permette di distinguere overhead, reset, merge e retry da contesa.

#backend-breakdown-chart(openmp-random)

== CUDA - profilo temporale

La run CUDA usa una #platform(cuda-random) con #workers(cuda-random). Sul grafo `random` l'esecuzione backend dura #duration(cuda-random.loop), mentre la baseline sequenziale CPU misurata nello stesso report dura #duration(sequential-cpu-seconds(cuda-random)).

#figure(
  table(
    columns: (auto, auto),
    align: (left, right),
    table.header([*Voce*], [*Tempo*]),
    [Tempo totale della run], [#duration(cuda-random.total)],
    [Esecuzione backend CUDA], [#duration(cuda-random.loop)],
    [Algoritmo device CUDA], [#duration(algorithm-seconds(cuda-random))],
    [Verifica sequenziale CPU], [#duration(sequential-cpu-seconds(cuda-random))],
    [Sottosezioni CUDA strumentate], [#duration(profiled-seconds(cuda-random))],
    [Residuo del loop non attribuito], [#duration(unprofiled-mst-seconds(cuda-random))],
    [Tempo fuori esecuzione backend], [#duration(setup-before-loop-seconds(cuda-random))],
  ),
  caption: [Scomposizione dei tempi CUDA sul grafo `random`.],
) <tab:cuda-timing-gap>

La @tab:cuda-timing-gap separa l'esecuzione backend CUDA, il tempo algoritmo device, la verifica sequenziale CPU e il residuo non attribuito dalla strumentazione. Le sottosezioni CUDA strumentate coprono il #percent(profiled-seconds(cuda-random), cuda-random.loop) dell'esecuzione backend.

Il termine dominante del profilo CUDA è il setup device registrato nel breakdown. La scansione degli archi richiede #duration(timing-value(cuda-random, "scan_seconds")), mentre il tempo algoritmo device richiede #duration(algorithm-seconds(cuda-random)). Il limite osservato sul tempo backend resta l'ammortamento dei costi fissi descritti dopo @eq:cuda-cost-optimality.

Lo speedup sperimentale CUDA sul grafo `random`, misurato sul tempo algoritmo, è $T_s / T_p = #calc.round(empirical-speedup(cuda-random), digits: 2)$. Sul tempo backend completo il rapporto è #calc.round(backend-loop-speedup(cuda-random), digits: 2), perché $|E| = #cuda-random.edges$ deve ammortizzare anche setup, lanci kernel e copie.

#backend-breakdown-chart(cuda-random)

La run CUDA sul punto principale `random` conferma parzialmente la lettura del modello: il tempo algoritmo device è molto inferiore alla baseline sequenziale CPU, mentre il tempo backend completo resta dominato dai costi fissi di setup, copie e lanci kernel.

== Confronto complessivo

Il confronto complessivo parte dal grafo `random`, perché è l'unico punto in cui tutti i backend sono misurati sulla stessa taglia non banale e con la stessa baseline sequenziale CPU.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    table.header([*Backend*], [*Algoritmo*], [*Tempo backend*], [*Baseline CPU*], [*Speedup algoritmo*]),
    ..random-runs
      .map(item => (
        [#backend-label(item.backend)],
        [#duration(algorithm-seconds(item))],
        [#duration(item.loop)],
        [#duration(sequential-cpu-seconds(item))],
        [#calc.round(empirical-speedup(item), digits: 2)],
      ))
      .flatten(),
  ),
  caption: [Confronto sul grafo `random` tra tempo algoritmo parallelo, tempo backend e baseline sequenziale CPU misurata nello stesso report.],
) <tab:random-speedup>

La @tab:random-speedup separa due letture. Il tempo algoritmo misura il nucleo parallelo confrontabile con il modello; il tempo backend conserva invece setup, copie e overhead necessari per eseguire davvero il backend. Questa distinzione è decisiva soprattutto per CUDA, dove il tempo device è molto piccolo ma il costo completo resta dominato dal setup.

#random-total-chart

#random-breakdown-stacked-chart

Il grafico dei totali mostra il costo osservato dall'esterno, mentre il breakdown in pila separa scan, reduce, contract e residuo/overhead sul grafo `random`. In questo modo si vede subito se il loop è dominato dal lavoro sugli archi o da fasi ausiliarie legate al backend.

#backend-speedup-theory-chart("mpi")

#backend-speedup-theory-chart("openmp")

#backend-speedup-theory-chart("cuda")

Le tre figure precedenti confrontano ciascun backend con il proprio limite ideale. La figura seguente porta lo stesso confronto su un unico piano: per ciascun backend mostra lo speedup misurato e quello previsto dal modello del Capitolo 2, a parità di densità e di grado di parallelismo. Il confronto è omogeneo perché entrambe le quantità sono lo stesso rapporto $T_s slash T_p$, valutato rispettivamente sui dati e sulla formula:

#cross-backend-speedup-theory-chart

L'analisi di isoefficienza del Capitolo 2 individua, per MPI e OpenMP, come la densità $frac(|E|, |V|)$ deve scalare con $p$ perché l'efficienza resti almeno $E_("min") = 1/2$ (@tab:theory-summary). Il grafico seguente mostra l'efficienza misurata $E_p = S_p / p$ senza sovrapporre la soglia di riferimento:

#efficiency-isoefficiency-chart

#let density-cell(item) = {
  if item == none {
    [--]
  } else {
    [#calc.round(edge-density(item), digits: 0)]
  }
}

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*Backend*], [*Soglia teorica*], [*Primo speedup utile*], [*Prima mezza efficienza*]),
    [MPI],
    [#isoefficiency-threshold-label("mpi")],
    [#density-cell(random-sweep-first-speedup-crossover("mpi"))],
    [#density-cell(random-sweep-first-half-efficiency("mpi"))],

    [OpenMP],
    [#isoefficiency-threshold-label("openmp")],
    [#density-cell(random-sweep-first-speedup-crossover("openmp"))],
    [#density-cell(random-sweep-first-half-efficiency("openmp"))],

    [CUDA],
    [#isoefficiency-threshold-label("cuda")],
    [#density-cell(random-sweep-first-speedup-crossover("cuda"))],
    [#density-cell(random-sweep-first-half-efficiency("cuda"))],
  ),
  caption: [Soglie osservate nella sweep `random`. Le ultime due colonne riportano la densità $frac(|E|, |V|)$ del primo punto misurato che soddisfa la condizione indicata; la prima riporta invece la soglia teorica di @tab:theory-summary, espressa come puro numero (valutando $p log p$ o $p$ al $p$ effettivo di ciascuna run) -- le due cose non sono direttamente comparabili in valore assoluto, si veda il testo seguente.],
) <tab:random-sweep-crossovers>

Nella @tab:random-sweep-crossovers, `primo speedup utile` indica il primo punto della sweep in cui lo speedup algoritmo raggiunge almeno il pareggio con la baseline sequenziale CPU. Per esempio, un valore `2` significa che la prima densità misurata con speedup almeno pari a uno è $frac(|E|, |V|) = 2$. `Prima mezza efficienza` usa invece la soglia sperimentale $S >= p / 2$ per MPI e OpenMP; per CUDA la soglia usa il numero di SM rilevato, quindi può restare non raggiunta nei punti disponibili.

La colonna `soglia teorica` non va letta come una previsione del valore delle altre due. Le soglie di @tab:theory-summary sono leggi di scala asintotiche -- dicono come $|E| slash |V|$ deve crescere con $p$ perché l'efficienza resti almeno $1/2$, a meno di una costante moltiplicativa che $Theta$ non determina -- non valori di incrocio puntuali. Il numero riportato in tabella nasce dal sostituire il $p$ misurato nella legge di scala, ma resta un coefficiente di scala, non una stima della densità di incrocio: le ultime due colonne, al contrario, sono misure dirette su un'unica run a $p$ fissato. Il confronto sensato tra le tre colonne è quindi qualitativo (l'ordine relativo tra backend), non numerico riga per riga.

Le figure della sweep seguono quindi due letture distinte. I grafici di speedup separati per backend confrontano ogni curva misurata con la linea ideale dello stesso modello; il grafico misurato/teorico aggregato confronta direttamente i backend sullo stesso piano; il grafico di efficienza mostra il comportamento osservato a $p$ fissato. Per CUDA, la distanza dalla linea ideale basata sugli SM mostra che il parallelismo disponibile non basta da solo: atomiche, accessi ai rappresentanti e sincronizzazioni tra kernel restano nel percorso critico.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right),
    table.header([*Backend*], [*Grafo*], [*Vertici*], [*Archi*], [Round], [*Totale*]),
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

La @tab:all-times riporta i tempi complessivi per tutte le combinazioni backend-grafo disponibili. Sul grafo piccolo `dense16` non è corretto affermare che OpenMP sia favorito rispetto a CUDA: CUDA misura #duration(run("cuda", "dense16").total), mentre OpenMP misura #duration(run("openmp", "dense16").total). Il backend che beneficia in modo netto dei grafi piccoli è MPI, con #duration(run("mpi", "dense16").total), perché non paga setup hardware paragonabili a quelli GPU e non attiva una DSU condivisa con contesa tra thread.

Tutte le run incluse riportano verifica sequenziale CPU con esito `ok` nei JSON sorgente, quindi il confronto temporale riguarda implementazioni che producono lo stesso MST del verificatore sequenziale CPU.

Il confronto complessivo conferma solo in parte il modello del Capitolo 2. La riduzione collettiva limita MPI, OpenMP non riesce a trasformare la soglia asintotica favorevole in speedup reale nella run misurata, e CUDA mostra il vantaggio del parallelismo sugli archi soprattutto quando si isola il tempo algoritmo dai costi fissi del backend.

= Capitolo 5 - Conclusioni

Il confronto tra MPI, OpenMP e CUDA mostra che lo stesso algoritmo di Boruvka non ha un unico andamento prestazionale una volta fissato il modello di esecuzione. La parte comune resta la stessa: ogni round cerca il miglior arco uscente da ciascuna componente e poi contrae le componenti collegate. Cambia però il modo in cui si realizza il passaggio dalla scan locale alla decisione globale. MPI divide gli archi tra processi e paga una `MPI_Allreduce` su $|V|$ chiavi; OpenMP divide la scan tra thread ma combina buffer locali in memoria condivisa; CUDA evita una riduzione esplicita tra copie locali e usa kernel device con aggiornamenti atomici su strutture globali. Questa differenza spiega perché le equazioni del Capitolo 2 non sono tre riscritture cosmetiche della stessa formula, ma tre modelli diversi dello stesso schema algoritmico.

I risultati seguono il modello teorico soprattutto nella posizione dei colli di bottiglia. Per MPI, il termine $|V| dot log p$ di @eq:mpi-time si manifesta nella riduzione collettiva: sul grafo `random` la `MPI_Allreduce` costa #duration(mpi-random-reduce), cioè il #percent(mpi-random-reduce, mpi-random.loop) del loop. Per OpenMP, @eq:omp-time prevede una soglia più favorevole di MPI, $frac(|E|, |V|) >= p$ invece di $p log p$, ma le misure mostrano che la costante della riduzione piatta, la gestione dei buffer e la contesa sulla DSU condivisa impediscono a questa previsione asintotica di tradursi automaticamente in speedup: sul grafo `random` lo speedup algoritmo resta #calc.round(empirical-speedup(openmp-random), digits: 2). Per CUDA, il modello ideale @eq:cuda-sm-speedup descrive bene il vantaggio strutturale del parallelismo sugli archi, ma resta un limite superiore: lo speedup sul solo algoritmo device è #calc.round(empirical-speedup(cuda-random), digits: 2), mentre sul backend completo scende a #calc.round(backend-loop-speedup(cuda-random), digits: 2) perché setup, copie, lanci kernel, accessi globali e atomiche non sono inclusi nel limite basato solo sugli SM.

L'efficienza chiarisce un aspetto che lo speedup da solo nasconde. MPI e OpenMP usano rispettivamente #worker-count(mpi-random) e #worker-count(openmp-random) worker, quindi uno speedup moderato può ancora corrispondere a una quota significativa del parallelismo disponibile. CUDA, invece, viene normalizzato su #worker-count(cuda-random) SM: anche quando il tempo algoritmo device migliora molto rispetto alla baseline CPU, l'efficienza classica $E_p = S_p slash p$ resta più bassa perché il denominatore rappresenta un grado di parallelismo fisico molto più grande. Questo non rende CUDA meno utile; indica solo che speedup assoluto ed efficienza normalizzata rispondono a domande diverse.

È quindi corretta, con questa precisazione, l'idea che il lavoro presenti tre approcci tecnologici allo stesso algoritmo. Non cambia la specifica del problema: tutte le run producono un MST verificato contro il riferimento sequenziale CPU. Cambia invece la decomposizione operativa del round. Una modifica apparentemente piccola, come scegliere una riduzione collettiva MPI invece di una riduzione piatta OpenMP o di un `atomicMin` CUDA, cambia il termine non-scan dominante e quindi cambia soglie, overhead e forma dello speedup. In questo senso le tre implementazioni sono utili proprio perché isolano come memoria distribuita, memoria condivisa e SIMT trasformano lo stesso Boruvka in tre profili di scalabilità distinti.

La stessa lettura vale anche oltre questo algoritmo. In molti problemi paralleli le tre tecnologie non sono alternative esclusive: possono essere combinate in una strategia ibrida, usando MPI per distribuire il lavoro tra nodi, OpenMP per sfruttare i core CPU dentro ogni nodo e CUDA per accelerare i kernel più regolari e massivamente paralleli sulla GPU. Questa combinazione può ridurre i tempi solo se la partizione dei dati e le comunicazioni lasciano abbastanza lavoro locale da ammortizzare sincronizzazioni, copie e lanci kernel. Per Boruvka, una possibile direzione ibrida sarebbe distribuire partizioni del grafo con MPI, usare CUDA per la scan degli archi locali e mantenere una fase di riduzione globale per riconciliare i candidati tra nodi; il beneficio dipenderebbe però dal costo di comunicare candidati e frontiere rispetto al lavoro sugli archi.

Dal punto di vista teorico, scalando a grafi ancora più grandi ci si aspetta infine di incontrare un limite di banda. Boruvka visita ripetutamente molti archi e, nella scan, il lavoro per arco è relativamente semplice: leggere estremi e peso, trovare i rappresentanti, confrontare una chiave e aggiornare un candidato. In un regime abbastanza grande, il tempo non può essere ridotto solo aumentando processi, thread o SM, perché diventa necessario muovere una quantità crescente di dati tra memoria, rete e unità di calcolo. Per MPI questo limite corrisponde al traffico di comunicazione richiesto dalle riduzioni e dalla riconciliazione dei candidati; per OpenMP alla banda della memoria condivisa e alle scritture concorrenti sui buffer; per CUDA alla banda della memoria globale e al throughput delle atomiche. Questa è una previsione di scalabilità del modello, non una proprietà dimostrata direttamente dalla campagna sperimentale: indica il collo di bottiglia che ci si aspetta quando il parallelismo computazionale smette di essere il fattore dominante.
