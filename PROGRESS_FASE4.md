# Fase 4 (HOG+SVM) — stato lavori

Note di ripresa per la prossima sessione. Non è un documento di consegna, è promemoria.

## Cosa è confermato e stabile

- `include/hog.hpp` / `src/hog.cpp`: `makeHog()`, `computeFeatures()`, `trainSvm()`, `classify()`
  implementati. `trainSvm` usa **`cv::ml::SVMSGD`** (non `cv::ml::SVM`, cambiato su richiesta
  esplicita a metà conversazione). Due fix reali dentro `trainSvm`, non negoziabili:
  - le label devono essere `CV_32F` (non `CV_32S` come per `cv::ml::SVM`), altrimenti
    assertion failure a runtime.
  - `TermCriteria` **solo `MAX_ITER`** (niente componente `EPS`). Con `EPS` attivo a 1e-5
    (la spec originale) la convergenza scatta troppo presto su dataset "duri" (es. con hard
    negative) e produce un modello non addestrato — identico da 100k a 1M iterazioni. Verificato
    che su pos+neg originali (dataset "facile") il bug non si manifestava mai, quindi il modello
    di Fase 4 base è valido; sarebbe esploso silenziosamente al primo giro di HNM.
  - `classify()` prende `cv::Ptr<cv::ml::SVMSGD>`, non `cv::Ptr<cv::ml::SVM>` (SVMSGD non eredita
    da SVM, sono classi sorelle sotto `StatModel`).
  - **NOTA per chi userà `predict(..., RAW_OUTPUT)` su SVMSGD**: non funziona, restituisce la
    stessa etichetta discreta del predict normale (sospetto: `RAW_OUTPUT` e `UPDATE_MODEL`
    condividono lo stesso valore di enum in OpenCV). Lo score continuo va ricostruito a mano:
    `score = svm->getWeights().dot(descriptor) + svm->getShift()`.

- `src/color_proposals.cpp`: soglie S/V alzate 60/40→80/60 nelle tre `inRange` (il prato secco
  cade nella stessa banda di tinta del cono in certe condizioni di luce — verificato con
  istogrammi HSV). Aggiunto filtro area massima 1% immagine in `findCandidateBoxes` (elimina i
  blob giganti erba+cono fusi dalla closing morfologica; matematicamente non può scartare un vero
  positivo). Testato e scartato: togliere l'opening, cambiarne il kernel, togliere la closing,
  filtro aspect ratio sui blob — tutte peggiorative o neutre, vedi cronologia conversazione.

- `data/svm_hog_sgd.yml`: modello Fase 4 "base" (pos+neg puliti, no hard negative). Numeri validi.
- `data/svm_hog_sgd_hnm.yml`: modello con hard negative mining (25000 patch, minate dai
  candidati **aspect-normalizzati** delle immagini di train, IoU<0.1 da qualsiasi GT, score>0
  col modello base). **Questo modello è probabilmente da rifare** — vedi sotto.

## La scoperta importante di questo giro (non ancora sfruttata)

Misurato con `tools/funnel_analysis.cpp` (imbuto GT→coperti da un candidato IoU≥0.3→sopra
soglia→sopravvivono a NMS) che il collo di bottiglia **non è solo** la generazione dei candidati:
a thr=0, dei candidati ben localizzati (IoU≥0.3 con un cono vero) solo il 42.6% supera la soglia
(57% scartato dal classificatore). La causa: **mismatch di distribuzione train/inferenza**. I
positivi di training sono crop puliti sulle bbox GT (mediana score +52.1 sul test set di Fase 4);
i candidati reali del detector sono ritagli di blob colore, decentrati/parziali (mediana score
solo -5.2 anche quando ben localizzati). L'aspect-ratio normalization NON risolve questo (tasso
di accettazione identico con o senza, 57.0% vs 57.2% — il guadagno di recall che avevo attribuito
alla normalizzazione veniva in realtà tutto dalla copertura, non dall'accettazione: correzione
fatta in conversazione, verificata isolando le due variabili).

## Piano concordato con l'utente (in corso, non finito)

**Riallenare sui candidati, non sulle ground truth** (procedura standard tipo R-CNN):
1. Far girare `colorProposals()` (invariata) su tutte le immagini di **train**.
2. Etichettare ogni candidato per IoU con le GT: **>0.5 positivo, <0.3 negativo, in mezzo
   scartato** (ambiguo).
3. Ritagliare+ridimensionare a 16x24 **il candidato così com'è**, niente aspect-normalization
   (positivi e negativi useranno la stessa procedura di crop dell'inferenza — la normalizzazione
   diventa superflua per costruzione).
4. Riallenare con `trainSvm` (invariata) su questi nuovi pos/neg.
5. Rimisurare con `funnel_analysis` in modalità `raw` (non `aspect`): aspettativa dell'utente è
   che il 57% di scarto condizionale scenda parecchio, e che con copertura ~50% si arrivi a una
   recall finale vicina al 40% (il doppio di adesso), senza toccare i proposal.
6. Se servono più positivi: aggiungere jitter sulle bbox GT (2-3 varianti per box, shift/resize
   random 10-20%) come materiale extra, opzionale.

## Prossimo passo immediato

`tools/extract_proposal_patches.cpp` è **scritto ma non ancora compilato né eseguito**. Implementa
esattamente il punto 1-3 sopra: scrive in `data/patches/pos_proposal` e `data/patches/neg_proposal`.

Comando per compilare (dalla cartella `build/`):
```
g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
  ../tools/extract_proposal_patches.cpp ../src/color_proposals.cpp ../src/utils.cpp \
  -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml \
  -o extract_proposal_patches
./extract_proposal_patches 70000
```
Poi riallenare (serve un piccolo tool nuovo tipo `train_hnm.cpp` ma puntato su
`pos_proposal`/`neg_proposal` invece che su `pos`/`neg_plus_hard` — non ancora scritto), poi
`funnel_analysis <nuovo_modello.yml> raw 0 150 7` per confrontare con i numeri di questo file.

## File ad-hoc in `tools/` (non collegati a CMakeLists, "al volo" su richiesta esplicita)

- `detector_common.hpp`: candidati aspect-normalizzati (da riconsiderare se il retraining sui
  proposal rende la normalizzazione inutile) + NMS condivisi.
- `eval_proposals.cpp`: recall dei soli candidati (candidateRecall) su campione di test.
- `mine_hard_negatives.cpp`, `train_hnm.cpp`: pipeline di hard negative mining (il modello HNM
  attuale). Da rivalutare se si scarta l'approccio in favore del retraining sui proposal.
- `eval_detector.cpp`: valutazione end-to-end (precision/recall/F1 per soglia) con candidati
  aspect-normalizzati.
- `funnel_analysis.cpp`: l'imbuto per stadio + percentili score, usato per la diagnosi sopra.
  Supporta modalità `raw` (blob grezzi di colorProposals) e `aspect` (normalizzati).
- `visualize_detections.cpp`: genera immagini con GT (verde) e detection (rosso) per ispezione
  visiva. Preso a modello `../data/svm_hog_sgd_hnm.yml`, soglia 0, candidati aspect di default.
- `extract_proposal_patches.cpp`: **il pezzo nuovo**, vedi sopra.

## Numeri di riferimento (150 immagini test, seed 7, per confronto quando si rifà tutto)

| Config | Coverage (stadio 2) | Accettazione | Recall finale (thr=0) | Precision (thr=0) |
|---|---|---|---|---|
| Raw + baseline | 45.2% | 57.2% | 25.4% | ~13% (stimato da altro run) |
| Aspect + baseline | 50.5% | 57.0% | 28.3% | ~13% |
| Raw + HNM | 45.2% | 35.5% | 15.8% | — |
| Aspect + HNM | 50.5% | 42.6% | 21.2% | ~68% |

Fase 4 "pulita" (crop puliti su GT, non detection reale): accuracy 94.86%, precision 96.21%,
recall 91.32% sul test set — questo è il tetto teorico del classificatore su input puliti, il
gap con i numeri sopra è la storia di questo documento.

## Aggiornamento — riallenamento sui proposal + fix preprocessing (questa sessione)

Eseguito il piano rimasto in sospeso sopra, con due correzioni non previste nel piano originale.

**1. `extract_proposal_patches.cpp` eseguito, ma la soglia `isPos>0.5` del piano andava
corretta.** Prima iterazione (isPos>0.5, isNeg<0.3, come da piano): risultato **peggiore** del
baseline (acceptance 52.6% vs 57.2%, recall finale 23.3% vs 25.4%, raw, thr=0, 150 img test).
Causa: la soglia "coperto" usata da `funnel_analysis` per misurare la recall è IoU≥0.3, ma il
training scartava come "ambiguo" tutto ciò che stava tra 0.3 e 0.5 — cioè esattamente i candidati
decentrati/parziali che poi in valutazione contano come "coperti" ma il classificatore non aveva
mai visto come positivi. Corretto a **isPos≥0.3, isNeg<0.2** (allineato alla definizione di
"coperto"): positivi passati da 11960 a 24746 (con flip), risultato **acceptance 57.2%→78.6%,
recall finale 25.4%→34.9%** (raw, thr=0, 150 img test). Modello: `data/svm_hog_sgd_proposal.yml`
(nuovo tool `tools/train_proposal.cpp`, non collegato a CMakeLists come gli altri).

**2. Con la classificazione non più il collo di bottiglia, il tetto è tornato a essere la
coverage dei proposal (45.2%) — cioè il vero preprocessing (`color_proposals.cpp`), non la
classificazione.** Diagnosi con due tool nuovi (`tools/coverage_diag.cpp`,
`tools/mask_coverage_diag.cpp`) sulle GT box del test set non coperte da nessun candidato
(55%): mediana altezza 18px contro 30px delle coperte (i coni piccoli/lontani sono il problema,
non il colore — mediana tinta/saturazione delle mancate è simile alle coperte). Del 55% mancato:
9.3% ha zero pixel di maschera colore nella box anche PRIMA della morfologia (irrecuperabile con
soglie HSV più permissive, vedi sotto), ma **15% ha maschera non vuota prima e vuota dopo** —
ucciso dalla pulizia morfologica, non dalla soglia colore. Causa individuata: i coni piccoli
producono maschere frammentate in isole di pochi pixel (compressione/antialiasing), e la pipeline
faceva opening (rimuove rumore) PRIMA della closing (fonde le due metà del cono) — quindi le
isole venivano cancellate una per una prima che la closing avesse la possibilità di fonderle in
un blob abbastanza grande da sopravvivere.

Fix verificato con `tools/proposal_tuning.cpp` (duplica la pipeline colorMask+morph senza
toccare `color_proposals.cpp`, per confrontare varianti su tutto il test set — 600 immagini —
prima di portare qualcosa in src): **invertito l'ordine (closing poi opening) e allargato il
kernel di closing 3x7→5x9**. Risultato sull'intero test set: coverage 45.7%→48.5%, **e**
candidati/immagine 521→477 (meno rumore, non di più — dominanza netta, non un compromesso).
Altre varianti provate e scartate: closing senza opening dopo (coverage uguale ma 1465
candidati/immagine, troppo rumore), kernel di closing 7x11 (coverage inferiore, il gap è troppo
largo e fonde con lo sfondo). Portato in `src/color_proposals.cpp` (funzione `morphCleaning`).

Sweep sulle soglie S/V della `colorMask` (stesso tool, altra ipotesi: la saturazione mediana
delle box mancate, 66, è già sotto l'attuale soglia 80): **negativo**. Abbassare sat/val peggiora
sempre la coverage (sat=65,val=60 → 44.1%; sat=50,val=40 → crolla a 35.2%), perché più pixel
d'erba entrano nella maschera e, dopo la closing, si fondono con i coni veri in blob giganti che
il filtro area 1% scarta — lo stesso meccanismo per cui le soglie erano già state alzate da
60/40 a 80/60 in un giro precedente. **80/60 confermato come ottimo locale**, non toccato.

Ripetuta l'estrazione proposal + riallenamento con la nuova `color_proposals.cpp` (stessa
procedura, patch rigenerate: 25626 positivi). Risultato finale, confrontato su tutto il test set
(600 immagini, raw, seed 7) contro i due modelli precedenti **rivalutati anch'essi sulla nuova
`color_proposals.cpp`** (per un confronto onesto):

| Modello | thr=0 | thr=10 | thr=20 |
|---|---|---|---|
| baseline (`svm_hog_sgd.yml`) | P=0.12 R=0.28 | P=0.23 R=0.23 | P=0.42 R=0.19 |
| HNM (`svm_hog_sgd_hnm.yml`) | P=0.71 R=0.20 | P=0.93 R=0.16 | P=0.98 R=0.13 |
| **proposal (`svm_hog_sgd_proposal.yml`)** | **P=0.29 R=0.34** | **P=0.60 R=0.27** | **P=0.86 R=0.20** |

Il modello proposal domina gli altri due su tutta la curva precision/recall (a parità di recall,
precisione più alta; a parità di precisione, recall più alta) — non è un compromesso, è un
miglioramento netto. `data/svm_hog_sgd_proposal.yml` è ora il modello di riferimento, non più
`svm_hog_sgd_hnm.yml`.

**Nota**: `cv::ml::SVMSGD` non fissa un seed, quindi due training sugli stessi dati danno modelli
leggermente diversi (osservato: stesso setup, thr=0 raw variava 34.9%↔36.1% di recall tra due
run). I numeri sopra sono indicativi entro ±1-2 punti, non esatti al decimale.

### File nuovi in `tools/` (ad-hoc, non collegati a CMakeLists)

- `train_proposal.cpp`: allena su `pos_proposal`/`neg_proposal` (sostituisce concettualmente
  `train_hnm.cpp`, che resta per confronto/storia ma non è più la pipeline consigliata).
- `coverage_diag.cpp`: per le GT box non coperte, dimensione e colore mediano — ha isolato la
  dimensione (non il colore) come differenza principale.
- `mask_coverage_diag.cpp`: frazione di pixel-maschera dentro le GT box prima/dopo la morfologia
  — ha isolato l'opening-prima-della-closing come causa delle box uccise dalla pulizia.
- `proposal_tuning.cpp`: duplica colorMask+morph per confrontare varianti (ordine morfologia,
  kernel, soglie S/V) su tutto il test set senza toccare `color_proposals.cpp` finché una
  variante non è verificata.

### Prossimo passo possibile

Il 9.3% di GT box con zero pixel di maschera colore anche prima della morfologia è probabilmente
irrecuperabile con questo approccio a soglie HSV fisse (lo sweep S/V lo conferma indirettamente:
non c'è margine ad abbassare le soglie senza peggiorare tutto il resto). Il prossimo guadagno di
coverage, se serve, richiede probabilmente un cambio di approccio più radicale (es. detection
multi-scala, o una banda di colore separata/più permissiva solo per blob sotto una certa
dimensione), non un altro giro di tuning sulle soglie attuali.

## Aggiornamento — coverage spinta a 80%+ con multi-scala centrata sul blob (sessione successiva)

L'utente ha chiesto esplicitamente di portare la coverage del proposal all'80-90%, sperimentando
liberamente in `tools/` senza vincolarsi a quanto già fatto. Approccio che ha funzionato, dopo
vari tentativi scartati (vedi sotto): **niente più "un box = un blob"**. Un blob della maschera
colore, specie per coni piccoli/lontani, è spesso un frammento (compressione/antialiasing: la
maschera copre solo una parte del cono, non tutto), quindi la sua bounding box ha quasi sempre la
TAGLIA sbagliata anche quando la POSIZIONE (centroide) è corretta. Invece di tassellare tutta la
regione con una griglia (esploso a centinaia di migliaia di candidati/immagine per coverage
comunque non oltre il 85-95%, scartato: la dilatazione necessaria a "riempire" l'area intorno al
blob fonde anche coni distinti tra loro), si genera **un box per blob per altezza**, a un piccolo
set di altezze assolute fisse (non relative alla taglia del blob!), centrato sul centroide del
blob, senza nessuna dilatazione/morfologia (i blob restano quelli grezzi della maschera colore,
la loro posizione è già abbastanza precisa).

Parametri finali (misurati con `tools/dense_grid_proposals.cpp` su tutto il test set, 600
immagini): altezze assolute `{10, 18, 30, 50, 85, 140, 220}` (log-spaziate, tarate sui percentili
reali delle altezze GT — mediana 22px, p90 94px), blob scartati sotto 3px di area (rumore a
singolo pixel, contributo trascurabile alla coverage). Risultato: **coverage 45.2%→81.5%**,
candidati/immagine 521→~18500 (confermato anche con la vera `src/color_proposals.cpp`, non solo
il duplicato in tools: `funnel_analysis` su un campione di 30 immagini dà addirittura 87.9%).
Portato in `src/color_proposals.cpp`, sostituendo `morphCleaning`+`findCandidateBoxes` a bbox
singola (rimossa la pulizia morfologica: con questo approccio non serve più).

Tentativi scartati durante la ricerca (tutti in `tools/dense_grid_proposals.cpp`, mai portati in
src): tiling a griglia completa della bbox dilatata (85-95% coverage ma 100-300k
candidati/immagine); più dilatazione prima del centroide (PEGGIORA la coverage nell'approccio
centrato: fonde blob di coni diversi, sposta il centroide fuori da entrambi); snap del centroide
a una griglia grossolana per deduplicare (non riduce granché i candidati, i blob sono
genuinamente sparsi in posizioni diverse, non ammassati).

**Costo non gratis, e serio**: con ~18500 candidati/immagine invece di ~500, riallenato il
classificatore sugli stessi target IoU di prima (`extract_proposal_patches.cpp`, invariato:
272804 positivi con flip, 100000 negativi, su tutte le 1400 immagini di train) e rimisurato
end-to-end con `eval_detector` su tutto il test set (600 immagini, 26 minuti di run): **recall
alta e in linea con la coverage (79.3% a thr=0, 80.5% a thr=-10) ma precision crollata a livelli
impraticabili** (thr=0: precision 0.38%, thr=50 cioè soglia molto alta: precision 7.8%, ~33
FP/immagine anche li'). Lo SVM lineare su HOG, che prima doveva scartare ~500 candidati/immagine,
ora ne deve scartare ~18500, e a quella scala anche una specificità molto alta produce comunque
migliaia di falsi positivi assoluti per immagine — è il classico problema "ago nel pagliaio" dei
detector a scansione densa, che di solito si risolve con cascate a piu' stadi o classificatori
non lineari, non con piu' dati per un modello lineare.

**Stato**: l'obiettivo richiesto esplicitamente (coverage del proposal all'80-90%) è raggiunto e
verificato sul test set completo. La pipeline end-to-end (classificazione + soglia + NMS), però,
non è più utilizzabile così com'è — va vista come un problema separato e successivo, non ancora
affrontato in questo giro. Prossimi passi possibili, in ordine di probabile costo/beneficio:
1. Cascata a due stadi: prima un filtro economico e molto permissivo (es. lo score dell'attuale
   SVM lineare, soglia bassissima) per abbattere i candidati da 18500 a poche centinaia, poi un
   secondo classificatore più espressivo (SVM non lineare / kernel RBF, o piccola rete) solo su
   quelli sopravvissuti.
2. Ridurre la densità dei candidati mantenendo la coverage: es. altezze assolute più mirate per
   blob (stimare un range di taglie plausibili dalla dimensione del blob invece di generarle
   tutte sempre), o un filtro di forma/aspect ratio del blob prima di espanderlo.
3. NMS più aggressivo prima possibile nella pipeline (per candidato, non per score) per ridurre
   il volume assoluto indipendentemente dal classificatore.

## Tentativo scartato — proposal geometrico (linee/triangoli) invece che color-blob

Su richiesta esplicita dell'utente, testato un approccio del tutto diverso in
`tools/triangle_proposals.cpp` (non portato in src, **negativo**): sfruttare che i coni sono
triangoli in silhouette (due lati obliqui che convergono all'apice, base orizzontale), cercare le
linee vere con `cv::HoughLinesP` su Canny, accoppiare una linea "sinistra" (dx/dy>0) con una
"destra" (dx/dy<0) il cui apice combacia, e usare il bbox del triangolo risultante come candidato
— idea: la taglia verrebbe dalla geometria reale invece che indovinata come nel multi-scala
assoluto sopra.

**Risultato: molto peggio su entrambi gli assi.** Coverage 6.6%-17.1% (contro l'81.5% del color
puro) e candidati/immagine da 200mila a 6 milioni (contro ~18500), con tempi proibitivi (4-17
minuti anche solo per 20-60 immagini di test). Motivo, dopo diagnosi: in una scena naturale
(erba, ghiaia, texture) `HoughLinesP` produce moltissimi segmenti spuri non legati ai coni, e i
coni reali — mediana altezza 22px nel dataset — sono troppo piccoli/compressi perché Canny+Hough
ne trovi i due lati come segmenti puliti e appaiabili: quasi tutti i triangoli trovati sono falsi
positivi geometrici (due segmenti spuri che per caso convergono), non i lati veri di un cono.
Provato anche a fondere i blob colore prima di cercare le linee (per ridurre la ridondanza tra
ROI sovrapposte di frammenti vicini, che nella prima versione spiegava da sola i milioni di
candidati) e a restringere l'angolo accettato a 8-35° dalla verticale (piu' vicino alla vera
geometria di un cono): migliora un po' ma resta lontanissimo dall'essere utilizzabile.

**Conclusione**: l'approccio color-blob (`src/color_proposals.cpp` attuale, 81.5% coverage) resta
la soluzione. L'idea geometrica potrebbe funzionare su immagini a risoluzione più alta / coni più
grandi nell'inquadratura, ma su questo dataset (coni piccoli, spesso lontani, compressione JPEG)
l'informazione di bordo netto necessaria a Hough non è affidabilmente presente.

## Cascata a due stadi (implementata) + validazione su vero test set esterno + ottimizzazione parametri

Il problema aperto lasciato dalla sessione precedente: con ~18500 candidati/immagine (multi-scala
assoluta), un unico SVM lineare non riesce a tenere la precisione (thr=0: 0.4%) perche' deve
scartare troppi negativi in assoluto per ogni vero positivo — vedi spiegazione estesa data
all'utente in conversazione (non ripetuta qui, concetto: precisione dipende dal tasso di errore
assoluto su un numero enorme di negativi, non dalla percentuale, e un iperpiano lineare non riesce
a comprimere ulteriormente quel tasso di errore data la varieta' di finestre generate). Soluzione:
**cascata a due stadi**.

**Stadio 1** (`tools/cascade_stage1_sweep.cpp` per la scelta della soglia): il classificatore
lineare gia' allenato (`svm_hog_sgd_proposal.yml`), usato a una soglia bassa e permissiva SOLO per
abbattere il volume di candidati, non per decidere. Sweep di soglie sul test set sintetico:
thr=10 -> 5000 candidati/immagine (retention 93.6%), thr=15 -> 3150/immagine (89.3%), thr=20 ->
1943/immagine (83.4%).

**Stadio 2** (`tools/train_stage2.cpp`): SVM non lineare (`cv::ml::SVM`, kernel RBF, non SVMSGD —
RBF ha molta piu' capacita' di un iperpiano) allenato SOLO sui candidati che lo stadio 1
lascerebbe passare (stessa lezione del giro precedente: training deve vedere la distribuzione di
inferenza). Patch filtrate da `pos_proposal`/`neg_proposal` gia' estratte, sottocampionate a
15000/classe per contenere il tempo di training (RBF e' O(n^2)-ish), iperparametri C/gamma scelti
con `trainAuto` (5-fold CV).

**Verificato che funziona**: sul test set sintetico, a parita' di recall (~0.30-0.33), precisione
0.05 (SVM lineare unico, thr=40) -> 0.41 (cascata) — **8x meglio**. Vedi tabella di ottimizzazione
sotto per i numeri completi.

### Validazione su un vero test set esterno (non sintetico)

L'utente ha fornito un secondo test set, mai visto durante nessun tuning: `test_set/segmentation_test/`,
24 immagini reali Formula Student (sigle team: amz, BME, ka, mms, tuwr — sembra un sottoinsieme di
FSOCO), annotazioni Supervisely a maschera bitmap (base64+zlib+PNG, offset "origin"), 400 coni.
Convertite in bbox con `tools/convert_supervisely_to_csv.py` (decodifica base64->zlib->PNG, bbox
= estensione dei pixel non-zero della maschera + origin) -> `data/dataset_real.csv` +
`data/test_real.txt`. Verificate visivamente (bbox disegnate, allineate correttamente sui coni,
non sui barili striati rosso/bianco nelle stesse foto).

**Risultato: la pipeline tarata SOLO sul sintetico generalizza bene sul reale, anzi meglio in
alcuni aspetti**: coverage proposal 96.3% (contro 81.5% sintetico — probabile perche' nelle foto
vere i coni sono in media piu' grandi/vicini). Non e' overfitting sui dettagli del rendering
sintetico. Tool: `tools/eval_real.cpp` (stesse metriche di `eval_cascade.cpp`, puntato al test set
reale), `tools/visualize_real.cpp` (disegna GT verde / detection rosse, salva in
`data/detection_samples_real/`).

### Ottimizzazione dei parametri della cascata

Allenati 3 stadi-2 con soglie stadio-1 diverse (5, 10, 15 — vedi `svm_stage2_rbf_thr{5,10,15}.yml`)
e confrontati con lo stesso sweep di soglie stadio-2 (-0.01 a 0.3) sia sul sintetico (campione 60
immagini, poi confermato su tutte le 600) sia sul reale (tutte le 24). Nota: modelli diversi hanno
scale di score diverse (C scelto da trainAuto varia: 0.1 per thr1=10/15, 2.5 per thr1=5) — la
soglia stadio-2 ottimale NON e' trasferibile tra modelli diversi, va ritrovata ogni volta.

| stadio-1 thr | candidati/immagine sopravvissuti | miglior F1 sintetico (60 img) | miglior F1 sintetico (600 img, conferma) | miglior F1 reale (24 img) |
|---|---|---|---|---|
| 5  | 7782 | 0.334 (thr2=0.13) | — | 0.508 (thr2=0.15) |
| 10 | 5014 | 0.341 (thr2=0.01) | — | 0.496 (thr2=0.01) |
| **15** | **3146** | **0.365 (thr2=0.01)** | **0.365 (thr2=0.01, confermato)** | 0.505 (thr2=0.01) |

**thr1=15 vince**: miglior F1 sul sintetico (chiaro, non marginale: 0.365 contro 0.33-0.34),
sostanzialmente a pari merito col migliore sul reale (0.505 contro 0.508 di thr1=5, differenza nel
rumore di 24 immagini), **e** usa meno della meta' dei candidati di thr1=10 per lo stadio 2 (2263-3146
contro 5014/immagine) — piu' economico E piu' preciso, non un compromesso.

**Configurazione finale adottata**:
- `data/svm_hog_sgd_proposal.yml` — stadio 1, soglia **15**
- `data/svm_stage2_rbf.yml` (= `svm_stage2_rbf_thr15.yml`) — stadio 2, soglia **~0.01** per il
  punto bilanciato (F1 migliore); soglia piu' alta (0.015-0.02) se si preferisce piu' precisione a
  scapito della recall, vedi tabelle sopra per il resto della curva.

Numeri finali confermati sull'intero test set sintetico (600 immagini, thr1=15, thr2=0.01):
**precision=0.407, recall=0.331** (TP=3970, FP=5793, FN=8040) — contro lo 0.05 di precisione del
singolo SVM lineare alla stessa recall. Sul reale (24 immagini, stessa config): precision=0.413,
recall=0.648.

Provato anche, su suggerimento dell'utente, uno smoothing piu' aggressivo prima di Canny per
uccidere il rumore di texture (erba/ghiaia) che genera i segmenti spuri: Gaussian 7x7/11x11,
bilateral filter (edge-preserving), median blur, con soglie Canny piu' basse in proporzione.
Risultato: **non risolve, conferma la diagnosi**. Gaussian 11x11 abbatte i candidati (24k/immagine,
da centinaia di migliaia) ma la coverage crolla ANCHE lei (4.9%) — lo smoothing forte uccide il
poco di bordo vero rimasto insieme al rumore. Bilateral/median preservano un po' piu' di bordo
(coverage ~9%, il doppio) ma il rumore torna a esplodere (1-2 milioni di candidati/immagine). In
nessun caso ci si avvicina al color-blob (81.5%): il segnale (i due lati netti del cono) non e'
semplicemente presente in modo affidabile a queste risoluzioni, non è questione di rapporto
segnale/rumore risolvibile con un filtro. Filone chiuso.
