# cones-detection

Pipeline di detection coni (tipo Formula Student) via computer vision classica: proposal
per-colore + classificazione HOG/SVM a cascata. Nessun deep learning.

## Stato attuale (branch `hog`)

Pipeline completa (detection -> classificazione -> segmentazione), tutta in `src/`:

1. **Generazione candidati** (`src/color_proposals.cpp`, `colorProposals()`): white balance ->
   maschera colore HSV (blu/arancione/giallo) -> per ogni blob grezzo della maschera, box
   aspect-normalizzati a 7 altezze assolute fisse `{10,18,30,50,85,140,220}px`, centrati sul
   centroide del blob. **Coverage (IoU>=0.3 con GT) verificata: 81.5% sul test set sintetico,
   96.3% su un vero test set esterno** (vedi sotto). ~18.500 candidati/immagine.

2. **Classificazione a cascata** (`src/detector.cpp`, `detectCones()`). Il descrittore di ogni patch
   (`describePatch()` in `src/hog.cpp`) è HOG sul grigio **più** un istogramma H-S 12x4 pesato 1.5:
   il solo HOG butta via il colore, che è l'unica cosa che distingue un cono da una macchia di
   asfalto della stessa forma.
   - **Stadio 1**: SVM lineare su HOG (`cv::ml::SVMSGD`, modello `data/svm_hog_sgd_proposal.yml`),
     soglia **15** — filtro economico e permissivo, abbatte i candidati a ~2500-3150/immagine
     mantenendo ~89% della recall raggiungibile.
   - **Stadio 2**: SVM non lineare (`cv::ml::SVM`, kernel RBF, modello
     `data/svm_stage2_rbf_balanced.yml`), soglia **~0.01** (punto di miglior F1; più alta = più
     precisione, meno recall) — allenato SOLO sui candidati che lo stadio 1 lascia passare
     (stessa distribuzione di inferenza). Poi NMS (IoU 0.1) e allargamento asimmetrico del box
     (`kPadTop/Bottom/Side` = 0 / 0.4 / 0.1), che serve solo a dare contesto alla segmentazione.

### Cosa ha spostato l'ago

Provato in una sandbox separata prima di toccare `src/` (protocollo e numeri in `EXPERIMENTS.md`):

| modifica | effetto sulla macro F1 |
|---|---|
| **istogramma di colore nel descrittore** | 0.370 -> 0.436 |
| **peso del colore a 1.5** invece di 1.0 | 0.436 -> 0.479 |
| **un giro di hard negative mining** | 0.479 -> 0.493 |
| patch più grandi (24x36, 32x48) | 0.479 -> 0.354, scartata |

(misure sul protocollo veloce a 1500 immagini, confrontabili solo fra loro)

Sul progetto vero sono serviti altri due aggiustamenti, entrambi sullo stadio 2:

| modifica | macro F1 |
|---|---|
| 15000 campioni per classe, C scelto da `trainAuto` (= 0.1) | 0.405 |
| 15000 campioni, C fissato a 2.5 | 0.435 |
| **5000 campioni, C fissato a 2.5, soglia 0.1** | **0.545** |

Due lezioni: l'SVM RBF finale lavora **meglio con meno campioni** (5000 invece di 15000, a parità di
C), e `trainAuto` è una trappola perché sceglie C in base a quanti campioni riceve — un C diverso
sposta gli score su un'altra scala e la soglia dello stadio 2, che è una costante, smette di
valere. Con C fisso il training è anche 6 minuti più veloce, perché salta la cross validation.

3. **Classificazione del colore** (`src/classifier.cpp`, `classifier()`): per ogni box, CLAHE sul
   canale V, maschere HSV giallo/blu/arancione/nero limitate a un trapezio (taglia gli angoli alti
   del box), voto a maggioranza di pixel. Se vince l'arancione, `classifyOrangeCone()` distingue
   small/big contando le fasce bianche orizzontali.

   Il training (`main train`) fa anche un giro di **hard negative mining**: i candidati che lo
   stadio 1 accetta per sbaglio tornano fra i negativi e lo stadio 1 viene riallenato con quelli.

4. **Segmentazione** (`src/segmentation.cpp`, `segmentCone()`): k-means++ (K=3) in HSV sul ritaglio,
   il cluster del cono è quello più votato dalla maschera colore della classe predetta; poi closing
   verticale (kernel = 30% dell'altezza del ROI) per ricucire i blob separati dalle fasce bianche,
   componente connessa più grande, contorno esterno riempito. I ROI più bassi di 80px vengono
   ingranditi prima di segmentare. mIoU per singolo cono: **0.67**.

5. **Box finale** (`src/detector.cpp`, `tightenToMask()`): il box riportato in output non è quello
   allargato del punto 2 ma la bounding box della maschera del punto 4, con un margine del 5%.
   **IoU con il box vero: 0.59 usando il box allargato, 0.70 usando la maschera.**

### Le tre metriche della specifica

`tools/eval_pipeline.cpp`, pipeline completa sul vero test set esterno (24 immagini FSOCO-style,
398 coni, mai usate per il tuning):

| metrica | valore |
|---|---|
| **Macro-averaged F1** (IoU>=0.5) | **0.545** |
| **mAP@0.5:0.95** | **0.143** |
| **mIoU** (Jaccard sui pixel) | **0.288** |

Ottenuti rilanciando `main train` da zero su tutte le 5755 immagini annotate, con il test set tenuto
fuori da `dataset/`. Il descrittore include l'istogramma di colore e il training fa un giro di hard
negative mining (vedi sotto).

Dettaglio per classe:

| classe | P | R | F1 | coni |
|---|---|---|---|---|
| giallo | 0.767 | 0.556 | 0.645 | 178 |
| blu | 0.717 | 0.553 | 0.625 | 188 |
| small orange | 0.467 | 0.500 | 0.483 | 14 |
| big orange | 0.600 | 0.333 | 0.429 | 18 |

Le due classi arancioni valgono metà del punteggio pur essendo l'8% dei coni: è l'effetto voluto
dalla media macro, e restano il punto debole. Attenzione a non confondere questo mIoU (sui pixel, su
tutta l'immagine, quindi paga i coni non trovati e i falsi positivi) con quello riportato da
`eval_segmentation` (0.64), che è la media degli IoU dei soli coni già trovati: misurano cose diverse.

**Dove si perde**, misurato con l'imbuto per classe di `eval_pipeline`: i proposal di colore coprono
il 97% dei coni (gli arancioni al 100%), la cascata ne lascia passare il 71%, e la metà di quelli
che restano non arriva a IoU 0.5 con il box vero. Il collo di bottiglia è la cascata SVM, non il
colore e non il classificatore, che azzecca il 96% di quello che riceve.

### Metriche di supporto (non nella specifica)

Utili per capire dove la pipeline perde, ma non sono i numeri da riportare. Tutte misurate *a
detection perfetta*, cioè dando ai moduli i box del ground truth invece di quelli della pipeline
(`tools/eval_segmentation.cpp`):

- **Segmentazione**: mIoU per singolo cono **0.67** (0.58 sotto i 25px, 0.73 sopra i 60px).
- **Classificazione**: accuratezza **0.93** — giallo 171/178, blu 184/188, small orange 5/14,
  big orange 9/18. Il grosso degli errori residui è fra le due classi arancioni.
- **Localizzazione**: IoU del box con quello vero, **0.59** col box allargato del punto 2 contro
  **0.70** con quello ricavato dalla maschera.

## Due test set

- **Training**: `dataset/`, costruito da `main dataset` unendo i due archivi FSOCO e togliendo le
  immagini del test set. 5755 immagini annotate, ground truth in `data/dataset.csv`
  (image,x1,y1,x2,y2).
- **Reale** (validazione fuori distribuzione, mai usato per tuning): `test_set/segmentation_test/`
  (24 immagini, squadre Formula Student — amz/BME/ka/mms/tuwr), annotazioni Supervisely a maschera
  bitmap, convertite in bbox da `tools/convert_supervisely_to_csv.py` in `data/dataset_real.csv` +
  `data/test_real.txt`. La pipeline tarata solo sul sintetico generalizza bene qui (anzi meglio in
  coverage) — non sembra overfitting sui dettagli del rendering sintetico.

## Dove guardare per i dettagli

- **`PROGRESS_FASE4.md`**: cronologia completa di questa fase (Fase 4, HOG+SVM), decisioni prese e
  scartate con le misure, prossimi passi possibili. Il documento di riferimento per riprendere il
  lavoro.
- **`tuning_log.md`**: log di un giro di tuning precedente e più vecchio dei soli parametri
  colore/morfologia di `colorProposals` (S/V, bande H, merge union-find). Superato dalle modifiche
  più recenti descritte in `PROGRESS_FASE4.md` (multi-scala assoluta), tenuto per riferimento
  storico.
- **`Pipeline.md`**: specifica task originale del progetto.
- **`tools/`**: tutti gli strumenti ad-hoc di sperimentazione/valutazione/training della cascata,
  **non collegati a CMakeLists** (compilazione manuale, comando riportato in testa a ogni file).
  Non toccano `src/` finché una variante non è verificata e "vince". File chiave:
  - `train_proposal.cpp` / `train_stage2.cpp`: training stadio 1 / stadio 2.
  - `eval_detector.cpp` / `eval_cascade.cpp` / `eval_real.cpp`: valutazione precision/recall,
    rispettivamente per il vecchio classificatore singolo, la cascata su sintetico, la cascata sul
    reale.
  - `funnel_analysis.cpp`: imbuto per stadio (coverage -> soglia -> NMS) sul sintetico.
  - `eval_segmentation.cpp` + `segmentation_variants.hpp`: mIoU della sola segmentazione contro le
    maschere bitmap vere del test set reale (398 coni), con le varianti provate tenute come
    parametri. `./eval_segmentation <variante> [cartella_output] [pad]`; la variante 0 è quella in
    `src/`. Riporta anche l'accuratezza della classificazione con la matrice di confusione.
  - `eval_pipeline.cpp`: precision/recall/F1 della pipeline completa così com'è in `src/`, sul test
    set reale, confrontando il box largo con quello ristretto sulla maschera.
  - `visualize_real.cpp` / `visualize_detections.cpp`: disegnano GT (verde) e detection (rosso) su
    immagini campione, salvate in `data/detection_samples_real/` / `data/detection_samples/`.
  - `convert_supervisely_to_csv.py`: converte le annotazioni del test set reale in CSV.

## Build e uso

`cmake` + `make` nella cartella `build/` compila il target `main`. Le modalità:

```
main detect <immagine> [output.png]   pipeline completa su una immagine
main train                            estrae le patch e allena i due stadi della cascata
main eval                             macro F1, mAP@0.5:0.95 e mIoU sul test set reale
main dataset                          unisce le due cartelle FSOCO in dataset/
main label                            ridisegna le annotazioni in labeled/
main csv <dataset_dir>                estrae le bounding box in data/dataset.csv
```

Da zero l'ordine è: `main dataset` -> `main csv ../dataset` -> `main train` -> `main eval`. I primi
due servono solo se `dataset/` e `data/dataset.csv` non ci sono già.

Non c'è nessuno split train/test da fare: il test set è una cartella separata
(`test_set/segmentation_test/`), quindi **tutte** le immagini annotate di `dataset/` vanno nel
training. `main dataset` si occupa di tenere fuori da `dataset/` le immagini del test set, che nel
dataset FSOCO di partenza stanno negli stessi archivi e con gli stessi nomi: senza quel filtro
finivano nel training set e il modello si allenava su una parte del test.

`main train` sovrascrive `data/svm_hog_sgd_proposal.yml` e `data/svm_stage2_rbf_balanced.yml`, cioè
i due modelli che `main detect` carica: conviene farne una copia prima. Lo stadio 2 usa `trainAuto`
con 5-fold cross validation su 15.000 patch per classe, quindi è la parte lenta di gran lunga.

Due note sulle soglie, imparate riallenando davvero da zero:

- **La soglia dello stadio 2 va con il modello.** Ora C è fissato in `trainStage2()`, quindi la
  scala degli score non si sposta più da un training all'altro e `kStage2Thr = 0.1` resta valida.
  Se si cambiano C, il numero di campioni o le feature, va ritarata: `./eval_pipeline 0 0.1 <soglia>`
  fa lo sweep. Storicamente questo è costato tre training buttati.
- **Vecchia nota, ora rientrata:** `trainAuto` sceglie da sé C e
  gamma, quindi gli score di un modello riallenato stanno su una scala diversa: riusare la soglia
  del modello precedente (0.01) su un modello nuovo fa scendere la macro F1 da 0.45 a **0.06**, e
  sembra che la pipeline sia rotta quando invece è solo mal tagliata (il mAP, che non dipende dalla
  soglia, resta 0.12). Per questo `trainStage2()` misura la soglia migliore sulle stesse patch e la
  scrive in `data/svm_stage2_rbf_balanced.yml.thr`, che `detectCones()` legge all'avvio. Se il file
  manca si torna alla costante `kStage2Thr`, e la cosa viene stampata.
- `trainStage2()` filtra le patch con la soglia dello stadio 1 (15, la stessa di `detectCones`). Se
  lo stadio 1 venisse riallenato su dati molto diversi i suoi score cambierebbero scala e il filtro
  potrebbe non lasciar passare nulla; in quel caso la funzione si ferma con un messaggio esplicito.

### Cosa serve, oltre al repo

`.gitignore` esclude `data/`, `dataset/` e `test_set/`, quindi chi clona ottiene il codice ma **non**
i modelli allenati né le immagini, e nessuna delle tre modalità principali parte. Servono, dai dati
FSOCO: `dataset/` (le 2000 immagini sintetiche più i JSON) per `main train`, e
`test_set/segmentation_test/` per `main eval`. I due modelli `.yml` si rigenerano con `main train`
in circa 7 minuti.

## Prossimi passi possibili

In ordine di valore atteso sulla macro F1, che è la metrica principale:

- **La cascata SVM è il collo di bottiglia.** I proposal di colore coprono il 97% dei coni, la
  cascata ne lascia passare il 71%, e i due SVM sono allenati solo sul dataset sintetico. Riallenarli
  includendo patch dal reale è la leva più grande rimasta. Attenzione: `data/test_real.txt` è
  l'unico set fuori distribuzione, se lo si usa per il training non resta niente per validare.
- **I coni arancioni** (F1 0.250 e 0.400 contro ~0.58 di giallo e blu) valgono metà della macro F1
  pur essendo l'8% dei coni. Il classificatore ormai ne sbaglia pochi (small orange 5/14, big
  orange 9/18 a detection perfetta): il problema è che la cascata ne lascia passare la metà.
- **Localizzazione**: a IoU 0.95 il mAP è 0, e circa metà dei coni che la cascata trova non arriva a
  IoU 0.5. Il box finale viene dalla maschera, quindi qui si guadagna migliorando la segmentazione
  dei coni piccoli (mIoU 0.58 sotto i 25px contro 0.73 sopra i 60px).
- Il 9.3% di GT box senza alcun pixel di colore rilevabile resta irrecuperabile con le soglie HSV
  attuali (limite noto, non un bug).

### Cosa è già stato provato

**Ancoraggio verticale del box** (centrare il box al 42% dall'alto del blob invece che sul
centroide): validato solo a livello geometrico, mai portato in `src/`, e i tool e i modelli relativi
sono stati rimossi quando si è deciso di non proseguire. La stessa idea è poi rientrata dalla porta
principale sotto forma di `kPadTop = 0`, che sposta il box verso il basso e vale +0.05 di macro F1.

### Tentativi già fatti che NON hanno funzionato

Documentati per non ripeterli; i parametri sono ancora nei file di varianti in `tools/`.

- **Rigettare i box senza abbastanza pixel colorati** (`classifier_variants.hpp`, varianti 3-6): la
  frazione di pixel colorati non separa i coni dai falsi positivi, perché dentro un box allargato
  anche un cono vero occupa poca area. Macro F1 da 0.454 a 0.361 già al 5%.
- **Togliere all'arancione il ruolo di ramo `else`** (variante 2): nessun effetto. Nei falsi positivi
  l'arancione vince davvero per numero di pixel, perché la sua banda H 0-15 copre terra e asfalto
  rossastro.
- **Alzare la soglia della NMS** (0.2-0.5): peggiora sempre, 0.1 è già l'ottimo.
- **Spostare la soglia dello stadio 2**: 0.01 è l'ottimo per la macro F1. Abbassarla aiuta di poco il
  mAP (+0.008) e distrugge la F1; alzarla distrugge entrambe.
- **Allargare il padding** per dare più contesto alla segmentazione: peggiora, la maschera aggancia
  altri oggetti. È la direzione opposta (ridurlo) quella che ha pagato.
