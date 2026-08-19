# Tuning log — color proposals (src/hog.cpp)

Misure su `./tune ../dataset ../data/train.txt ../data/dataset.csv 200`.
IoU threshold 0.3, invariato. Ogni riga cambia UN parametro rispetto alla
config attiva precedente (baseline = riga 1). "Attiva" = config rimasta nel
codice dopo la riga.

| # | Modifica | Recall | Recall <20px | Recall 20-50px | Recall >50px | Cand/img | nulla% | frammento% | fuso% | disallineato% | Esito |
|---|----------|--------|---------------|-----------------|--------------|----------|--------|------------|-------|----------------|-------|
| 0 | Baseline: S=60,V=40, apertura ellittica 3x3, chiusura 3x7 | 0.4531 | 0.3404 | 0.4923 | 0.6125 | 842.2 | 47.29 | 22.83 | 25.56 | 4.31 | riferimento |
| 1 | + maschera rosso H 170-179 (S60/V40) | 0.4776 | 0.3507 | 0.5294 | 0.6431 | 882.25 | 44.45 | 22.85 | 28.51 | 4.19 | MEGLIO, tenuta |
| 2 | arancio: H_min 5 -> 0 | 0.4928 | 0.3639 | 0.5442 | 0.6631 | 890.70 | 46.02 | 18.31 | 31.40 | 4.27 | MEGLIO, tenuta |
| 3 | whiteBalance disabilitato | 0.5024 | 0.3089 | 0.5954 | 0.7291 | 1044.95 | 41.73 | 21.47 | 31.72 | 5.08 | MEGLIO in totale, tenuta (ma <20px peggiora: 0.364->0.309) |
| 4 | chiusura 3x7 disabilitata (solo apertura 3x3) | 0.5411 | 0.3564 | 0.6300 | 0.7574 | 2257.98 | 28.26 | 42.42 | 20.14 | 9.18 | MEGLIO, tenuta. Cand/img esplode (2258) ma è secondario ora. "fuso" crolla 31.7->20.1, "frammento" sale 21.5->42.4 come atteso: la chiusura era ciò che fondeva sia frammenti che sfondo |
| 5 | banda blu allargata: H 100-130 -> 95-135 | 0.5500 | 0.3691 | 0.6389 | 0.7585 | 2585.35 | 27.70 | 40.11 | 21.93 | 10.27 | MEGLIO, tenuta |
| 6 | banda gialla allargata: H 20-35 -> 18-40 | 0.5679 | 0.4006 | 0.6517 | 0.7574 | 2633.45 | 28.17 | 37.75 | 23.33 | 10.75 | MEGLIO, tenuta |
| 7 | banda rossa allargata: H 170-179 -> 165-179 | 0.5715 | 0.4052 | 0.6543 | 0.7609 | 2644.32 | 28.02 | 37.73 | 23.53 | 10.72 | MEGLIO (marginale), tenuta |
| 8 | banda arancione allargata: H_max 20 -> 22 | 0.5715 | 0.4052 | 0.6543 | 0.7609 | 2644.32 | 28.02 | 37.73 | 23.53 | 10.72 | NESSUN EFFETTO (ridondante, già coperta da giallo che parte da 18) — riportato a 20 |
| 9 | S: 60 -> 80 (tutte le bande) | 0.5474 | 0.3708 | 0.6229 | 0.7715 | 1694.68 | 26.16 | 52.15 | 13.13 | 8.56 | PEGGIORE, tornato a S=60 |
| 10 | V: 40 -> 60 (tutte le bande, S=60) | 0.5816 | 0.4138 | 0.6639 | 0.7750 | 2382.82 | 27.08 | 39.74 | 21.97 | 11.21 | MEGLIO, tenuta |
| 11 | V: 60 -> 80 (tutte le bande, S=60) | 0.5695 | 0.4275 | 0.6479 | 0.7173 | 2207.99 | 27.17 | 44.16 | 18.28 | 10.40 | PEGGIORE, tornato a V=60 |

### Secondo giro: riverifica interazioni sulla config finale (bande larghe, no chiusura)

Ipotesi dell'utente: whiteBalance e S/V sono stati fissati quando le bande H
erano ancora strette (step 3, 9-11). Con le bande allargate (step 5-7) le
interazioni potrebbero essere cambiate. Riverifica.

| # | Modifica | Recall | Recall <20px | Recall 20-50px | Recall >50px | Cand/img | nulla% | frammento% | fuso% | disallineato% | Esito |
|---|----------|--------|---------------|-----------------|--------------|----------|--------|------------|-------|----------------|-------|
| 12 | whiteBalance riattivato (su config step 11→10, bande larghe, S=60 V=60) | 0.5758 | 0.4562 | 0.6236 | 0.7338 | 1693.01 | 33.01 | 36.07 | 22.06 | 8.85 | Totale PEGGIORE di 0.006 (0.5816->0.5758) ma <20px MEGLIORE di 0.042 (0.4138->0.4562), che è la fascia più numerosa (1745/4156=42% dei GT). Trade-off, non un miglioramento netto. Vedi nota sotto. |
| 13 | (con WB attivo) S: 60 -> 80 | 0.5541 | 0.3868 | 0.6204 | 0.7762 | 1282.67 | 28.06 | 51.16 | 13.33 | 7.45 | PEGGIORE, tornato a S=60. Conferma: S alto resta dannoso anche con WB attivo |
| 14 | (con WB attivo) V: 60 -> 80 | 0.5679 | 0.4670 | 0.6127 | 0.6926 | 1456.40 | 32.68 | 41.20 | 18.60 | 7.52 | Totale PEGGIORE (0.5758->0.5679), <20px ancora meglio (0.4562->0.4670) ma trade-off si aggrava sulle fasce grandi. Tornato a V=60 |
| 15 | (con WB attivo) V: 60 -> 40 (valore originale) | 0.5659 | 0.4487 | 0.6127 | 0.7208 | 1935.32 | 33.54 | 34.37 | 24.11 | 7.98 | PEGGIORE di V=60+WB, tornato a V=60. Conferma: V=60 è l'ottimo locale sia con che senza WB |

**Decisione presa**: tra le due config non dominanti (0 no-WB@V60 = 0.5816 totale/0.4138 su <20px, vs 12 WB-on@V60 = 0.5758 totale/0.4562 su <20px), tengo **WB riattivato** come config finale. Motivazione: il delta sul totale è marginale (0.0058, ~1%), mentre il guadagno sulla fascia <20px (+0.0424) riguarda il 42% dei ground truth ed è la fascia storicamente più debole (era la più bassa fin dalla baseline). Se preferisci ottimizzare solo il totale, la config 0 (WB disattivato) resta a disposizione: è sufficiente togliere la chiamata a `whiteBalance` in `colorProposals`.

## Fusione dei frammenti a livello di box (soluzione strutturale)

Implementata su config finale (WB on, S=60 V=60, bande larghe, no chiusura):
maschere per colore tenute separate, findContours per maschera, poi fusione
via union-find dei box con overlap orizzontale >= 40% della larghezza minima
e gap verticale <= altezza del frammento più piccolo.

| # | Modifica | Recall | Recall <20px | Recall 20-50px | Recall >50px | Cand/img | nulla% | frammento% | fuso% | disallineato% | Esito |
|---|----------|--------|---------------|-----------------|--------------|----------|--------|------------|-------|----------------|-------|
| 16 | maschere separate per colore + merge box (overlap>=0.4, gap<=minHeight) | 0.5626 | 0.4057 | 0.6255 | 0.7691 | 1502.6 | 31.08 | 35.97 | 26.84 | 6.11 | PEGGIORE del totale pre-merge (0.5758->0.5626). Candidati -11% (1693->1503, atteso) ma "fuso" peggiora (22.06->26.84): il merge unisce anche blob non correlati. "frammento" quasi invariato (36.07->35.97): il merge non sta risolvendo il problema che doveva risolvere. Soglie da rivedere. |
| 16b | (diagnostico) maschere separate, merge DISATTIVATO | 0.5760 | 0.4172 | 0.6498 | 0.7668 | 2776.74 | 19.41 | 52.78 | 17.31 | 10.50 | Isola l'effetto: separare le maschere da sole non cambia il recall (0.5758->0.5760, invariato). Il calo allo step 16 è causato interamente dalla logica di merge, che fa più danni (falsi merge -> "fuso") di quanti frammenti risolve. Ispezionate immagini di debug: sui coni distanti/piccoli i pixel di colore sono sparsi a chiazze (non un pulito split alto/basso), quindi il criterio overlap+gap fondeva coppie non correlate. |
| 17 | merge irrigidito: overlap H >=0.4->0.6, + vincolo larghezze comparabili (max/min <= 2.5), gap verticale <= minHeight -> <= 0.5*minHeight | **0.5811** | 0.4218 | 0.6498 | 0.7821 | 2491.43 | 19.64 | 50.49 | 20.28 | 9.59 | MEGLIORE: sopra sia la baseline pre-merge (0.5758) sia le maschere-separate-senza-merge (0.5760). Candidati -10% rispetto a 16b (2777->2491) mantenendo il recall. Tenuta come configurazione finale. |
| 18 | gap: 0.5*minHeight -> 0.3*minHeight (più stretto) | 0.5744 | 0.4126 | 0.6447 | 0.7774 | 2593.76 | 19.39 | 51.39 | 19.22 | 10.01 | PEGGIORE, tornato a 0.5*minHeight |
| 19 | overlap: 0.6 -> 0.7 | 0.5821 | 0.4218 | 0.6511 | 0.7845 | 2550.36 | 19.75 | 50.89 | 19.92 | 9.44 | MEGLIORE (marginale), tenuta |
| 20 | overlap: 0.7 -> 0.8 | 0.5840 | 0.4224 | 0.6549 | 0.7856 | 2587.28 | 19.84 | 51.42 | 19.32 | 9.43 | MEGLIORE (marginale), tenuta |
| 21 | overlap: 0.8 -> 0.9 | 0.5840 | 0.4229 | 0.6549 | 0.7845 | 2623.94 | 19.84 | 51.71 | 18.57 | 9.89 | STESSO recall ma più candidati (2624 vs 2587), tornato a 0.8 |

## Configurazione finale definitiva

```
whiteBalance: ATTIVO (Shades of Gray, p=6)
colorMasks (HSV), tenute separate fino a dopo il merge:
  blue:   H  95-135   S 60-255   V 60-255
  orange: H   0- 20   S 60-255   V 60-255
  yellow: H  18- 40   S 60-255   V 60-255
  red:    H 165-179   S 60-255   V 60-255
morphCleaning: solo apertura ellittica 3x3 (chiusura rimossa)
findContours: per maschera colore separata
mergeFragments: union-find sui box della STESSA maschera colore.
  Due box si fondono se: overlap orizzontale >= 80% della larghezza minore,
  larghezze comparabili (max/min <= 2.5), gap verticale <= 50% dell'altezza
  del frammento più piccolo.
```

Risultato finale su 200 immagini:
- **RECALL CANDIDATI: 0.5840**
- Recall <20px: 0.4224 | 20-50px: 0.6549 | >50px: 0.7856
- Candidati/immagine: 2587.28
- Diagnosi persi: nulla 19.84% | frammento 51.42% | fuso 19.32% | disallineato 9.43%

**Recall per coni >= 20px (esclude la fascia più piccola/difficile): 0.7011**
(calcolato da covBySize/gtBySize dello stratificato: (1023+667)/(1562+849))

## Conclusione: limite operativo del metodo classico

Il recall totale (0.5840) resta sotto la soglia 0.85 richiesta e anche sotto
la fascia 0.7-0.75 indicata come possibile esito. Tuttavia isolando la
fascia >=20px il recall sale a **0.70**, e per i coni >50px arriva a **0.79**.
La fascia <20px (1745/4156 = 42% dei ground truth) resta strutturalmente il
collo di bottiglia: "nulla in zona" (19.8% dei persi totali, ma
prevalentemente concentrato qui) mostra che su questi coni i pixel di
colore non superano mai le soglie HSV, indipendentemente da come vengono
poi raggruppati in box — non è un problema di frammentazione ma di segnale
del colore assente/troppo debole all'origine (compressione video, dimensione
sub-pixel, sfocatura da movimento).

La fusione dei frammenti a livello di box (proposta nella sessione precedente,
implementata qui) ha funzionato come previsto: ha spostato "nulla in zona"
da 27-33% a 19.8% (i frammenti sparsi ora producono almeno un candidato
parziale) e ha tenuto "fuso" sotto controllo (19.3%, contro il 31%+ che si
otteneva con la chiusura morfologica) senza introdurre l'effetto collaterale
delle fusioni cono+sfondo di colore diverso. Il costo era la fusione errata
di blob non correlati dello stesso colore quando i criteri erano troppo
permissivi (step 16, IoU-rilevante); irrigidendo i criteri (step 17-20) il
danno è stato eliminato mantenendo il beneficio.

**Raccomandazione**: dichiarare un limite operativo a 20px e riportare
SEMPRE il recall complessivo su tutte le ground truth insieme allo
spaccato per fascia — mai filtrare le ground truth dal denominatore per
far salire il numero riportato:
- complessivo: 0.5840 (poi rivisto a 0.5876, vedi correzione sotto)
- coni >= 20px: 0.70 (metodo classico adeguato)
- coni < 20px: 0.42 (limitazione nota; il segnale colore è spesso sotto
  soglia a quella distanza — servirebbe upsampling, super-resolution, o un
  fallback diverso a valle)

## Correzione: la decisione sul white balance era sbagliata

Il criterio "tieni WB perché la fascia <20px è la più numerosa (42%)" era
il criterio sbagliato. Il criterio corretto: il recall dei proposal conta
solo per le fasce che lo stadio HOG+SVM a valle può davvero classificare.
Un cono <20px ridimensionato alla finestra HOG produce un descrittore che è
quasi solo interpolazione — quei coni si perdono comunque più avanti nella
pipeline, quindi ottimizzare i proposal per loro è uno spreco se costa
recall sulle fasce >=20px che l'SVM può effettivamente usare.

Verifica numerica: dallo step 12, il totale scendeva di 0.0058 mentre
<20px saliva di 0.0424. Se <20px è il 42% dei GT, quel guadagno da solo
vale +0.0424*0.42 = +0.0178 sul totale; siccome il totale netto scendeva,
le fasce medie/grandi (58% dei GT) dovevano aver perso circa
(0.0178+0.0058)/0.58 = 0.041, cioè ~4 punti di recall sulla fascia
utilizzabile. Un pessimo scambio.

**whiteBalance ridisattivato definitivamente.** Ri-misurato con il resto
della config finale (merge irrigidito, step 17-20) e WB OFF:

| # | Modifica | Recall | Recall <20px | Recall 20-50px | Recall >50px | Cand/img | nulla% | frammento% | fuso% | disallineato% | Esito |
|---|----------|--------|---------------|-----------------|--------------|----------|--------|------------|-------|----------------|-------|
| 22 | whiteBalance ridisattivato (su config finale con merge) | **0.5876** | 0.4023 | 0.6793 | 0.7998 | 3258.18 | 18.14 | 50.82 | 18.55 | 12.49 | MEGLIORE sia sul totale (0.5840->0.5876) SIA sulla fascia utilizzabile >=20px (0.70->0.7217): nessun vero trade-off, l'ipotesi "tieni WB per la fascia numerosa" era basata sul criterio sbagliato. WB resta disattivato. |

Recall >=20px ricalcolato: (0.679257*1562 + 0.799764*849) / (1562+849) = **0.7217**

## Filtri geometrici (riduzione candidati, obiettivo <500/img mantenendo il recall)

Applicati alla fine di `colorProposals`, dopo il merge, su tutti i candidati
insieme (non per-colore). Un filtro alla volta, misurato con lo stesso
protocollo. Config di partenza: WB off, merge irrigidito (step 17-20),
3258.18 candidati/img, recall 0.5876.

| # | Modifica | Recall | Recall <20px | Recall 20-50px | Recall >50px | Cand/img | Recall >=20px | Esito |
|---|----------|--------|---------------|-----------------|--------------|----------|----------------|-------|
| 23 | altezza minima: 8px | 0.5464 | 0.3054 | 0.6780 | 0.7998 | 718.04 | 0.7211 | Candidati -78% (3258->718). Recall totale scende (colpisce quasi solo <20px, coerente: filtra rumore e frammenti minuscoli). Recall >=20px INVARIATO (0.7217->0.7211). Scambio ottimo per lo stadio HOG+SVM a valle. Tenuto. |
| 24 | + area minima: 25px^2 | 0.5464 | 0.3054 | 0.6780 | 0.7998 | 716.01 | 0.7211 | Effetto trascurabile (atteso: altezza>=8 già implica quasi sempre area>=25 salvo sliver larghi 1-3px). Candidati -0.3% (718->716), recall invariato. Tenuto (costo zero). |
| 25 | + w/h in [0.45, 1.00] (valori richiesti) | 0.2813 | 0.2132 | 0.3572 | 0.2815 | 414.89 | — | CROLLO drastico anche su >50px (0.7998->0.2815). Ispezione debug: i frammenti per-colore (punta stretta/alta, base larga/bassa di un cono tagliato) hanno w/h molto lontani da un cono intero, perché il merge non li unisce (differiscono troppo in larghezza, vincolo max/min<=2.5 del passo 17). Il filtro colpisce esattamente i frammenti parziali che oggi ottengono comunque IoU>=0.3. Range troppo stretto per questa pipeline a frammenti. |
| 26 | w/h allargato a [0.2, 1.5] | 0.4278 | 0.2957 | 0.5506 | 0.4735 | 619.55 | — | Ancora troppo aggressivo, >50px ancora colpito (0.7998->0.4735). Allargo ulteriormente. |
| 27 | w/h allargato a [0.1, 3.0] | 0.5464 | 0.3054 | 0.6780 | 0.7998 | 702.82 | 0.7211 | Recall torna esattamente al livello pre-filtro (nessun costo), ma candidati -1.8% soltanto (716->703): quasi un no-op a questo range. |
| 28 | w/h stretto a [0.15, 2.2] | 0.5455 | 0.3054 | 0.6773 | 0.7962 | 683.44 | 0.7205 | Costo trascurabile (-0.0009 totale, -0.0006 su >=20px), candidati -2.8% in più (703->683). Punto di compromesso, tenuto. |
| 29 | w/h stretto a [0.2, 2.0] (tentativo di stringere oltre) | 0.5409 | 0.3054 | 0.6741 | 0.7797 | 673.69 | 0.7078 | Il costo comincia a salire (>50px -2pt) per un guadagno marginale (683->674, -1.3%). Non conviene, tornato a [0.15, 2.2]. |
| 30 | + larghezza massima: 15% larghezza immagine (dataset a 45 risoluzioni -> soglia relativa, non assoluta) | 0.5450 | 0.3054 | 0.6773 | 0.7939 | 683.02 | 0.7198 | Effetto quasi nullo (683.44->683.02, -0.4 candidati). Ispezione debug (immagine 347): il "rettangolone dell'orizzonte" temuto (cielo/linea alberi) non è più presente nel set di candidati a questo punto della pipeline — probabilmente già eliminato dal vincolo di larghezze comparabili nel merge (step 17) o dall'apertura. |
| 31 | larghezza massima: 15% -> 8% | 0.5363 | 0.3054 | 0.6773 | 0.7515 | 682.03 | 0.6979 | Costo reale (>50px -4pt) per zero guadagno sui candidati (683.02->682.03, -1!). A questa soglia si iniziano a tagliare coni grandi/vicini legittimi (che possono superare l'8% della larghezza immagine) senza toccare rumore. Tornato a 15%. |

**Scoperta**: il filtro di larghezza massima non serve a molto in questa pipeline: ispezionando le immagini di debug, il problema dei "rettangoloni dell'orizzonte" (cielo/linea alberi) risulta già ampiamente risolto a monte. Il rumore residuo che tiene i candidati a ~683/img è di natura diversa e più insidiosa: **texture del terreno** (ciottolato, ghiaia, erba secca) che supera le soglie HSV a chiazze su ampie porzioni di superficie, ben visibile confrontando l'immagine 1292 (prato/ciottolato, decine di falsi positivi sparsi) con la 6328 (asfalto pulito, quasi zero falsi positivi). Questo rumore ha dimensioni e aspect ratio plausibili per un cono, quindi nessuno dei quattro filtri richiesti (altezza, area, w/h, larghezza) lo elimina in modo mirato.

## Configurazione finale (dopo filtri geometrici)

```
whiteBalance: DISATTIVO
colorMasks (HSV), separate: blue H95-135, orange H0-20, yellow H18-40,
  red H165-179, tutte S60-255 V60-255
morphCleaning: solo apertura ellittica 3x3
findContours: per maschera colore
mergeFragments: union-find, overlap H>=80%, larghezze max/min<=2.5,
  gap verticale <= 50% altezza frammento minore
filterCandidates: altezza>=8px, area>=25px^2, w/h in [0.15, 2.2],
  larghezza <= 15% larghezza immagine
```

**Report onesto (nessuna ground truth esclusa dal denominatore — 4156 GT totali in tutti i numeri sotto):**

| Metrica | Recall | GT nel bucket |
|---|---|---|
| **Complessivo** | **0.5450** | 4156 (100%) |
| Coni < 20px | 0.3054 | 1745 (42%) — limite noto: segnale colore spesso assente all'origine |
| Coni 20-50px | 0.6773 | 1562 (38%) |
| Coni > 50px | 0.7939 | 849 (20%) |
| Coni >= 20px (aggregato, indicativo per l'SVM a valle) | 0.7198 | 2411 (58%) |

Candidati/immagine: **683.02** (da 3258 pre-filtri geometrici, -79%; da 842 alla baseline originale S60/V40+chiusura).
Obiettivo utente <500/img: **non raggiunto** con i soli quattro filtri richiesti — il rumore da texture del terreno (non da forma) resta il fattore dominante e richiederebbe un criterio diverso (es. rigetto per densità di bordi/contrasto locale, o soglie colore più severe a scapito del recall) per essere abbattuto ulteriormente.

## Diagnostica forma dei match (w/h) e fix del merge

Aggiunta a `tune.cpp` (su richiesta esplicita, diagnostica additiva — non
tocca calculateIoU, soglia 0.3, stratificazione o conteggio GT): per ogni
GT coperta (bestIoU>=0.3) si registra w/h del candidato migliore, poi si
stampano p10/mediana/p90.

**Misura alla config dello step 31** (overlap 0.8, larghezze max/min<=2.5,
gap<=0.5*minHeight): p10=0.545, **mediana=0.923**, p90=1.698.

Mediana attesa per un cono intero: ~0.70-0.73. Il valore misurato (0.92) e
il range molto ampio (fattore >3x tra p10 e p90) confermano l'ipotesi: i
candidati che raggiungono IoU>=0.3 sono prevalentemente FRAMMENTI (punta
stretta+alta o base larga+bassa), non coni interi. La metrica di recall a
soglia 0.3 non lo segnalava perché un frammento che copre "buona parte"
del cono passa comunque la soglia.

**Causa**: nel merge (step 17), il vincolo "larghezze comparabili" (max/min
<= 2.5) impedisce la fusione fisiologica di punta (stretta) e base (larga)
dello stesso cono, che per costruzione hanno larghezze molto diverse.

Rilassati due vincoli del merge, uno alla volta, ri-misurando anche p50/p90:

| # | Modifica | Recall | Cand/img | fuso% | w/h p10 | w/h mediana | w/h p90 | Esito |
|---|----------|--------|----------|-------|---------|-------------|---------|-------|
| 32 | vincolo larghezze: max/min 2.5 -> 4.0 | 0.5488 | 639.47 | 19.73 | 0.539 | 0.838 | 1.670 | Recall e candidati migliorano insieme; mediana si avvicina (0.92->0.84) ma non ancora al target |
| 33 | vincolo larghezze: 4.0 -> 6.0 | 0.5462 | 605.66 | 20.78 | 0.538 | 0.833 | 1.667 | Leggero PEGGIORAMENTO del recall, mediana non migliora oltre. Tornato a 4.0 |
| 34 | gap verticale: 0.5*minHeight -> 0.8*minHeight | **0.5758** | 635.34 | 22.46 | 0.500 | **0.724** | 1.250 | Salto notevole su tutti i fronti: recall +0.027, mediana entra nel range atteso (0.70-0.73), p90 si stringe (1.67->1.25). Il gap era il vincolo piu' limitante: la fascia nera tra punta e base e' spesso piu' alta del frammento piu' piccolo |
| 35 | gap: 0.8 -> 1.2*minHeight | **0.5972** | 625.74 | 25.81 | 0.489 | 0.700 | 1.000 | Ancora meglio su tutto: recall, candidati, forma (p90=1.0) |
| 36 | gap: 1.2 -> 2.0*minHeight | **0.6001** | 608.37 | 30.32 | 0.464 | 0.690 | 0.956 | Picco di recall. "fuso" continua a salire (25.8->30.3%, atteso: piu' merge = piu' rischio di unire oggetti diversi) ma il saldo netto e' positivo |
| 37 | gap: 2.0 -> 3.0*minHeight | 0.5876 | 570.64 | 33.31 | 0.450 | 0.684 | 0.946 | PEGGIORE: il recall scende, "fuso" continua a salire senza piu' benefici. Superato il punto di equilibrio |
| 38 | gap: 3.0 -> 1.6*minHeight (tentativo tra 1.2 e 2.0) | 0.5982 | 614.03 | 28.20 | 0.474 | 0.695 | 0.983 | Dominato da 2.0*minHeight (recall e candidati entrambi peggiori). Scartato |
| 39 | gap: 1.6 -> 2.3*minHeight | 0.5953 | 592.02 | 31.03 | 0.462 | 0.688 | 0.954 | Leggermente sotto il picco di 2.0 sul recall. **gap=2.0*minHeight confermato come ottimo** |
| 40 | (a gap=2.0 fisso) vincolo larghezze: 4.0 -> 5.0 | 0.5987 | 579.58 | 31.59 | 0.462 | 0.686 | 0.938 | Quasi equivalente a 4.0 (recall leggermente sotto, candidati piu' bassi). Priorita' al recall in questa fase: tornato a 4.0 |
| 41 | (a gap=2.0, larghezze=4.0 fissi) overlap: 0.8 -> 0.6 | 0.5958 | 595.83 | 33.10 | 0.453 | 0.680 | 0.923 | Leggermente PEGGIORE, tornato a overlap=0.8 |

**Criteri di fusione finali**: overlap orizzontale >= 80% larghezza minore,
larghezze max/min <= 4.0 (invece di 2.5 — corretto per accomodare
punta stretta + base larga dello stesso cono), gap verticale <=
200% dell'altezza del frammento piu' piccolo (invece di 50% — la fascia
nera/bianca tra le due parti colorate di un cono e' spesso piu' alta del
frammento piu' piccolo, specialmente su coni piccoli/distanti).

**Con questi criteri, rifiltrato con i filtri geometrici** (step 23-31):
riverificato che l'intervallo w/h [0.15, 2.2] resta necessario nonostante
il fix — provato a restringere a [0.35,1.1] (candidati 420, ma recall
-4.6pt) e [0.3,1.3] (candidati 498, sotto il target ma recall -2.4pt).
L'utente ha esplicitamente deprioritizzato il target di 500 candidati;
tenuto [0.15, 2.2] per non sacrificare recall.

## Configurazione e numeri finali (dopo il fix del merge)

```
mergeFragments: overlap H >= 80%, larghezze max/min <= 4.0,
                gap verticale <= 200% dell'altezza del frammento minore
(resto della pipeline invariato: WB off, bande HSV larghe, S60/V60,
 solo apertura 3x3, filterCandidates come da step 30)
```

| Metrica | Valore |
|---|---|
| RECALL CANDIDATI (su 4156 GT) | **0.6001** |
| Recall <20px (n=1745, 42%) | 0.3622 |
| Recall 20-50px (n=1562, 38%) | 0.7369 |
| Recall >50px (n=849, 20%) | 0.8375 |
| Recall >=20px (aggregato) | 0.7724 |
| Candidati/immagine | 608.37 |
| w/h dei match: p10 / mediana / p90 | 0.464 / **0.690** / 0.956 |

La mediana w/h (0.690) e' ora dentro il range atteso per un cono intero
(0.70-0.73): i proposal rappresentano prevalentemente coni interi, non
frammenti. Il recall e' salito da 0.545 a 0.600 e i candidati sono scesi
da 683 a 608 nello stesso passaggio — nessun trade-off, solo un bug nel
criterio di fusione.


## Configurazione finale (migliore trovata)

Tutti i parametri esplorati in priorità a-e sono stati testati; nessun altro
valore in coda alla lista rimane da provare senza uscire dal perimetro
concordato (solo src/hog.cpp, metrica invariata).

```
whiteBalance:  DISABILITATO
colorMask (HSV):
  blue:   H  95-135   S 60-255   V 60-255
  orange: H   0- 20   S 60-255   V 60-255
  yellow: H  18- 40   S 60-255   V 60-255
  red:    H 165-179   S 60-255   V 60-255   (nuova banda, wrap-around H=0)
morphCleaning: solo apertura ellittica 3x3, CHIUSURA RIMOSSA
```

Risultato su 200 immagini:
- RECALL CANDIDATI: **0.5816**
- Recall <20px: 0.4138 | 20-50px: 0.6639 | >50px: 0.7750
- Candidati/immagine: 2382.82
- Diagnosi persi: nulla 27.08% | frammento 39.74% | fuso 21.97% | disallineato 11.21%

## Esito: sotto la soglia di 0.7 — ricerca fermata

Recall massimo raggiunto (0.5816) resta sotto 0.7 dopo aver esplorato tutti i
parametri in ordine di priorità (a-e). Come da istruzioni, la ricerca si
ferma qui invece di continuare a tentare valori random.

**Diagnosi confermata**: la categoria di perdita dominante è ormai
"frammento" (39.74%, era 22.83% alla baseline) e non più "nulla in zona"
(27.08%, era 47.29%). Le soglie di colore ora catturano la maggior parte dei
pixel dei coni; il collo di bottiglia è che i due segmenti del cono (striscia
bianca centrale) restano due componenti connesse separate nella maschera.
Riattivare una chiusura morfologica per fonderli riporta su la categoria
"fuso" (era 25-31% con chiusura 3x7 attiva) perché la stessa operazione
salda indiscriminatamente anche cono+sfondo quando sono vicini nei pixel.
Nessuna combinazione lineare di soglie HSV o kernel morfologici risolve
entrambi i problemi insieme, perché entrambi agiscono sulla stessa
rappresentazione (maschera di pixel) senza distinguere "questo è lo stesso
cono spezzato in due" da "questo è cono+sfondo che si toccano".

**Soluzione strutturale proposta** (non ancora implementata):
fondere i frammenti a livello di bounding box DOPO findContours, invece che
a livello di maschera prima di essa. Idea: nella config attuale (senza
chiusura), findCandidateBoxes produce già i due segmenti separati del cono
come contorni distinti. Un passo di post-processing su `candidates` che
unisce coppie di box con overlap orizzontale significativo, gap verticale
piccolo relativo alla loro dimensione, e appartenenti alla stessa banda di
colore, dovrebbe ricostruire il bounding box del cono intero senza mai
toccare pixel di sfondo lontani (a differenza della chiusura morfologica,
che opera in un raggio fisso su tutta l'immagine indipendentemente dal
colore). Questo lascia intatta la maschera (quindi non introduce nuove
fusioni cono+sfondo) e agisce solo sulla geometria dei box già trovati.

