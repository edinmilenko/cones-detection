#pragma once
#include <string>

// unisce fsoco_bounding_boxes_train e fsoco_segmentation_train in dataset/, rinumerando le
// coppie immagine/annotazione condivise (1.jpg, 1.json, 2.jpg, 2.json, ...).
void dataLoader();

// estrae le bounding box (annotazioni "rectangle") da dataset/ in dataset.csv (img,x1,y1,x2,y2).
void csvDatasetMaker(std::string datasetDir);

// split train/test deterministico (seed fisso) su un sottoinsieme di devSize immagini.
void makeSplit(std::string csvPath, std::string outDir, int devSize = 2000, double trainFrac = 0.7, int seed = 42);

// disegna le annotazioni (rettangoli o maschere bitmap) di dataset/ e salva in labeled/, per
// ispezione visiva.
void labeler();
