# Task to complete for us: 

- data loader, create a "dataset" folder saving the image with bounding box with coordinates extracted from the json (for us, not to be delivered).

- labeler: takes (img, its json) pairs and draws the bounding boxes (&segmentations?) on top and then saves them in "labeled" folder

- function to evaluate scores of bounding boxes found vs ground truth, used after pipeline for evaluation, some code from labeler will be recycled

# PIPELINE FOR CONES DETECTION

2 opzioni: nella root crei cartella "dataset" tutte le coppie immagini-json (se ho 10k foto avrai dentro dataset 10k coppie 1.json-1.png, 2.json-2.png).xx (altra opzione sarebbe avere un vettore di coppie di paths immagine e json per tenere il dataset "invariato", da capire se necessario altrimenti si evita)


dataset/
├── cone_00001.jpg
├── cone_00001.json
├── cone_00002.jpg
├── cone_00002.json
└── ...

        ↓

Ground Truth Loader (labeler)
        ↓
┌───────────────────────┐
│ image + annotations   │
│ bbox + segmentation   │
└───────────────────────┘

        ↓

    Detection
        │
      ┌─┴─────────────────┐
      ↓                   ↓
Viola-Jones        Template Matching, paired with data augmentation with rotation, cropping, light stretching
      │                   │
      └─────────┬─────────┘
                ↓
            same evaluator
                ↓
      IoU / Precision / Recall
                ↓
          performance comparison
                ↓
        predicted bounding boxes
                ↓
        Color Classification, count all the pixels with a certain color and within a certain range of color
                ↓
        Segmentation, based on the label (color) found from classification.
                ↓
        Evaluation


----

                    DATASET (subset per tentativi veloci..)
                       │
                       ▼
             analyze_vj_dataset
                       │
                 2264 bbox
                       │
                       ▼
             prepare_vj_dataset
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 annotations.txt             negatives.txt
          │                         │
          │                    negatives/
          │                         │
          └──────────┬──────────────┘
                     │
                     ▼
              OpenCV 3.4.20
                     │
                     ▼
          opencv_createsamples
                     │
                     ▼
                positives.vec
                     │
                     ▼
           opencv_traincascade
                     │
                     ▼
                cascade.xml
                     │
                     ▼
              OpenCV 4.13
                     │
                     ▼
            cv::CascadeClassifier
                     │
                     ▼
             TEST SET SEPARATO