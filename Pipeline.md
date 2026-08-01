# Task to complete for us: 

- data loader, create a "dataset" folder saving the image with bounding box with coordinates extracted from the json (for us, not to be delivered).

- labeler: takes (img, its json) pairs and draws the bounding boxes (&segmentations?) on top and then saves them in "labeled" folder

dataset/
  cone_00001.jpg
  cone_00001.json
  cone_00002.png
  cone_00002.json
        ↓
labeled/
  cone_00001.jpg
  cone_00002.png

- function to evaluate scores of bounding boxes found vs ground truth, used after pipeline for evaluation, some code from labeler will be recycled

# PIPELINE FOR CONES DETECTION

2 opzioni: nella root crei cartella "dataset" tutte le coppie immagini-json (se ho 10k foto avrai dentro dataset 10k coppie 1.json-1.png, 2.json-2.png).xx (altra opzione sarebbe avere un vettore di coppie di paths immagine e json per tenere il dataset "invariato", da capire se necessario altrimenti si evita)

1. Conversion of images from RGB to grayscale.

2\. Two option:

2.a) Viola-Jones approach

2.b) Template matching (paired with data augmentation with rotation, cropping, light stretching)

3\) Classification. For classification count all the pixels with a certain color and within a certain range of color.

4\) Segmentation based on the label (color) found from classification.

5\) Metrics computation (P.S. not all the cones need to be detected, the main ones).