# PIPELINE FOR CONES DETECTION



Task to complete: data loader, create a folder saving the image with bounding box with coordinates extracted from the json (for us, not to be delivered).

2 opzioni: nella root crei cartella "dataset" tutte le coppie immagini-json (se ho 10k foto avrai dentro dataset 10k coppie 1.json-1.png, 2.json-2.png).



1. Conversion of images from RGB to grayscale.



2\. Two option:



2.a) Viola-Jones approach



2.b) Template matching (paired with data augmentation with rotation, cropping, light stretching)



3\) Classification. For classification count all the pixels with a certain color and within a certain range of color.



4\) Segmentation based on the label (color) found from classification.



5\) Metrics computation (P.S. not all the cones need to be detected, the main ones).

