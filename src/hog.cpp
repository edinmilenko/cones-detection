#include "hog.hpp"
#include "patch_geometry.hpp"

// unica definizione dei parametri HOG: training e detection leggono da qui,
// cosi' un descrittore di dimensione sbagliata non puo' nascere da un mismatch
// tra due punti del codice che sono andati fuori sincrono.
cv::HOGDescriptor makeHog(){
    return cv::HOGDescriptor(kPatchSize, cv::Size(8, 8), cv::Size(4, 4), cv::Size(4, 4), 9);
}

// TODO: computeFeatures, trainSvm, classify
