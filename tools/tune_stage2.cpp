// Standalone file, not wired into CMakeLists: retrains ONLY stage 2 on the patches already on disk,
// with the number of samples per class passed on the command line. Retraining the whole cascade to
// try one value would mean re-extracting a million patches for nothing.
//
// Why this matters: trainAuto picks C by cross validation, and how many samples it gets changes what
// it picks. With 5000 per class it chose C=2.5, with 15000 it chose C=0.1, twenty-five times more
// regularised, and the whole cascade got softer.
//
// Ad-hoc build (from the build/ folder):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/tune_stage2.cpp ../src/hog.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o tune_stage2
//   ./tune_stage2 <samplesPerClass> [fixedC] [outModel.yml]

#include "hog.hpp"
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv){
    const int perClass = argc > 1 ? std::atoi(argv[1]) : 5000;
    const double fixedC = argc > 2 ? std::atof(argv[2]) : 0.0;
    const std::string outPath = argc > 3 ? argv[3] : "../data/svm_stage2_rbf_balanced.yml";

    std::cout << "stage 2 with " << perClass << " samples per class, C=" << fixedC << " -> " << outPath << '\n';
    trainStage2("../data/patches/positives", "../data/patches/negatives",
                "../data/svm_hog_sgd_proposal.yml", outPath, 15.0f, perClass, fixedC);
    return 0;
}
