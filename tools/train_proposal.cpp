// File a parte, non collegato a CMakeLists: richiama trainSvm() (invariata, gia' presente in
// src/hog.cpp) sulle patch estratte da extract_proposal_patches.cpp (candidati reali del
// detector, non crop puliti sulle GT). Vedi PROGRESS_FASE4.md per il perche'.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/train_proposal.cpp ../src/hog.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o train_proposal
//   ./train_proposal

#include "hog.hpp"
#include <iostream>

int main(){
    const std::string posDir = "../data/patches/pos_proposal";
    const std::string negDir = "../data/patches/neg_proposal";
    const std::string modelPath = "../data/svm_hog_sgd_proposal.yml";

    std::cout << "training su " << posDir << " / " << negDir << "\n";
    trainSvm(posDir, negDir, modelPath);
    std::cout << "modello salvato in " << modelPath << "\n";
    return 0;
}
