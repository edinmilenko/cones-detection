// ad-hoc, non collegato a CMakeLists: come train_proposal.cpp, ma con le cartelle e il
// modello passati da linea di comando invece che hardcoded, per allenare piu' varianti
// (es. geometrie di ancoraggio diverse) senza duplicare il file.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/train_proposal_generic.cpp ../src/hog.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o train_proposal_generic
//   ./train_proposal_generic <pos_dir> <neg_dir> <output.yml>

#include "hog.hpp"
#include <iostream>

int main(int argc, char** argv){
    if(argc < 4){
        std::cerr << "uso: train_proposal_generic <pos_dir> <neg_dir> <output.yml>\n";
        return 1;
    }
    std::cout << "training su " << argv[1] << " / " << argv[2] << "\n";
    trainSvm(argv[1], argv[2], argv[3]);
    std::cout << "modello salvato in " << argv[3] << "\n";
    return 0;
}
