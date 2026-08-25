// File a parte, non collegato a CMakeLists: unisce i negativi originali (../data/patches/neg)
// con gli hard negative appena minati (../data/patches/neg_hard) in un'unica cartella e
// richiama trainSvm() (invariata, gia' presente in src/hog.cpp) per riaddestrare il modello.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/train_hnm.cpp ../src/hog.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o train_hnm
//   ./train_hnm

#include "hog.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

static void linkOrCopyAll(const fs::path& srcDir, const fs::path& dstDir){
    for(const auto& entry : fs::directory_iterator(srcDir)){
        if(!entry.is_regular_file()) continue;
        fs::path dst = dstDir / entry.path().filename();
        if(fs::exists(dst)) continue;
        std::error_code ec;
        fs::create_hard_link(entry.path(), dst, ec);
        if(ec) fs::copy_file(entry.path(), dst, fs::copy_options::overwrite_existing);
    }
}

int main(){
    const fs::path negDir = "../data/patches/neg";
    const fs::path negHardDir = "../data/patches/neg_hard";
    const fs::path negCombinedDir = "../data/patches/neg_plus_hard";

    fs::create_directories(negCombinedDir);
    std::cout << "unisco " << negDir << " e " << negHardDir << " in " << negCombinedDir << '\n';
    linkOrCopyAll(negDir, negCombinedDir);
    linkOrCopyAll(negHardDir, negCombinedDir);

    size_t n = std::distance(fs::directory_iterator(negCombinedDir), fs::directory_iterator());
    std::cout << "negativi combinati: " << n << '\n';

    trainSvm("../data/patches/pos", negCombinedDir.string(), "../data/svm_hog_sgd_hnm.yml");
    return 0;
}
