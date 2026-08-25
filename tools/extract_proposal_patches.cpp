// File a parte, non collegato a CMakeLists: alleniamo il classificatore sulla stessa
// distribuzione su cui verra' poi usato. Fa girare colorProposals() (invariata) su tutte le
// immagini di TRAIN, etichetta ogni candidato per IoU con le bbox reali (>0.5 positivo, <0.3
// negativo, in mezzo scartato perche' ambiguo), e salva le patch 16x24 risultanti. Niente
// normalizzazione di aspect ratio: il candidato viene ritagliato e ridimensionato esattamente
// come accadra' in inferenza, quindi training e detection vedono la stessa distribuzione di
// crop (decentrati, tagliati, con contesto variabile) invece dei crop puliti sulle bbox GT.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/extract_proposal_patches.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml \
//     -o extract_proposal_patches
//   ./extract_proposal_patches [target_negativi]

#include "color_proposals.hpp"
#include "hog.hpp"
#include "utils.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv){
    int targetNeg = argc > 1 ? std::atoi(argv[1]) : 70000;

    const std::string datasetDir = "../dataset";
    const std::string csvPath = "../data/dataset.csv";
    const std::string splitFile = "../data/train.txt";
    const std::string posOutDir = "../data/patches/pos_proposal";
    const std::string negOutDir = "../data/patches/neg_proposal";

    std::filesystem::create_directories(posOutDir);
    std::filesystem::create_directories(negOutDir);

    std::unordered_map<std::string, std::vector<cv::Rect>> gt;
    {
        std::ifstream f(csvPath);
        std::string l;
        while(std::getline(f, l)){
            std::stringstream ss(l);
            std::string img, x1s, y1s, x2s, y2s;
            std::getline(ss, img, ',');
            std::getline(ss, x1s, ',');
            std::getline(ss, y1s, ',');
            std::getline(ss, x2s, ',');
            std::getline(ss, y2s, ',');
            int x1 = std::stoi(x1s), y1 = std::stoi(y1s), x2 = std::stoi(x2s), y2 = std::stoi(y2s);
            gt[img].push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
    }

    std::vector<std::string> trainImgs;
    {
        std::ifstream f(splitFile);
        std::string l;
        while(std::getline(f, l)) if(!l.empty()) trainImgs.push_back(l);
    }
    std::mt19937 rng(42);
    std::shuffle(trainImgs.begin(), trainImgs.end(), rng);

    int posSaved = 0, negSaved = 0, usedImgs = 0;
    for(const auto& imgName : trainImgs){
        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        usedImgs++;

        const auto& gtBoxes = gt[imgName];
        std::vector<cv::Rect> cands = colorProposals(bgr);
        std::string stem = imgName.substr(0, imgName.find_last_of('.'));

        int localIdx = 0;
        for(const auto& c : cands){
            if(c.width < 4 || c.height < 4) continue;

            double bestIou = 0.0;
            for(const auto& gb : gtBoxes) bestIou = std::max(bestIou, calculateIoU(c, gb));

            // soglie riallineate alla definizione di "coperto" usata da funnel_analysis
            // (IoU>=0.3): con isPos>0.5 la maggior parte dei candidati "coperti" del mondo
            // reale (decentrati, IoU tipica 0.3-0.5) restava scartata come ambigua e il
            // classificatore non imparava mai ad accettarli. Prima iterazione (isPos>0.5,
            // isNeg<0.3): acceptance 52.6%, recall finale 23.3% (thr=0, raw, 150 img test) —
            // peggio del baseline (57.2% / 25.4%). Vedi PROGRESS_FASE4.md.
            bool isPos = bestIou >= 0.3;
            bool isNeg = bestIou < 0.2;
            if(!isPos && !isNeg) continue; // zona ambigua, scartata
            if(isNeg && negSaved >= targetNeg) continue; // quota negativi raggiunta, continua a cercare positivi

            cv::Mat crop = gray(c);
            cv::Mat patch;
            int interp = (c.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(crop, patch, kPatchSize, 0, 0, interp);

            const std::string& outDir = isPos ? posOutDir : negOutDir;
            cv::imwrite(outDir + "/" + stem + "_" + std::to_string(localIdx) + ".png", patch);
            if(isPos){
                cv::Mat flipped;
                cv::flip(patch, flipped, 1);
                cv::imwrite(outDir + "/" + stem + "_" + std::to_string(localIdx) + "_flip.png", flipped);
                posSaved += 2;
            } else {
                negSaved++;
            }
            localIdx++;
        }

        if(usedImgs % 100 == 0){
            std::cout << "  ..." << usedImgs << " immagini, pos=" << posSaved << " neg=" << negSaved << "\n";
        }
    }

    std::cout << "fatto: " << usedImgs << " immagini di train, positivi=" << posSaved
               << " (con flip), negativi=" << negSaved << "\n";
    std::cout << "positivi in: " << posOutDir << "\nnegativi in: " << negOutDir << "\n";
    return 0;
}
