// File a parte, non collegato a CMakeLists: usa il modello SVMSGD attuale per trovare, sulle
// immagini di TRAIN (non test), i candidati proposti da colorProposals() che il modello
// classifica (a torto) come cono ma che non sono vicini a nessuna bbox reale. Li salva come
// patch 16x24 in ../data/patches/neg_hard, da usare come hard negative per il retraining.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I../tools -I/usr/include/opencv4 \
//     ../tools/mine_hard_negatives.cpp ../src/hog.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o mine_hard_negatives
//   ./mine_hard_negatives [target_count]

#include "detector_common.hpp"
#include "hog.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv){
    int target = argc > 1 ? std::atoi(argv[1]) : 25000;

    const std::string datasetDir = "../dataset";
    const std::string csvPath = "../data/dataset.csv";
    const std::string splitFile = "../data/train.txt";
    const std::string modelPath = "../data/svm_hog_sgd.yml";
    const std::string outDir = "../data/patches/neg_hard";

    std::filesystem::create_directories(outDir);

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
    std::mt19937 rng(123);
    std::shuffle(trainImgs.begin(), trainImgs.end(), rng);

    cv::Ptr<cv::ml::SVMSGD> svm = cv::ml::SVMSGD::load(modelPath);

    int saved = 0, usedImgs = 0;
    for(const auto& imgName : trainImgs){
        if(saved >= target) break;

        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        usedImgs++;

        const auto& gtBoxes = gt[imgName];
        std::vector<cv::Rect> cands = detector::proposeCandidates(bgr);

        for(const auto& c : cands){
            if(saved >= target) break;

            double bestIou = 0.0;
            for(const auto& gb : gtBoxes) bestIou = std::max(bestIou, calculateIoU(c, gb));
            if(bestIou >= 0.1) continue; // troppo vicino a un cono vero, non e' un hard negative pulito

            cv::Mat crop = gray(c);
            cv::Mat patch;
            int interp = (c.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(crop, patch, kPatchSize, 0, 0, interp);

            if(classify(svm, patch) <= 0) continue; // il modello lo scarta gia' correttamente

            std::string stem = imgName.substr(0, imgName.find_last_of('.'));
            cv::imwrite(outDir + "/" + stem + "_" + std::to_string(saved) + ".png", patch);
            saved++;
        }

        if(usedImgs % 100 == 0){
            std::cout << "  ..." << usedImgs << " immagini, " << saved << " hard negative\n";
        }
    }

    std::cout << "hard negative salvati: " << saved << " da " << usedImgs << " immagini di train -> " << outDir << '\n';
    return 0;
}
