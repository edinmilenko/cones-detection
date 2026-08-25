// File a parte, non collegato a CMakeLists: prototipa varianti della pipeline colorMask+morph
// (senza toccare src/color_proposals.cpp) e misura coverage (IoU>=0.3) + conteggio medio
// candidati/immagine sul test set intero, per confrontarle onestamente prima di decidere se
// portare qualcosa in src/. Diagnosi che ha motivato questo esperimento (mask_coverage_diag.cpp,
// coverage_diag.cpp): il 55% delle GT box del test set non e' coperta da nessun candidato;
// il 15% e' uccisa interamente dalla morfologia (mask non vuota prima, vuota dopo), soprattutto
// box piccole (mediana altezza 18px vs 30px delle coperte) la cui maschera colore e' probabile
// sia frammentata in isole piccole invece che un blob solido (compressione/antialiasing a
// distanza) — l'opening attuale gira PRIMA della closing e le cancella prima che la closing
// possa fonderle. Qui si prova a invertire l'ordine.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/proposal_tuning.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
//     -o proposal_tuning
//   ./proposal_tuning [numero_immagini] [seed]

#include "utils.hpp"
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

cv::Mat whiteBalance(const cv::Mat& imgBGR, double p){
    cv::Mat imgF; imgBGR.convertTo(imgF, CV_32F);
    cv::Mat expImg; cv::pow(imgF, p, expImg);
    cv::Scalar m = cv::mean(expImg);
    double e[3];
    for(int i = 0; i < 3; i++){ e[i] = std::pow(m[i], 1.0/p); if(e[i] < 1e-6) e[i] = 1e-6; }
    double gainMean = (e[0]+e[1]+e[2])/3;
    cv::Scalar gain(gainMean/e[0], gainMean/e[1], gainMean/e[2]);
    cv::multiply(imgF, gain, imgF);
    cv::Mat out; imgF.convertTo(out, CV_8U);
    return out;
}

cv::Mat colorMask(const cv::Mat& imgBGR, int satThr, int valThr){
    cv::Mat hsv; cv::cvtColor(imgBGR, hsv, cv::COLOR_BGR2HSV);
    cv::Mat blue, orange, yellow, mask;
    cv::inRange(hsv, cv::Scalar(100, satThr, valThr), cv::Scalar(130, 255, 255), blue);
    cv::inRange(hsv, cv::Scalar(  5, satThr, valThr), cv::Scalar( 20, 255, 255), orange);
    cv::inRange(hsv, cv::Scalar( 20, satThr, valThr), cv::Scalar( 35, 255, 255), yellow);
    cv::bitwise_or(blue, orange, mask);
    cv::bitwise_or(mask, yellow, mask);
    return mask;
}

std::vector<cv::Rect> findCandidateBoxes(const cv::Mat& mask){
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double maxArea = 0.01 * mask.rows * mask.cols;
    std::vector<cv::Rect> candidates;
    for(auto& c : contours){
        cv::Rect r = cv::boundingRect(c);
        if(r.area() > maxArea) continue;
        candidates.push_back(r);
    }
    return candidates;
}

// variante attuale in src/color_proposals.cpp: opening poi closing
std::vector<cv::Rect> variantBaseline(const cv::Mat& imgBGR){
    cv::Mat corrected = whiteBalance(imgBGR, 6.0);
    cv::Mat mask = colorMask(corrected, 80, 60);
    cv::Mat elemOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::Mat elemClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, elemOpen);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, elemClose);
    return findCandidateBoxes(mask);
}

// vincitore del giro precedente (closing 5x9 poi opening 3x3): gia' portato in
// src/color_proposals.cpp. Qui riparte lo sweep sulle soglie S/V della colorMask: nella diagnosi
// (coverage_diag.cpp) la saturazione mediana delle GT box mancate (66) era gia' sotto l'attuale
// soglia 80, quindi probabile che la soglia sia troppo aggressiva per coni piccoli/lontani dove
// il colore si diluisce con lo sfondo per via di anti-aliasing/compressione.
std::vector<cv::Rect> morphWinner(const cv::Mat& mask){
    cv::Mat out = mask.clone();
    cv::Mat elemOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::Mat elemClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,9));
    cv::morphologyEx(out, out, cv::MORPH_CLOSE, elemClose);
    cv::morphologyEx(out, out, cv::MORPH_OPEN, elemOpen);
    return findCandidateBoxes(out);
}

std::vector<cv::Rect> variantSatVal(const cv::Mat& imgBGR, int satThr, int valThr){
    cv::Mat corrected = whiteBalance(imgBGR, 6.0);
    cv::Mat mask = colorMask(corrected, satThr, valThr);
    return morphWinner(mask);
}

} // namespace

int main(int argc, char** argv){
    int nSample = argc > 1 ? std::atoi(argv[1]) : 150;
    unsigned seed = argc > 2 ? (unsigned)std::atoi(argv[2]) : 7;

    const std::string datasetDir = "../dataset";
    const std::string csvPath = "../data/dataset.csv";
    const std::string splitFile = "../data/test.txt";

    std::unordered_map<std::string, std::vector<cv::Rect>> gt;
    {
        std::ifstream f(csvPath); std::string l;
        while(std::getline(f, l)){
            std::stringstream ss(l);
            std::string img, x1s, y1s, x2s, y2s;
            std::getline(ss, img, ','); std::getline(ss, x1s, ',');
            std::getline(ss, y1s, ','); std::getline(ss, x2s, ',');
            std::getline(ss, y2s, ',');
            int x1 = std::stoi(x1s), y1 = std::stoi(y1s), x2 = std::stoi(x2s), y2 = std::stoi(y2s);
            gt[img].push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
    }
    std::vector<std::string> testImgs;
    { std::ifstream f(splitFile); std::string l; while(std::getline(f, l)) if(!l.empty()) testImgs.push_back(l); }
    std::mt19937 rng(seed);
    std::shuffle(testImgs.begin(), testImgs.end(), rng);
    if((int)testImgs.size() > nSample) testImgs.resize(nSample);

    struct Variant { std::string name; std::function<std::vector<cv::Rect>(const cv::Mat&)> fn; };
    std::vector<Variant> variants = {
        {"baseline (sat=80,val=60, open->close)", variantBaseline},
        {"sat=80,val=60 (vincitore morfologia)",  [](const cv::Mat& b){ return variantSatVal(b, 80, 60); }},
        {"sat=65,val=60",                         [](const cv::Mat& b){ return variantSatVal(b, 65, 60); }},
        {"sat=80,val=45",                         [](const cv::Mat& b){ return variantSatVal(b, 80, 45); }},
        {"sat=65,val=45",                         [](const cv::Mat& b){ return variantSatVal(b, 65, 45); }},
        {"sat=50,val=40",                         [](const cv::Mat& b){ return variantSatVal(b, 50, 40); }},
    };

    for(auto& v : variants){
        long long nGt = 0, nCovered = 0, nCands = 0;
        int nImg = 0;
        for(const auto& imgName : testImgs){
            auto it = gt.find(imgName);
            if(it == gt.end() || it->second.empty()) continue;
            cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
            if(bgr.empty()) continue;
            nImg++;

            std::vector<cv::Rect> cands = v.fn(bgr);
            nCands += (long long)cands.size();

            for(const auto& gb : it->second){
                nGt++;
                double bestIou = 0.0;
                for(const auto& c : cands) bestIou = std::max(bestIou, calculateIoU(c, gb));
                if(bestIou >= 0.3) nCovered++;
            }
        }
        std::cout << v.name << ":  coverage=" << (100.0*nCovered/nGt) << "%  ("
                   << nCovered << "/" << nGt << ")   candidati/immagine=" << (double(nCands)/nImg) << "\n";
    }

    return 0;
}
