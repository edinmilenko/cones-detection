// Standalone file, not wired into CMakeLists: prototypes variants of the colorMask+morph pipeline
// (without touching src/color_proposals.cpp) and measures coverage (IoU>=0.3) plus the average
// candidate count per image over the whole test set, to compare them fairly before deciding whether
// to take anything into src/. What prompted the experiment (mask_coverage_diag.cpp,
// coverage_diag.cpp): 55% of the GT boxes in the test set are covered by no candidate at all, and
// 15% are wiped out entirely by the morphology (mask not empty before, empty after), mostly small
// boxes (median height 18px against 30px for the covered ones) whose color mask is probably
// fragmented into little islands rather than one solid blob (compression and antialiasing at
// distance) — the current opening runs BEFORE the closing and erases them before the closing
// possa fonderle. Qui si prova a invertire l'ordine.
//
// Ad-hoc build (from the build/ folder):
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

// the variant currently in src/color_proposals.cpp: opening then closing
std::vector<cv::Rect> variantBaseline(const cv::Mat& imgBGR){
    cv::Mat corrected = whiteBalance(imgBGR, 6.0);
    cv::Mat mask = colorMask(corrected, 80, 60);
    cv::Mat elemOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::Mat elemClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, elemOpen);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, elemClose);
    return findCandidateBoxes(mask);
}

// winner of the previous round (closing 5x9 then opening 3x3): already taken into
// src/color_proposals.cpp. The sweep over the S/V thresholds of colorMask starts again here: in the
// diagnosis (coverage_diag.cpp) the median saturation of the missed GT boxes (66) was already below
// the current threshold of 80, so the threshold is probably too aggressive for small or far cones,
// where the color bleeds into the background through anti-aliasing and compression.
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
                   << nCovered << "/" << nGt << ")   candidates/image=" << (double(nCands)/nImg) << "\n";
    }

    return 0;
}
