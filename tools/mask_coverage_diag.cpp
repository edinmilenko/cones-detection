// File a parte, non collegato a CMakeLists: duplica (senza toccare color_proposals.cpp) la
// pipeline whiteBalance+colorMask+morphCleaning per misurare, dentro ogni GT box non coperta da
// un candidato (vedi coverage_diag.cpp), la frazione di pixel che passano il filtro colore PRIMA
// e DOPO la pulizia morfologica. Se la frazione pre-morfologia e' gia' bassa, il problema e' la
// soglia HSV; se e' alta ma crolla dopo l'opening/closing, il problema e' la morfologia che
// cancella i blob piccoli.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/mask_coverage_diag.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
//     -o mask_coverage_diag
//   ./mask_coverage_diag [numero_immagini] [seed]

#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
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

cv::Mat colorMask(const cv::Mat& imgBGR){
    cv::Mat hsv; cv::cvtColor(imgBGR, hsv, cv::COLOR_BGR2HSV);
    cv::Mat blue, orange, yellow, mask;
    cv::inRange(hsv, cv::Scalar(100, 80, 60), cv::Scalar(130, 255, 255), blue);
    cv::inRange(hsv, cv::Scalar(  5, 80, 60), cv::Scalar( 20, 255, 255), orange);
    cv::inRange(hsv, cv::Scalar( 20, 80, 60), cv::Scalar( 35, 255, 255), yellow);
    cv::bitwise_or(blue, orange, mask);
    cv::bitwise_or(mask, yellow, mask);
    return mask;
}

cv::Mat morphCleaning(const cv::Mat& mask){
    cv::Mat elemOpening = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat elemClosing = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,7));
    cv::Mat newMask;
    cv::morphologyEx(mask, newMask, cv::MORPH_OPEN, elemOpening);
    cv::morphologyEx(newMask, newMask, cv::MORPH_CLOSE, elemClosing);
    return newMask;
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

    std::vector<double> preMissed, postMissed, preCovered, postCovered;
    std::vector<int> heightMissed;

    for(const auto& imgName : testImgs){
        auto it = gt.find(imgName);
        if(it == gt.end() || it->second.empty()) continue;
        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;

        cv::Mat corrected = whiteBalance(bgr, 6.0);
        cv::Mat maskPre = colorMask(corrected);
        cv::Mat maskPost = morphCleaning(maskPre);

        // ricalcola i candidati con la stessa logica di findCandidateBoxes per sapere quali GT restano scoperte
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(maskPost, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double maxArea = 0.01 * maskPost.rows * maskPost.cols;
        std::vector<cv::Rect> cands;
        for(auto& c : contours){
            cv::Rect r = cv::boundingRect(c);
            if(r.area() > maxArea) continue;
            cands.push_back(r);
        }

        for(const auto& gb : it->second){
            cv::Rect clipped = gb & cv::Rect(0, 0, bgr.cols, bgr.rows);
            if(clipped.width <= 0 || clipped.height <= 0) continue;

            double fracPre = cv::countNonZero(maskPre(clipped)) / double(clipped.area());
            double fracPost = cv::countNonZero(maskPost(clipped)) / double(clipped.area());

            double bestIou = 0.0;
            for(const auto& c : cands) bestIou = std::max(bestIou, calculateIoU(c, gb));

            if(bestIou < 0.3){
                preMissed.push_back(fracPre); postMissed.push_back(fracPost);
                heightMissed.push_back(gb.height);
            } else {
                preCovered.push_back(fracPre); postCovered.push_back(fracPost);
            }
        }
    }

    auto stat = [](const std::string& name, std::vector<double> v){
        if(v.empty()){ std::cout << name << ": n=0\n"; return; }
        std::sort(v.begin(), v.end());
        double sum = 0; for(double x : v) sum += x;
        std::cout << name << ": n=" << v.size() << " min=" << v.front() << " p25=" << v[v.size()/4]
                   << " median=" << v[v.size()/2] << " p75=" << v[v.size()*3/4] << " max=" << v.back()
                   << " mean=" << (sum/v.size()) << "\n";
    };

    std::cout << "frazione di pixel della GT box che passano il filtro colore (0..1)\n\n";
    stat("PRE-morfologia,  coperte", preCovered);
    stat("PRE-morfologia,  MANCATE", preMissed);
    stat("POST-morfologia, coperte", postCovered);
    stat("POST-morfologia, MANCATE", postMissed);

    int zeroPreMissed = 0, zeroPostMissed = 0;
    for(double v : preMissed) if(v < 1e-9) zeroPreMissed++;
    for(double v : postMissed) if(v < 1e-9) zeroPostMissed++;
    std::cout << "\nmancate con fracPRE==0 (nessun pixel di colore utile nella box): " << zeroPreMissed
               << "/" << preMissed.size() << " (" << (100.0*zeroPreMissed/preMissed.size()) << "%)\n";
    std::cout << "mancate con fracPRE>0 ma fracPOST==0 (uccise dalla morfologia): "
               << (zeroPostMissed - zeroPreMissed) << "/" << preMissed.size() << " ("
               << (100.0*(zeroPostMissed - zeroPreMissed)/preMissed.size()) << "%)\n";

    return 0;
}
