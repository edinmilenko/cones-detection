// Standalone file, not wired into CMakeLists: for every GT box in the test set that no
// colorProposals() candidate covers at IoU>=0.3, prints the height and width of the box and the
// median hue/saturation/value of its pixels, to tell whether the miss is a size problem (box too
// small, killed by the opening) or a color one (outside
// dalle bande HSV filtrate da colorMask).
//
// Ad-hoc build (from the build/ folder):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/coverage_diag.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
//     -o coverage_diag
//   ./coverage_diag [numero_immagini] [seed]

#include "color_proposals.hpp"
#include "utils.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv){
    int nSample = argc > 1 ? std::atoi(argv[1]) : 150;
    unsigned seed = argc > 2 ? (unsigned)std::atoi(argv[2]) : 7;

    const std::string datasetDir = "../dataset";
    const std::string csvPath = "../data/dataset.csv";
    const std::string splitFile = "../data/test.txt";

    std::unordered_map<std::string, std::vector<cv::Rect>> gt;
    {
        std::ifstream f(csvPath);
        std::string l;
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

    std::vector<int> coveredH, missedH, coveredW, missedW;
    std::vector<double> missedHue, missedSat, missedVal, coveredHue, coveredSat, coveredVal;
    int totalGt = 0, missed = 0;

    for(const auto& imgName : testImgs){
        auto it = gt.find(imgName);
        if(it == gt.end() || it->second.empty()) continue;
        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;
        cv::Mat hsv; cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        std::vector<cv::Rect> cands = colorProposals(bgr);

        for(const auto& gb : it->second){
            totalGt++;
            double bestIou = 0.0;
            for(const auto& c : cands) bestIou = std::max(bestIou, calculateIoU(c, gb));

            cv::Rect clipped = gb & cv::Rect(0, 0, hsv.cols, hsv.rows);
            std::vector<double> hs, ss, vs;
            if(clipped.width > 0 && clipped.height > 0){
                cv::Mat roi = hsv(clipped);
                for(int y = 0; y < roi.rows; y++)
                    for(int x = 0; x < roi.cols; x++){
                        cv::Vec3b p = roi.at<cv::Vec3b>(y, x);
                        hs.push_back(p[0]); ss.push_back(p[1]); vs.push_back(p[2]);
                    }
            }
            auto median = [](std::vector<double> v) -> double {
                if(v.empty()) return NAN;
                std::sort(v.begin(), v.end());
                return v[v.size()/2];
            };

            if(bestIou < 0.3){
                missed++;
                missedH.push_back(gb.height); missedW.push_back(gb.width);
                missedHue.push_back(median(hs)); missedSat.push_back(median(ss)); missedVal.push_back(median(vs));
            } else {
                coveredH.push_back(gb.height); coveredW.push_back(gb.width);
                coveredHue.push_back(median(hs)); coveredSat.push_back(median(ss)); coveredVal.push_back(median(vs));
            }
        }
    }

    auto stat = [](const std::string& name, std::vector<int> v){
        if(v.empty()){ std::cout << name << ": n=0\n"; return; }
        std::sort(v.begin(), v.end());
        double sum = 0; for(int x : v) sum += x;
        std::cout << name << ": n=" << v.size() << " min=" << v.front() << " p25=" << v[v.size()/4]
                   << " median=" << v[v.size()/2] << " p75=" << v[v.size()*3/4] << " max=" << v.back()
                   << " mean=" << (sum/v.size()) << "\n";
    };
    auto statd = [](const std::string& name, std::vector<double> v){
        v.erase(std::remove_if(v.begin(), v.end(), [](double x){ return std::isnan(x); }), v.end());
        if(v.empty()){ std::cout << name << ": n=0\n"; return; }
        std::sort(v.begin(), v.end());
        double sum = 0; for(double x : v) sum += x;
        std::cout << name << ": n=" << v.size() << " min=" << v.front() << " p25=" << v[v.size()/4]
                   << " median=" << v[v.size()/2] << " p75=" << v[v.size()*3/4] << " max=" << v.back()
                   << " mean=" << (sum/v.size()) << "\n";
    };

    std::cout << "total GT: " << totalGt << ", not covered (IoU<0.3): " << missed
               << " (" << (100.0*missed/totalGt) << "%)\n\n";
    std::cout << "--- dimensioni box ---\n";
    stat("height  covered", coveredH);
    stat("height  MISSED", missedH);
    stat("width covered", coveredW);
    stat("width MISSED", missedW);
    std::cout << "\n--- median color inside the box (HSV, OpenCV scale 0-179/0-255/0-255) ---\n";
    statd("hue covered", coveredHue);
    statd("hue MISSED", missedHue);
    statd("sat covered", coveredSat);
    statd("sat MISSED", missedSat);
    statd("val covered", coveredVal);
    statd("val MISSED", missedVal);

    return 0;
}
