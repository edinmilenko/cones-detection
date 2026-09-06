// Standalone file, not wired into CMakeLists: first step of the two-stage cascade. Stage 1 is the
// linear classifier already trained (data/svm_hog_sgd_proposal.yml, SVMSGD over HOG), run at a LOW,
// permissive threshold whose only job is to cut the number of candidates down (from ~18500 per
// image to far fewer) before handing the survivors to a second, more expressive classifier.
// This tool only picks that threshold: over a range of thresholds it measures how many candidates
// survive per image on average and how much "coverage" is left (GT covered by a candidate above
// threshold, IoU>=0.3) — the trade-off to settle before training stage 2.
//
// Ad-hoc build (from the build/ folder):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/cascade_stage1_sweep.cpp ../src/hog.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o cascade_stage1_sweep
//   ./cascade_stage1_sweep [numero_immagini] [seed]

#include "hog.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "color_proposals.hpp"
#include "utils.hpp"

int main(int argc, char** argv){
    int nSample = argc > 1 ? std::atoi(argv[1]) : 100;
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

    cv::Ptr<cv::ml::SVMSGD> svm = cv::ml::SVMSGD::load("../data/svm_hog_sgd_proposal.yml");
    cv::Mat W = svm->getWeights();
    float shift = svm->getShift();
    cv::HOGDescriptor hog = makeHog();

    std::vector<float> thresholds = {0, 5, 10, 15, 20, 25, 30, 40, 50, 60};
    std::vector<long long> survivors(thresholds.size(), 0), coveredSurv(thresholds.size(), 0);
    long long nGt = 0, nCovered = 0, totalCands = 0;
    int nImg = 0;

    for(const auto& imgName : testImgs){
        auto it = gt.find(imgName);
        if(it == gt.end() || it->second.empty()) continue;
        const auto& gtBoxes = it->second;

        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;
        cv::Mat gray; cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        nImg++;

        std::vector<cv::Rect> cands = colorProposals(bgr);
        totalCands += (long long)cands.size();

        std::vector<float> scores(cands.size());
        for(size_t i = 0; i < cands.size(); i++){
            cv::Mat crop = gray(cands[i]);
            cv::Mat patch;
            int interp = (cands[i].height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(crop, patch, kPatchSize, 0, 0, interp);
            std::vector<float> descriptor;
            hog.compute(patch, descriptor);
            float s = shift;
            for(size_t j = 0; j < descriptor.size(); j++) s += W.at<float>(0, (int)j) * descriptor[j];
            scores[i] = s;
        }

        for(const auto& gb : gtBoxes){
            nGt++;
            bool covered = false;
            for(const auto& c : cands) if(calculateIoU(c, gb) >= 0.3){ covered = true; break; }
            if(covered) nCovered++;
        }

        for(size_t t = 0; t < thresholds.size(); t++){
            for(size_t i = 0; i < cands.size(); i++){
                if(scores[i] <= thresholds[t]) continue;
                survivors[t]++;
            }
            for(const auto& gb : gtBoxes){
                bool coveredAboveThr = false;
                for(size_t i = 0; i < cands.size(); i++){
                    if(scores[i] > thresholds[t] && calculateIoU(cands[i], gb) >= 0.3){ coveredAboveThr = true; break; }
                }
                if(coveredAboveThr) coveredSurv[t]++;
            }
        }
    }

    std::cout << "images: " << nImg << ", candidates/image (unfiltered): " << (double(totalCands)/nImg) << "\n";
    std::cout << "coverage unfiltered (IoU>=0.3): " << (100.0*nCovered/nGt) << "%\n\n";
    for(size_t t = 0; t < thresholds.size(); t++){
        std::cout << "thr=" << thresholds[t]
                   << ": survivors/image=" << (double(survivors[t])/nImg)
                   << "  coverage-after-filter=" << (100.0*coveredSurv[t]/nGt) << "%"
                   << "  (" << (100.0*coveredSurv[t]/std::max<long long>(nCovered,1)) << "% of the unfiltered one)\n";
    }

    return 0;
}
