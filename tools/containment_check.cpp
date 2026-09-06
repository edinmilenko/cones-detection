// ad-hoc, not wired into CMakeLists: the two-stage cascade is trained and scored on IoU, but what
// the next step actually needs (segmentation, done by another module) is different: a box bigger
// than the cone is fine, the segmentation trims it, while a box that CUTS the cone is not, that
// part is gone. Measures, on the boxes the REAL cascade picks (not an oracle), how much area of
// GT stays inside the chosen box (containment = area(box & GT) / area(GT)), before and after a
// post-hoc padding of the final box (no retraining: the box already chosen is just widened).
//
// Ad-hoc build (from the build/ folder):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/containment_check.cpp ../src/hog.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o containment_check
//   ./containment_check [pad_top_frac] [pad_bottom_frac] [pad_side_frac] [out_dir]

#include "color_proposals.hpp"
#include "hog.hpp"
#include "utils.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {
constexpr float kStage1Thr = 15.0f;
constexpr float kStage2Thr = 0.01f;
constexpr double kNmsIou = 0.1;

cv::Rect padBox(const cv::Rect& b, double padTop, double padBottom, double padSide, cv::Size imgSize){
    int extraSide = (int)std::lround(b.width * padSide);
    int extraTop = (int)std::lround(b.height * padTop);
    int extraBottom = (int)std::lround(b.height * padBottom);
    int x = std::max(0, b.x - extraSide);
    int y = std::max(0, b.y - extraTop);
    int x2 = std::min(imgSize.width, b.x + b.width + extraSide);
    int y2 = std::min(imgSize.height, b.y + b.height + extraBottom);
    return cv::Rect(cv::Point(x, y), cv::Point(x2, y2));
}

double containment(const cv::Rect& box, const cv::Rect& gt){
    return double((box & gt).area()) / double(gt.area());
}
} // namespace

int main(int argc, char** argv){
    double padTop = argc > 1 ? std::atof(argv[1]) : 0.0;
    double padBottom = argc > 2 ? std::atof(argv[2]) : 0.0;
    double padSide = argc > 3 ? std::atof(argv[3]) : 0.0;
    std::string outDir = argc > 4 ? argv[4] : "";
    std::string splitFile = argc > 5 ? argv[5] : "../data/test_real.txt";
    if(!outDir.empty()) std::filesystem::create_directories(outDir);

    const std::string datasetDir = "../test_set/segmentation_test/img";
    const std::string csvPath = "../data/dataset_real.csv";

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
    std::vector<std::string> imgs;
    { std::ifstream f(splitFile); std::string l; while(std::getline(f, l)) if(!l.empty()) imgs.push_back(l); }

    cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load("../data/svm_hog_sgd_proposal.yml");
    cv::Ptr<cv::ml::SVM> stage2 = cv::ml::SVM::load("../data/svm_stage2_rbf_balanced.yml");

    std::vector<double> containments;
    long long tp = 0, fp = 0, fn = 0;
    for(const auto& imgName : imgs){
        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;
        cv::Mat gray; cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        for(const auto& c : colorProposals(bgr)){
            cv::Mat patch;
            int interp = c.height > kPatchSize.height ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(gray(c), patch, kPatchSize, 0, 0, interp);
            if(rawScore(stage1, patch) <= kStage1Thr) continue;
            float s2 = rbfScore(stage2, patch);
            if(s2 <= kStage2Thr) continue;
            boxes.push_back(c);
            scores.push_back(s2);
        }
        std::vector<int> kept = nonMaxSuppression(boxes, scores, kNmsIou);

        const auto& gtBoxes = gt[imgName];
        std::vector<bool> matched(gtBoxes.size(), false);
        std::vector<cv::Rect> paddedBoxes;
        for(int idx : kept){
            cv::Rect padded = padBox(boxes[idx], padTop, padBottom, padSide, bgr.size());
            paddedBoxes.push_back(padded);

            // the IoU>=0.3 matching here uses the box AFTER padding: that is what actually comes
            // out of the pipeline and what the next step would be scored on.
            double bestIou = 0; int bestJ = -1;
            for(size_t j = 0; j < gtBoxes.size(); j++){
                if(matched[j]) continue;
                double i_ = calculateIoU(padded, gtBoxes[j]);
                if(i_ > bestIou){ bestIou = i_; bestJ = (int)j; }
            }
            if(bestIou >= 0.3){
                tp++; matched[bestJ] = true;
                containments.push_back(containment(padded, gtBoxes[bestJ]));
            } else fp++;
        }
        for(bool m : matched) if(!m) fn++;

        if(!outDir.empty()){
            cv::Mat out = bgr.clone();
            for(const auto& box : gtBoxes) cv::rectangle(out, box, cv::Scalar(0, 200, 0), 2);
            for(const auto& box : paddedBoxes) cv::rectangle(out, box, cv::Scalar(0, 0, 255), 2);
            std::string stem = imgName.substr(0, imgName.find_last_of('.'));
            cv::imwrite(outDir + "/" + stem + "_det.png", out);
        }
    }

    std::sort(containments.begin(), containments.end());
    auto pct = [&](double p){ return containments[(size_t)(p * (containments.size()-1))]; };
    int full95 = 0, full100 = 0;
    for(double c : containments){ if(c >= 0.95) full95++; if(c >= 0.999) full100++; }

    double precision = (tp+fp)>0 ? double(tp)/(tp+fp) : 0.0;
    double recall = (tp+fn)>0 ? double(tp)/(tp+fn) : 0.0;
    std::cout << "pad top=" << padTop << " bottom=" << padBottom << " side=" << padSide << ":\n";
    std::cout << "  precision=" << precision << " recall=" << recall << " TP=" << tp << " FP=" << fp << " FN=" << fn << "\n";
    std::cout << "  containment of the TP: min=" << containments.front() << " p25=" << pct(0.25)
               << " p50=" << pct(0.5) << " p75=" << pct(0.75) << " max=" << containments.back()
               << "  (cone fully inside the box >=95%: " << (100.0*full95/containments.size())
               << "%, >=99.9%: " << (100.0*full100/containments.size()) << "%)\n";
    return 0;
}
