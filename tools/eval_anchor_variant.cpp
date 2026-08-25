// ad-hoc, non collegato a CMakeLists: valuta end-to-end (precision/recall/F1, NMS fisso a 0.1)
// una cascata allenata su una geometria di ancoraggio specifica, sweepando la soglia stadio2
// (non trasferibile tra modelli diversi allenati separatamente).
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/eval_anchor_variant.cpp ../src/hog.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o eval_anchor_variant
//   ./eval_anchor_variant <frac> <stage1.yml> <stage2.yml>

#include "hog.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

const std::vector<double> kHeights = {10, 18, 30, 50, 85, 140, 220};
const double kAspect = static_cast<double>(kPatchSize.width) / kPatchSize.height;
constexpr float kStage1Thr = 15.0f;
constexpr double kNmsIou = 0.1;

cv::Mat whiteBalance(const cv::Mat& imgBGR, double p){
    cv::Mat imgF;
    imgBGR.convertTo(imgF, CV_32F);
    cv::Mat expImg;
    cv::pow(imgF, p, expImg);
    cv::Scalar m = cv::mean(expImg);
    double e[3];
    for(int i = 0; i < 3; i++){
        e[i] = std::pow(m[i], 1.0 / p);
        if(e[i] < 1e-6) e[i] = 1e-6;
    }
    double gainMean = (e[0] + e[1] + e[2]) / 3;
    cv::Scalar gain(gainMean / e[0], gainMean / e[1], gainMean / e[2]);
    cv::multiply(imgF, gain, imgF);
    cv::Mat out;
    imgF.convertTo(out, CV_8U);
    return out;
}

cv::Mat colorMask(const cv::Mat& imgBGR){
    cv::Mat hsv;
    cv::cvtColor(imgBGR, hsv, cv::COLOR_BGR2HSV);
    cv::Mat blue, orange, yellow, mask;
    cv::inRange(hsv, cv::Scalar(100, 80, 60), cv::Scalar(130, 255, 255), blue);
    cv::inRange(hsv, cv::Scalar(5, 80, 60), cv::Scalar(20, 255, 255), orange);
    cv::inRange(hsv, cv::Scalar(20, 80, 60), cv::Scalar(35, 255, 255), yellow);
    cv::bitwise_or(blue, orange, mask);
    cv::bitwise_or(mask, yellow, mask);
    return mask;
}

std::vector<cv::Rect> blobs(const cv::Mat& imgBGR){
    cv::Mat corrected = whiteBalance(imgBGR, 6.0);
    cv::Mat mask = colorMask(corrected);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double maxArea = 0.01 * mask.rows * mask.cols;
    std::vector<cv::Rect> out;
    for(auto& c : contours){
        cv::Rect r = cv::boundingRect(c);
        if(r.area() > maxArea || r.area() < 3) continue;
        out.push_back(r);
    }
    return out;
}

std::vector<cv::Rect> anchorAt(const cv::Rect& blob, cv::Size imgSize, double frac){
    std::vector<cv::Rect> out;
    double cx = blob.x + blob.width / 2.0;
    double cy = blob.y + blob.height / 2.0;
    for(double h : kHeights){
        double w = h * kAspect;
        int nx = (int)std::lround(cx - w / 2.0);
        int ny = (int)std::lround(cy - h * frac);
        int nw = (int)std::lround(w), nh = (int)std::lround(h);
        nx = std::max(0, nx);
        ny = std::max(0, ny);
        nw = std::min(nw, imgSize.width - nx);
        nh = std::min(nh, imgSize.height - ny);
        if(nw >= 4 && nh >= 4) out.push_back(cv::Rect(nx, ny, nw, nh));
    }
    return out;
}

std::vector<cv::Rect> candidatesAt(const cv::Mat& imgBGR, double frac){
    std::vector<cv::Rect> out;
    for(auto& b : blobs(imgBGR)){
        auto g = anchorAt(b, imgBGR.size(), frac);
        out.insert(out.end(), g.begin(), g.end());
    }
    return out;
}

} // namespace

int main(int argc, char** argv){
    if(argc < 4){
        std::cerr << "uso: eval_anchor_variant <frac> <stage1.yml> <stage2.yml>\n";
        return 1;
    }
    double frac = std::atof(argv[1]);
    std::string stage1Path = argv[2];
    std::string stage2Path = argv[3];
    std::string splitFile = argc > 4 ? argv[4] : "../data/test_real.txt";

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

    cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load(stage1Path);
    cv::Ptr<cv::ml::SVM> stage2 = cv::ml::SVM::load(stage2Path);

    struct ImgData { std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<cv::Rect> gtBoxes; };
    std::vector<ImgData> data;

    for(const auto& imgName : imgs){
        auto it = gt.find(imgName);
        if(it == gt.end() || it->second.empty()) continue;
        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;
        cv::Mat gray; cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        ImgData d;
        d.gtBoxes = it->second;
        for(const auto& c : candidatesAt(bgr, frac)){
            cv::Mat patch;
            int interp = c.height > kPatchSize.height ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(gray(c), patch, kPatchSize, 0, 0, interp);

            if(rawScore(stage1, patch) <= kStage1Thr) continue;
            float s2 = rbfScore(stage2, patch);

            d.boxes.push_back(c);
            d.scores.push_back(s2);
        }
        data.push_back(std::move(d));
    }

    for(float thr2 : {-0.02f, -0.01f, -0.005f, 0.f, 0.005f, 0.01f, 0.015f, 0.02f, 0.03f, 0.05f, 0.08f,
                       0.1f, 0.13f, 0.15f, 0.18f, 0.2f, 0.25f, 0.3f}){
        long long tp = 0, fp = 0, fn = 0;
        for(const auto& d : data){
            std::vector<cv::Rect> keptBoxes;
            std::vector<float> keptScores;
            for(size_t i = 0; i < d.boxes.size(); i++){
                if(d.scores[i] > thr2){ keptBoxes.push_back(d.boxes[i]); keptScores.push_back(d.scores[i]); }
            }
            std::vector<int> nmsIdx = nonMaxSuppression(keptBoxes, keptScores, kNmsIou);

            std::vector<bool> matched(d.gtBoxes.size(), false);
            for(int idx : nmsIdx){
                double bestIou = 0.0; int bestJ = -1;
                for(size_t j = 0; j < d.gtBoxes.size(); j++){
                    if(matched[j]) continue;
                    double i_ = calculateIoU(keptBoxes[idx], d.gtBoxes[j]);
                    if(i_ > bestIou){ bestIou = i_; bestJ = (int)j; }
                }
                if(bestIou >= 0.3){ tp++; matched[bestJ] = true; }
                else fp++;
            }
            for(bool m : matched) if(!m) fn++;
        }
        double precision = (tp + fp) > 0 ? double(tp) / double(tp + fp) : 0.0;
        double recall = (tp + fn) > 0 ? double(tp) / double(tp + fn) : 0.0;
        double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
        std::cout << "frac=" << frac << " thr2=" << thr2 << ": precision=" << precision
                   << " recall=" << recall << " f1=" << f1
                   << "  TP=" << tp << " FP=" << fp << " FN=" << fn << '\n';
    }
    return 0;
}
