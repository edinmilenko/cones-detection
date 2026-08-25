// ad-hoc, non collegato a CMakeLists: esegue la cascata VERA (stadio1+stadio2 gia' allenati,
// stessi modelli e soglie di src/main.cpp) sui candidati generati con un frac di ancoraggio
// scelto, per vedere cosa sceglie davvero il classificatore — non il "migliore possibile"
// contro il GT come anchor_visual.cpp, ma l'output reale della pipeline. I modelli sono
// allenati su patch frac=0.5: usarli con frac=0.42 e' un test onesto ma non equivalente a un
// retraining sulla nuova geometria.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/detect_anchor_compare.cpp ../src/hog.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o detect_anchor_compare
//   ./detect_anchor_compare <frac> <output_dir>

#include "hog.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
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

const std::vector<double> kHeights = {10, 18, 30, 50, 85, 140, 220};
const double kAspect = static_cast<double>(kPatchSize.width) / kPatchSize.height;
constexpr float kStage1Thr = 15.0f;
constexpr float kStage2Thr = 0.01f;
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
    double frac = argc > 1 ? std::atof(argv[1]) : 0.5;
    std::string outDir = argc > 2 ? argv[2] : "../data/detect_anchor_compare";
    std::filesystem::create_directories(outDir);

    const std::string datasetDir = "../test_set/segmentation_test/img";
    const std::string csvPath = "../data/dataset_real.csv";
    const std::string splitFile = "../data/test_real.txt";

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

    long long totalTp = 0, totalFp = 0, totalFn = 0;
    for(const auto& imgName : imgs){
        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;
        cv::Mat gray; cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        for(const auto& c : candidatesAt(bgr, frac)){
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
        for(int idx : kept){
            double bestIou = 0; int bestJ = -1;
            for(size_t j = 0; j < gtBoxes.size(); j++){
                if(matched[j]) continue;
                double i_ = calculateIoU(boxes[idx], gtBoxes[j]);
                if(i_ > bestIou){ bestIou = i_; bestJ = (int)j; }
            }
            if(bestIou >= 0.3){ totalTp++; matched[bestJ] = true; } else totalFp++;
        }
        for(bool m : matched) if(!m) totalFn++;

        cv::Mat out = bgr.clone();
        for(const auto& box : gtBoxes) cv::rectangle(out, box, cv::Scalar(0, 200, 0), 2);
        for(int idx : kept) cv::rectangle(out, boxes[idx], cv::Scalar(0, 0, 255), 2);
        std::string stem = imgName.substr(0, imgName.find_last_of('.'));
        cv::imwrite(outDir + "/" + stem + "_det.png", out);
    }

    double precision = (totalTp + totalFp) > 0 ? double(totalTp) / (totalTp + totalFp) : 0.0;
    double recall = (totalTp + totalFn) > 0 ? double(totalTp) / (totalTp + totalFn) : 0.0;
    std::cout << "frac=" << frac << ": precision=" << precision << " recall=" << recall
               << " TP=" << totalTp << " FP=" << totalFp << " FN=" << totalFn
               << " -> immagini in " << outDir << "\n";
    return 0;
}
