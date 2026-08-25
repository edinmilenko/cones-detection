// ad-hoc, non collegato a CMakeLists: come anchor_tuning.cpp, ma disegna per ogni GT il
// miglior candidato geometrico (nessun classificatore) a frac=0.50 (attuale, rosso) e
// frac=0.42 (proposto, blu), per vedere ad occhio il guadagno di localizzazione.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/anchor_visual.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml \
//     -o anchor_visual
//   ./anchor_visual <immagine.png> <csv_gt> <output.png>

#include "hog.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

const std::vector<double> kHeights = {10, 18, 30, 50, 85, 140, 220};
const double kAspect = static_cast<double>(kPatchSize.width) / kPatchSize.height;

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

// centro del box al frac*h dal bordo superiore del blob (frac=0.5 = centroide, l'attuale;
// frac<0.5 sposta il box verso il basso rispetto al blob)
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

// per ogni GT, il candidato con IoU piu' alto tra tutti quelli generati con quel frac
cv::Rect bestCandidate(const cv::Rect& gtBox, const std::vector<cv::Rect>& cands){
    cv::Rect best;
    double bestIou = -1;
    for(auto& c : cands){
        double i = calculateIoU(c, gtBox);
        if(i > bestIou){ bestIou = i; best = c; }
    }
    return best;
}

} // namespace

int main(int argc, char** argv){
    if(argc < 4){
        std::cerr << "uso: anchor_visual <immagine.png> <csv_gt> <output.png>\n";
        return 1;
    }
    std::string imgPath = argv[1];
    std::string csvPath = argv[2];
    std::string outPath = argv[3];

    cv::Mat bgr = cv::imread(imgPath);
    if(bgr.empty()){ std::cerr << "immagine non trovata: " << imgPath << "\n"; return 1; }
    std::string imgName = std::filesystem::path(imgPath).filename().string();

    std::vector<cv::Rect> gtBoxes;
    std::ifstream f(csvPath);
    std::string line;
    while(std::getline(f, line)){
        std::stringstream ss(line);
        std::string img, x1s, y1s, x2s, y2s;
        std::getline(ss, img, ','); std::getline(ss, x1s, ',');
        std::getline(ss, y1s, ','); std::getline(ss, x2s, ',');
        std::getline(ss, y2s, ',');
        if(img != imgName) continue;
        int x1 = std::stoi(x1s), y1 = std::stoi(y1s), x2 = std::stoi(x2s), y2 = std::stoi(y2s);
        gtBoxes.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
    }

    std::vector<cv::Rect> blobList = blobs(bgr);
    std::vector<cv::Rect> candsOld, candsNew;
    for(auto& b : blobList){
        auto o = anchorAt(b, bgr.size(), 0.5);
        auto n = anchorAt(b, bgr.size(), 0.42);
        candsOld.insert(candsOld.end(), o.begin(), o.end());
        candsNew.insert(candsNew.end(), n.begin(), n.end());
    }

    cv::Mat out = bgr.clone();
    for(auto& gtBox : gtBoxes){
        cv::rectangle(out, gtBox, cv::Scalar(0, 200, 0), 2);
        cv::rectangle(out, bestCandidate(gtBox, candsOld), cv::Scalar(0, 0, 255), 1);
        cv::rectangle(out, bestCandidate(gtBox, candsNew), cv::Scalar(255, 120, 0), 1);
    }
    cv::imwrite(outPath, out);
    std::cout << "salvato in " << outPath << " (verde=GT, rosso=frac0.50 attuale, blu=frac0.42 proposto)\n";
    return 0;
}
