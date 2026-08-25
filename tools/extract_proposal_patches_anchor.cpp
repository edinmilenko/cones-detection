// ad-hoc, non collegato a CMakeLists: come extract_proposal_patches.cpp, ma con l'ancoraggio
// verticale del box parametrizzato (frac invece del centroide fisso al 50%), per confrontare
// piu' valori prima di decidere se portare il cambio in src/color_proposals.cpp. Duplica la
// pipeline di generazione blob (whiteBalance+colorMask+findContours), senza toccare src/.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/extract_proposal_patches_anchor.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml \
//     -o extract_proposal_patches_anchor
//   ./extract_proposal_patches_anchor <frac> <pos_out_dir> <neg_out_dir> [target_negativi]

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
        std::cerr << "uso: extract_proposal_patches_anchor <frac> <pos_out_dir> <neg_out_dir> [target_negativi]\n";
        return 1;
    }
    double frac = std::atof(argv[1]);
    std::string posOutDir = argv[2];
    std::string negOutDir = argv[3];
    int targetNeg = argc > 4 ? std::atoi(argv[4]) : 100000;

    const std::string datasetDir = "../dataset";
    const std::string csvPath = "../data/dataset.csv";
    const std::string splitFile = "../data/train.txt";

    std::filesystem::create_directories(posOutDir);
    std::filesystem::create_directories(negOutDir);

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
    std::mt19937 rng(42);
    std::shuffle(trainImgs.begin(), trainImgs.end(), rng);

    int posSaved = 0, negSaved = 0, usedImgs = 0;
    for(const auto& imgName : trainImgs){
        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        usedImgs++;

        const auto& gtBoxes = gt[imgName];
        std::vector<cv::Rect> cands = candidatesAt(bgr, frac);
        std::string stem = imgName.substr(0, imgName.find_last_of('.'));

        int localIdx = 0;
        for(const auto& c : cands){
            if(c.width < 4 || c.height < 4) continue;

            double bestIou = 0.0;
            for(const auto& gb : gtBoxes) bestIou = std::max(bestIou, calculateIoU(c, gb));

            bool isPos = bestIou >= 0.3;
            bool isNeg = bestIou < 0.2;
            if(!isPos && !isNeg) continue;
            if(isNeg && negSaved >= targetNeg) continue;

            cv::Mat crop = gray(c);
            cv::Mat patch;
            int interp = (c.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(crop, patch, kPatchSize, 0, 0, interp);

            const std::string& outDir = isPos ? posOutDir : negOutDir;
            cv::imwrite(outDir + "/" + stem + "_" + std::to_string(localIdx) + ".png", patch);
            if(isPos){
                cv::Mat flipped;
                cv::flip(patch, flipped, 1);
                cv::imwrite(outDir + "/" + stem + "_" + std::to_string(localIdx) + "_flip.png", flipped);
                posSaved += 2;
            } else {
                negSaved++;
            }
            localIdx++;
        }

        if(usedImgs % 200 == 0){
            std::cout << "  ..." << usedImgs << " immagini, pos=" << posSaved << " neg=" << negSaved << "\n";
        }
    }

    std::cout << "fatto (frac=" << frac << "): " << usedImgs << " immagini di train, positivi="
               << posSaved << " (con flip), negativi=" << negSaved << "\n";
    return 0;
}
