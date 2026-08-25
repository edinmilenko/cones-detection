// ad-hoc, non collegato a CMakeLists: duplica la pipeline di generazione blob di
// color_proposals.cpp (senza toccarlo) per confrontare strategie di ancoraggio verticale dei
// box generati da ogni blob. Osservazione di sessione: i box finali tagliano spesso la base
// del cono pur avendo margine vuoto sopra la punta — ipotesi: il centroide del blob colore e'
// piu' in alto del centro vero del cono (la base, spesso bianca/in ombra, non supera le soglie
// HSV), quindi centrare il box sul centroide lo sposta sempre verso l'alto.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/anchor_tuning.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml \
//     -o anchor_tuning
//   ./anchor_tuning [numero_immagini] [seed]

#include "hog.hpp"
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

struct Stat {
    double sumBestIou = 0;
    int nGt = 0, coveredAt03 = 0, coveredAt05 = 0;
    long long totalCands = 0;
};

} // namespace

int main(int argc, char** argv){
    bool real = argc > 1 && std::string(argv[1]) == "real";
    int nSample = argc > 2 ? std::atoi(argv[2]) : 150;
    unsigned seed = argc > 3 ? (unsigned)std::atoi(argv[3]) : 7;

    std::string csvPath = real ? "../data/dataset_real.csv" : "../data/dataset.csv";
    std::string splitFile = real ? "../data/test_real.txt" : "../data/test.txt";
    std::string imgDir = real ? "../test_set/segmentation_test/img" : "../dataset";

    std::unordered_map<std::string, std::vector<cv::Rect>> gt;
    std::ifstream f(csvPath);
    std::string line;
    while(std::getline(f, line)){
        std::stringstream ss(line);
        std::string img, x1s, y1s, x2s, y2s;
        std::getline(ss, img, ','); std::getline(ss, x1s, ',');
        std::getline(ss, y1s, ','); std::getline(ss, x2s, ',');
        std::getline(ss, y2s, ',');
        int x1 = std::stoi(x1s), y1 = std::stoi(y1s), x2 = std::stoi(x2s), y2 = std::stoi(y2s);
        gt[img].push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
    }
    std::vector<std::string> imgs;
    { std::ifstream sf(splitFile); std::string l; while(std::getline(sf, l)) if(!l.empty()) imgs.push_back(l); }
    std::mt19937 rng(seed);
    std::shuffle(imgs.begin(), imgs.end(), rng);
    if((int)imgs.size() > nSample) imgs.resize(nSample);

    auto run = [&](double frac, const std::string& label){
        Stat s;
        for(auto& imgName : imgs){
            auto it = gt.find(imgName);
            if(it == gt.end() || it->second.empty()) continue;
            cv::Mat bgr = cv::imread(imgDir + "/" + imgName);
            if(bgr.empty()) continue;

            std::vector<cv::Rect> cands;
            for(auto& b : blobs(bgr)){
                auto g = anchorAt(b, bgr.size(), frac);
                cands.insert(cands.end(), g.begin(), g.end());
            }
            s.totalCands += (long long)cands.size();

            for(auto& g : it->second){
                double best = 0;
                for(auto& c : cands) best = std::max(best, calculateIoU(c, g));
                s.sumBestIou += best;
                s.nGt++;
                if(best >= 0.3) s.coveredAt03++;
                if(best >= 0.5) s.coveredAt05++;
            }
        }
        std::cout << label << ": media best-IoU per GT=" << (s.sumBestIou / s.nGt)
                   << "  coverage@0.3=" << (100.0 * s.coveredAt03 / s.nGt) << "%"
                   << "  coverage@0.5=" << (100.0 * s.coveredAt05 / s.nGt) << "%"
                   << "  candidati/immagine=" << (double(s.totalCands) / imgs.size()) << "\n";
    };

    run(0.5, "frac=0.50 (centroide, attuale)");
    run(0.42, "frac=0.42");
    run(0.35, "frac=0.35");
    run(0.25, "frac=0.25");
    run(0.15, "frac=0.15");
    run(0.0, "frac=0.00 (ancorato al top del blob)");

    return 0;
}
