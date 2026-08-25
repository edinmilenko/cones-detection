// File a parte, non collegato a CMakeLists: approccio geometrico invece che color-blob puro.
// Idea dell'utente: i coni sono triangoli (in silhouette): due lati obliqui che convergono
// verso l'apice in alto, base orizzontale in basso. Invece di indovinare la taglia del box
// (dense_grid_proposals.cpp, altezze assolute alla cieca), si cercano le linee vere con Hough,
// si accoppia una linea "sinistra" (scende da sinistra verso l'apice, dx/dy>0 in coordinate
// immagine) con una "destra" (dx/dy<0) il cui apice combacia, e il bbox del triangolo risultante
// e' il candidato. Il colore resta usato SOLO per delimitare le regioni di ricerca (dove cercare
// le linee), non per definire il box come nella pipeline attuale — la taglia/posizione esatta
// viene dalla geometria, non dal blob.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/triangle_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
//     -o triangle_proposals
//   ./triangle_proposals [numero_immagini] [seed]

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

const double kAspect = static_cast<double>(kPatchSize.width) / kPatchSize.height;

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

struct Segment { cv::Point top, bottom; double dxdy; };

// dx/dy>0 = "lato sinistro" del cono (sale verso destra andando verso l'apice), dx/dy<0 =
// "lato destro". Scarta segmenti troppo corti o troppo vicini all'orizzontale/verticale (bordi
// del terreno, ombre, texture d'erba: quasi tutte le linee spurie in una scena naturale sono
// vicine all'orizzontale).
std::vector<Segment> classifySegments(const std::vector<cv::Vec4i>& lines, double minAngleDeg, double maxAngleDeg){
    std::vector<Segment> out;
    for(const auto& l : lines){
        cv::Point p1(l[0], l[1]), p2(l[2], l[3]);
        cv::Point top = (p1.y <= p2.y) ? p1 : p2;
        cv::Point bottom = (p1.y <= p2.y) ? p2 : p1;
        int dy = bottom.y - top.y;
        if(dy < 3) continue; // quasi orizzontale, o degenere
        double dxdy = double(bottom.x - top.x) / dy;
        double angleFromVertical = std::atan(std::abs(dxdy)) * 180.0 / CV_PI;
        if(angleFromVertical < minAngleDeg || angleFromVertical > maxAngleDeg) continue;
        out.push_back({top, bottom, dxdy});
    }
    return out;
}

enum class Smooth { None, Gaussian, Bilateral, Median };

struct Config {
    std::string name;
    Smooth smooth;
    int smoothParam; // Gaussian/Median: dimensione kernel (dispari); Bilateral: diametro
    int cannyLo, cannyHi;
    int houghThr, minLineLen, maxLineGap;
    double minAngleDeg, maxAngleDeg; // range accettato per i lati del cono, gradi dalla verticale
    double apexTolFrac;   // tolleranza orizzontale tra gli apici delle due linee, frazione dell'altezza combinata
    double roiMarginMult; // ROI di ricerca intorno al blob colore: multiplo della sua altezza
    double minColorFrac;  // frazione minima di pixel-maschera dentro il bbox candidato finale
};

std::vector<cv::Rect> runConfig(const cv::Mat& imgBGR, const Config& cfg){
    cv::Mat corrected = whiteBalance(imgBGR, 6.0);
    cv::Mat mask = colorMask(corrected);

    cv::Mat gray; cv::cvtColor(corrected, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    switch(cfg.smooth){
        case Smooth::None:      blurred = gray; break;
        case Smooth::Gaussian:  cv::GaussianBlur(gray, blurred, cv::Size(cfg.smoothParam, cfg.smoothParam), 0); break;
        case Smooth::Bilateral: cv::bilateralFilter(gray, blurred, cfg.smoothParam, cfg.smoothParam * 2.0, cfg.smoothParam / 2.0); break;
        case Smooth::Median:    cv::medianBlur(gray, blurred, cfg.smoothParam); break;
    }
    cv::Mat edges; cv::Canny(blurred, edges, cfg.cannyLo, cfg.cannyHi);

    std::vector<cv::Vec4i> linesRaw;
    cv::HoughLinesP(edges, linesRaw, 1, CV_PI/180, cfg.houghThr, cfg.minLineLen, cfg.maxLineGap);
    std::vector<Segment> segments = classifySegments(linesRaw, cfg.minAngleDeg, cfg.maxAngleDeg);

    // fonde i blob vicini in regioni di ricerca prima di cercare le linee: senza questo, migliaia
    // di frammenti generano ROI quasi identiche e sovrapposte, ripetendo lo stesso pairing di
    // linee decine di volte (spiega i milioni di candidati/immagine della prima versione).
    cv::Mat merged;
    cv::dilate(mask, merged, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9,9)));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(merged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> candidates;
    for(auto& c : contours){
        cv::Rect blob = cv::boundingRect(c);
        if(blob.area() < 3) continue;

        double cx = blob.x + blob.width / 2.0;
        double cy = blob.y + blob.height / 2.0;
        double roiH = std::max(20.0, blob.height * cfg.roiMarginMult);
        double roiW = roiH * kAspect * 1.5;
        cv::Rect roi(
            std::max(0, (int)std::lround(cx - roiW/2)),
            std::max(0, (int)std::lround(cy - roiH/2)),
            (int)std::lround(roiW), (int)std::lround(roiH));
        roi.width = std::min(roi.width, imgBGR.cols - roi.x);
        roi.height = std::min(roi.height, imgBGR.rows - roi.y);

        // segmenti il cui midpoint cade nella ROI
        std::vector<const Segment*> inRoi;
        for(const auto& s : segments){
            cv::Point mid((s.top.x + s.bottom.x)/2, (s.top.y + s.bottom.y)/2);
            if(roi.contains(mid)) inRoi.push_back(&s);
        }

        for(const auto* sl : inRoi){
            if(sl->dxdy <= 0) continue; // lato sinistro: dx/dy>0
            for(const auto* sr : inRoi){
                if(sr->dxdy >= 0) continue; // lato destro: dx/dy<0

                double combinedH = std::max(sl->bottom.y, sr->bottom.y) - std::min(sl->top.y, sr->top.y);
                if(combinedH < 6) continue;
                double apexTol = std::max(4.0, combinedH * cfg.apexTolFrac);
                if(std::abs(sl->top.x - sr->top.x) > apexTol) continue;

                int x0 = std::min({sl->top.x, sl->bottom.x, sr->top.x, sr->bottom.x});
                int x1 = std::max({sl->top.x, sl->bottom.x, sr->top.x, sr->bottom.x});
                int y0 = std::min({sl->top.y, sr->top.y});
                int y1 = std::max({sl->bottom.y, sr->bottom.y});
                cv::Rect box(x0, y0, x1 - x0, y1 - y0);
                box &= cv::Rect(0, 0, imgBGR.cols, imgBGR.rows);
                if(box.width < 4 || box.height < 4) continue;

                if(cfg.minColorFrac > 0){
                    double frac = cv::countNonZero(mask(box)) / double(box.area());
                    if(frac < cfg.minColorFrac) continue;
                }
                candidates.push_back(box);
            }
        }
    }
    return candidates;
}

} // namespace

int main(int argc, char** argv){
    int nSample = argc > 1 ? std::atoi(argv[1]) : 60;
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

    using S = Smooth;
    std::vector<Config> configs = {
        {"I: gauss7, canny30-100",   S::Gaussian,  7, 30, 100, 15, 6, 4, 8, 35, 0.3, 2.0, 0.0},
        {"J: gauss11, canny30-100",  S::Gaussian, 11, 30, 100, 15, 6, 4, 8, 35, 0.3, 2.0, 0.0},
        {"K: bilateral9, canny30-100", S::Bilateral, 9, 30, 100, 15, 6, 4, 8, 35, 0.3, 2.0, 0.0},
        {"L: median5, canny30-100",  S::Median,    5, 30, 100, 15, 6, 4, 8, 35, 0.3, 2.0, 0.0},
        {"M: gauss7, canny50-150",   S::Gaussian,  7, 50, 150, 15, 6, 4, 8, 35, 0.3, 2.0, 0.0},
    };

    for(auto& cfg : configs){
        long long nGt = 0, nCovered = 0, nCands = 0;
        int nImg = 0;
        for(const auto& imgName : testImgs){
            auto it = gt.find(imgName);
            if(it == gt.end() || it->second.empty()) continue;
            cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
            if(bgr.empty()) continue;
            nImg++;

            std::vector<cv::Rect> cands = runConfig(bgr, cfg);
            nCands += (long long)cands.size();

            for(const auto& gb : it->second){
                nGt++;
                double bestIou = 0.0;
                for(const auto& c : cands) bestIou = std::max(bestIou, calculateIoU(c, gb));
                if(bestIou >= 0.3) nCovered++;
            }
        }
        std::cout << cfg.name << ":  coverage=" << (100.0*nCovered/nGt) << "%  ("
                   << nCovered << "/" << nGt << ")   candidati/immagine=" << (double(nCands)/nImg) << "\n";
    }

    return 0;
}
