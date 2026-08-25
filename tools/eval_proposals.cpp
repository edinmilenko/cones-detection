// File a parte, non collegato a CMakeLists: misura la recall di colorProposals() (IoU>=0.3,
// via candidateRecall() gia' nel repo) su un campione di immagini del test set, per validare
// le modifiche fatte a color_proposals.cpp (soglie S/V alzate 60/40 -> 80/60, filtro area
// massima 1% sui candidati).
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/eval_proposals.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml \
//     -o eval_proposals
//   ./eval_proposals [numero_immagini] [seed]

#include "color_proposals.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv){
    int nSample = argc > 1 ? std::atoi(argv[1]) : 80;
    unsigned seed = argc > 2 ? (unsigned)std::atoi(argv[2]) : 42;

    const std::string datasetDir = "../dataset";
    const std::string csvPath = "../data/dataset.csv";
    const std::string splitFile = "../data/test.txt";

    std::vector<std::string> testImgs;
    {
        std::ifstream f(splitFile);
        std::string l;
        while(std::getline(f, l)) if(!l.empty()) testImgs.push_back(l);
    }

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

    std::mt19937 rng(seed);
    std::shuffle(testImgs.begin(), testImgs.end(), rng);
    if((int)testImgs.size() > nSample) testImgs.resize(nSample);

    double recallSum = 0.0;
    int nUsed = 0;
    int bigBlobImgs = 0;
    long long totalCandidates = 0;

    for(const auto& imgName : testImgs){
        auto it = gt.find(imgName);
        if(it == gt.end() || it->second.empty()) continue;

        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;

        std::vector<cv::Rect> candidates = colorProposals(bgr);
        double r = candidateRecall(candidates, it->second, 0.3);
        if(r < 0) continue;

        recallSum += r;
        nUsed++;
        totalCandidates += (long long)candidates.size();

        double imgArea = double(bgr.rows) * bgr.cols;
        for(const auto& c : candidates){
            if(c.area() > 0.03 * imgArea){ bigBlobImgs++; break; }
        }
    }

    std::cout << "immagini valutate: " << nUsed << '\n';
    std::cout << "recall media (IoU>=0.3): " << (recallSum / nUsed) << '\n';
    std::cout << "immagini con almeno un blob >3% area immagine: " << bigBlobImgs << '\n';
    std::cout << "candidati medi per immagine: " << (double(totalCandidates) / nUsed) << '\n';

    return 0;
}
