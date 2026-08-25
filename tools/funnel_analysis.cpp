// File a parte, non collegato a CMakeLists: imbuto recall per stadio (proposal -> soglia ->
// NMS) e percentili dello score raw per candidati "buoni" (IoU>=0.3 con un cono vero) vs
// "cattivi" (IoU<0.05 con qualsiasi cono), sulla stessa lista di immagini per ogni
// combinazione candidati/modello, per isolare dove si perde davvero la recall.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I../tools -I/usr/include/opencv4 \
//     ../tools/funnel_analysis.cpp ../src/hog.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o funnel_analysis
//   ./funnel_analysis <model.yml> <raw|aspect> [soglia] [numero_immagini] [seed]

#include "detector_common.hpp"
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

namespace {

std::vector<cv::Rect> rawBlobs(const cv::Mat& imgBGR){
    return colorProposals(imgBGR);
}

double percentile(std::vector<float> v, double p){
    if(v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    double frac = idx - lo;
    return v[lo] * (1 - frac) + v[hi] * frac;
}

} // namespace

int main(int argc, char** argv){
    if(argc < 3){
        std::cerr << "uso: funnel_analysis <model.yml> <raw|aspect> [soglia] [numero_immagini] [seed]\n";
        return 1;
    }
    std::string modelPath = argv[1];
    std::string mode = argv[2];
    float thr = argc > 3 ? std::atof(argv[3]) : 0.0f;
    int nSample = argc > 4 ? std::atoi(argv[4]) : 150;
    unsigned seed = argc > 5 ? (unsigned)std::atoi(argv[5]) : 7;

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
            std::getline(ss, img, ',');
            std::getline(ss, x1s, ',');
            std::getline(ss, y1s, ',');
            std::getline(ss, x2s, ',');
            std::getline(ss, y2s, ',');
            int x1 = std::stoi(x1s), y1 = std::stoi(y1s), x2 = std::stoi(x2s), y2 = std::stoi(y2s);
            gt[img].push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
    }

    std::vector<std::string> testImgs;
    {
        std::ifstream f(splitFile);
        std::string l;
        while(std::getline(f, l)) if(!l.empty()) testImgs.push_back(l);
    }
    std::mt19937 rng(seed);
    std::shuffle(testImgs.begin(), testImgs.end(), rng);
    if((int)testImgs.size() > nSample) testImgs.resize(nSample);

    cv::Ptr<cv::ml::SVMSGD> svm = cv::ml::SVMSGD::load(modelPath);
    cv::Mat W = svm->getWeights();
    float shift = svm->getShift();
    cv::HOGDescriptor hog = makeHog();

    long long nGt = 0, nCovered = 0, nSurvivesThr = 0, nSurvivesNms = 0;
    std::vector<float> matchedScores, unmatchedScores;

    for(const auto& imgName : testImgs){
        auto it = gt.find(imgName);
        if(it == gt.end() || it->second.empty()) continue;
        const auto& gtBoxes = it->second;

        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> cands = (mode == "aspect") ? detector::proposeCandidates(bgr) : rawBlobs(bgr);

        std::vector<float> scores(cands.size());
        for(size_t i = 0; i < cands.size(); i++){
            const auto& c = cands[i];
            cv::Mat crop = gray(c);
            cv::Mat patch;
            int interp = (c.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(crop, patch, kPatchSize, 0, 0, interp);
            std::vector<float> descriptor;
            hog.compute(patch, descriptor);
            float s = shift;
            for(size_t j = 0; j < descriptor.size(); j++) s += W.at<float>(0, (int)j) * descriptor[j];
            scores[i] = s;

            double bestIou = 0.0;
            for(const auto& gb : gtBoxes) bestIou = std::max(bestIou, calculateIoU(c, gb));
            if(bestIou >= 0.3) matchedScores.push_back(s);
            else if(bestIou < 0.05) unmatchedScores.push_back(s);
        }

        // stage 2 & 3, per GT box di questa immagine
        std::vector<bool> covered(gtBoxes.size(), false), survivesThr(gtBoxes.size(), false);
        for(size_t g = 0; g < gtBoxes.size(); g++){
            for(size_t i = 0; i < cands.size(); i++){
                if(calculateIoU(cands[i], gtBoxes[g]) >= 0.3){
                    covered[g] = true;
                    if(scores[i] > thr) survivesThr[g] = true;
                }
            }
        }

        // stage 4: NMS sui candidati sopra soglia, poi matching greedy per punteggio (stessa logica di eval_detector)
        std::vector<cv::Rect> keptBoxes;
        std::vector<float> keptScores;
        for(size_t i = 0; i < cands.size(); i++){
            if(scores[i] > thr){ keptBoxes.push_back(cands[i]); keptScores.push_back(scores[i]); }
        }
        std::vector<int> nmsIdx = detector::nms(keptBoxes, keptScores, 0.4);
        std::vector<bool> matchedAfterNms(gtBoxes.size(), false);
        for(int idx : nmsIdx){
            double bestIou = 0.0; int bestJ = -1;
            for(size_t g = 0; g < gtBoxes.size(); g++){
                if(matchedAfterNms[g]) continue;
                double i_ = calculateIoU(keptBoxes[idx], gtBoxes[g]);
                if(i_ > bestIou){ bestIou = i_; bestJ = (int)g; }
            }
            if(bestIou >= 0.3) matchedAfterNms[bestJ] = true;
        }

        for(size_t g = 0; g < gtBoxes.size(); g++){
            nGt++;
            if(covered[g]) nCovered++;
            if(survivesThr[g]) nSurvivesThr++;
            if(matchedAfterNms[g]) nSurvivesNms++;
        }
    }

    std::cout << "=== IMBUTO (" << mode << ", modello=" << modelPath << ", thr=" << thr << ", n_img=" << nSample << ") ===\n";
    std::cout << "1. GT totali:                          " << nGt << "\n";
    std::cout << "2. coperte da un candidato (IoU>=0.3):  " << nCovered << "  (" << (100.0*nCovered/nGt) << "%)\n";
    std::cout << "3. sopravvivono alla soglia:            " << nSurvivesThr << "  (" << (100.0*nSurvivesThr/nGt) << "% del totale, "
               << (100.0*nSurvivesThr/std::max<long long>(nCovered,1)) << "% di quelle coperte)\n";
    std::cout << "4. sopravvivono alla NMS:               " << nSurvivesNms << "  (" << (100.0*nSurvivesNms/nGt) << "% del totale, "
               << (100.0*nSurvivesNms/std::max<long long>(nSurvivesThr,1)) << "% di quelle sopra soglia)\n";

    std::cout << "\n=== SCORE RAW: candidati che matchano un cono (IoU>=0.3), n=" << matchedScores.size() << " ===\n";
    for(double p : {0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0}){
        std::cout << "  p" << p << ": " << percentile(matchedScores, p) << "\n";
    }
    std::cout << "=== SCORE RAW: candidati senza nessun cono vicino (IoU<0.05), n=" << unmatchedScores.size() << " ===\n";
    for(double p : {0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0}){
        std::cout << "  p" << p << ": " << percentile(unmatchedScores, p) << "\n";
    }

    return 0;
}
