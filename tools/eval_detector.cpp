// Standalone file, not wired into CMakeLists: scores the complete detection pipeline
// (colorProposals + aspect ratio normalization + classification + threshold + NMS) against the
// against the real bboxes, over a sample of test images, for one model and a series of thresholds.
//
// Ad-hoc build (from the build/ folder):
//   g++ -std=gnu++17 -O2 -I../include -I../tools -I/usr/include/opencv4 \
//     ../tools/eval_detector.cpp ../src/hog.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o eval_detector
//   ./eval_detector <model.yml> [raw|aspect] [numero_immagini] [seed]

#include "detector_common.hpp"
#include "hog.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "uso: eval_detector <model.yml> [numero_images] [seed]\n";
        return 1;
    }
    std::string modelPath = argv[1];
    std::string mode = argc > 2 ? argv[2] : "aspect";
    int nSample = argc > 3 ? std::atoi(argv[3]) : 100;
    unsigned seed = argc > 4 ? (unsigned)std::atoi(argv[4]) : 7;

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

    // cache: candidates + raw scores per image, so the threshold sweep does not recompute HOG each time
    struct ImgData { std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<cv::Rect> gtBoxes; };
    std::vector<ImgData> data;

    for(const auto& imgName : testImgs){
        auto it = gt.find(imgName);
        if(it == gt.end() || it->second.empty()) continue;

        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> cands = (mode == "raw") ? colorProposals(bgr) : detector::proposeCandidates(bgr);
        ImgData d;
        d.gtBoxes = it->second;
        for(const auto& c : cands){
            cv::Mat crop = gray(c);
            cv::Mat patch;
            int interp = (c.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(crop, patch, kPatchSize, 0, 0, interp);
            std::vector<float> descriptor;
            hog.compute(patch, descriptor);
            float score = shift;
            for(size_t j = 0; j < descriptor.size(); j++) score += W.at<float>(0, (int)j) * descriptor[j];
            d.boxes.push_back(c);
            d.scores.push_back(score);
        }
        data.push_back(std::move(d));
    }

    std::cout << "images evaluated: " << data.size() << " (model: " << modelPath << ", mode=" << mode << ")\n";

    for(float thr : {-10.f, 0.f, 10.f, 20.f, 30.f, 40.f, 50.f}){
        long long tp = 0, fp = 0, fn = 0;
        for(const auto& d : data){
            std::vector<cv::Rect> keptBoxes;
            std::vector<float> keptScores;
            for(size_t i = 0; i < d.boxes.size(); i++){
                if(d.scores[i] > thr){ keptBoxes.push_back(d.boxes[i]); keptScores.push_back(d.scores[i]); }
            }
            std::vector<int> nmsIdx = detector::nms(keptBoxes, keptScores, 0.4);

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
        std::cout << "  thr=" << thr << ": precision=" << precision << " recall=" << recall
                   << "  TP=" << tp << " FP=" << fp << " FN=" << fn << '\n';
    }

    return 0;
}
