// Ad-hoc variant of eval_cascade.cpp: like eval_real_nms_sweep.cpp but on the synthetic test set,
// to check that the gain from a more aggressive NMS is not an artefact of the mere 24
// images of the real test set. stage2_thr fixed at 0.01 (the best F1 already found), sweeping the
// NMS IoU threshold.
//
// Ad-hoc build (from the build/ folder):
//   g++ -std=gnu++17 -O2 -I../include -I../tools -I/usr/include/opencv4 \
//     ../tools/eval_cascade_nms_sweep.cpp ../src/hog.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o eval_cascade_nms_sweep
//   ./eval_cascade_nms_sweep [stage1_thr] [numero_immagini] [seed] [stage2_model.yml]

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
    float stage1Thr = argc > 1 ? std::atof(argv[1]) : 15.0f;
    int nSample = argc > 2 ? std::atoi(argv[2]) : 100;
    unsigned seed = argc > 3 ? (unsigned)std::atoi(argv[3]) : 7;
    std::string stage2Path = argc > 4 ? argv[4] : "../data/svm_stage2_rbf.yml";

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

    cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load("../data/svm_hog_sgd_proposal.yml");
    cv::Mat W = stage1->getWeights();
    float shift = stage1->getShift();
    cv::Ptr<cv::ml::SVM> stage2 = cv::ml::SVM::load(stage2Path);
    cv::HOGDescriptor hog = makeHog();

    struct ImgData { std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<cv::Rect> gtBoxes; };
    std::vector<ImgData> data;

    long long totalCands = 0, survivedStage1 = 0;
    for(const auto& imgName : testImgs){
        auto it = gt.find(imgName);
        if(it == gt.end() || it->second.empty()) continue;

        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;
        cv::Mat gray; cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> cands = colorProposals(bgr);
        totalCands += (long long)cands.size();

        ImgData d;
        d.gtBoxes = it->second;
        for(const auto& c : cands){
            cv::Mat crop = gray(c);
            cv::Mat patch;
            int interp = (c.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(crop, patch, kPatchSize, 0, 0, interp);
            std::vector<float> descriptor;
            hog.compute(patch, descriptor);

            float s1 = shift;
            for(size_t j = 0; j < descriptor.size(); j++) s1 += W.at<float>(0, (int)j) * descriptor[j];
            if(s1 <= stage1Thr) continue; // dropped by stage 1
            survivedStage1++;

            cv::Mat featRow(1, (int)descriptor.size(), CV_32F, descriptor.data());
            cv::Mat raw;
            stage2->predict(featRow, raw, cv::ml::StatModel::RAW_OUTPUT);
            // cv::ml::SVM: RAW_OUTPUT is the signed distance from the hyperplane; by construction
            // (positive label=1 used in training) MORE NEGATIVE values mean more towards the
            // positive class. The sign is flipped so that "higher score = more cone", as in stage 1
            // and the rest of the codebase.
            float s2 = -raw.at<float>(0, 0);

            d.boxes.push_back(c);
            d.scores.push_back(s2);
        }
        data.push_back(std::move(d));
    }

    std::cout << "images evaluated: " << data.size() << " (stage1_thr=" << stage1Thr << ")\n";
    std::cout << "candidates/image pre-stage1: " << (double(totalCands)/data.size())
               << "  survivors/image: " << (double(survivedStage1)/data.size()) << "\n\n";

    const float thr2 = 0.01f;
    std::vector<ImgData> filtered;
    for(const auto& d : data){
        ImgData fd; fd.gtBoxes = d.gtBoxes;
        for(size_t i = 0; i < d.boxes.size(); i++){
            if(d.scores[i] > thr2){ fd.boxes.push_back(d.boxes[i]); fd.scores.push_back(d.scores[i]); }
        }
        filtered.push_back(std::move(fd));
    }

    std::cout << "sweep threshold IoU NMS a stage2_thr=" << thr2 << " fixed:\n";
    for(double iouThr : {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5}){
        long long tp = 0, fp = 0, fn = 0;
        for(const auto& d : filtered){
            std::vector<int> nmsIdx = detector::nms(d.boxes, d.scores, iouThr);

            std::vector<bool> matched(d.gtBoxes.size(), false);
            for(int idx : nmsIdx){
                double bestIou = 0.0; int bestJ = -1;
                for(size_t j = 0; j < d.gtBoxes.size(); j++){
                    if(matched[j]) continue;
                    double i_ = calculateIoU(d.boxes[idx], d.gtBoxes[j]);
                    if(i_ > bestIou){ bestIou = i_; bestJ = (int)j; }
                }
                if(bestIou >= 0.3){ tp++; matched[bestJ] = true; }
                else fp++;
            }
            for(bool m : matched) if(!m) fn++;
        }
        double precision = (tp + fp) > 0 ? double(tp) / double(tp + fp) : 0.0;
        double recall = (tp + fn) > 0 ? double(tp) / double(tp + fn) : 0.0;
        double f1 = (precision + recall) > 0 ? 2*precision*recall/(precision+recall) : 0.0;
        std::cout << "  nms_iou=" << iouThr << ": precision=" << precision << " recall=" << recall
                   << " f1=" << f1 << "  TP=" << tp << " FP=" << fp << " FN=" << fn << '\n';
    }

    return 0;
}
