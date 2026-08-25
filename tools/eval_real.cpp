// File a parte, non collegato a CMakeLists: come eval_cascade.cpp, ma puntato al vero test set
// esterno (test_set/segmentation_test, convertito in CSV da convert_supervisely_to_csv.py) invece
// del test set sintetico (dataset/ + data/test.txt) usato per tutto il tuning fin qui. Verifica
// se le soglie/parametri tarati sul dataset sintetico reggono su immagini reali.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I../tools -I/usr/include/opencv4 \
//     ../tools/eval_real.cpp ../src/hog.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o eval_real
//   ./eval_real [stage1_thr] [stage2_model.yml]

#include "detector_common.hpp"
#include "hog.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <sstream>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv){
    float stage1Thr = argc > 1 ? std::atof(argv[1]) : 15.0f;
    std::string stage2Path = argc > 2 ? argv[2] : "../data/svm_stage2_rbf.yml";

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
    std::vector<std::string> testImgs;
    { std::ifstream f(splitFile); std::string l; while(std::getline(f, l)) if(!l.empty()) testImgs.push_back(l); }

    cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load("../data/svm_hog_sgd_proposal.yml");
    cv::Mat W = stage1->getWeights();
    float shift = stage1->getShift();
    cv::Ptr<cv::ml::SVM> stage2 = cv::ml::SVM::load(stage2Path);
    cv::HOGDescriptor hog = makeHog();

    struct ImgData { std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<cv::Rect> gtBoxes; };
    std::vector<ImgData> data;

    long long nGt = 0, nCovered = 0, totalCands = 0, survivedStage1 = 0;
    for(const auto& imgName : testImgs){
        auto it = gt.find(imgName);
        if(it == gt.end() || it->second.empty()) continue;
        const auto& gtBoxes = it->second;

        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()){ std::cerr << "immagine non trovata: " << imgName << "\n"; continue; }
        cv::Mat gray; cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> cands = colorProposals(bgr);
        totalCands += (long long)cands.size();

        for(const auto& gb : gtBoxes){
            nGt++;
            for(const auto& c : cands) if(calculateIoU(c, gb) >= 0.3){ nCovered++; break; }
        }

        ImgData d;
        d.gtBoxes = gtBoxes;
        for(const auto& c : cands){
            cv::Mat crop = gray(c);
            cv::Mat patch;
            int interp = (c.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(crop, patch, kPatchSize, 0, 0, interp);
            std::vector<float> descriptor;
            hog.compute(patch, descriptor);

            float s1 = shift;
            for(size_t j = 0; j < descriptor.size(); j++) s1 += W.at<float>(0, (int)j) * descriptor[j];
            if(s1 <= stage1Thr) continue;
            survivedStage1++;

            cv::Mat featRow(1, (int)descriptor.size(), CV_32F, descriptor.data());
            cv::Mat raw;
            stage2->predict(featRow, raw, cv::ml::StatModel::RAW_OUTPUT);
            float s2 = -raw.at<float>(0, 0);

            d.boxes.push_back(c);
            d.scores.push_back(s2);
        }
        data.push_back(std::move(d));
    }

    std::cout << "=== TEST SET REALE (" << data.size() << " immagini, " << nGt << " GT box) ===\n";
    std::cout << "candidati/immagine: " << (double(totalCands)/data.size()) << "\n";
    std::cout << "coverage (IoU>=0.3, nessun classificatore): " << (100.0*nCovered/nGt) << "%\n";
    std::cout << "sopravvissuti allo stadio 1/immagine: " << (double(survivedStage1)/data.size()) << "\n\n";

    for(float thr2 : {-0.01f, -0.005f, 0.f, 0.005f, 0.01f, 0.015f, 0.02f, 0.03f, 0.05f, 0.08f, 0.1f, 0.13f, 0.15f, 0.18f, 0.2f, 0.25f, 0.3f}){
        long long tp = 0, fp = 0, fn = 0;
        for(const auto& d : data){
            std::vector<cv::Rect> keptBoxes;
            std::vector<float> keptScores;
            for(size_t i = 0; i < d.boxes.size(); i++){
                if(d.scores[i] > thr2){ keptBoxes.push_back(d.boxes[i]); keptScores.push_back(d.scores[i]); }
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
        std::cout << "  stage2_thr=" << thr2 << ": precision=" << precision << " recall=" << recall
                   << "  TP=" << tp << " FP=" << fp << " FN=" << fn << '\n';
    }

    return 0;
}
