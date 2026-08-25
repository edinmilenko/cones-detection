// File a parte, non collegato a CMakeLists: come visualize_detections.cpp, ma per il vero test
// set esterno (test_set/segmentation_test) e con la cascata a due stadi attuale (stadio1
// lineare -> stadio2 RBF) invece del singolo classificatore lineare aspect-normalizzato usato
// dalla versione originale del tool. Disegna GT (verde) e detection accettate dopo NMS (rosso)
// su tutte le immagini del test set reale.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I../tools -I/usr/include/opencv4 \
//     ../tools/visualize_real.cpp ../src/hog.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o visualize_real
//   ./visualize_real [stage1_thr] [stage2_thr] [nms_iou] [out_dir]

#include "detector_common.hpp"
#include "hog.hpp"
#include <filesystem>
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
    float stage2Thr = argc > 2 ? std::atof(argv[2]) : 0.01f;
    double nmsIou = argc > 3 ? std::atof(argv[3]) : 0.4;

    const std::string datasetDir = "../test_set/segmentation_test/img";
    const std::string csvPath = "../data/dataset_real.csv";
    const std::string splitFile = "../data/test_real.txt";
    const std::string outDir = argc > 4 ? argv[4] : "../data/detection_samples_real";
    std::filesystem::create_directories(outDir);

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
    cv::Mat W = stage1->getWeights();
    float shift = stage1->getShift();
    cv::Ptr<cv::ml::SVM> stage2 = cv::ml::SVM::load("../data/svm_stage2_rbf.yml");
    cv::HOGDescriptor hog = makeHog();

    for(const auto& imgName : imgs){
        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()){ std::cerr << "skip (non leggibile): " << imgName << "\n"; continue; }
        cv::Mat gray; cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> cands = colorProposals(bgr);

        std::vector<cv::Rect> validBoxes;
        std::vector<float> validScores;
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

            cv::Mat featRow(1, (int)descriptor.size(), CV_32F, descriptor.data());
            cv::Mat raw;
            stage2->predict(featRow, raw, cv::ml::StatModel::RAW_OUTPUT);
            float s2 = -raw.at<float>(0, 0);
            if(s2 <= stage2Thr) continue;

            validBoxes.push_back(c);
            validScores.push_back(s2);
        }

        std::vector<int> kept = detector::nms(validBoxes, validScores, nmsIou);

        cv::Mat out = bgr.clone();
        for(const auto& box : gt[imgName]) cv::rectangle(out, box, cv::Scalar(0, 200, 0), 2);      // GT verde
        for(int idx : kept) cv::rectangle(out, validBoxes[idx], cv::Scalar(0, 0, 255), 2);          // predetti rosso

        std::string stem = imgName.substr(0, imgName.find_last_of('.'));
        std::string outPath = outDir + "/" + stem + "_det.png";
        cv::imwrite(outPath, out);
        std::cout << imgName << ": GT=" << gt[imgName].size() << " candidati=" << cands.size()
                   << " dopo cascata+NMS=" << kept.size() << " -> " << outPath << "\n";
    }

    return 0;
}
