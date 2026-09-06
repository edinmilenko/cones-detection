// Standalone file, not wired into CMakeLists: the final detection pipeline --
// colorProposals() -> aspect ratio normalization -> classification with the SVMSGD model retrained
// through hard negative mining -> threshold on the raw score -> NMS.
// Disegna GT (verde) e detection accettate (rosso) su qualche immagine di test.
//
// History, for whoever reads this later: the very first version used HOGDescriptor::detectMultiScale
// (exhaustive sliding window): hundreds of thousands of windows per image, minutes to test a single
// one, false positives everywhere because it skipped the proposal stage. The second version wired in
// colorProposals() as it was: no more boxes on sky and trees, but candidate recall around 50% and,
// worse, the classifier rejected most candidates even when they sat squarely on a real cone,
// because a color blob has an arbitrary shape (often just the colored band, not the whole cone)
// very unlike the clean crops it had been trained on. This version normalizes every blob to the
// aspect ratio of kPatchSize before classifying it, and uses a model retrained with hard negative
// mining (the false positives of the old model on the color candidates of the TRAIN images, added
// as negatives).
//
// Ad-hoc build (from the build/ folder):
//   g++ -std=gnu++17 -O2 -I../include -I../tools -I/usr/include/opencv4 \
//     ../tools/visualize_detections.cpp ../src/hog.cpp ../src/color_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o visualize_detections
//   ./visualize_detections [image_count] [score_threshold] [model.yml]

#include "detector_common.hpp"
#include "hog.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv){
    int limit = argc > 1 ? std::atoi(argv[1]) : 24;
    float scoreThreshold = argc > 2 ? std::atof(argv[2]) : 0.0f;
    std::string modelPath = argc > 3 ? argv[3] : "../data/svm_hog_sgd_hnm.yml";

    const std::string datasetDir = "../dataset";
    const std::string csvPath = "../data/dataset.csv";
    const std::string splitFile = "../data/test.txt";
    const std::string outDir = "../data/detection_samples";
    std::filesystem::create_directories(outDir);

    std::set<std::string> testSet;
    {
        std::ifstream f(splitFile);
        std::string l;
        while(std::getline(f, l)) if(!l.empty()) testSet.insert(l);
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
            if(testSet.count(img) == 0) continue;
            int x1 = std::stoi(x1s), y1 = std::stoi(y1s), x2 = std::stoi(x2s), y2 = std::stoi(y2s);
            gt[img].push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
    }

    std::vector<std::pair<std::string, int>> byCount;
    for(const auto& [img, boxes] : gt) byCount.push_back({img, (int)boxes.size()});
    std::sort(byCount.begin(), byCount.end(), [](const auto& a, const auto& b){ return a.second > b.second; });

    std::vector<std::string> chosen;
    for(size_t i = 0; i < byCount.size() && (int)chosen.size() < limit; i++) chosen.push_back(byCount[i].first);

    cv::Ptr<cv::ml::SVMSGD> svm = cv::ml::SVMSGD::load(modelPath);
    cv::Mat W = svm->getWeights();
    float shift = svm->getShift();
    cv::HOGDescriptor hog = makeHog();

    for(const auto& imgName : chosen){
        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()){
            std::cerr << "skip (unreadable): " << imgName << "\n";
            continue;
        }
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> candidates = detector::proposeCandidates(bgr);

        std::vector<cv::Rect> validBoxes;
        std::vector<float> validScores;
        for(const auto& c : candidates){
            cv::Mat crop = gray(c);
            cv::Mat patch;
            int interp = (c.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(crop, patch, kPatchSize, 0, 0, interp);

            std::vector<float> descriptor;
            hog.compute(patch, descriptor);
            float score = shift;
            for(size_t j = 0; j < descriptor.size(); j++) score += W.at<float>(0, (int)j) * descriptor[j];

            if(score > scoreThreshold){
                validBoxes.push_back(c);
                validScores.push_back(score);
            }
        }

        std::vector<int> kept = detector::nms(validBoxes, validScores);

        cv::Mat color;
        cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
        for(const auto& box : gt[imgName]) cv::rectangle(color, box, cv::Scalar(0, 200, 0), 2);   // GT verde
        for(int idx : kept) cv::rectangle(color, validBoxes[idx], cv::Scalar(0, 0, 255), 2);       // predetti rosso

        std::string stem = imgName.substr(0, imgName.find_last_of('.'));
        std::string outPath = outDir + "/" + stem + "_det.png";
        cv::imwrite(outPath, color);
        std::cout << imgName << ": GT=" << gt[imgName].size() << " candidates=" << candidates.size()
                   << " above threshold=" << validBoxes.size() << " after NMS=" << kept.size()
                   << " -> " << outPath << "\n";
    }

    return 0;
}
