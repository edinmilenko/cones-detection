// Throwaway sanity check: works out the sign convention of RAW_OUTPUT for cv::ml::SVM
// (plain predict against raw) on a few known pos/neg patches, before trusting the sign in
// eval_cascade.cpp.
#include "hog.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>

namespace fs = std::filesystem;

int main(){
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("../data/svm_stage2_rbf.yml");
    cv::HOGDescriptor hog = makeHog();

    auto testDir = [&](const std::string& dir, int n){
        int i = 0;
        for(const auto& e : fs::directory_iterator(dir)){
            if(!e.is_regular_file()) continue;
            if(i++ >= n) break;
            cv::Mat img = cv::imread(e.path().string(), cv::IMREAD_GRAYSCALE);
            std::vector<float> descriptor;
            hog.compute(img, descriptor);
            cv::Mat featRow(1, (int)descriptor.size(), CV_32F, descriptor.data());
            cv::Mat labelOut, rawOut;
            svm->predict(featRow, labelOut);
            svm->predict(featRow, rawOut, cv::ml::StatModel::RAW_OUTPUT);
            std::cout << dir << " label=" << labelOut.at<float>(0,0) << " raw=" << rawOut.at<float>(0,0) << "\n";
        }
    };
    testDir("../data/patches/pos_proposal", 5);
    testDir("../data/patches/neg_proposal", 5);
    return 0;
}
