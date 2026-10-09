//Author: Alessandro Martini
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>


inline const cv::Size PATCH_SIZE(16, 24);

cv::HOGDescriptor makeHog();

std::vector<float> describePatch(const cv::Mat& patch);

void trainSvm(const std::string& posDir, const std::string& negDir, const std::string& modelPath);

void trainStage2(const std::string& posDir, const std::string& negDir, const std::string& stage1Path, const std::string& modelPath, float stage1Thr = 15.0f, int maxPerClass = 5000);

float rawScore(const cv::Ptr<cv::ml::SVMSGD>& svm, const cv::Mat& patch);

float rbfScore(const cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& patch);
