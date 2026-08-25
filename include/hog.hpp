#pragma once
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

// standard patch size (positives, negatives and the HOG window): single source of truth,
// shared by patches.cpp (resizes) and this file (describes).
inline const cv::Size kPatchSize(16, 24);

// only place where HOG parameters are defined: the saved SVM model expects descriptors of
// an exact size, so training and detection must both go through here.
cv::HOGDescriptor makeHog();

// patches (already resized to kPatchSize) -> matrix of HOG descriptors, one row per patch.
cv::Mat computeFeatures(const std::vector<std::string>& files);

void trainSvm(const std::string& posDir, const std::string& negDir, const std::string& modelPath);

// note: cv::ml::SVMSGD doesn't derive from cv::ml::SVM (sibling hierarchies under
// StatModel), so the pointer type follows whatever model trainSvm actually produces.
float classify(const cv::Ptr<cv::ml::SVMSGD>& svm, const cv::Mat& patch);

// continuous score (not just the +-1 label from classify): predict(..., RAW_OUTPUT)
// doesn't work for SVMSGD, so it's reconstructed by hand from the weights and shift.
float rawScore(const cv::Ptr<cv::ml::SVMSGD>& svm, const cv::Mat& patch);

// continuous score for a cv::ml::SVM (RBF kernel): here RAW_OUTPUT works, but returns the
// signed distance with the opposite convention used elsewhere (higher = more likely a
// cone), so it's negated.
float rbfScore(const cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& patch);
