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

// descriptor of one patch (BGR, already kPatchSize): HOG on the grey version plus a small H-S
// histogram. HOG alone throws the colour away, and colour is what separates a cone from a piece of
// asphalt with the same shape: adding the histogram took the macro F1 from 0.37 to 0.48.
// Every stage goes through here, so training and detection cannot disagree on the layout.
std::vector<float> describePatch(const cv::Mat& patch);

void trainSvm(const std::string& posDir, const std::string& negDir, const std::string& modelPath);

// second stage of the cascade: an RBF SVM trained only on the patches stage 1 lets through, which
// is the distribution it will actually see at inference.
// C is fixed rather than left to trainAuto, and the samples are capped at 5000 per class. Both
// numbers were measured (tools/tune_stage2.cpp): trainAuto picks a different C depending on how many
// samples it gets, which moves the scores onto another scale and silently invalidates the stage 2
// threshold. Fixing C also drops the 6 minutes of cross validation from the training.
void trainStage2(const std::string& posDir, const std::string& negDir, const std::string& stage1Path, const std::string& modelPath, float stage1Thr = 15.0f, int maxPerClass = 5000, double fixedC = 2.5);

// note: cv::ml::SVMSGD doesn't derive from cv::ml::SVM (sibling hierarchies under
// StatModel), so the pointer type follows whatever model trainSvm actually produces.
// continuous score of the linear stage: predict(..., RAW_OUTPUT)
// doesn't work for SVMSGD, so it's reconstructed by hand from the weights and shift.
float rawScore(const cv::Ptr<cv::ml::SVMSGD>& svm, const cv::Mat& patch);

// continuous score for a cv::ml::SVM (RBF kernel): here RAW_OUTPUT works, but returns the
// signed distance with the opposite convention used elsewhere (higher = more likely a
// cone), so it's negated.
float rbfScore(const cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& patch);
