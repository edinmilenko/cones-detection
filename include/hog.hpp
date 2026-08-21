#pragma once
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

// unico punto in cui vengono definiti i parametri di HOG: il modello SVM salvato
// si aspetta descrittori di dimensione esatta, quindi training e detection devono
// per forza passare da qui.
cv::HOGDescriptor makeHog();

// patch (gia' ridimensionate a kPatchSize) -> matrice di descrittori HOG, una riga per patch.
cv::Mat computeFeatures(const std::vector<std::string>& files);

void trainSvm(const std::string& posDir, const std::string& negDir, const std::string& modelPath);

float classify(const cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& patch);
