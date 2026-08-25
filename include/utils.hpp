#pragma once

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

//returns iou
struct Pred {
    std::string imgName;
    cv::Rect bbox;
    double score;
};

struct GT {
    std::string imgName;
    cv::Rect bbox;
    bool matched = false;
};
double calculateIoU(cv::Rect a, cv::Rect b);
void evaluatePredictions(std::vector<Pred> predictions, std::vector<GT> ground, double iouThreshold);

//keeps only the highest-scoring box among those that overlap above iouThr
std::vector<int> nonMaxSuppression(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, double iouThr);

