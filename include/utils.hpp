//Author: Edin Milenko
#pragma once

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>


double calculateIoU(cv::Rect a, cv::Rect b);

std::vector<int> nonMaxSuppression(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, double iouThr);

struct Pred {
    std::string imgName;
    cv::Rect bbox;
    int cls;
    double score;
};

struct GT {
    std::string imgName;
    cv::Rect bbox;
    int cls;
};

double macroF1(const std::vector<Pred>& preds, const std::vector<GT>& ground, const std::vector<int>& classes, double iouThr = 0.5);

double meanAveragePrecision(const std::vector<Pred>& preds, const std::vector<GT>& ground, const std::vector<int>& classes);

struct PixelIoU {
    std::map<int, long long> tp, fp, fn;
    void add(const cv::Mat& predLabels, const cv::Mat& gtLabels, const std::vector<int>& classes);
    double mean(const std::vector<int>& classes) const;
};
