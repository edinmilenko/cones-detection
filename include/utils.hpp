#pragma once

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

//returns iou
double calculateIoU(cv::Rect a, cv::Rect b);

//same thing for two binary masks of equal size
double maskIoU(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask);

//fraction of ground truth boxes that at least one proposal covers above iouThr
double candidateRecall(const std::vector<cv::Rect>& proposals, const std::vector<cv::Rect>& groundTruth, double iouThr = 0.3);

//keeps only the highest-scoring box among those that overlap above iouThr
std::vector<int> nonMaxSuppression(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, double iouThr);

// one detection and one annotated cone, as the metrics below want them
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

// F1 of each class on its own, then averaged with no weights: a class with 14 cones counts as
// much as one with 188.
double macroF1(const std::vector<Pred>& preds, const std::vector<GT>& ground, const std::vector<int>& classes, double iouThr = 0.5);

// AP per class (101-point interpolation, as COCO does) averaged over the ten IoU thresholds from
// 0.50 to 0.95. Needs preds sorted by nothing in particular: it sorts them by score itself.
double meanAveragePrecision(const std::vector<Pred>& preds, const std::vector<GT>& ground, const std::vector<int>& classes);

// Jaccard over mask pixels, TP/(TP+FP+FN) per class. Fed one image at a time so the label maps of
// the whole test set never have to be in memory together.
struct PixelIoU {
    std::map<int, long long> tp, fp, fn;
    void add(const cv::Mat& predLabels, const cv::Mat& gtLabels, const std::vector<int>& classes);
    double mean(const std::vector<int>& classes) const;
};
