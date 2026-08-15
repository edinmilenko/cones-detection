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
static double calculateIoU(cv::Rect a, cv::Rect b);
void evaluatePredictions(std::vector<Pred> predictions, std::vector<GT> ground, double iouThreshold);

