#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// color proposals -> stage1 linear filter (cheap) -> stage2 RBF -> NMS -> pad the chosen box.
// scores, if given, receives the stage2 score of each box, which mAP needs to rank them.
std::vector<cv::Rect> detectCones(const cv::Mat& bgr, std::vector<float>* scores = nullptr);

// one cone as the whole pipeline sees it
struct Cone {
    cv::Rect roi;    // padded box the segmentation ran inside
    cv::Rect box;    // box we report, tightened on the mask
    cv::Mat mask;    // mask of the cone, in roi coordinates
    int cls;
    float score;
};

// detectCones -> classifier -> segmentCone -> tightenToMask, for every cone in the image
std::vector<Cone> detectAndSegment(const cv::Mat& bgr);

// end-to-end: detectCones -> classifier -> segmentCone, then draws box, label and mask of each
// cone and saves the image.
void detectPipeline(const std::string& imagePath, const std::string& outPath);

// runs the whole pipeline over the annotated test set and prints macro F1, mAP@0.5:0.95 and mIoU.
void evaluatePipeline(const std::string& imgDir, const std::string& annDir, const std::string& splitFile);
