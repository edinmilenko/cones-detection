#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// final pipeline: color proposals -> stage1 linear filter (cheap) -> stage2 RBF
// (classification) -> NMS -> pad the chosen box.
std::vector<cv::Rect> detectCones(const cv::Mat& bgr);

// loads the image, runs detectCones, draws the boxes and saves the result.
void detectPipeline(const std::string& imagePath, const std::string& outPath);
