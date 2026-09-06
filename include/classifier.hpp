#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

enum class ConeClass {
    YELLOW = 1,
    BLUE = 2,
    SMALL_ORANGE = 3,
    BIG_ORANGE = 4
};

std::vector<int> classifier(const cv::Mat& image, const std::vector<cv::Rect>& bboxes);

ConeClass classifyOrangeCone(const cv::Mat& hsv_roi);

// Label and BGR color used to draw a class on the detection output
std::string className(int predictedClass);
cv::Scalar classColor(int predictedClass);
