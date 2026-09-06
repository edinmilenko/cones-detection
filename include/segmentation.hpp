#pragma once

#include <opencv2/opencv.hpp>
#include "classifier.hpp"

cv::Mat segmentCone(const cv::Mat& roi, int predictedClass);
