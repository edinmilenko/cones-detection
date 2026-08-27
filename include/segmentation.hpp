#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include <opencv2/opencv.hpp>
#include "classifier.hpp" 

cv::Mat segmentCone(const cv::Mat& roi, int predictedClass);

double calculateIoU(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask);

#endif