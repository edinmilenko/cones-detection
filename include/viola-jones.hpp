#ifndef VIOLA_JONES_HPP
#define VIOLA_JONES_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Minimal OpenCV Viola-Jones cone detector.
// The actual cascade is loaded from an external XML file and reused for all detections.
std::vector<cv::Rect> detectCones(
    const cv::Mat& image,
    cv::CascadeClassifier& cascade,
    double scaleFactor = 1.1,
    int minNeighbors = 3,
    cv::Size minSize = cv::Size(),
    cv::Size maxSize = cv::Size());

bool loadCascade(cv::CascadeClassifier& cascade, const std::string& xmlPath);

cv::Mat drawDetections(
    const cv::Mat& image,
    const std::vector<cv::Rect>& detections,
    const cv::Scalar& color = cv::Scalar(0, 255, 0));

#endif // VIOLA_JONES_HPP
