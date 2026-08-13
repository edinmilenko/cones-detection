// Runtime Viola-Jones wrapper: loads an OpenCV cascade XML, runs
// detectMultiScale on an image, and offers a separate debug-drawing helper.
#include "viola-jones.hpp"

#include <stdexcept>

namespace {

cv::Mat toGray(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Optional histogram equalization can help with uneven lighting.
    // It is intentionally easy to remove if the cascade behaves better without it.
    // cv::equalizeHist(gray, gray);

    return gray;
}

} // namespace

bool loadCascade(cv::CascadeClassifier& cascade, const std::string& xmlPath) {
    return cascade.load(xmlPath);
}

std::vector<cv::Rect> detectCones(
    const cv::Mat& image,
    cv::CascadeClassifier& cascade,
    double scaleFactor,
    int minNeighbors,
    cv::Size minSize,
    cv::Size maxSize) {
    if (image.empty()) {
        throw std::invalid_argument("Cannot detect cones in an empty image");
    }
    if (cascade.empty()) {
        throw std::runtime_error("CascadeClassifier was not loaded from an XML file");
    }

    const cv::Mat gray = toGray(image);

    std::vector<cv::Rect> detections;
    cascade.detectMultiScale(
        gray,
        detections,
        scaleFactor,
        minNeighbors,
        cv::CASCADE_SCALE_IMAGE,
        minSize,
        maxSize);

    return detections;
}

cv::Mat drawDetections(
    const cv::Mat& image,
    const std::vector<cv::Rect>& detections,
    const cv::Scalar& color) {
    cv::Mat debug = image.clone();

    for (const cv::Rect& rect : detections) {
        cv::rectangle(debug, rect, color, 2, cv::LINE_AA);
        cv::putText(debug,
                    "cone",
                    cv::Point(rect.x, std::max(20, rect.y - 8)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv::LINE_AA);
    }

    return debug;
}
