// Runtime Viola-Jones wrapper: loads an OpenCV cascade XML, runs
// detectMultiScale on an image, and offers a separate debug-drawing helper.
#include "viola-jones.hpp"

#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>

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

// Compute Intersection over Union between two rectangles
double iou(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect intersection = a & b;
    if (intersection.area() == 0) {
        return 0.0;
    }

    const double aArea = static_cast<double>(a.area());
    const double bArea = static_cast<double>(b.area());
    const double unionArea = aArea + bArea - intersection.area();

    return intersection.area() / unionArea;
}

// Compute center distance between two rectangles
double centerDistance(const cv::Rect& a, const cv::Rect& b) {
    const cv::Point centerA(a.x + a.width / 2, a.y + a.height / 2);
    const cv::Point centerB(b.x + b.width / 2, b.y + b.height / 2);
    return std::sqrt(std::pow(centerA.x - centerB.x, 2) + std::pow(centerA.y - centerB.y, 2));
}

// Compute adaptive center distance threshold based on bounding box size
// For small cones (~20px), threshold is ~30px
// For large cones (~400px), threshold is ~150px
// Formula: 0.75 × max(width, height) gives a reasonable spacing threshold
double adaptiveCenterDistanceThreshold(const cv::Rect& referenceBox) {
    return 0.75 * std::max(referenceBox.width, referenceBox.height);
}

// Non-Maximum Suppression for dense cone detection
// Uses adaptive threshold based on bounding box size
// NOTE: This is a starting point - thresholds should be tuned based on validation results
std::vector<cv::Rect> nonMaximumSuppression(
    const std::vector<cv::Rect>& detections,
    double iouThreshold,
    double defaultMinCenterDistance) {
    if (detections.empty()) {
        return detections;
    }

    std::vector<cv::Rect> filtered;
    std::vector<bool> suppressed(detections.size(), false);

    // Sort by area (proxy for confidence) - larger area = higher confidence
    std::vector<size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return detections[a].height * detections[a].width >
               detections[b].height * detections[b].width;
    });

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        if (suppressed[idx]) continue;

        const cv::Rect& current = detections[idx];
        
        // Adaptive center distance threshold based on current box size
        // This ensures small cones use smaller thresholds, large cones use larger ones
        const double adaptiveDistance = adaptiveCenterDistanceThreshold(current);
        
        filtered.push_back(current);

        // Suppress all overlapping boxes
        for (size_t j = 0; j < indices.size(); ++j) {
            if (i == j || suppressed[indices[j]]) continue;

            const cv::Rect& other = detections[indices[j]];
            
            // Adaptive NMS: suppress if BOTH conditions are met:
            // 1. High IoU overlap (cones are very close/occluded)
            // 2. Center distance is small (relative to cone size)
            // Formula: 0.75 × max(width, height) gives ~30px for small cones, ~150px for large ones
            if (iou(current, other) > iouThreshold &&
                centerDistance(current, other) < adaptiveDistance) {
                suppressed[indices[j]] = true;
            }
        }
    }

    return filtered;
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

    // Adaptive NMS for cones of varying sizes:
    // - IoU 0.4: only suppress boxes with significant overlap
    // - Center distance: adaptive (0.75 × max dimension)
    //   - Small cone (20px): ~15px threshold
    //   - Large cone (400px): ~300px threshold
    // NOTE: These values are starting points - tune based on validation results
    detections = nonMaximumSuppression(detections, 0.4, 40.0);

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
