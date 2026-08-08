#ifndef LABELER_HPP
#define LABELER_HPP

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

enum class AnnotationGeometry {
    Rectangle,
    Bitmap,
};

// Represents either a rectangle, or a bitmap mask positioned at origin.
struct Annotation {
    AnnotationGeometry geometry = AnnotationGeometry::Rectangle;
    cv::Point topLeft;
    cv::Point bottomRight;
    cv::Mat mask;
    cv::Point origin;
    std::string label;
};

std::vector<Annotation> readAnnotations(const fs::path& annotationPath);
bool drawAnnotation(cv::Mat& image, const Annotation& annotation);
void labeler();

#endif // LABELER_HPP
