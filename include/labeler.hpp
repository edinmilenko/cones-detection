#ifndef LABELER_HPP
#define LABELER_HPP

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// represents a single bounding box annotation
struct Annotation {
    cv::Point topLeft;
    cv::Point bottomRight;
    std::string label;
};

std::vector<Annotation> readAnnotations(const fs::path& annotationPath);
bool drawAnnotation(cv::Mat& image, const Annotation& annotation);

#endif // LABELER_HPP