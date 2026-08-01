#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "labeler.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

// sample
// {
//   "objects": [{
//     "classTitle": "seg_blue_cone",
//     "geometryType": "rectangle",
//     "points": { "exterior": [[left, top], [right, bottom]] }
//   }]
// }
std::vector<Annotation> readAnnotations(const fs::path& annotationPath) {
    std::ifstream input(annotationPath);
    if (!input) {
        throw std::runtime_error("Cannot open annotation file");
    }

    const json document = json::parse(input);
    std::vector<Annotation> annotations;
    const auto& objects = document.at("objects");

    // da capire come va cambiato se invece è bitmap
    for (const json& object : objects) {
        if (object.value("geometryType", "") != "rectangle") {
            continue;
        }

        const json& corners = object.at("points").at("exterior");
        if (!corners.is_array() || corners.size() != 2 ||
            !corners[0].is_array() || !corners[1].is_array() ||
            corners[0].size() != 2 || corners[1].size() != 2) {
            throw std::runtime_error("Rectangle must have exactly two [x, y] corners");
        }

        annotations.push_back({
            {corners[0][0].get<int>(), corners[0][1].get<int>()},
            {corners[1][0].get<int>(), corners[1][1].get<int>()},
            object.value("classTitle", "cone"),
        });
    }

    if (annotations.empty() && !objects.empty()) {
        throw std::runtime_error("No rectangle annotations found");
    }

    return annotations;
}

bool drawAnnotation(cv::Mat& image, const Annotation& annotation) {
    const cv::Point p1 = annotation.topLeft;
    const cv::Point p2 = annotation.bottomRight;

    // rectangle from the two points, handling unordered points and inclusive bounds
    const cv::Rect box(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
                       std::abs(p1.x - p2.x) + 1, std::abs(p1.y - p2.y) + 1);

    // intersect with image bounds to clamp it
    const cv::Rect imageBounds(0, 0, image.cols, image.rows);
    const cv::Rect validBox = box & imageBounds;

    if (validBox.area() <= 0) {
        return false;
    }

    cv::rectangle(image, validBox, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::putText(image, annotation.label, {validBox.x, std::max(18, validBox.y - 6)},
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    return true;
}
