// doesnt still work for segmentation

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "labeler.hpp"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype>
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

void labeler() {
    const fs::path datasetDir = "../dataset";
    const fs::path labeledDir = "../labeled";

    if (!fs::exists(datasetDir)) {
        throw std::runtime_error("Dataset folder does not exist: " + datasetDir.string());
    }

    if (fs::exists(labeledDir)) {
        std::cout << "The folder labeled already exists, deleting..." << std::endl;
        fs::remove_all(labeledDir);
    }

    if (!fs::create_directory(labeledDir)) {
        throw std::runtime_error("Failed to create labeled directory");
    }

    std::vector<fs::path> imageFiles;
    for (const auto& entry : fs::directory_iterator(datasetDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
            imageFiles.push_back(entry.path());
        }
    }

    std::sort(imageFiles.begin(), imageFiles.end());

    size_t processedImages = 0;
    size_t skippedImages = 0;

    for (const fs::path& imagePath : imageFiles) {
        const fs::path jsonPath = datasetDir / (imagePath.stem().string() + ".json");

        if (!fs::exists(jsonPath)) {
            std::cout << "Missing annotation for image: " << imagePath.filename() << std::endl;
            ++skippedImages;
            continue;
        }

        cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cout << "Failed to read image: " << imagePath << std::endl;
            ++skippedImages;
            continue;
        }

        try {
            const std::vector<Annotation> annotations = readAnnotations(jsonPath);

            for (const Annotation& annotation : annotations) {
                drawAnnotation(image, annotation);
            }

            const fs::path outputPath = labeledDir / imagePath.filename();
            if (!cv::imwrite(outputPath.string(), image)) {
                throw std::runtime_error("Failed to write output image: " + outputPath.string());
            }

            ++processedImages;
            std::cout << "Wrote " << outputPath << std::endl;
        } catch (const std::exception& e) {
            ++skippedImages;
            std::cout << "Skipping " << imagePath.filename() << ": " << e.what() << std::endl;
        }
    }

    std::cout << "Labeling completed. Processed " << processedImages
              << " image(s), skipped " << skippedImages << "." << std::endl;
}
