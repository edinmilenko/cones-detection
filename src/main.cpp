#include "viola-jones.hpp"

#include <opencv2/core.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    try {
        // This executable is the runtime side of the project. It only consumes a
        // finished cascade.xml through OpenCV 4; dataset preparation, splitting,
        // .vec creation, and cascade training are separate offline tools.
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0]
                      << " <input_image> <cascade.xml> <config.json> [output_image]\n";
            return 1;
        }

        const std::string imagePath = argv[1];
        const std::string cascadePath = argv[2];
        const std::string configPath = argv[3];
        const std::string outputPath = argc >= 5 ? argv[4] : "detected_cones.jpg";

        const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Cannot read input image: " + imagePath);
        }

        cv::CascadeClassifier cascade;
        if (!loadCascade(cascade, cascadePath)) {
            throw std::runtime_error("Failed to load cascade XML: " + cascadePath);
        }

        // OpenCV nativamente supporta il parsing JSON
        cv::FileStorage fs(configPath, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Cannot open configuration file: " + configPath);
        }

        // Lettura iperparametri con fallback per evitare abort a runtime
        const double scaleFactor = fs["scaleFactor"].empty() ? 1.1 : static_cast<double>(fs["scaleFactor"]);
        const int minNeighbors = fs["minNeighbors"].empty() ? 3 : static_cast<int>(fs["minNeighbors"]);
        
        std::vector<int> minSizeVec;
        if (!fs["minSize"].empty()) {
            fs["minSize"] >> minSizeVec;
        }

        std::vector<int> maxSizeVec;
        if (!fs["maxSize"].empty()) {
            fs["maxSize"] >> maxSizeVec;
        }

        cv::Size minSize(20, 20); // Base fallback
        if (minSizeVec.size() == 2) {
            minSize = cv::Size(minSizeVec[0], minSizeVec[1]);
        }

        cv::Size maxSize(200, 200); // Base fallback
        if (maxSizeVec.size() == 2) {
            maxSize = cv::Size(maxSizeVec[0], maxSizeVec[1]);
        }

        fs.release();

        const std::vector<cv::Rect> detections = detectCones(
            image,
            cascade,
            scaleFactor,
            minNeighbors,
            minSize,
            maxSize);

        const cv::Mat debugImage = drawDetections(image, detections);
        if (!cv::imwrite(outputPath, debugImage)) {
            throw std::runtime_error("Failed to write output image: " + outputPath);
        }

        std::cout << "Detected " << detections.size() << " cone(s).\n";
        std::cout << "Applied Inference Config from: " << configPath << "\n"
                  << " - scaleFactor: " << scaleFactor << "\n"
                  << " - minNeighbors: " << minNeighbors << "\n"
                  << " - minSize: [" << minSize.width << ", " << minSize.height << "]\n"
                  << " - maxSize: [" << maxSize.width << ", " << maxSize.height << "]\n";
        std::cout << "Saved debug image to " << outputPath << "\n";
        
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << "\n";
        return 1;
    }
}