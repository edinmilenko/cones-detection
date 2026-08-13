#include "viola-jones.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    try {
        // This executable is the runtime side of the project. It only consumes a
        // finished cascade.xml through OpenCV 4; dataset preparation, splitting,
        // .vec creation, and cascade training are separate offline tools.
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0]
                      << " <input_image> <cascade.xml> [output_image]" << std::endl;
            return 1;
        }

        const std::string imagePath = argv[1];
        const std::string cascadePath = argv[2];
        const std::string outputPath = argc >= 4 ? argv[3] : "detected_cones.jpg";

        const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Cannot read input image: " + imagePath);
        }

        cv::CascadeClassifier cascade;
        if (!loadCascade(cascade, cascadePath)) {
            throw std::runtime_error("Failed to load cascade XML: " + cascadePath);
        }

        // Detection-time parameters are deliberately kept separate from the
        // training pipeline. They provide the first runtime baseline and can be
        // tuned later using validation data, without retraining the cascade.
        const double scaleFactor = 1.1;
        const int minNeighbors = 3;
        const cv::Size minSize(20, 20);
        const cv::Size maxSize(200, 200);

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

        std::cout << "Detected " << detections.size() << " cone(s)." << std::endl;
        std::cout << "Saved debug image to " << outputPath << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
}
