#include "labeler.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char* argv[]) {
    // path hardcodato per test, da cambiare
    const fs::path imagePath = "../dataset/example.jpg";
    const fs::path jsonPath = "../dataset/example.json";
    const fs::path outputPath = "../labeled_example.jpg";

    try {
        // read image
        cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to read image: " + imagePath.string());
        }
        std::cout << "Successfully read image: " << imagePath << std::endl;

        // read json annotations
        const std::vector<Annotation> annotations = readAnnotations(jsonPath);
        std::cout << "Found " << annotations.size() << " annotations in " << jsonPath << std::endl;

        // draw bbox based on annotations
        for (const Annotation& annotation : annotations) {
            drawAnnotation(image, annotation);
        }
        std::cout << "Drew " << annotations.size() << " bounding boxes." << std::endl;

        // saving
        if (!cv::imwrite(outputPath.string(), image)) {
            throw std::runtime_error("Failed to write output image: " + outputPath.string());
        }
        std::cout << "Successfully saved labeled image to: " << outputPath << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}