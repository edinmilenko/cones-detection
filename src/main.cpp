#include "viola-jones.hpp"

#include <opencv2/core.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <set>

namespace fs = std::filesystem;

namespace {

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return value;
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

bool isSupportedImage(const fs::path& path) {
    const std::string extension = lowercase(path.extension().string());
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        // This executable is the runtime side of the project. It only consumes a
        // finished cascade.xml through OpenCV 4; dataset preparation, splitting,
        // .vec creation, and cascade training are separate offline tools.
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0]
                      << " <input_image_or_list> <cascade.xml> <config.json> [output_image_or_output_dir]\n";
            std::cerr << "\nSingle image mode: " << argv[0] << " <image.jpg> <cascade.xml> <config.json> [output.jpg]\n";
            std::cerr << "\nBatch mode: " << argv[0] << " <list_of_images.txt> <cascade.xml> <config.json> <output_dir>\n";
            std::cerr << "\nDirectory mode: " << argv[0] << " <directory_with_images/> <cascade.xml> <config.json> <output_dir>\n";
            return 1;
        }

        const std::string inputPath = argv[1];
        const std::string cascadePath = argv[2];
        const std::string configPath = argv[3];
        const fs::path outputPath = argc >= 5 ? fs::path(argv[4]) : fs::path("detected_cones.jpg");

        // Detect mode: single image, list file, or directory
        std::vector<std::string> imagesToProcess;
        std::string outputMode; // "single" or "batch"

        if (fs::exists(inputPath)) {
            if (fs::is_regular_file(inputPath)) {
                // Could be a single image or a list file
                const std::string ext = lowercase(fs::path(inputPath).extension().string());
                if (ext == ".txt") {
                    // Read list of images from file
                    std::ifstream listFile(inputPath);
                    if (!listFile.is_open()) {
                        throw std::runtime_error("Cannot open image list file: " + inputPath);
                    }
                    std::string line;
                    const fs::path listDirectory = fs::path(inputPath).parent_path();
                    while (std::getline(listFile, line)) {
                        line = trim(line);
                        if (!line.empty() && line[0] != '#') {
                            const fs::path imagePath(line);
                            imagesToProcess.push_back(
                                (imagePath.is_absolute() ? imagePath : listDirectory / imagePath).lexically_normal().string());
                        }
                    }
                    if (imagesToProcess.empty()) {
                        throw std::runtime_error("No valid image paths found in list file: " + inputPath);
                    }
                    outputMode = "batch";
                } else {
                    // Single image
                    imagesToProcess.push_back(inputPath);
                    outputMode = "single";
                }
            } else if (fs::is_directory(inputPath)) {
                // Directory mode - process all images in directory
                outputMode = "batch";
                for (const auto& entry : fs::directory_iterator(inputPath)) {
                    if (entry.is_regular_file()) {
                        if (isSupportedImage(entry.path())) {
                            imagesToProcess.push_back(entry.path().string());
                        }
                    }
                }
                if (imagesToProcess.empty()) {
                    throw std::runtime_error("No images found in directory: " + inputPath);
                }
                std::sort(imagesToProcess.begin(), imagesToProcess.end());
            }
        } else {
            throw std::runtime_error("Input path does not exist: " + inputPath);
        }

        // Load cascade
        cv::CascadeClassifier cascade;
        if (!loadCascade(cascade, cascadePath)) {
            throw std::runtime_error("Failed to load cascade XML: " + cascadePath);
        }

        // Load configuration
        cv::FileStorage config(configPath, cv::FileStorage::READ);
        if (!config.isOpened()) {
            throw std::runtime_error("Cannot open configuration file: " + configPath);
        }

        const double scaleFactor = config["scaleFactor"].empty() ? 1.1 : static_cast<double>(config["scaleFactor"]);
        const int minNeighbors = config["minNeighbors"].empty() ? 3 : static_cast<int>(config["minNeighbors"]);
        
        std::vector<int> minSizeVec;
        if (!config["minSize"].empty()) {
            config["minSize"] >> minSizeVec;
        }

        std::vector<int> maxSizeVec;
        if (!config["maxSize"].empty()) {
            config["maxSize"] >> maxSizeVec;
        }

        cv::Size minSize(20, 20);
        if (minSizeVec.size() == 2) {
            minSize = cv::Size(minSizeVec[0], minSizeVec[1]);
        }

        cv::Size maxSize(200, 200);
        if (maxSizeVec.size() == 2) {
            maxSize = cv::Size(maxSizeVec[0], maxSizeVec[1]);
        }

        config.release();

        // Process images
        std::cout << "Processing " << imagesToProcess.size() << " image(s)...\n";
        std::cout << "Cascade: " << cascadePath << "\n";
        std::cout << "Config: " << configPath << "\n";

        if (outputMode == "single") {
            // Single image mode - backward compatible
            const std::string imagePath = imagesToProcess[0];
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (image.empty()) {
                throw std::runtime_error("Cannot read input image: " + imagePath);
            }

            const std::vector<cv::Rect> detections = detectCones(
                image,
                cascade,
                scaleFactor,
                minNeighbors,
                minSize,
                maxSize);

            const cv::Mat debugImage = drawDetections(image, detections);
            if (!cv::imwrite(outputPath.string(), debugImage)) {
                throw std::runtime_error("Failed to write output image: " + outputPath.string());
            }

            std::cout << "Detected " << detections.size() << " cone(s).\n";
            std::cout << "Applied Inference Config from: " << configPath << "\n"
                      << " - scaleFactor: " << scaleFactor << "\n"
                      << " - minNeighbors: " << minNeighbors << "\n"
                      << " - minSize: [" << minSize.width << ", " << minSize.height << "]\n"
                      << " - maxSize: [" << maxSize.width << ", " << maxSize.height << "]\n";
            std::cout << "Saved debug image to " << outputPath << "\n";
        } else {
            // Batch mode
            std::cout << "Batch mode enabled\n";

            if (argc < 5) {
                throw std::runtime_error("Batch mode requires an output directory");
            }
            if (fs::exists(outputPath) && !fs::is_directory(outputPath)) {
                throw std::runtime_error("Batch output path is not a directory: " + outputPath.string());
            }
            if (!fs::exists(outputPath)) {
                fs::create_directories(outputPath);
            }

            std::size_t processedCount = 0;
            std::size_t failedCount = 0;
            std::size_t totalDetections = 0;

            std::set<std::string> outputNames;
            for (const std::string& imagePath : imagesToProcess) {
                const std::string basename = fs::path(imagePath).stem().string();
                std::cout << "\nProcessing: " << imagePath << " ... ";

                cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
                if (image.empty()) {
                    std::cerr << "Failed to read image\n";
                    ++failedCount;
                    continue;
                }

                std::vector<cv::Rect> detections;
                try {
                    detections = detectCones(
                        image,
                        cascade,
                        scaleFactor,
                        minNeighbors,
                        minSize,
                        maxSize);
                    totalDetections += detections.size();
                } catch (const std::exception& e) {
                    std::cerr << "Detection failed: " << e.what() << "\n";
                    ++failedCount;
                    continue;
                }

                std::string outputName = basename + "_detected.jpg";
                for (std::size_t suffix = 2; !outputNames.insert(outputName).second; ++suffix) {
                    outputName = basename + "_detected_" + std::to_string(suffix) + ".jpg";
                }
                const fs::path imageOutputPath = outputPath / outputName;

                cv::Mat debugImage = drawDetections(image, detections);
                if (!cv::imwrite(imageOutputPath.string(), debugImage)) {
                    std::cerr << "Failed to write output image: " << imageOutputPath << "\n";
                    ++failedCount;
                    continue;
                }

                std::cout << "OK (" << detections.size() << " cones)\n";
                ++processedCount;
            }

            std::cout << "\n========================================\n";
            std::cout << "Batch Processing Complete\n";
            std::cout << "========================================\n";
            std::cout << "Total images: " << imagesToProcess.size() << "\n";
            std::cout << "Processed: " << processedCount << "\n";
            std::cout << "Failed: " << failedCount << "\n";
            std::cout << "Total detections: " << totalDetections << "\n";
            std::cout << "Output directory: " << outputPath << "\n";
            if (failedCount != 0) {
                return 2;
            }
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << "\n";
        return 1;
    }
}