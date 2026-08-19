#include "viola-jones.hpp"

#include <opencv2/core.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

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
        const std::string outputPath = argc >= 5 ? argv[4] : "detected_cones";

        // Detect mode: single image, list file, or directory
        std::vector<std::string> imagesToProcess;
        std::string outputMode; // "single" or "batch"

        if (fs::exists(inputPath)) {
            if (fs::is_regular_file(inputPath)) {
                // Could be a single image or a list file
                const std::string ext = fs::path(inputPath).extension().string();
                if (ext == ".txt") {
                    // Read list of images from file
                    std::ifstream listFile(inputPath);
                    if (!listFile.is_open()) {
                        throw std::runtime_error("Cannot open image list file: " + inputPath);
                    }
                    std::string line;
                    while (std::getline(listFile, line)) {
                        if (!line.empty() && line[0] != '#') {
                            imagesToProcess.push_back(line);
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
                        const std::string ext = entry.path().extension().string();
                        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                            imagesToProcess.push_back(entry.path().string());
                        }
                    }
                }
                if (imagesToProcess.empty()) {
                    throw std::runtime_error("No images found in directory: " + inputPath);
                }
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
        cv::FileStorage fs(configPath, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Cannot open configuration file: " + configPath);
        }

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

        cv::Size minSize(20, 20);
        if (minSizeVec.size() == 2) {
            minSize = cv::Size(minSizeVec[0], minSizeVec[1]);
        }

        cv::Size maxSize(200, 200);
        if (maxSizeVec.size() == 2) {
            maxSize = cv::Size(maxSizeVec[0], maxSizeVec[1]);
        }

        fs.release();

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
        } else {
            // Batch mode
            std::cout << "Batch mode enabled\n";

            // Determine output mode: single output or directory
            bool isDirectoryOutput = fs::is_directory(outputPath);
            if (!isDirectoryOutput && !fs::exists(outputPath)) {
                fs::create_directories(outputPath);
            }

            std::size_t processedCount = 0;
            std::size_t failedCount = 0;
            std::size_t totalDetections = 0;

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

                // Generate output path
                std::string outputPathStr;
                if (isDirectoryOutput) {
                    outputPathStr = outputPath + "/" + basename + "_detected.jpg";
                } else {
                    outputPathStr = outputPath;
                }

                cv::Mat debugImage = drawDetections(image, detections);
                if (!cv::imwrite(outputPathStr, debugImage)) {
                    std::cerr << "Failed to write output image: " << outputPathStr << "\n";
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
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << "\n";
        return 1;
    }
}