// Offline Haar-training data preparer: converts image/JSON annotations into
// OpenCV annotation manifests and reproducible 24x24 background patches.
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr int kNegativeWindowSize = 24;
// This is intentionally modest for the first experiment. Increase it later if
// training needs more background variety; the fixed RNG seed keeps a run reproducible.
constexpr int kNegativeSamplesPerImage = 8;
// Negatives must avoid not only the GT box but also a small safety border around it.
constexpr int kNegativeMarginPx = 6;

std::vector<fs::path> listImageFiles(const fs::path& datasetDir) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(datasetDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            files.push_back(entry.path());
        }
    }
    // A stable order is part of reproducibility because it controls both the
    // generated patch numbering and the sequence consumed by the RNG.
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<cv::Rect> readGroundTruthBoxes(const fs::path& jsonPath, int imageWidth, int imageHeight) {
    std::ifstream input(jsonPath);
    if (!input) {
        throw std::runtime_error("Cannot open annotation file: " + jsonPath.string());
    }

    const json document = json::parse(input);
    std::vector<cv::Rect> boxes;

    for (const auto& object : document.value("objects", json::array())) {
        if (object.value("geometryType", "") != "rectangle") {
            continue;
        }

        const auto& corners = object.at("points").at("exterior");
        if (!corners.is_array() || corners.size() != 2 || !corners[0].is_array() || !corners[1].is_array()) {
            continue;
        }

        const int x1 = static_cast<int>(corners[0][0].get<int>());
        const int y1 = static_cast<int>(corners[0][1].get<int>());
        const int x2 = static_cast<int>(corners[1][0].get<int>());
        const int y2 = static_cast<int>(corners[1][1].get<int>());

        const int left = std::min(x1, x2);
        const int top = std::min(y1, y2);
        const int right = std::max(x1, x2);
        const int bottom = std::max(y1, y2);

        // JSON corners are inclusive, whereas cv::Rect stores width and height.
        const int width = std::max(0, right - left + 1);
        const int height = std::max(0, bottom - top + 1);

        if (width <= 0 || height <= 0) {
            continue;
        }

        const cv::Rect box(left, top, width, height);
        if (box.x < 0 || box.y < 0 || box.x + box.width > imageWidth || box.y + box.height > imageHeight) {
            continue;
        }

        boxes.push_back(box);
    }

    return boxes;
}

bool isValidNegativeCandidate(const cv::Rect& candidate, const std::vector<cv::Rect>& groundTruth, int marginPx) {
    for (const auto& gt : groundTruth) {
        // Expanding ground truth, rather than the candidate, makes the intended
        // exclusion region explicit: no background crop may touch this region.
        cv::Rect expandedGt = gt;
        expandedGt.x -= marginPx;
        expandedGt.y -= marginPx;
        expandedGt.width += 2 * marginPx;
        expandedGt.height += 2 * marginPx;

        if ((candidate & expandedGt).area() > 0) {
            return false;
        }
    }
    return true;
}

std::string relativePathString(const fs::path& path) {
    const fs::path relative = fs::relative(path, fs::current_path());
    return relative.empty() ? path.string() : relative.string();
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <dataset_dir> <output_dir>" << std::endl;
            return 1;
        }

        const fs::path datasetDir = argv[1];
        const fs::path outputDir = argv[2];

        if (!fs::exists(datasetDir) || !fs::is_directory(datasetDir)) {
            throw std::runtime_error("Dataset directory does not exist: " + datasetDir.string());
        }

        const fs::path trainingDir = outputDir;
        const fs::path negativesDir = trainingDir / "negatives";
        fs::create_directories(negativesDir);

        const fs::path annotationsPath = trainingDir / "annotations.txt";
        const fs::path negativesListPath = trainingDir / "negatives.txt";
        // Provenance is needed later to keep every negative patch in the same
        // train/validation/test split as the image from which it was cropped.
        const fs::path negativeSourcesPath = trainingDir / "negative_sources.txt";
        // This includes images without GT boxes: they still contribute negatives
        // and must therefore participate in an image-level split.
        const fs::path datasetImagesPath = trainingDir / "dataset_images.txt";

        std::ofstream annotationsFile(annotationsPath, std::ios::trunc);
        std::ofstream negativesFile(negativesListPath, std::ios::trunc);
        std::ofstream negativeSourcesFile(negativeSourcesPath, std::ios::trunc);
        std::ofstream datasetImagesFile(datasetImagesPath, std::ios::trunc);

        if (!annotationsFile || !negativesFile || !negativeSourcesFile || !datasetImagesFile) {
            throw std::runtime_error("Cannot create training manifest files in: " + trainingDir.string());
        }

        const std::vector<fs::path> imageFiles = listImageFiles(datasetDir);
        for (const fs::path& imagePath : imageFiles) {
            datasetImagesFile << relativePathString(imagePath) << '\n';
        }
        std::size_t processedImages = 0;
        std::size_t missingJson = 0;
        std::size_t invalidBoxes = 0;
        std::size_t positiveObjects = 0;
        std::size_t negativeSamples = 0;

        // Do not change this seed casually: it makes generated patch locations
        // and manifests reproducible across machines with the same dataset.
        std::mt19937 rng(12345);

        for (const auto& imagePath : imageFiles) {
            const fs::path jsonPath = imagePath;
            fs::path jsonFile = jsonPath;
            jsonFile.replace_extension(".json");

            if (!fs::exists(jsonFile)) {
                ++missingJson;
                continue;
            }

            cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                continue;
            }

            ++processedImages;

            std::vector<cv::Rect> groundTruth;
            try {
                groundTruth = readGroundTruthBoxes(jsonFile, image.cols, image.rows);
            } catch (const std::exception&) {
                continue;
            }

            std::vector<cv::Rect> validPositiveBoxes;
            for (const cv::Rect& box : groundTruth) {
                if (box.width <= 0 || box.height <= 0) {
                    ++invalidBoxes;
                    continue;
                }
                if (box.x < 0 || box.y < 0 || box.x + box.width > image.cols || box.y + box.height > image.rows) {
                    ++invalidBoxes;
                    continue;
                }
                validPositiveBoxes.push_back(box);
            }

            if (!validPositiveBoxes.empty()) {
                // OpenCV's annotation format is:
                // image_path object_count x y width height [x y width height ...]
                std::vector<std::string> tokens;
                tokens.push_back(relativePathString(imagePath));
                tokens.push_back(std::to_string(validPositiveBoxes.size()));

                for (const cv::Rect& box : validPositiveBoxes) {
                    tokens.push_back(std::to_string(box.x));
                    tokens.push_back(std::to_string(box.y));
                    tokens.push_back(std::to_string(box.width));
                    tokens.push_back(std::to_string(box.height));
                }

                annotationsFile << [&tokens]() {
                    std::ostringstream oss;
                    for (std::size_t i = 0; i < tokens.size(); ++i) {
                        if (i > 0) {
                            oss << ' ';
                        }
                        oss << tokens[i];
                    }
                    return oss.str();
                }() << '\n';

                positiveObjects += validPositiveBoxes.size();
            }

            std::uniform_int_distribution<int> xDist(0, std::max(0, image.cols - kNegativeWindowSize));
            std::uniform_int_distribution<int> yDist(0, std::max(0, image.rows - kNegativeWindowSize));
            std::size_t attempts = 0;
            std::size_t generatedThisImage = 0;

            while (generatedThisImage < kNegativeSamplesPerImage && attempts < 200) {
                ++attempts;

                const int x = xDist(rng);
                const int y = yDist(rng);
                const cv::Rect candidate(x, y, kNegativeWindowSize, kNegativeWindowSize);

                if (!isValidNegativeCandidate(candidate, groundTruth, kNegativeMarginPx)) {
                    continue;
                }

                std::ostringstream fileName;
                fileName << std::setw(6) << std::setfill('0') << (negativeSamples + 1) << ".png";

                const fs::path outputPatch = negativesDir / fileName.str();
                if (!cv::imwrite(outputPatch.string(), image(candidate))) {
                    throw std::runtime_error("Cannot write negative patch: " + outputPatch.string());
                }

                const std::string patchPath = relativePathString(outputPatch);
                // Keep negatives.txt compatible with OpenCV's trainer (one patch
                // path per line), and store provenance separately for splitting.
                negativesFile << patchPath << '\n';
                negativeSourcesFile << patchPath << ' ' << relativePathString(imagePath) << '\n';

                ++generatedThisImage;
                ++negativeSamples;
            }
        }

        std::cout << "Images processed: " << processedImages << "\n";
        std::cout << "Positive objects written: " << positiveObjects << "\n";
        std::cout << "Negative samples generated: " << negativeSamples << "\n";
        std::cout << "Images with missing JSON: " << missingJson << "\n";
        std::cout << "Invalid GT boxes skipped: " << invalidBoxes << "\n";
        std::cout << "Annotations file: " << annotationsPath << "\n";
        std::cout << "Negatives list: " << negativesListPath << "\n";
        std::cout << "Negative sources file: " << negativeSourcesPath << "\n";
        std::cout << "Dataset images file: " << datasetImagesPath << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
}
