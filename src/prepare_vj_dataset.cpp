// Offline Haar-training data preparer: converts image/JSON annotations into
// OpenCV annotation manifests and reproducible 36x36 background patches.
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

// Dimensione patch aumentata a 36x36 per ospitare finestre di training 24x36 o 36x24.
constexpr int kNegativePatchSize = 36;
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
        const int width = std::max(0, std::max(x1, x2) - left + 1);
        const int height = std::max(0, std::max(y1, y2) - top + 1);

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
        fs::create_directories(trainingDir / "negatives");

        // Utilizziamo due file separati per il manifest
        std::ofstream standingFile(trainingDir / "annotations_standing.txt", std::ios::trunc);
        std::ofstream fallenFile(trainingDir / "annotations_fallen.txt", std::ios::trunc);
        std::ofstream negativesFile(trainingDir / "negatives.txt", std::ios::trunc);
        std::ofstream negativeSourcesFile(trainingDir / "negative_sources.txt", std::ios::trunc);
        std::ofstream datasetImagesFile(trainingDir / "dataset_images.txt", std::ios::trunc);

        if (!standingFile || !fallenFile || !negativesFile || !negativeSourcesFile || !datasetImagesFile) {
            throw std::runtime_error("Cannot create training manifest files");
        }

        const std::vector<fs::path> imageFiles = listImageFiles(datasetDir);
        for (const fs::path& imagePath : imageFiles) datasetImagesFile << relativePathString(imagePath) << '\n';

        std::size_t processedImages = 0, missingJson = 0, invalidBoxes = 0;
        std::size_t standingCount = 0, fallenCount = 0, negativeSamples = 0;
        std::mt19937 rng(12345);

        for (const auto& imagePath : imageFiles) {
            fs::path jsonFile = imagePath;
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

            std::vector<cv::Rect> standingBoxes, fallenBoxes;
            for (const cv::Rect& box : groundTruth) {
                if (box.width <= 0 || box.height <= 0 || box.x < 0 || box.y < 0 || box.x + box.width > image.cols || box.y + box.height > image.rows) {
                    ++invalidBoxes;
                    continue;
                }
                if (box.height >= box.width) {
                    standingBoxes.push_back(box);
                } else {
                    fallenBoxes.push_back(box);
                }
            }

            auto writeManifest = [](std::ofstream& file, const std::string& path, const std::vector<cv::Rect>& boxes) {
                if (boxes.empty()) return;
                file << path << " " << boxes.size();
                for (const auto& box : boxes) file << " " << box.x << " " << box.y << " " << box.width << " " << box.height;
                file << '\n';
            };

            writeManifest(standingFile, relativePathString(imagePath), standingBoxes);
            writeManifest(fallenFile, relativePathString(imagePath), fallenBoxes);
            
            standingCount += standingBoxes.size();
            fallenCount += fallenBoxes.size();

            std::uniform_int_distribution<int> xDist(0, std::max(0, image.cols - kNegativePatchSize));
            std::uniform_int_distribution<int> yDist(0, std::max(0, image.rows - kNegativePatchSize));
            std::size_t attempts = 0, generatedThisImage = 0;

            while (generatedThisImage < kNegativeSamplesPerImage && attempts < 200) {
                ++attempts;
                const cv::Rect candidate(xDist(rng), yDist(rng), kNegativePatchSize, kNegativePatchSize);
                if (!isValidNegativeCandidate(candidate, groundTruth, kNegativeMarginPx)) continue;

                std::ostringstream fileName;
                fileName << std::setw(6) << std::setfill('0') << (negativeSamples + 1) << ".png";
                const fs::path outputPatch = trainingDir / "negatives" / fileName.str();
                
                if (!cv::imwrite(outputPatch.string(), image(candidate))) throw std::runtime_error("Cannot write negative patch");
                
                const std::string patchPath = relativePathString(outputPatch);
                negativesFile << patchPath << '\n';
                negativeSourcesFile << patchPath << ' ' << relativePathString(imagePath) << '\n';
                ++generatedThisImage; ++negativeSamples;
            }
        }

        std::cout << "Images processed: " << processedImages << "\n"
                  << "Standing objects: " << standingCount << "\n"
                  << "Fallen objects: " << fallenCount << "\n"
                  << "Negative samples (36x36): " << negativeSamples << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}