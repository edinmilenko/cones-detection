// Annotation utility: provides the project-specific image labeling workflow
// and writes the JSON annotations later consumed by the offline dataset tools.
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <zlib.h>
#include "labeler.hpp"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::vector<uchar> decodeBase64(const std::string& encoded) {
    static const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uchar> decoded;
    std::uint32_t accumulator = 0;
    int bits = -8;

    for (const unsigned char character : encoded) {
        if (std::isspace(character)) {
            continue;
        }
        if (character == '=') {
            break;
        }

        const size_t index = alphabet.find(character);
        if (index == std::string::npos) {
            throw std::runtime_error("Bitmap data is not valid base64");
        }

        accumulator = (accumulator << 6) + static_cast<std::uint32_t>(index);
        bits += 6;
        if (bits >= 0) {
            decoded.push_back(static_cast<uchar>((accumulator >> bits) & 0xff));
            bits -= 8;
        }
    }
    return decoded;
}

std::vector<uchar> decompressZlib(const std::vector<uchar>& compressed) {
    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(compressed.data()));
    stream.avail_in = static_cast<uInt>(compressed.size());

    if (inflateInit(&stream) != Z_OK) {
        throw std::runtime_error("Cannot initialize bitmap decompressor");
    }

    std::vector<uchar> decompressed;
    constexpr size_t chunkSize = 16 * 1024;
    int result = Z_OK;
    do {
        const size_t previousSize = decompressed.size();
        decompressed.resize(previousSize + chunkSize);
        stream.next_out = reinterpret_cast<Bytef*>(decompressed.data() + previousSize);
        stream.avail_out = static_cast<uInt>(chunkSize);
        result = inflate(&stream, Z_NO_FLUSH);
        decompressed.resize(previousSize + chunkSize - stream.avail_out);
    } while (result == Z_OK);

    inflateEnd(&stream);
    if (result != Z_STREAM_END) {
        throw std::runtime_error("Cannot decompress bitmap data");
    }
    return decompressed;
}

cv::Mat decodeBitmapMask(const json& bitmap) {
    const std::vector<uchar> compressed = decodeBase64(bitmap.at("data").get<std::string>());
    if (compressed.empty()) {
        throw std::runtime_error("Bitmap data is empty");
    }

    const std::vector<uchar> png = decompressZlib(compressed);
    const cv::Mat decoded = cv::imdecode(png, cv::IMREAD_UNCHANGED);
    if (decoded.empty()) {
        throw std::runtime_error("Bitmap data does not contain a valid PNG mask");
    }

    cv::Mat mask;
    if (decoded.channels() == 4) {
        cv::extractChannel(decoded, mask, 3);
    } else if (decoded.channels() == 1) {
        mask = decoded;
    } else {
        cv::cvtColor(decoded, mask, cv::COLOR_BGR2GRAY);
    }
    cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
    return mask;
}

} // namespace

std::vector<Annotation> readAnnotations(const fs::path& annotationPath) {
    std::ifstream input(annotationPath);
    if (!input) {
        throw std::runtime_error("Cannot open annotation file");
    }

    const json document = json::parse(input);
    std::vector<Annotation> annotations;
    const auto& objects = document.at("objects");

    for (const json& object : objects) {
        const std::string geometryType = object.value("geometryType", "");
        const std::string label = object.value("classTitle", "cone");

        if (geometryType == "rectangle") {
            const json& corners = object.at("points").at("exterior");
            if (!corners.is_array() || corners.size() != 2 ||
                !corners[0].is_array() || !corners[1].is_array() ||
                corners[0].size() != 2 || corners[1].size() != 2) {
                throw std::runtime_error("Rectangle must have exactly two [x, y] corners");
            }

            annotations.push_back({
                AnnotationGeometry::Rectangle,
                {corners[0][0].get<int>(), corners[0][1].get<int>()},
                {corners[1][0].get<int>(), corners[1][1].get<int>()},
                {},
                {},
                label,
            });
            continue;
        }

        if (geometryType == "bitmap") {
            const json& bitmap = object.at("bitmap");
            const json& origin = bitmap.at("origin");
            if (!origin.is_array() || origin.size() != 2) {
                throw std::runtime_error("Bitmap origin must be [x, y]");
            }

            annotations.push_back({
                AnnotationGeometry::Bitmap,
                {},
                {},
                decodeBitmapMask(bitmap),
                {origin[0].get<int>(), origin[1].get<int>()},
                label,
            });
        }
    }

    if (annotations.empty() && !objects.empty()) {
        throw std::runtime_error("No supported annotations found");
    }

    return annotations;
}

bool drawAnnotation(cv::Mat& image, const Annotation& annotation) {
    if (annotation.geometry == AnnotationGeometry::Bitmap) {
        const cv::Rect maskBounds(annotation.origin.x, annotation.origin.y,
                                  annotation.mask.cols, annotation.mask.rows);
        const cv::Rect imageBounds(0, 0, image.cols, image.rows);
        const cv::Rect validBounds = maskBounds & imageBounds;
        if (validBounds.area() <= 0) {
            return false;
        }

        const cv::Rect maskRegion(validBounds.x - maskBounds.x, validBounds.y - maskBounds.y,
                                  validBounds.width, validBounds.height);
        const cv::Mat mask = annotation.mask(maskRegion);
        cv::Mat destination = image(validBounds);
        const cv::Mat green(destination.size(), destination.type(), cv::Scalar(0, 255, 0));
        cv::Mat overlay;
        cv::addWeighted(destination, 0.55, green, 0.45, 0.0, overlay);
        overlay.copyTo(destination, mask);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE,
                         validBounds.tl());
        if (contours.empty()) {
            return false;
        }
        cv::drawContours(image, contours, -1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        const cv::Rect box = cv::boundingRect(mask) + validBounds.tl();
        cv::putText(image, annotation.label, {box.x, std::max(18, box.y - 6)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        return true;
    }

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
