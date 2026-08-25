#include "viola-jones.hpp"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <zlib.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

// couple of functions directly taken from labeler.cpp / HNM), this needs refactor
std::vector<uchar> decodeBase64(const std::string& encoded) {
    static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uchar> decoded;
    std::uint32_t accumulator = 0;
    int bits = -8;
    for (const unsigned char c : encoded) {
        if (std::isspace(c)) continue;
        if (c == '=') break;
        const size_t index = alphabet.find(c);
        if (index == std::string::npos) throw std::runtime_error("Invalid base64");
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
    if (inflateInit(&stream) != Z_OK) throw std::runtime_error("Cannot init decompressor");
    std::vector<uchar> decompressed;
    constexpr size_t chunkSize = 16 * 1024;
    int result = Z_OK;
    do {
        const size_t prevSize = decompressed.size();
        decompressed.resize(prevSize + chunkSize);
        stream.next_out = reinterpret_cast<Bytef*>(decompressed.data() + prevSize);
        stream.avail_out = static_cast<uInt>(chunkSize);
        result = inflate(&stream, Z_NO_FLUSH);
        decompressed.resize(prevSize + chunkSize - stream.avail_out);
    } while (result == Z_OK);
    inflateEnd(&stream);
    return decompressed;
}

cv::Mat decodeBitmapMask(const json& bitmap) {
    const std::vector<uchar> compressed = decodeBase64(bitmap.at("data").get<std::string>());
    const std::vector<uchar> png = decompressZlib(compressed);
    const cv::Mat decoded = cv::imdecode(png, cv::IMREAD_UNCHANGED);
    cv::Mat mask;
    if (decoded.channels() == 4) cv::extractChannel(decoded, mask, 3);
    else if (decoded.channels() == 1) mask = decoded;
    else cv::cvtColor(decoded, mask, cv::COLOR_BGR2GRAY);
    cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
    return mask;
}

std::vector<cv::Rect> readGroundTruth(const fs::path& jsonPath, int imgWidth, int imgHeight) {
    std::ifstream in(jsonPath);
    if (!in) return {};
    json doc = json::parse(in, nullptr, false);
    if (doc.is_discarded()) return {};

    std::vector<cv::Rect> boxes;
    for (const auto& obj : doc.value("objects", json::array())) {
        std::string geom = obj.value("geometryType", "");
        if (geom == "rectangle") {
            const auto& pts = obj.at("points").at("exterior");
            int x1 = pts[0][0].get<int>(), y1 = pts[0][1].get<int>();
            int x2 = pts[1][0].get<int>(), y2 = pts[1][1].get<int>();
            boxes.emplace_back(std::max(0, std::min(x1, x2)), std::max(0, std::min(y1, y2)),
                               std::min(imgWidth, std::max(x1, x2)) - std::max(0, std::min(x1, x2)),
                               std::min(imgHeight, std::max(y1, y2)) - std::max(0, std::min(y1, y2)));
        } else if (geom == "bitmap") {
            cv::Mat mask = decodeBitmapMask(obj.at("bitmap"));
            int ox = obj.at("bitmap").at("origin")[0].get<int>();
            int oy = obj.at("bitmap").at("origin")[1].get<int>();
            cv::Rect validBounds = cv::Rect(ox, oy, mask.cols, mask.rows) & cv::Rect(0, 0, imgWidth, imgHeight);
            if (validBounds.area() > 0) boxes.push_back(validBounds);
        }
    }
    return boxes;
}

double calculateIoU(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b;
    if (intersection.area() == 0) return 0.0;
    return static_cast<double>(intersection.area()) / (a.area() + b.area() - intersection.area());
}
// -----------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <val_images.txt> <cascade.xml> <config.json>\n";
        return 1;
    }

    std::string listPath = argv[1];
    std::string cascadePath = argv[2];
    std::string configPath = argv[3];

    cv::CascadeClassifier cascade;
    if (!loadCascade(cascade, cascadePath)) return 1;

    cv::FileStorage config(configPath, cv::FileStorage::READ);
    double scaleFactor = config["scaleFactor"].empty() ? 1.15 : (double)config["scaleFactor"];
    int minNeighbors = config["minNeighbors"].empty() ? 8 : (int)config["minNeighbors"];
    std::vector<int> minSizeVec, maxSizeVec;
    config["minSize"] >> minSizeVec; config["maxSize"] >> maxSizeVec;
    cv::Size minSize(minSizeVec[0], minSizeVec[1]);
    cv::Size maxSize(maxSizeVec[0], maxSizeVec[1]);

    std::ifstream listFile(listPath);
    if (!listFile) {
        std::cerr << "Cannot open " << listPath << "\n";
        return 1;
    }

    std::string line;
    int total_tp = 0, total_fp = 0, total_fn = 0, total_images = 0;
    const double IOU_THRESHOLD = 0.5;

    while (std::getline(listFile, line)) {
        if (line.empty() || line[0] == '#') continue;
        fs::path imgPath = line;
        cv::Mat img = cv::imread(imgPath.string());
        if (img.empty()) continue;

        total_images++;
        fs::path gtPath = imgPath;
        gtPath.replace_extension(".json");
        std::vector<cv::Rect> gtBoxes = readGroundTruth(gtPath, img.cols, img.rows);
        std::vector<cv::Rect> preds = detectCones(img, cascade, scaleFactor, minNeighbors, minSize, maxSize);

        int tp = 0;
        std::vector<bool> gt_matched(gtBoxes.size(), false);
        
        // Greedy matching
        for (const auto& pred : preds) {
            double best_iou = 0.0;
            int best_gt_idx = -1;
            for (size_t i = 0; i < gtBoxes.size(); ++i) {
                if (gt_matched[i]) continue;
                double iou = calculateIoU(pred, gtBoxes[i]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_gt_idx = static_cast<int>(i);
                }
            }
            if (best_iou >= IOU_THRESHOLD && best_gt_idx != -1) {
                tp++;
                gt_matched[best_gt_idx] = true;
            }
        }

        int fp = static_cast<int>(preds.size()) - tp;
        int fn = static_cast<int>(gtBoxes.size()) - tp;
        
        total_tp += tp;
        total_fp += fp;
        total_fn += fn;
    }

    double precision = total_tp > 0 ? static_cast<double>(total_tp) / (total_tp + total_fp) : 0.0;
    double recall = total_tp > 0 ? static_cast<double>(total_tp) / (total_tp + total_fn) : 0.0;
    double fppi = total_images > 0 ? static_cast<double>(total_fp) / total_images : 0.0;

    std::cout << "--- Validation Metrics ---\n";
    std::cout << "Images analyzed: " << total_images << "\n";
    std::cout << "True Positives (TP): " << total_tp << "\n";
    std::cout << "False Positives (FP): " << total_fp << "\n";
    std::cout << "False Negatives (FN): " << total_fn << "\n";
    std::cout << "Precision: " << std::fixed << std::setprecision(4) << precision << "\n";
    std::cout << "Recall: " << std::fixed << std::setprecision(4) << recall << "\n";
    std::cout << "FPPI: " << std::fixed << std::setprecision(4) << fppi << "\n";

    return 0;
}