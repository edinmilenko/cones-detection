#include "viola-jones.hpp"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

// Includes libreries used in labeler.cpp to decode masks
#include <zlib.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

/*
    THE FOLLOWING FUNCTIONS decodeBase64, decompressZlib e decodeBitmapMask ARE EXACTLY THOSE FROM src/labeler.cpp
    reason is to support "geometryType": "bitmap" and avoiding having cones as negatives
    THIS WILL BE REFACTORED TO AVOID DUPLICATE CODE
*/
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
// HERE ENDS THE NEED TO REFACTOR, no more comes from src/labeler.cpp

/*
    Measures accuracy of prediction wrt groud thuth. If we'd use IoM here, big false positives that happen to have
    inside them a true small cone would register a IoM = 1.0 and that would be a true positive ignoring it.
    Using IoU the big area of union would be close to 0+ labeling it correctly as a false positive (hard negative)
    that classifier must learn to suppress
*/
double calculateIoU(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b;
    if (intersection.area() == 0) return 0.0;
    return static_cast<double>(intersection.area()) / (a.area() + b.area() - intersection.area());
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
            int left = std::max(0, std::min(x1, x2));
            int top = std::max(0, std::min(y1, y2));
            int right = std::min(imgWidth, std::max(x1, x2));
            int bottom = std::min(imgHeight, std::max(y1, y2));
            boxes.emplace_back(left, top, right - left, bottom - top);
        } else if (geom == "bitmap") {
            // estracts bouding boxes from mask (requires decodeBitmapMask brougth from labeler.cpp)
            cv::Mat mask = decodeBitmapMask(obj.at("bitmap"));
            int ox = obj.at("bitmap").at("origin")[0].get<int>();
            int oy = obj.at("bitmap").at("origin")[1].get<int>();
            cv::Rect validBounds = cv::Rect(ox, oy, mask.cols, mask.rows) & cv::Rect(0, 0, imgWidth, imgHeight);
            if (validBounds.area() > 0) boxes.push_back(validBounds);
        }
    }
    return boxes;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <dataset_dir> <cascade.xml> <config.json> <output_dir>\n";
        return 1;
    }
    
    fs::path datasetDir = argv[1];
    std::string cascadePath = argv[2];
    std::string configPath = argv[3];
    fs::path outputDir = argv[4];

    cv::CascadeClassifier cascade;
    if (!loadCascade(cascade, cascadePath)) return 1;

    cv::FileStorage config(configPath, cv::FileStorage::READ);
    double scaleFactor = config["scaleFactor"].empty() ? 1.15 : (double)config["scaleFactor"];
    int minNeighbors = config["minNeighbors"].empty() ? 8 : (int)config["minNeighbors"];
    
    std::vector<int> minSizeVec, maxSizeVec;
    config["minSize"] >> minSizeVec;
    config["maxSize"] >> maxSizeVec;
    cv::Size minSize(minSizeVec[0], minSizeVec[1]);
    cv::Size maxSize(maxSizeVec[0], maxSizeVec[1]);

    fs::create_directories(outputDir / "hnm_patches");
    fs::create_directories(outputDir / "json_predictions");
    std::ofstream manifest(outputDir / "train_negatives_hnm.txt", std::ios::trunc);

    int hnCount = 0;

    for (const auto& entry : fs::directory_iterator(datasetDir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".png") continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        fs::path gtPath = entry.path();
        gtPath.replace_extension(".json");
        std::vector<cv::Rect> gtBoxes = readGroundTruth(gtPath, img.cols, img.rows);

        auto preds = detectCones(img, cascade, scaleFactor, minNeighbors, minSize, maxSize);

        json predJson;
        predJson["objects"] = json::array();

        for (const auto& pred : preds) {
            // just saves preditions into JSON file cuz modular pipeline uses it
            json obj;
            obj["classTitle"] = "cone";
            obj["geometryType"] = "rectangle";
            obj["points"]["exterior"] = {{pred.x, pred.y}, {pred.x + pred.width, pred.y + pred.height}};
            predJson["objects"].push_back(obj);

            double maxIoU = 0.0;
            for (const auto& gt : gtBoxes) {
                maxIoU = std::max(maxIoU, calculateIoU(pred, gt));
            }

            // hard negatives are pred boxes that dont intersecate ground truth
            if (maxIoU < 0.1) {
                cv::Mat patch;
                cv::resize(img(pred), patch, cv::Size(36, 36)); // Dimensione fissata dal prepare_vj_dataset
                
                std::ostringstream patchName;
                patchName << "hn_" << std::setw(6) << std::setfill('0') << hnCount++ << ".png";
                fs::path patchPath = outputDir / "hnm_patches" / patchName.str();
                
                cv::imwrite(patchPath.string(), patch);
                manifest << fs::relative(patchPath, fs::current_path()).string() << "\n";
            }
        }

        std::ofstream jsonOut(outputDir / "json_predictions" / gtPath.filename());
        jsonOut << predJson.dump(4);
    }

    std::cout << "HNM done. Generated " << hnCount << " hard negative patches.\n";
    return 0;
}