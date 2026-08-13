// Offline dataset inspector: reads image/JSON pairs and reports bounding-box
// statistics used to choose and validate the Viola-Jones training window.
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct BBox {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    int width = 0;
    int height = 0;
    double aspectRatio = 0.0;
    double area = 0.0;
};

struct Stats {
    std::size_t imageCount = 0;
    std::size_t bboxCount = 0;

    double minWidth = 0.0;
    double maxWidth = 0.0;
    double meanWidth = 0.0;
    double medianWidth = 0.0;

    double minHeight = 0.0;
    double maxHeight = 0.0;
    double meanHeight = 0.0;
    double medianHeight = 0.0;

    double minArea = 0.0;
    double maxArea = 0.0;
    double meanArea = 0.0;
    double medianArea = 0.0;

    double minAspect = 0.0;
    double maxAspect = 0.0;
    double meanAspect = 0.0;
    double medianAspect = 0.0;

    double p10Width = 0.0;
    double p25Width = 0.0;
    double p50Width = 0.0;
    double p75Width = 0.0;
    double p90Width = 0.0;
    double p95Width = 0.0;

    double p10Height = 0.0;
    double p25Height = 0.0;
    double p50Height = 0.0;
    double p75Height = 0.0;
    double p90Height = 0.0;
    double p95Height = 0.0;

    std::size_t invalidCount = 0;
    std::size_t widthLt10 = 0;
    std::size_t widthLt20 = 0;
    std::size_t widthLt30 = 0;
    std::size_t widthLt50 = 0;
    std::size_t heightLt10 = 0;
    std::size_t heightLt20 = 0;
    std::size_t heightLt30 = 0;
    std::size_t heightLt50 = 0;
};

std::vector<fs::path> listDatasetImageFiles(const fs::path& datasetDir) {
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
    std::sort(files.begin(), files.end());
    return files;
}

double percentile(const std::vector<double>& values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    if (values.size() == 1) {
        return values.front();
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    if (p <= 0.0) {
        return sorted.front();
    }
    if (p >= 100.0) {
        return sorted.back();
    }

    const double rank = (sorted.size() - 1) * (p / 100.0);
    const std::size_t lower = static_cast<std::size_t>(std::floor(rank));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(rank));

    if (lower == upper) {
        return sorted[lower];
    }

    const double fraction = rank - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

double medianValue(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t mid = sorted.size() / 2;

    if (sorted.size() % 2 == 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2.0;
    }
    return sorted[mid];
}

std::vector<BBox> readBoxesForImage(const fs::path& jsonPath, const cv::Size& imageSize) {
    std::ifstream input(jsonPath);
    if (!input) {
        throw std::runtime_error("Cannot open JSON annotation: " + jsonPath.string());
    }

    const json document = json::parse(input);
    std::vector<BBox> boxes;

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

        BBox box{};
        box.x1 = left;
        box.y1 = top;
        box.x2 = right;
        box.y2 = bottom;
        box.width = std::max(0, right - left + 1);
        box.height = std::max(0, bottom - top + 1);
        box.aspectRatio = (box.width > 0 && box.height > 0) ? (static_cast<double>(box.width) / box.height) : 0.0;
        box.area = static_cast<double>(box.width) * box.height;

        if (box.x1 < 0 || box.y1 < 0 || box.x2 >= imageSize.width || box.y2 >= imageSize.height) {
            // Keep invalid boxes in stats, but do not silently ignore them.
            // The caller can count them as invalid.
        }

        boxes.push_back(box);
    }

    return boxes;
}

void updateStats(Stats& stats, const std::vector<BBox>& boxes) {
    if (boxes.empty()) {
        return;
    }

    std::vector<double> widths;
    std::vector<double> heights;
    std::vector<double> areas;
    std::vector<double> aspects;

    for (const auto& box : boxes) {
        widths.push_back(static_cast<double>(box.width));
        heights.push_back(static_cast<double>(box.height));
        areas.push_back(box.area);
        aspects.push_back(box.aspectRatio);

        if (box.width < 10) {
            ++stats.widthLt10;
        }
        if (box.width < 20) {
            ++stats.widthLt20;
        }
        if (box.width < 30) {
            ++stats.widthLt30;
        }
        if (box.width < 50) {
            ++stats.widthLt50;
        }

        if (box.height < 10) {
            ++stats.heightLt10;
        }
        if (box.height < 20) {
            ++stats.heightLt20;
        }
        if (box.height < 30) {
            ++stats.heightLt30;
        }
        if (box.height < 50) {
            ++stats.heightLt50;
        }
    }

    const auto widthMinMax = std::minmax_element(widths.begin(), widths.end());
    const auto heightMinMax = std::minmax_element(heights.begin(), heights.end());
    const auto areaMinMax = std::minmax_element(areas.begin(), areas.end());
    const auto aspectMinMax = std::minmax_element(aspects.begin(), aspects.end());

    stats.minWidth = *widthMinMax.first;
    stats.maxWidth = *widthMinMax.second;
    stats.meanWidth = std::accumulate(widths.begin(), widths.end(), 0.0) / static_cast<double>(widths.size());
    stats.medianWidth = medianValue(widths);

    stats.minHeight = *heightMinMax.first;
    stats.maxHeight = *heightMinMax.second;
    stats.meanHeight = std::accumulate(heights.begin(), heights.end(), 0.0) / static_cast<double>(heights.size());
    stats.medianHeight = medianValue(heights);

    stats.minArea = *areaMinMax.first;
    stats.maxArea = *areaMinMax.second;
    stats.meanArea = std::accumulate(areas.begin(), areas.end(), 0.0) / static_cast<double>(areas.size());
    stats.medianArea = medianValue(areas);

    stats.minAspect = *aspectMinMax.first;
    stats.maxAspect = *aspectMinMax.second;
    stats.meanAspect = std::accumulate(aspects.begin(), aspects.end(), 0.0) / static_cast<double>(aspects.size());
    stats.medianAspect = medianValue(aspects);

    stats.p10Width = percentile(widths, 10.0);
    stats.p25Width = percentile(widths, 25.0);
    stats.p50Width = percentile(widths, 50.0);
    stats.p75Width = percentile(widths, 75.0);
    stats.p90Width = percentile(widths, 90.0);
    stats.p95Width = percentile(widths, 95.0);

    stats.p10Height = percentile(heights, 10.0);
    stats.p25Height = percentile(heights, 25.0);
    stats.p50Height = percentile(heights, 50.0);
    stats.p75Height = percentile(heights, 75.0);
    stats.p90Height = percentile(heights, 90.0);
    stats.p95Height = percentile(heights, 95.0);
}

void printSummary(const Stats& stats) {
    std::cout << "\nSummary for Haar window choice\n";
    std::cout << "-----------------------------\n";
    std::cout << "Likely reasonable base windows to test: 20x20, 24x24, 32x32\n";
    std::cout << "- if most boxes are below ~30 px in height, a 20x20/24x24 window may be closer to the data\n";
    std::cout << "- if the median height is ~30-60 px, 24x24 or 32x32 are reasonable starting points\n";
    std::cout << "- if many boxes are still under 10-20 px, those may be too small for a stable Haar detector without aggressive scale pyramids\n";
    std::cout << "- no automatic configuration change is performed here; this is an analysis-only tool\n";
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <dataset_dir>" << std::endl;
            return 1;
        }

        const fs::path datasetDir = argv[1];
        if (!fs::exists(datasetDir) || !fs::is_directory(datasetDir)) {
            throw std::runtime_error("Dataset directory does not exist: " + datasetDir.string());
        }

        const std::vector<fs::path> imageFiles = listDatasetImageFiles(datasetDir);
        Stats stats;
        std::vector<double> allWidths;
        std::vector<double> allHeights;
        std::vector<double> allAreas;
        std::vector<double> allAspects;

        std::size_t analyzedImages = 0;
        std::size_t invalidBoxes = 0;

        for (const auto& imageFile : imageFiles) {
            fs::path jsonFile = imageFile;
            jsonFile.replace_extension(".json");
            if (!fs::exists(jsonFile)) {
                continue;
            }

            cv::Mat image = cv::imread(imageFile.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                continue;
            }

            ++analyzedImages;

            const auto boxes = readBoxesForImage(jsonFile, image.size());
            if (boxes.empty()) {
                continue;
            }

            stats.bboxCount += boxes.size();

            for (const auto& box : boxes) {
                if (box.x1 < 0 || box.y1 < 0 || box.x2 >= image.cols || box.y2 >= image.rows) {
                    ++invalidBoxes;
                }

                allWidths.push_back(static_cast<double>(box.width));
                allHeights.push_back(static_cast<double>(box.height));
                allAreas.push_back(box.area);
                allAspects.push_back(box.aspectRatio);
            }
        }

        std::cout << "Dataset statistics for Viola-Jones training window choice\n";
        std::cout << "====================================================\n";
        std::cout << "Images analyzed: " << analyzedImages << "\n";
        std::cout << "Total bounding boxes: " << stats.bboxCount << "\n";
        std::cout << "Out-of-bounds boxes: " << invalidBoxes << "\n\n";

        if (!allWidths.empty()) {
            updateStats(stats, std::vector<BBox>{});
            std::sort(allWidths.begin(), allWidths.end());
            std::sort(allHeights.begin(), allHeights.end());
            std::sort(allAreas.begin(), allAreas.end());
            std::sort(allAspects.begin(), allAspects.end());

            const auto widthMinMax = std::minmax_element(allWidths.begin(), allWidths.end());
            const auto heightMinMax = std::minmax_element(allHeights.begin(), allHeights.end());
            const auto areaMinMax = std::minmax_element(allAreas.begin(), allAreas.end());
            const auto aspectMinMax = std::minmax_element(allAspects.begin(), allAspects.end());

            stats.minWidth = *widthMinMax.first;
            stats.maxWidth = *widthMinMax.second;
            stats.meanWidth = std::accumulate(allWidths.begin(), allWidths.end(), 0.0) / static_cast<double>(allWidths.size());
            stats.medianWidth = medianValue(allWidths);

            stats.minHeight = *heightMinMax.first;
            stats.maxHeight = *heightMinMax.second;
            stats.meanHeight = std::accumulate(allHeights.begin(), allHeights.end(), 0.0) / static_cast<double>(allHeights.size());
            stats.medianHeight = medianValue(allHeights);

            stats.minArea = *areaMinMax.first;
            stats.maxArea = *areaMinMax.second;
            stats.meanArea = std::accumulate(allAreas.begin(), allAreas.end(), 0.0) / static_cast<double>(allAreas.size());
            stats.medianArea = medianValue(allAreas);

            stats.minAspect = *aspectMinMax.first;
            stats.maxAspect = *aspectMinMax.second;
            stats.meanAspect = std::accumulate(allAspects.begin(), allAspects.end(), 0.0) / static_cast<double>(allAspects.size());
            stats.medianAspect = medianValue(allAspects);

            stats.p10Width = percentile(allWidths, 10.0);
            stats.p25Width = percentile(allWidths, 25.0);
            stats.p50Width = percentile(allWidths, 50.0);
            stats.p75Width = percentile(allWidths, 75.0);
            stats.p90Width = percentile(allWidths, 90.0);
            stats.p95Width = percentile(allWidths, 95.0);

            stats.p10Height = percentile(allHeights, 10.0);
            stats.p25Height = percentile(allHeights, 25.0);
            stats.p50Height = percentile(allHeights, 50.0);
            stats.p75Height = percentile(allHeights, 75.0);
            stats.p90Height = percentile(allHeights, 90.0);
            stats.p95Height = percentile(allHeights, 95.0);

            std::cout << "Width: min=" << stats.minWidth << ", max=" << stats.maxWidth
                      << ", mean=" << stats.meanWidth << ", median=" << stats.medianWidth << "\n";
            std::cout << "Height: min=" << stats.minHeight << ", max=" << stats.maxHeight
                      << ", mean=" << stats.meanHeight << ", median=" << stats.medianHeight << "\n";
            std::cout << "Area: min=" << stats.minArea << ", max=" << stats.maxArea
                      << ", mean=" << stats.meanArea << ", median=" << stats.medianArea << "\n";
            std::cout << "Aspect ratio (w/h): min=" << stats.minAspect << ", max=" << stats.maxAspect
                      << ", mean=" << stats.meanAspect << ", median=" << stats.medianAspect << "\n\n";

            std::cout << "Percentiles and thresholds for width (px)\n";
            std::cout << "P10=" << stats.p10Width << ", P25=" << stats.p25Width << ", P50=" << stats.p50Width
                      << ", P75=" << stats.p75Width << ", P90=" << stats.p90Width << ", P95=" << stats.p95Width << "\n";
            std::cout << "Percentiles and thresholds for height (px)\n";
            std::cout << "P10=" << stats.p10Height << ", P25=" << stats.p25Height << ", P50=" << stats.p50Height
                      << ", P75=" << stats.p75Height << ", P90=" << stats.p90Height << ", P95=" << stats.p95Height << "\n\n";

            std::cout << "Width percentages below thresholds\n";
            std::cout << " <10 px: " << (stats.bboxCount ? (100.0 * static_cast<double>(std::count_if(allWidths.begin(), allWidths.end(), [](double v) { return v < 10.0; })) / static_cast<double>(stats.bboxCount)) : 0.0) << "%\n";
            std::cout << " <20 px: " << (stats.bboxCount ? (100.0 * static_cast<double>(std::count_if(allWidths.begin(), allWidths.end(), [](double v) { return v < 20.0; })) / static_cast<double>(stats.bboxCount)) : 0.0) << "%\n";
            std::cout << " <30 px: " << (stats.bboxCount ? (100.0 * static_cast<double>(std::count_if(allWidths.begin(), allWidths.end(), [](double v) { return v < 30.0; })) / static_cast<double>(stats.bboxCount)) : 0.0) << "%\n";
            std::cout << " <50 px: " << (stats.bboxCount ? (100.0 * static_cast<double>(std::count_if(allWidths.begin(), allWidths.end(), [](double v) { return v < 50.0; })) / static_cast<double>(stats.bboxCount)) : 0.0) << "%\n\n";

            std::cout << "Height percentages below thresholds\n";
            std::cout << " <10 px: " << (stats.bboxCount ? (100.0 * static_cast<double>(std::count_if(allHeights.begin(), allHeights.end(), [](double v) { return v < 10.0; })) / static_cast<double>(stats.bboxCount)) : 0.0) << "%\n";
            std::cout << " <20 px: " << (stats.bboxCount ? (100.0 * static_cast<double>(std::count_if(allHeights.begin(), allHeights.end(), [](double v) { return v < 20.0; })) / static_cast<double>(stats.bboxCount)) : 0.0) << "%\n";
            std::cout << " <30 px: " << (stats.bboxCount ? (100.0 * static_cast<double>(std::count_if(allHeights.begin(), allHeights.end(), [](double v) { return v < 30.0; })) / static_cast<double>(stats.bboxCount)) : 0.0) << "%\n";
            std::cout << " <50 px: " << (stats.bboxCount ? (100.0 * static_cast<double>(std::count_if(allHeights.begin(), allHeights.end(), [](double v) { return v < 50.0; })) / static_cast<double>(stats.bboxCount)) : 0.0) << "%\n\n";

            printSummary(stats);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
}
