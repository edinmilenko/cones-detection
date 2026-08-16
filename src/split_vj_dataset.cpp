// Offline dataset splitter: partitions complete source images into train,
// validation, and test manifests while validating negative-patch provenance.
// Updated to handle split standing/fallen manifests and 36x36 negative patches.
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kNegativePatchSize = 36;

struct SplitPercentages {
    // defaults percentages, not fixed image counts
    int train = 70;
    int validation = 15;
    int test = 15;
};

struct NegativeSource {
    // One record mirrors a line in negative_sources.txt: cropped patch ->
    // original dataset image. This prevents negative-patch leakage.
    std::string patchPath;
    std::string sourcePath;
};

std::string firstToken(const std::string& line) {
    std::istringstream stream(line);
    std::string token;
    stream >> token;
    return token;
}

std::size_t validateAnnotationLine(const std::string& line) {
    std::istringstream stream(line);
    std::string imagePath;
    std::size_t objectCount = 0;
    if (!(stream >> imagePath >> objectCount) || imagePath.empty() || !fs::exists(imagePath)) {
        throw std::runtime_error("Invalid annotation image path: " + line);
    }

    std::vector<int> coordinates;
    int value = 0;
    while (stream >> value) {
        coordinates.push_back(value);
    }
    // annotations.txt follows OpenCV's 2 + 4N token contract.
    if (coordinates.size() != objectCount * 4) {
        throw std::runtime_error("Invalid annotation record: " + line);
    }
    return objectCount;
}

void validateSinglePathLine(const std::string& line, const std::string& fileName) {
    if (line.empty() || firstToken(line) != line) {
        throw std::runtime_error("Expected exactly one path per line in " + fileName + ": " + line);
    }
}

// Fallback per file mancanti nel caso in cui non ci siano oggetti di un certo tipo
std::vector<std::string> readLinesSafe(const fs::path& path) {
    std::vector<std::string> lines;
    if (!fs::exists(path)) return lines; 
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open: " + path.string());
    }
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<std::string> readLines(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open: " + path.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<NegativeSource> readNegativeSources(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open: " + path.string());
    }

    std::vector<NegativeSource> records;
    std::unordered_set<std::string> patches;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty()) {
            continue;
        }

        std::istringstream stream(line);
        NegativeSource record;
        std::string extra;
        if (!(stream >> record.patchPath >> record.sourcePath) || (stream >> extra)) {
            throw std::runtime_error("Invalid negative source record at line " + std::to_string(lineNumber));
        }
        if (!patches.insert(record.patchPath).second) {
            throw std::runtime_error("Duplicate negative patch in negative_sources.txt: " + record.patchPath);
        }
        records.push_back(std::move(record));
    }
    return records;
}

std::array<std::size_t, 3> splitCounts(std::size_t total, const SplitPercentages& percentages) {
    const std::array<int, 3> values{percentages.train, percentages.validation, percentages.test};
    std::array<std::size_t, 3> counts{};
    std::array<int, 3> remainders{};
    std::size_t assigned = 0;

    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::size_t scaled = total * static_cast<std::size_t>(values[i]);
        counts[i] = scaled / 100;
        remainders[i] = static_cast<int>(scaled % 100);
        assigned += counts[i];
    }

    std::array<std::size_t, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end(), [&remainders](std::size_t left, std::size_t right) {
        return remainders[left] > remainders[right];
    });
    // Largest-remainder allocation preserves the requested proportions while
    // ensuring every input image is assigned exactly once.
    for (std::size_t i = assigned; i < total; ++i) {
        ++counts[order[(i - assigned) % order.size()]];
    }
    return counts;
}

void writeLines(const fs::path& path, const std::vector<std::string>& lines) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot create: " + path.string());
    }
    for (const std::string& line : lines) {
        output << line << '\n';
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc != 2 && argc != 3 && argc != 6) {
            std::cerr << "Usage: " << argv[0]
                      << " <training_dir> [seed] [train_percent val_percent test_percent]\n";
            return 1;
        }

        const fs::path trainingDir = argv[1];
        const unsigned int seed = argc >= 3 ? static_cast<unsigned int>(std::stoul(argv[2])) : 12345U;
        SplitPercentages percentages;
        if (argc == 6) {
            percentages.train = std::stoi(argv[3]);
            percentages.validation = std::stoi(argv[4]);
            percentages.test = std::stoi(argv[5]);
        }
        if (percentages.train < 0 || percentages.validation < 0 || percentages.test < 0 ||
            percentages.train + percentages.validation + percentages.test != 100) {
            throw std::runtime_error("Split percentages must be non-negative and sum to 100");
        }

        const std::vector<std::string> standingLines = readLinesSafe(trainingDir / "annotations_standing.txt");
        const std::vector<std::string> fallenLines = readLinesSafe(trainingDir / "annotations_fallen.txt");
        const std::vector<std::string> negativeLines = readLines(trainingDir / "negatives.txt");
        const std::vector<NegativeSource> negativeSources = readNegativeSources(trainingDir / "negative_sources.txt");
        const std::vector<std::string> datasetImageLines = readLines(trainingDir / "dataset_images.txt");

        if (negativeLines.size() != negativeSources.size()) {
            throw std::runtime_error("negatives.txt and negative_sources.txt have different line counts");
        }

        // dataset_images.txt is the authoritative population for the split. It
        // includes images without boxes, which can still provide negatives.
        std::unordered_set<std::string> datasetImages;
        for (const std::string& imagePath : datasetImageLines) {
            validateSinglePathLine(imagePath, "dataset_images.txt");
            if (!fs::exists(imagePath)) {
                throw std::runtime_error("Missing dataset image: " + imagePath);
            }
            if (!datasetImages.insert(imagePath).second) {
                throw std::runtime_error("Duplicate dataset image: " + imagePath);
            }
        }

        std::unordered_set<std::string> negativePaths;
        for (const NegativeSource& record : negativeSources) {
            if (!negativePaths.insert(record.patchPath).second) {
                throw std::runtime_error("Duplicate negative patch: " + record.patchPath);
            }
            if (!fs::exists(record.patchPath) || !fs::exists(record.sourcePath)) {
                throw std::runtime_error("Missing negative patch or source image: " + record.patchPath);
            }
            if (datasetImages.count(record.sourcePath) != 1) {
                throw std::runtime_error("Negative source is not in dataset_images.txt: " + record.sourcePath);
            }
            
            const cv::Mat patch = cv::imread(record.patchPath, cv::IMREAD_UNCHANGED);
            if (patch.empty() || patch.cols != kNegativePatchSize || patch.rows != kNegativePatchSize) {
                throw std::runtime_error("Negative patch is not " + std::to_string(kNegativePatchSize) + "x" + std::to_string(kNegativePatchSize) + ": " + record.patchPath);
            }
        }

        std::unordered_set<std::string> negativeListPaths;
        for (const std::string& negativeLine : negativeLines) {
            const std::string patchPath = firstToken(negativeLine);
            if (!negativeListPaths.insert(patchPath).second) {
                throw std::runtime_error("Duplicate negative patch in negatives.txt: " + patchPath);
            }
            if (negativePaths.count(patchPath) != 1) {
                throw std::runtime_error("negatives.txt contains an untracked patch: " + negativeLine);
            }
        }
        if (negativeListPaths != negativePaths) {
            throw std::runtime_error("negatives.txt and negative_sources.txt reference different patches");
        }

        std::size_t standingObjects = 0;
        std::size_t fallenObjects = 0;
        std::unordered_set<std::string> annotationImages;

        for (const std::string& line : standingLines) {
            const std::string imagePath = firstToken(line);
            annotationImages.insert(imagePath);
            standingObjects += validateAnnotationLine(line);
            if (datasetImages.count(imagePath) != 1) {
                throw std::runtime_error("Annotation image is not in dataset_images.txt: " + imagePath);
            }
        }

        for (const std::string& line : fallenLines) {
            const std::string imagePath = firstToken(line);
            annotationImages.insert(imagePath);
            fallenObjects += validateAnnotationLine(line);
            if (datasetImages.count(imagePath) != 1) {
                throw std::runtime_error("Annotation image is not in dataset_images.txt: " + imagePath);
            }
        }

        // Split whole source images, never individual bounding boxes. This keeps
        // all positives and negatives from one scene in the same partition.
        std::vector<std::string> images(datasetImages.begin(), datasetImages.end());
        std::sort(images.begin(), images.end());
        std::mt19937 rng(seed);
        std::shuffle(images.begin(), images.end(), rng);
        const std::array<std::size_t, 3> counts = splitCounts(images.size(), percentages);

        const std::array<std::string, 3> names{"train", "val", "test"};
        std::array<std::unordered_set<std::string>, 3> splitImages;
        std::size_t offset = 0;
        for (std::size_t split = 0; split < names.size(); ++split) {
            for (std::size_t i = 0; i < counts[split]; ++i) {
                splitImages[split].insert(images[offset++]);
            }
        }

        for (std::size_t left = 0; left < splitImages.size(); ++left) {
            for (std::size_t right = left + 1; right < splitImages.size(); ++right) {
                for (const std::string& image : splitImages[left]) {
                    if (splitImages[right].count(image) != 0) {
                        throw std::runtime_error("Leakage detected for image: " + image);
                    }
                }
            }
        }

        const fs::path splitsDir = trainingDir / "splits";
        const fs::path manifestsDir = trainingDir / "manifests";
        fs::create_directories(splitsDir);
        fs::create_directories(manifestsDir);

        std::array<std::size_t, 3> splitStandingObjects{};
        std::array<std::size_t, 3> splitFallenObjects{};
        std::array<std::size_t, 3> splitNegativeSamples{};

        for (std::size_t split = 0; split < names.size(); ++split) {
            std::vector<std::string> splitStanding;
            std::vector<std::string> splitFallen;
            std::vector<std::string> splitNegatives;

            for (const std::string& line : standingLines) {
                if (splitImages[split].count(firstToken(line)) != 0) {
                    splitStanding.push_back(line);
                    splitStandingObjects[split] += validateAnnotationLine(line);
                }
            }
            for (const std::string& line : fallenLines) {
                if (splitImages[split].count(firstToken(line)) != 0) {
                    splitFallen.push_back(line);
                    splitFallenObjects[split] += validateAnnotationLine(line);
                }
            }

            for (const NegativeSource& record : negativeSources) {
                if (splitImages[split].count(record.sourcePath) != 0) {
                    splitNegatives.push_back(record.patchPath);
                    ++splitNegativeSamples[split];
                }
            }

            std::vector<std::string> imageList(splitImages[split].begin(), splitImages[split].end());
            std::sort(imageList.begin(), imageList.end());
            writeLines(splitsDir / (names[split] + "_images.txt"), imageList);
            writeLines(manifestsDir / (names[split] + "_annotations_standing.txt"), splitStanding);
            writeLines(manifestsDir / (names[split] + "_annotations_fallen.txt"), splitFallen);
            writeLines(manifestsDir / (names[split] + "_negatives.txt"), splitNegatives);
        }

        const fs::path reportPath = trainingDir / "split_report.txt";
        // The report is a durable checkpoint: colleagues can inspect dataset
        // integrity without re-running the splitter or parsing all manifests.
        std::ofstream report(reportPath, std::ios::trunc);
        if (!report) {
            throw std::runtime_error("Cannot create: " + reportPath.string());
        }
        report << "Dataset\n"
               << "  total images: " << images.size() << "\n"
               << "  annotated images: " << annotationImages.size() << "\n"
               << "  images without GT: " << images.size() - annotationImages.size() << "\n"
               << "  standing boxes: " << standingObjects << "\n"
               << "  fallen boxes: " << fallenObjects << "\n"
               << "  negative patches: " << negativeSources.size() << "\n"
               << "Split\n";
        for (std::size_t split = 0; split < names.size(); ++split) {
            report << "  " << names[split] << ": " << counts[split] << " images, "
                   << splitStandingObjects[split] << " standing, "
                   << splitFallenObjects[split] << " fallen, "
                   << splitNegativeSamples[split] << " negative patches\n";
        }
        report << "Leakage\n"
               << "  train intersection val: 0\n"
               << "  train intersection test: 0\n"
               << "  val intersection test: 0\n"
               << "Negative provenance\n"
               << "  orphan patches: 0\n"
               << "  missing sources: 0\n"
               << "  cross-split sources: 0\n";

        std::cout << "Split seed: " << seed << '\n';
        std::cout << "Annotated images: " << annotationImages.size() << '\n';
        std::cout << "Standing objects: " << standingObjects << '\n';
        std::cout << "Fallen objects: " << fallenObjects << '\n';
        std::cout << "Negative samples: " << negativeSources.size() << " (all " << kNegativePatchSize << "x" << kNegativePatchSize << ")\n";
        std::cout << "Images: " << images.size() << " (train=" << counts[0]
                  << ", val=" << counts[1] << ", test=" << counts[2] << ")\n";
        std::cout << "Anti-leakage check: passed\n";
        std::cout << "Split report: " << reportPath << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "An error occurred: " << error.what() << '\n';
        return 1;
    }
}