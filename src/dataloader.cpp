// Dataset utility: standalone tool to load and match image/annotation data.
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    try {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " <segmentation_dir> <bbox_dir> <output_dataset_dir>\n";
            return 1;
        }

        const fs::path pathSegmentation = argv[1];
        const fs::path pathBBox = argv[2];
        const fs::path destination = argv[3];

        if (fs::exists(destination)) {
            std::cout << "The folder " << destination << " already exists, deleting...\n";
            fs::remove_all(destination);
        } 
        
        if (fs::create_directories(destination)) {
            std::cout << "Successfully created folder " << destination << "\n";
        } else {
            throw std::runtime_error("Failed to create destination directory");
        }

        std::unordered_map<std::string, fs::path> imageFiles;
        std::unordered_map<std::string, fs::path> jsonFiles;

        auto processDirectory = [&](const fs::path& dir) {
            if (!fs::exists(dir)) {
                std::cerr << "Warning: Directory not found: " << dir << "\n";
                return;
            }
            for (auto const& entry : fs::recursive_directory_iterator{dir}) {
                if (entry.is_directory()) continue;
                std::string ext = entry.path().extension().string();
                
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                    imageFiles.emplace(entry.path().stem().string(), entry.path());
                } else if (ext == ".json") {
                    const std::string jsonName = entry.path().filename().string();
                    if (jsonName == "meta.json") continue;
                    // Fix: Rimossa l'estrazione substring fragile in favore dell'API standard
                    jsonFiles.emplace(entry.path().stem().string(), entry.path());
                }
            }
        };

        processDirectory(pathSegmentation);
        processDirectory(pathBBox);

        std::vector<std::string> sharedKeys;
        sharedKeys.reserve(std::min(imageFiles.size(), jsonFiles.size()));

        for (const auto& [stem, imagePath] : imageFiles) {
            if (jsonFiles.find(stem) != jsonFiles.end()) {
                sharedKeys.push_back(stem);
            } else {
                std::cout << "Missing JSON for source image: " << imagePath << "\n";
            }
        }

        std::sort(sharedKeys.begin(), sharedKeys.end(), [](const std::string& left, const std::string& right) {
            try {
                return std::stoll(left) < std::stoll(right);
            } catch (...) {
                return left < right;
            }
        });

        int fileNumber = 1;
        for (const std::string& key : sharedKeys) {
            const auto& oldImagePath = imageFiles.at(key);
            const auto& oldJsonPath = jsonFiles.at(key);

            std::string newImageName = std::to_string(fileNumber) + oldImagePath.extension().string();
            std::string newJsonName = std::to_string(fileNumber) + ".json";

            fs::copy(oldImagePath, destination / newImageName);
            fs::copy(oldJsonPath, destination / newJsonName);

            fileNumber++;
        }

        std::cout << "Dataset prepared with " << (fileNumber - 1) << " matched image/json pair(s).\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << "\n";
        return 1;
    }
}