#include "dataloader.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>


void dataLoader()
{
    // If dataset already exists, delete it and create a new one
    if (std::filesystem::exists("../dataset")) {
        std::cout << "The folder already exists, deleting..." << std::endl;
        std::filesystem::remove_all("../dataset");
    } 
    
    if (std::filesystem::create_directory("../dataset")) {
        std::cout << "Successfully created folder dataset" << std::endl;
    } else {
        throw std::runtime_error("Failed to create directory");
    }

    const std::string pathSegmentation = "../fsoco_segmentation_train-001/fsoco_segmentation_train";
    const std::string pathBBox = "../fsoco_bounding_boxes_train/fsoco_bounding_boxes_train";

    std::filesystem::path filePathSegmentation(pathSegmentation);
    std::filesystem::path filePathBBox(pathBBox);

    std::unordered_map<std::string, std::filesystem::path> imageFiles;
    std::unordered_map<std::string, std::filesystem::path> jsonFiles;

    // Iterating through the files and saving them in two vectors, one for images and one for json files
    // First we go through pathSegmentation then through pathBBox
    for (auto const& entry : std::filesystem::recursive_directory_iterator{filePathSegmentation}) {
        
        if (entry.is_directory()) continue;
        std::string ext = entry.path().extension().string();
        
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            imageFiles.emplace(entry.path().filename().string(), entry.path());
        } else if (ext == ".json") {
            const std::string jsonName = entry.path().filename().string();
            if (jsonName == "meta.json") {
                continue;
            }

            if (jsonName.size() <= 5 || jsonName.substr(jsonName.size() - 5) != ".json") {
                continue;
            }

            jsonFiles.emplace(jsonName.substr(0, jsonName.size() - 5), entry.path());
        }
    }
    
    for (auto const& entry : std::filesystem::recursive_directory_iterator{filePathBBox}) {
        
        if (entry.is_directory()) continue;
        std::string ext = entry.path().extension().string();
        
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            imageFiles.emplace(entry.path().filename().string(), entry.path());
        } else if (ext == ".json") {
            const std::string jsonName = entry.path().filename().string();
            if (jsonName == "meta.json") {
                continue;
            }

            if (jsonName.size() <= 5 || jsonName.substr(jsonName.size() - 5) != ".json") {
                continue;
            }

            jsonFiles.emplace(jsonName.substr(0, jsonName.size() - 5), entry.path());
        }
    }

    std::vector<std::string> sharedKeys;
    sharedKeys.reserve(std::min(imageFiles.size(), jsonFiles.size()));

    for (const auto& [stem, imagePath] : imageFiles) {
        if (jsonFiles.find(stem) != jsonFiles.end()) {
            sharedKeys.push_back(stem);
        } else {
            std::cout << "Missing JSON for source image: " << imagePath << std::endl;
        }
    }

    std::sort(sharedKeys.begin(), sharedKeys.end(), [](const std::string& left, const std::string& right) {
        try {
            return std::stoll(left) < std::stoll(right);
        } catch (...) {
            return left < right;
        }
    });

    std::filesystem::path destination = "../dataset";
    auto fileNumber = 1;
    
    // Saving only matched image/json pairs, renumbered as 1.jpg, 1.json, 2.jpg, 2.json, ...
    for (const std::string& key : sharedKeys) {
        const auto& oldImagePath = imageFiles.at(key);
        const auto& oldJsonPath = jsonFiles.at(key);

        std::string newImageName = std::to_string(fileNumber) + oldImagePath.extension().string();
        std::string newJsonName = std::to_string(fileNumber) + oldJsonPath.extension().string();

        std::filesystem::path newImagePath = destination / newImageName;
        std::filesystem::path newJsonPath = destination / newJsonName;

        std::filesystem::copy(oldImagePath, newImagePath);
        std::filesystem::copy(oldJsonPath, newJsonPath);

        fileNumber++;
    }

    std::cout << "Dataset prepared with " << (fileNumber - 1) << " matched image/json pair(s)." << std::endl;
}