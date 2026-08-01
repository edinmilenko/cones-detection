#include "dataloader.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>


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

    std::vector<std::filesystem::path> imageFiles;
    std::vector<std::filesystem::path> jsonFiles;

    // Iterating through the files and saving them in two vectors, one for images and one for json files
    // First we go through pathSegmentation then through pathBBox
    for (auto const& entry : std::filesystem::recursive_directory_iterator{filePathSegmentation}) {
        
        if (entry.is_directory()) continue;
        std::string ext = entry.path().extension().string();
        
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            imageFiles.push_back(entry.path());
        } else if (ext == ".json") {
            jsonFiles.push_back(entry.path());
        }
    }
    
    for (auto const& entry : std::filesystem::recursive_directory_iterator{filePathBBox}) {
        
        if (entry.is_directory()) continue;
        std::string ext = entry.path().extension().string();
        
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            imageFiles.push_back(entry.path());
        } else if (ext == ".json") {
            jsonFiles.push_back(entry.path());
        }
    }
    std::sort(imageFiles.begin(), imageFiles.end());
    std::sort(jsonFiles.begin(), jsonFiles.end());

    std::filesystem::path destination = "../dataset";
    auto fileNumber = 1;
    
    // Saving all the images and json files in the dataset folder with a new name, starting from 1.jpg, 1.json, 2.jpg, 2.json, ...
    for (size_t i = 0; i < imageFiles.size(); ++i) {
        
        const auto& oldImagePath = imageFiles[i];
        const auto& oldJsonPath = jsonFiles[i];

        std::string newImageName = std::to_string(fileNumber) + oldImagePath.extension().string();
        std::string newJsonName = std::to_string(fileNumber) + oldJsonPath.extension().string();

        std::filesystem::path newImagePath = destination / newImageName;
        std::filesystem::path newJsonPath = destination / newJsonName;

        std::filesystem::copy(oldImagePath, newImagePath);
        std::filesystem::copy(oldJsonPath, newJsonPath);

        fileNumber++;
    }
}