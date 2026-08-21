#include "patches.hpp"
#include "utils.hpp"
#include "patch_geometry.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <string>
#include <set>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <iostream>

//parse the dataset csv, keeping only the imgs listed in the split file.
//boxes shorter than minHeight are skipped and counted in discarded.
static std::unordered_map<std::string, std::vector<cv::Rect>> loadBoxes(const std::string& csvPath, const std::string& splitFile, int minHeight, int& discarded){
    //add all the filenames of the training to a set, used later to check if an image should be used for the training
    std::set<std::string> trainSet;
    std::ifstream split(splitFile);
    std::string line;
    while(std::getline(split, line)){
        if(line.empty()) continue;
        trainSet.insert(line);
    }

    std::ifstream csvFile(csvPath);
    std::string line1;
    std::unordered_map<std::string, std::vector<cv::Rect>> mp; //map to store the boxes foreach img
    discarded = 0;
    while(std::getline(csvFile, line1)){

        std::stringstream ss(line1);
        std::string imgName;
        std::getline(ss, imgName, ',');
        std::string x1s, y1s, x2s, y2s;

        std::getline(ss, x1s, ',');
        std::getline(ss, y1s, ',');
        std::getline(ss, x2s, ',');
        std::getline(ss, y2s, ',');

        int x1 = std::stoi(x1s);
        int y1 = std::stoi(y1s);
        int x2 = std::stoi(x2s);
        int y2 = std::stoi(y2s);

        if(trainSet.count(imgName) != 0){ //skip if the height is too small
            if((y2 - y1) < minHeight) discarded++;
            else mp[imgName].push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2,y2)));
        }
    }
    return mp;
}

void extractPositives(const std::string& datasetDir, const std::string& csvPath, const std::string& splitFile, const std::string& outDir, int minHeight){
    int counter = 0; //counter to see how many boxes are skipped
    std::unordered_map<std::string, std::vector<cv::Rect>> mp = loadBoxes(csvPath, splitFile, minHeight, counter);
    std::cout << "discarded boxes: " << counter << '\n';

    //save the positive samples. Open one img at a time
    std::filesystem::create_directories(outDir);
    int positiveCounter = 0;
    for(const auto& elem : mp){
        const std::string& imgPath = elem.first;
        const std::vector<cv::Rect>& boxes = elem.second;
        int boxCount = 0;
        cv::Mat img = cv::imread(datasetDir + "/" + imgPath, cv::IMREAD_GRAYSCALE);
        if(img.empty()) continue;
        std::string stem = imgPath.substr(0, imgPath.find_last_of('.'));

        for(const auto& box : boxes){

            cv::Rect safe = box & cv::Rect(0, 0, img.cols, img.rows);
            if (safe.width < 4 || safe.height < 4) continue; //if the box is out of the img it will fail so skip it.
            cv::Mat positive = img(safe);

            //resize to standard patch size
            int interp = (safe.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR; //use different interpolations for enlarging and shrinking
            cv::resize(positive, positive, kPatchSize, 0, 0, interp);
            cv::imwrite(outDir + "/" + stem + "_" + std::to_string(boxCount) + ".png", positive);
            cv::Mat flipped;
            cv::flip(positive, flipped, 1); //data augmentation keep flipped version also.
            cv::imwrite(outDir + "/" + stem + "_" + std::to_string(boxCount) + "_flip" + ".png", flipped);
            boxCount++;
            positiveCounter++;
        }
    }
    std::cout << "positive patches written: " << positiveCounter * 2 << " (" << positiveCounter << " boxes + flip)\n";
}

void extractNegatives(const std::string& datasetDir, const std::string& csvPath, const std::string& splitFile, const std::string& outDir, int numPatches){
    //keep ALL the boxes, including the ones too small to be positives:
    //they are still cones and must not end up among the negatives.
    int ignored = 0;
    std::unordered_map<std::string, std::vector<cv::Rect>> mp = loadBoxes(csvPath, splitFile, 0, ignored);
    if(mp.empty()) return;

    std::filesystem::create_directories(outDir);
    int patchesPerImg = std::max(1, numPatches / (int)mp.size());
    const int maxAttempts = patchesPerImg * 20;

    //one generator for the whole run, fixed seed so the negatives are reproducible
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> hDist(20, 120); //arbitrary: rectangles of 20/120 pixels
    std::uniform_real_distribution<double> ratioDist(0.55, 0.95); // width will be between 55-95% of height

    int negativeCounter = 0;
    for(const auto& elem : mp){

        const std::string& imgName = elem.first;
        cv::Mat img = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_GRAYSCALE);
        if(img.empty()) continue;
        std::string stem = imgName.substr(0, imgName.find_last_of('.'));

        int saved = 0;
        for(int i = 0; i < maxAttempts && saved < patchesPerImg; i++){
            int h = hDist(rng);
            int w = (int)std::lround(h * ratioDist(rng));
            if(h >= img.rows || w >= img.cols){
                continue; //if rect is out of bound skip;
            }
            //smart way to provide a valid box that doesn't go out of the img
            std::uniform_int_distribution<int> xDist(0, img.cols - w);
            std::uniform_int_distribution<int> yDist(0, img.rows - h);
            cv::Rect r(xDist(rng), yDist(rng), w, h);

            bool intersected = false;
            for(const auto& label : elem.second){
                if(calculateIoU(label, r) > 0.1){
                    intersected = true;
                    break;
                }
            }
            if(intersected) continue;

            cv::Mat negative = img(r);
            int interp = (r.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(negative, negative, kPatchSize, 0, 0, interp);

            cv::Scalar mean, stddev;
            cv::meanStdDev(negative, mean, stddev);
            if(stddev[0] < 5.0) continue; //patch troppo uniforme (cielo/padding nero), non informativa per l'SVM
            if(cv::countNonZero(negative) < 0.7 * negative.total()) continue; //dominata dal padding nero, bordo che non esiste nei proposal reali

            cv::imwrite(outDir + "/" + stem + "_" + std::to_string(saved) + ".png", negative);
            saved++;
            negativeCounter++;
        }
    }
    std::cout << "negative patches written: " << negativeCounter << '\n';
}
