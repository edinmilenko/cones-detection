//Author: Nicola Ranzolin
#include "patches.hpp"
#include "color_proposals.hpp"
#include <opencv2/ml.hpp>
#include "utils.hpp"
#include "hog.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <iostream>

//reads the dataset csv. dataset/ has no test set images inside, so everything here can be used for training.
static std::unordered_map<std::string, std::vector<cv::Rect>> loadBoxes(const std::string& csvPath){

    // one line per cone: image name, then the two corners of its box
    std::ifstream csvFile(csvPath);
    std::string line1;
    std::unordered_map<std::string, std::vector<cv::Rect>> mp; //map to store the boxes foreach img
    while(std::getline(csvFile, line1))
    {
        // the fields are separated by commas, we pull them out one at a time
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

        // every image collects all of its cones
        mp[imgName].push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2,y2)));
    }
    return mp;
}

//patches for the cascade: we run colorProposals on the training images and label each candidate by its IoU with the real boxes, so the classifier sees the same kind of crops it gets at detection time
void extractProposalPatches(const std::string& datasetDir, const std::string& csvPath, const std::string& posOutDir, const std::string& negOutDir, int maxPosPerImage, int maxNegPerImage){
    std::filesystem::create_directories(posOutDir);
    std::filesystem::create_directories(negOutDir);

    std::unordered_map<std::string, std::vector<cv::Rect>> gt = loadBoxes(csvPath);

    //all the annotated images, the test set is in another folder
    std::vector<std::string> trainImgs;
    std::unordered_map<std::string, std::vector<cv::Rect>>::const_iterator it;
    for(it = gt.begin(); it != gt.end(); ++it)
    {
        trainImgs.push_back(it->first);
    }
    // sorted first so the order does not depend on the map, then shuffled with a fixed seed
    std::sort(trainImgs.begin(), trainImgs.end());
    std::mt19937 rng(42);
    std::shuffle(trainImgs.begin(), trainImgs.end(), rng);
    std::cout << "training on " << trainImgs.size() << " images\n";

    int posSaved = 0, negSaved = 0, usedImgs = 0;
    for(const std::string& imgName : trainImgs)
    {
        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;

        usedImgs++;

        const std::vector<cv::Rect>& gtBoxes = gt[imgName];
        const std::string stem = imgName.substr(0, imgName.find_last_of('.'));

        //shuffle before capping, otherwise we always take the candidates of the same few blobs
        std::vector<cv::Rect> candidates = colorProposals(bgr);
        std::shuffle(candidates.begin(), candidates.end(), rng);

        int localIdx = 0;
        int posHere = 0;
        int negHere = 0;
        for(size_t c = 0; c < candidates.size(); c++)
        {
            const cv::Rect& candidate = candidates[c];
            if(candidate.width < 4 || candidate.height < 4) continue;

            // how well this candidate covers the cone it covers best
            double bestIou = 0.0;
            for(const cv::Rect& box : gtBoxes)
            {
                bestIou = std::max(bestIou, calculateIoU(candidate, box));
            }

            //same 0.3 we use everywhere else to say a cone is covered. With a higher value most real candidates ended up as ambiguous and the classifier never learned to accept them
            const bool isPos = bestIou >= 0.3;
            const bool isNeg = bestIou < 0.2;
            if(!isPos && !isNeg) continue;        
            // the negatives are capped per image, the positives are not: with the two classes even, stage 1 scored everything low and the cascade stopped detecting
            if(maxPosPerImage > 0 && isPos && posHere >= maxPosPerImage) continue;
            if(maxNegPerImage > 0 && isNeg && negHere >= maxNegPerImage) continue;

            // saved in colour, describePatch() uses the colour too
            cv::Mat patch;
            int interp = cv::INTER_LINEAR;
            if(candidate.height > PATCH_SIZE.height)
            {
                interp = cv::INTER_AREA;
            }
            cv::resize(bgr(candidate), patch, PATCH_SIZE, 0, 0, interp);

            std::string outDir = negOutDir;
            if(isPos)
            {
                outDir = posOutDir;
            }
            cv::imwrite(outDir + "/" + stem + "_" + std::to_string(localIdx) + ".png", patch);
            if(isPos)
            {
                // easy data augmentation
                cv::Mat flipped;
                cv::flip(patch, flipped, 1);
                cv::imwrite(outDir + "/" + stem + "_" + std::to_string(localIdx) + "_flip.png", flipped);
                posSaved += 2;
                posHere += 2;
            }
            else
            {
                negSaved++;
                negHere++;
            }
            localIdx++;
        }

        if(usedImgs % 100 == 0)
        {
            std::cout << "  ..." << usedImgs << " images, pos=" << posSaved << " neg=" << negSaved << '\n';
        }
    }

    std::cout << "proposal patches: " << usedImgs << " train images, positives=" << posSaved << " (with flip), negatives=" << negSaved << '\n';
}

//hard negatives: we run stage 1 on the images and keep the candidates it accepts that are on not cones
void mineHardNegatives(const std::string& datasetDir, const std::string& csvPath, const std::string& stage1Path, const std::string& negOutDir, float stage1Thr, int maxImages, int targetCount){
    cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load(stage1Path);
    if(stage1.empty())
    {
        throw std::runtime_error("mineHardNegatives: cannot load " + stage1Path);
    }

    std::unordered_map<std::string, std::vector<cv::Rect>> gt = loadBoxes(csvPath);

    std::vector<std::string> images;
    std::unordered_map<std::string, std::vector<cv::Rect>>::const_iterator it;
    for(it = gt.begin(); it != gt.end(); ++it)
    {
        images.push_back(it->first);
    }
    std::sort(images.begin(), images.end());

    //different seed from the extraction, so we start from other images
    std::mt19937 rng(7);
    std::shuffle(images.begin(), images.end(), rng);
    if(maxImages > 0 && (int)images.size() > maxImages)
    {
        images.resize(maxImages);
    }

    int saved = 0;
    for(size_t i = 0; i < images.size() && saved < targetCount; i++)
    {
        const std::string& imgName = images[i];
        cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()) continue;

        const std::vector<cv::Rect>& gtBoxes = gt[imgName];
        const std::string stem = imgName.substr(0, imgName.find_last_of('.'));

        int localIdx = 0;
        std::vector<cv::Rect> candidates = colorProposals(bgr);
        for(size_t c = 0; c < candidates.size() && saved < targetCount; c++)
        {
            const cv::Rect& candidate = candidates[c];
            if(candidate.width < 4 || candidate.height < 4) continue;

            double bestIou = 0.0;
            for(size_t g = 0; g < gtBoxes.size(); g++)
            {
                double iou = calculateIoU(candidate, gtBoxes[g]);
                if(iou > bestIou) bestIou = iou;
            }
            // too close to a real cone to be a clean negative
            if(bestIou >= 0.2) continue;

            cv::Mat patch;
            int interp = cv::INTER_LINEAR;
            if(candidate.height > PATCH_SIZE.height)
            {
                interp = cv::INTER_AREA;
            }
            cv::resize(bgr(candidate), patch, PATCH_SIZE, 0, 0, interp);
            // stage 1 already rejects it, so there is nothing to learn from it
            if(rawScore(stage1, patch) <= stage1Thr) continue;

            cv::imwrite(negOutDir + "/hard_" + stem + "_" + std::to_string(localIdx) + ".png", patch);
            localIdx++;
            saved++;
        }
    }
    std::cout << "hard negatives mined: " << saved << " into " << negOutDir << '\n';
}
