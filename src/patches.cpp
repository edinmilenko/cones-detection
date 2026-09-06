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

//parse the dataset csv. dataset/ never contains test set images (dataLoader keeps them out), so
//everything in here can be trained on. Boxes shorter than minHeight are skipped and counted.
std::unordered_map<std::string, std::vector<cv::Rect>> loadBoxes(const std::string& csvPath, int minHeight, int& discarded){

    std::ifstream csvFile(csvPath);
    std::string line1;
    std::unordered_map<std::string, std::vector<cv::Rect>> mp; //map to store the boxes foreach img
    discarded = 0;
    while(std::getline(csvFile, line1))
    {
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

        {
            if((y2 - y1) < minHeight) //skip if the height is too small
            {
                discarded++;
            }
            else
            {
                mp[imgName].push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2,y2)));
            }
        }
    }
    return mp;
}

//patches for the cascade: colorProposals() is run on the training images and every candidate is
//labelled by its IoU with the real boxes, so the classifier is trained on the same crops it will
//see at inference (off-centre, cut, with varying context) instead of clean crops on the GT boxes.
void extractProposalPatches(const std::string& datasetDir, const std::string& csvPath, const std::string& posOutDir, const std::string& negOutDir, int maxPosPerImage, int maxNegPerImage){
    std::filesystem::create_directories(posOutDir);
    std::filesystem::create_directories(negOutDir);

    int ignored = 0;
    std::unordered_map<std::string, std::vector<cv::Rect>> gt = loadBoxes(csvPath, 0, ignored);

    //every annotated image: there is no reason to hold any of them back, the test set is elsewhere
    std::vector<std::string> trainImgs;
    std::unordered_map<std::string, std::vector<cv::Rect>>::const_iterator it;
    for(it = gt.begin(); it != gt.end(); ++it)
    {
        trainImgs.push_back(it->first);
    }
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

        //shuffled before capping: colorProposals returns the candidates blob by blob, so taking
        //the first ones in order would always mean the same few cones of each image
        std::vector<cv::Rect> candidates = colorProposals(bgr);
        std::shuffle(candidates.begin(), candidates.end(), rng);

        int localIdx = 0;
        int posHere = 0;
        int negHere = 0;
        for(size_t c = 0; c < candidates.size(); c++)
        {
            const cv::Rect& candidate = candidates[c];
            if(candidate.width < 4 || candidate.height < 4) continue;

            double bestIou = 0.0;
            for(const cv::Rect& box : gtBoxes)
            {
                bestIou = std::max(bestIou, calculateIoU(candidate, box));
            }

            //the same 0.3 that counts as "covered" everywhere else: with a stricter threshold most
            //real candidates (off-centre, IoU 0.3-0.5) were dropped as ambiguous and the classifier
            //never learned to accept them
            const bool isPos = bestIou >= 0.3;
            const bool isNeg = bestIou < 0.2;
            if(!isPos && !isNeg) continue;                     //ambiguous, skipped
            if(maxPosPerImage > 0 && isPos && posHere >= maxPosPerImage) continue;
            if(maxNegPerImage > 0 && isNeg && negHere >= maxNegPerImage) continue;

            //saved in colour, because describePatch() looks at the colour too
            cv::Mat patch;
            int interp = (candidate.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(bgr(candidate), patch, kPatchSize, 0, 0, interp);

            const std::string& outDir = isPos ? posOutDir : negOutDir;
            cv::imwrite(outDir + "/" + stem + "_" + std::to_string(localIdx) + ".png", patch);
            if(isPos)
            {
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

//hard negatives: run the freshly trained stage 1 over the images and keep the candidates it accepts
//that sit on no cone at all. They are written into the same negatives folder, so retraining stage 1
//afterwards simply picks them up. One round is worth about +0.01 of macro F1 and +0.03 of mIoU.
void mineHardNegatives(const std::string& datasetDir, const std::string& csvPath, const std::string& stage1Path, const std::string& negOutDir, float stage1Thr, int maxImages, int targetCount){
    cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load(stage1Path);
    if(stage1.empty())
    {
        throw std::runtime_error("mineHardNegatives: cannot load " + stage1Path);
    }

    int ignored = 0;
    std::unordered_map<std::string, std::vector<cv::Rect>> gt = loadBoxes(csvPath, 0, ignored);

    std::vector<std::string> images;
    std::unordered_map<std::string, std::vector<cv::Rect>>::const_iterator it;
    for(it = gt.begin(); it != gt.end(); ++it)
    {
        images.push_back(it->first);
    }
    std::sort(images.begin(), images.end());

    //a different seed from the extraction, so the mining looks at other images first
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
            if(bestIou >= 0.2) continue;   //too close to a real cone to be a clean negative

            cv::Mat patch;
            int interp = (candidate.height > kPatchSize.height) ? cv::INTER_AREA : cv::INTER_LINEAR;
            cv::resize(bgr(candidate), patch, kPatchSize, 0, 0, interp);
            if(rawScore(stage1, patch) <= stage1Thr) continue;   //stage 1 already rejects it

            cv::imwrite(negOutDir + "/hard_" + stem + "_" + std::to_string(localIdx) + ".png", patch);
            localIdx++;
            saved++;
        }
    }
    std::cout << "hard negatives mined: " << saved << " into " << negOutDir << '\n';
}
