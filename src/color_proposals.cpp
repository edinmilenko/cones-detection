#include "color_proposals.hpp"
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "utils.hpp"

//method to implement whiteBalance for uneven illumination effects
static cv::Mat whiteBalance(const cv::Mat& imgBGR, double p){
    cv::Mat imgF;
    imgBGR.convertTo(imgF, CV_32F);

    //calculate for each channel the avg of the sum of the intensities ^ 6
    cv::Mat expImg;
    cv::pow(imgF, p, expImg);
    cv::Scalar m = cv::mean(expImg);

    //calculate for each channel the avg ^ 1/6
    double e[3];
    for(int i = 0; i < 3; i++){
        e[i] = std::pow(m[i], 1.0/p);
        if(e[i] < 1e-6){
            e[i] = 1e-6;
        }
    }

    double gainMean = (e[0] + e[1] + e[2]) / 3;
    cv::Scalar gain(gainMean / e[0], gainMean / e[1], gainMean / e[2]);
    cv::multiply(imgF, gain, imgF);

    cv::Mat out;
    imgF.convertTo(out, CV_8U);
    return out;
}

//filter the areas that could actually have a cone by color
static cv::Mat colorMask(const cv::Mat& imgBGR ){
    cv::Mat hsv;
    cv::cvtColor(imgBGR, hsv, cv::COLOR_BGR2HSV);

    cv::Mat blue, orange, yellow, mask;
    cv::inRange(hsv, cv::Scalar(100, 60, 40), cv::Scalar(130, 255, 255), blue);
    cv::inRange(hsv, cv::Scalar(  5, 60, 40), cv::Scalar( 20, 255, 255), orange);
    cv::inRange(hsv, cv::Scalar( 20, 60, 40), cv::Scalar( 35, 255, 255), yellow);

    //if an area is either blue, orange or yellow keep it
    cv::bitwise_or(blue, orange, mask);
    cv::bitwise_or(mask, yellow, mask);
    return mask;
}

//apply opening and closing to remove noise and to close the space between the two parts of the cones
static cv::Mat morphCleaning(const cv::Mat& mask){
    cv::Mat elemOpening = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)); //arbitrary, to be tuned
    cv::Mat elemClosing = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,7));

    //Opening before closing to remove the noisy single points first, the merge the two parts of the cone
    cv::Mat newMask;
    cv::morphologyEx(mask, newMask, cv::MORPH_OPEN, elemOpening);
    cv::morphologyEx(newMask, newMask, cv::MORPH_CLOSE, elemClosing);

    return newMask;
}

static std::vector<cv::Rect> findCandidateBoxes(const cv::Mat& mask){
    //find connected components in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> candidates;
    for(auto& elem : contours){
        cv::Rect r = cv::boundingRect(elem);
        candidates.push_back(r);
    }
    return candidates;
}


std::vector<cv::Rect> colorProposals(const cv::Mat& imgBGR){
    //white balance first
    cv::Mat correctedImg = whiteBalance(imgBGR, 6.0);

    //find mask and filter it
    cv::Mat mask = colorMask(correctedImg);
    cv::Mat newMask = morphCleaning(mask);
    cv::imwrite("../debug/mask.png", newMask);

    //find the candidates for cones
    std::vector<cv::Rect> candidates = findCandidateBoxes(newMask);
    return candidates;
}

//calculates the recall of the chosen candidates
double candidateRecall(const std::vector<cv::Rect>& proposals, const std::vector<cv::Rect>& groundTruth, double iouThr) {
    if (groundTruth.empty()) return -1.0;   // nothing to measure
    int covered = 0;
    for (const cv::Rect& gt : groundTruth) {
        for (const cv::Rect& p : proposals) {
            if (calculateIoU(p, gt) >= iouThr) { ++covered; break; }
        }
    }
    return double(covered) / groundTruth.size();
}
