#include "color_proposals.hpp"
#include "hog.hpp"
#include <algorithm>
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
    for(int i = 0; i < 3; i++)
    {
        e[i] = std::pow(m[i], 1.0/p);
        if(e[i] < 1e-6)
        {
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
    cv::inRange(hsv, cv::Scalar(100, 80, 60), cv::Scalar(130, 255, 255), blue);
    cv::inRange(hsv, cv::Scalar(  5, 80, 60), cv::Scalar( 20, 255, 255), orange);
    cv::inRange(hsv, cv::Scalar( 20, 80, 60), cv::Scalar( 35, 255, 255), yellow);

    //if an area is either blue, orange or yellow keep it
    cv::bitwise_or(blue, orange, mask);
    cv::bitwise_or(mask, yellow, mask);
    return mask;
}

//per blob, generate boxes at a few fixed heights centered on the centroid: a blob's own
//bounding box is often the wrong size (small/far cones are just a fragment of the cone)
static const std::vector<double> kProposalHeights = {10, 18, 30, 50, 85, 140, 220}; //tuned on GT height percentiles
static const int kMinBlobAreaPx = 3;
static const double kAspect = static_cast<double>(kPatchSize.width) / kPatchSize.height; // w/h

static std::vector<cv::Rect> centeredMultiScale(const cv::Rect& blob, cv::Size imgSize){
    std::vector<cv::Rect> out;
    double cx = blob.x + blob.width / 2.0;
    double cy = blob.y + blob.height / 2.0;
    for(double h : kProposalHeights)
    {
        double w = h * kAspect;
        int nx = static_cast<int>(std::lround(cx - w / 2.0));
        int ny = static_cast<int>(std::lround(cy - h / 2.0));
        int nw = static_cast<int>(std::lround(w));
        int nh = static_cast<int>(std::lround(h));
        nx = std::max(0, nx);
        ny = std::max(0, ny);
        nw = std::min(nw, imgSize.width - nx);
        nh = std::min(nh, imgSize.height - ny);
        if(nw >= 4 && nh >= 4)
        {
            out.push_back(cv::Rect(nx, ny, nw, nh));
        }
    }
    return out;
}

static std::vector<cv::Rect> findCandidateBoxes(const cv::Mat& mask, cv::Size imgSize){
    //find connected components in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //a real cone never exceeds 1% of the image area: discard huge blobs (e.g. grass with
    //the same color as the cone, merged after the mask)
    const double maxAreaFrac = 0.01;
    double maxArea = maxAreaFrac * mask.rows * mask.cols;

    std::vector<cv::Rect> candidates;
    for(auto& elem : contours)
    {
        cv::Rect r = cv::boundingRect(elem);
        if(r.area() > maxArea) continue;
        if(r.area() < kMinBlobAreaPx) continue;
        auto generated = centeredMultiScale(r, imgSize);
        candidates.insert(candidates.end(), generated.begin(), generated.end());
    }
    return candidates;
}

std::vector<cv::Rect> colorProposals(const cv::Mat& imgBGR){
    //white balance first
    cv::Mat correctedImg = whiteBalance(imgBGR, 6.0);

    //find mask
    cv::Mat mask = colorMask(correctedImg);

    //find the candidates for cones
    std::vector<cv::Rect> candidates = findCandidateBoxes(mask, imgBGR.size());
    return candidates;
}

//calculates the recall of the chosen candidates
double candidateRecall(const std::vector<cv::Rect>& proposals, const std::vector<cv::Rect>& groundTruth, double iouThr){
    if(groundTruth.empty())
    {
        return -1.0; // nothing to measure
    }
    int covered = 0;
    for(const cv::Rect& gt : groundTruth)
    {
        for(const cv::Rect& p : proposals)
        {
            if(calculateIoU(p, gt) >= iouThr)
            {
                ++covered;
                break;
            }
        }
    }
    return double(covered) / groundTruth.size();
}
