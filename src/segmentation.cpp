#include "segmentation.hpp"

cv::Mat segmentCone(const cv::Mat& roi, int predictedClass)
{
    cv::Mat hsv_roi;
    cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
    
    // Enhance contrast in V channel without washing out bright colors
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv_roi, hsv_channels); 
    //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    //clahe->apply(hsv_channels[2], hsv_channels[2]); 
    cv::merge(hsv_channels, hsv_roi);

    cv::Mat segm_roi = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);

    // Initial color segmentation
    switch(predictedClass)
    {
        case 1: { // Yellow (includes black to capture base/shadows)
            cv::Scalar lowerBoundYellow(15, 80, 50);
            cv::Scalar upperBoundYellow(40, 255, 255);
            //cv::Scalar lowerBoundBlack(0, 0, 0), upperBoundBlack(179, 50, 50);
            //cv::Mat maskYellow, maskBlack;
            cv::inRange(hsv_roi, lowerBoundYellow, upperBoundYellow, segm_roi);
            //cv::inRange(hsv_roi, lowerBoundBlack, upperBoundBlack, maskBlack);
            //segm_roi = maskYellow | maskBlack;
            break;
        }
        case 2: { // Blue
            cv::Scalar lowerBoundBlue(95, 85, 40), upperBoundBlue(130, 255, 255);
            cv::inRange(hsv_roi, lowerBoundBlue, upperBoundBlue, segm_roi);
            break;
        }
        case 3:
        case 4: { // Orange (Small & Big)
            cv::Scalar lowerBoundOrange(0, 60, 50), upperBoundOrange(15, 255, 255);
            cv::inRange(hsv_roi, lowerBoundOrange, upperBoundOrange, segm_roi);
            break;
        }
    }

    // Clean noise, then dilate to enclose the whole cone and its borders
    cv::morphologyEx(segm_roi, segm_roi, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    cv::Mat sure_backg;
    cv::dilate(segm_roi, sure_backg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 2); 

    cv::Mat maskWhite;
    cv::inRange(hsv_roi, cv::Scalar(0, 0, 150), cv::Scalar(179, 50, 255), maskWhite);
    
    sure_backg = sure_backg | maskWhite;

    // Use Distance Transform to find the absolute center of the cone
    cv::Mat sure_foreg; 
    cv::distanceTransform(segm_roi, sure_foreg, cv::DIST_L2, 5); 

    double minVal, maxVal;
    cv::minMaxLoc(sure_foreg, &minVal, &maxVal); 
    
    // Safety check: if no color is detected at all, return empty mask to prevent crash
    if (maxVal == 0) return segm_roi;

    // Threshold at 20% of max distance to isolate the core, then convert to 8-bit
    cv::threshold(sure_foreg, sure_foreg, 0.2 * maxVal, 255, cv::THRESH_BINARY);
    sure_foreg.convertTo(sure_foreg, CV_8U);

    // The uncertain border area is the difference between background and core
    cv::Mat unknown_region = sure_backg - sure_foreg;

    // Setup markers: 1 for background, 2 for cone core, 0 for unknown borders
    cv::Mat markers;
    cv::connectedComponents(sure_foreg, markers, 8, CV_32S);
    markers += 1; 
    markers.setTo(0, unknown_region);
    
    cv::watershed(roi, markers);
    segm_roi = (markers > 1);
    
    cv::morphologyEx(segm_roi, segm_roi, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    return segm_roi;
}


double calculateIoU(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask)
{
    cv::Mat intersectionMask, unionMask;
    cv::bitwise_and(predictedMask, groundTruthMask, intersectionMask);
    cv::bitwise_or(predictedMask, groundTruthMask, unionMask);
    
    double intersectionArea = cv::countNonZero(intersectionMask);
    double unionArea = cv::countNonZero(unionMask);

    if (unionArea == 0.0) {
        return 0.0; 
    }
    return intersectionArea / unionArea;
}