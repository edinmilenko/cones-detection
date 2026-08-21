#include "classifier.hpp"

ConeClass classifyOrangeCone(const cv::Mat& hsv_roi)
{
    cv::Scalar minimum(0, 0, 150);
    cv::Scalar maximum(179, 60, 255);

    cv::Mat whiteMask;
    cv::inRange(hsv_roi, minimum, maximum, whiteMask);
    // Applying closing for noise reduction and have more compact regions
    cv::morphologyEx(whiteMask, whiteMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3)));
    
    // FindContours will find one or two blobs, whether it is a small or big cone. We store the contours in vector "contours" as point coordinates
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(whiteMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int numberOfWhiteStrides = 0;
    double thresholdArea = hsv_roi.total() * 0.03;

    for (const auto& contour : contours)
    {
        if (cv::contourArea(contour) > thresholdArea)
        {
            numberOfWhiteStrides++;
        }
    }
    
    if (numberOfWhiteStrides >= 2)
    {
        return ConeClass::BIG_ORANGE;
    }
    else
    {
        return ConeClass::SMALL_ORANGE;
    }
}

std::vector<int> classifier(const cv::Mat& image, const std::vector<cv::Rect>& bboxes)
{
    std::vector<int> labels;
    // Number of vector we can store at most, since the length of bboxes is the number of cones detected in the image
    labels.reserve(bboxes.size());
    for(const auto& box : bboxes)
    {
        cv::Mat roi = image(box); 

        // Conversion from RGB to HSV to manipulate easily color ranges
        cv::Mat hsv_roi;
        cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);

        // For each Scalar we have on slot for Hue (which ranges from 0-179), Saturation (which ranges from 0-255) and Value (which ranges from 0-255)
        cv::Scalar lowerBoundYellow(25, 150, 100);
        cv::Scalar upperBoundYellow(35, 255, 255);
        cv::Scalar lowerBoundOrange(5, 150, 100);
        cv::Scalar upperBoundOrange(24, 255, 255);
        cv::Scalar lowerBoundBlue(90, 150, 100);
        cv::Scalar upperBoundBlue(130, 255, 255);

        // Create the binary mask filtering keeping only pixel if they are eqaul to the color we are analyzing
        // To filter we check every pixel to be in between lower bound and upper bound
        cv::Mat maskYellow;
        cv::Mat maskOrange;
        cv::Mat maskBlue;
        cv::inRange(hsv_roi, lowerBoundYellow, upperBoundYellow, maskYellow);
        cv::inRange(hsv_roi, lowerBoundOrange, upperBoundOrange, maskOrange);
        cv::inRange(hsv_roi, lowerBoundBlue, upperBoundBlue, maskBlue);

        //Count the number of pixel of each color to classify the cone
        int countYellow = cv::countNonZero(maskYellow);
        int countOrange = cv::countNonZero(maskOrange);
        int countBlue = cv::countNonZero(maskBlue);

        if (countYellow > countBlue && countYellow > countOrange)
        {
            labels.push_back(static_cast<int>(ConeClass::YELLOW));
        }
        else if (countBlue > countYellow && countBlue > countOrange)
        {
            labels.push_back(static_cast<int>(ConeClass::BLUE));
        }
        else
        {
            ConeClass orangeType = classifyOrangeCone(hsv_roi);
            labels.push_back(static_cast<int>(orangeType));
        }
    }
    return labels;
}
