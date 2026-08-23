#include "classifier.hpp"

ConeClass classifyOrangeCone(const cv::Mat& hsv_roi)
{
    cv::Scalar lowerBoundWhite(0, 0, 150);
    cv::Scalar upperBoundWhite(179, 100, 255);

    cv::Mat whiteMask;

    cv::inRange(hsv_roi, lowerBoundWhite, upperBoundWhite, whiteMask);
    // Applying closing for noise reduction and have more compact regions
    cv::morphologyEx(whiteMask, whiteMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    
    // FindContours will find one or two blobs, whether it is a small or big cone. We store the contours in vector "contours" as point coordinates
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(whiteMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int numberOfWhiteStrides = 0;
    double thresholdArea = hsv_roi.total() * 0.008;

    for (const auto& contour : contours)
    {
        // 1. Check the area first to immediately discard small noise
        if (cv::contourArea(contour) > thresholdArea)
        {
            // 2. If the area is large enough, compute the bounding box and Aspect Ratio
            cv::Rect contourBox = cv::boundingRect(contour);
            int width = contourBox.width;
            int height = contourBox.height;
            float ratio = static_cast<float>(width) / height;

            // 3. Verify that the shape is horizontal (width > height) to exclude vertical reflections
            if (ratio > 0.6f)
            {
                numberOfWhiteStrides++;
            }
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
        cv::Scalar lowerBoundYellow(16, 100, 50);
        cv::Scalar upperBoundYellow(40, 255, 255);
        cv::Scalar lowerBoundBlack(0, 0, 0);
        cv::Scalar upperBoundBlack(179, 50, 50);
        cv::Scalar lowerBoundOrange(0, 60, 50);
        cv::Scalar upperBoundOrange(15, 255, 255);
        cv::Scalar lowerBoundBlue(90, 50, 40);
        cv::Scalar upperBoundBlue(130, 255, 255);
        
        // Create the binary mask filtering keeping only pixel if they are eqaul to the color we are analyzing
        // To filter we check every pixel to be in between lower bound and upper bound
        // We combine yellow and black to combine some edge case where the yelow cone is cut-off from the scene and the brightness is bad
        cv::Mat maskYellow;
        cv::Mat maskBlack;
        cv::Mat maskOrange;
        cv::Mat maskBlue;

        cv::inRange(hsv_roi, lowerBoundYellow, upperBoundYellow, maskYellow);
        cv::inRange(hsv_roi, lowerBoundBlack, upperBoundBlack, maskBlack);
        cv::inRange(hsv_roi, lowerBoundOrange, upperBoundOrange, maskOrange);
        cv::inRange(hsv_roi, lowerBoundBlue, upperBoundBlue, maskBlue);

        //Count the number of pixel of each color to classify the cone
        int countYellow = cv::countNonZero(maskYellow);
        int countBlack = cv::countNonZero(maskBlack);
        int orangeConeScore = cv::countNonZero(maskOrange);
        int blueConeScore = cv::countNonZero(maskBlue);
        int yellowConeScore = countYellow;
        int totalPixels = hsv_roi.total();
        
        // Add black pixels to yellow score if yellow dominates or if black covers >10% of the area (saves shadowed/faded yellow cones)
        if (countYellow > orangeConeScore || countBlack > (totalPixels * 0.10)) 
        {
            yellowConeScore += countBlack;
        }

        // Require the dominant color to cover at least 2% of the bounding box to prevent far/small cones from being discarded as UNKNOWN
        int maxScore = std::max({yellowConeScore, blueConeScore, orangeConeScore});
        int minColorThreshold = totalPixels * 0.02;

        
        // We classify the cone
        // If no color reaches 5% threshold then it is unknown
        if (maxScore < minColorThreshold) 
        {
            labels.push_back(static_cast<int>(ConeClass::UNKNOWN));
        }
        else if (yellowConeScore > blueConeScore && yellowConeScore > orangeConeScore)
        {
            labels.push_back(static_cast<int>(ConeClass::YELLOW));
        }
        else if (blueConeScore > yellowConeScore && blueConeScore > orangeConeScore)
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
