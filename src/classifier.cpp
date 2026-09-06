#include "classifier.hpp"

ConeClass classifyOrangeCone(const cv::Mat& hsv_roi)
{
    cv::Scalar lowerBoundWhite(0, 0, 150);
    cv::Scalar upperBoundWhite(179, 100, 255);

    cv::Mat whiteMask;

    cv::inRange(hsv_roi, lowerBoundWhite, upperBoundWhite, whiteMask);
    // Applying opening for noise reduction and have more compact regions
    cv::morphologyEx(whiteMask, whiteMask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1)));
    
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
            if (ratio > 1.4f)
            {
                numberOfWhiteStrides++;
            }
        }
    }
    
    // Calculate the aspect ratio of the entire ROI bounding box (Height / Width)
    // A high ratio (> 2.0) means the bounding box is tall and narrow (typical of big cones) 
    // This is added because the cone might be very distant or very bright due to sun or external light
    float roiRatio = static_cast<float>(hsv_roi.rows) / static_cast<float>(hsv_roi.cols);

    if (numberOfWhiteStrides >= 2 || roiRatio > 2.0f)
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

        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv_roi, hsv_channels); // Split into Hue, Saturation, and Value channels

        // Apply CLAHE to the Value channel (index 2) to brighten shadows
        // without washing out already bright colors like yellow or orange tahnks to the threshold 2.0
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(hsv_channels[2], hsv_channels[2]); 

        // Merge the modified channels back into the HSV image
        cv::merge(hsv_channels, hsv_roi);

        // For each Scalar we have on slot for Hue (which ranges from 0-179), Saturation (which ranges from 0-255) and Value (which ranges from 0-255)
        cv::Scalar lowerBoundYellow(16, 100, 50);
        cv::Scalar upperBoundYellow(40, 255, 255);
        cv::Scalar lowerBoundBlack(0, 0, 0);
        cv::Scalar upperBoundBlack(179, 50, 50);
        cv::Scalar lowerBoundOrange(0, 60, 50);
        cv::Scalar upperBoundOrange(15, 255, 255);
        cv::Scalar lowerBoundBlue(95, 85, 40);
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

        // Cut off top corners
        int w = hsv_roi.cols;
        int h = hsv_roi.rows;

        // Create a black mask of the same size as the bounding box
        cv::Mat coneShapeMask = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
        
        // Define the 4 vertices of the trapezoid (inset top corners by 25%)
        std::vector<cv::Point> pts = {
            cv::Point(w * 0.25, 0),  
            cv::Point(w * 0.75, 0),  
            cv::Point(w, h),
            cv::Point(0, h)          
        };
        
        // Draw the white trapezoid on the mask
        cv::fillPoly(coneShapeMask, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(255));

        // Keep only the pixels inside the trapezoid, discarding the background corners and maintaining the same colors
        cv::bitwise_and(maskYellow, coneShapeMask, maskYellow);
        cv::bitwise_and(maskBlack, coneShapeMask, maskBlack);
        cv::bitwise_and(maskOrange, coneShapeMask, maskOrange);
        cv::bitwise_and(maskBlue, coneShapeMask, maskBlue);

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

        // We classify the cone
        if (yellowConeScore > blueConeScore && yellowConeScore > orangeConeScore)
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

std::string className(int predictedClass)
{
    switch (predictedClass)
    {
        case static_cast<int>(ConeClass::YELLOW): return "YELLOW";
        case static_cast<int>(ConeClass::BLUE): return "BLUE";
        case static_cast<int>(ConeClass::SMALL_ORANGE): return "S_ORANGE";
        case static_cast<int>(ConeClass::BIG_ORANGE): return "B_ORANGE";
        default: return "UNKNOWN";
    }
}

// BGR, so blue and orange look swapped
cv::Scalar classColor(int predictedClass)
{
    switch (predictedClass)
    {
        case static_cast<int>(ConeClass::YELLOW): return cv::Scalar(0, 255, 255);
        case static_cast<int>(ConeClass::BLUE): return cv::Scalar(255, 0, 0);
        case static_cast<int>(ConeClass::SMALL_ORANGE): return cv::Scalar(0, 165, 255);
        case static_cast<int>(ConeClass::BIG_ORANGE): return cv::Scalar(0, 100, 255);
        default: return cv::Scalar(255, 255, 255);
    }
}
