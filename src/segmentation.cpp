//Author: Edin Milenko
#include "segmentation.hpp"

//under this height there are too few pixels for kmeans and the morphology
static const int MIN_HEIGHT = 80;

//the white stripes cut the cone in separate blobs, we close them back before taking a component.
static const double CLOSE_FRACTION = 0.30;

//colour threshold -> kmeans -> biggest connected component
cv::Mat segmentCone(const cv::Mat& roi, int predictedClass)
{
    if(roi.empty()) return cv::Mat::zeros(1, 1, CV_8UC1);

    // a far cone is a handful of pixels, we blow it up before working on it
    cv::Mat work = roi;
    if(roi.rows < MIN_HEIGHT)
    {
        const double scale = static_cast<double>(MIN_HEIGHT) / roi.rows;
        cv::resize(roi, work, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }

    // HSV again, the colour of the cone is a range of hue and the brightness gets in the way
    cv::Mat hsvRoi;
    cv::cvtColor(work, hsvRoi, cv::COLOR_BGR2HSV);

    //colour mask of the predicted class, used as a hint for kmeans
    cv::Mat colorMask = cv::Mat::zeros(hsvRoi.size(), CV_8UC1);
    if(predictedClass == (int)ConeClass::YELLOW)
    {
        cv::inRange(hsvRoi, cv::Scalar(15, 80, 50), cv::Scalar(40, 255, 255), colorMask);
    }
    else if(predictedClass == (int)ConeClass::BLUE)
    {
        cv::inRange(hsvRoi, cv::Scalar(95, 85, 40), cv::Scalar(130, 255, 255), colorMask);
    }
    else if(predictedClass == (int)ConeClass::SMALL_ORANGE || predictedClass == (int)ConeClass::BIG_ORANGE)
    {
        //the two orange cones have the same colour, only the white stripes tell them apart
        cv::inRange(hsvRoi, cv::Scalar(0, 60, 50), cv::Scalar(15, 255, 255), colorMask);
    }

    // opening to drop the isolated pixels the threshold picks up around the cone
    cv::morphologyEx(colorMask, colorMask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    // nothing of the right colour in the box: there is nothing to segment
    if(cv::countNonZero(colorMask) == 0)
    {
        return cv::Mat::zeros(roi.size(), CV_8UC1);
    }

    // kmeans wants one row per pixel and floats, so we flatten the ROI
    cv::Mat samples = hsvRoi.reshape(1, hsvRoi.total());
    samples.convertTo(samples, CV_32F);

    //K=3: background, cone colour, white stripes
    const int K = 3;
    cv::Mat labels, centers;
    cv::kmeans(samples, K, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    // back to the shape of the ROI, so we can read the cluster of each pixel
    labels = labels.reshape(1, work.rows);

    // each cluster gets a vote for every pixel of the colour mask that falls inside it
    std::vector<int> colorVotes(K, 0);
    for(int y = 0; y < work.rows; y++)
    {
        for(int x = 0; x < work.cols; x++)
        {
            if(colorMask.at<uchar>(y, x) > 0) colorVotes[labels.at<int>(y, x)]++;
        }
    }

    // the most voted cluster is the cone
    int bestCluster = 0, maxVotes = -1;
    for(int i = 0; i < K; i++)
    {
        if(colorVotes[i] > maxVotes)
        {
            maxVotes = colorVotes[i];
            bestCluster = i;
        }
    }

    // the mask is that cluster and nothing else
    cv::Mat mask = cv::Mat::zeros(work.size(), CV_8UC1);
    for(int y = 0; y < work.rows; y++)
    {
        for(int x = 0; x < work.cols; x++)
        {
            if(labels.at<int>(y, x) == bestCluster) mask.at<uchar>(y, x) = 255;
        }
    }

    // a tall thin closing, so it glues the bands of the cone without eating its sides
    const int closeH = std::max(5, static_cast<int>(work.rows * CLOSE_FRACTION));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, closeH)));

    // of everything left we keep one blob, the cone is a single piece
    cv::Mat labeledImage, stats, centroids;
    const int nLabels = cv::connectedComponentsWithStats(mask, labeledImage, stats, centroids, 8, CV_32S);

    if(nLabels > 1)
    {
        int maxArea = 0;
        int maxLabel = 1;
        for(int i = 1; i < nLabels; i++) // label 0 is the background
        {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if(area > maxArea)
            {
                maxArea = area;
                maxLabel = i;
            }
        }

        // only the biggest one survives
        mask = (labeledImage == maxLabel);
        mask.convertTo(mask, CV_8UC1, 255.0);

        //filling the outer contour closes the stripes that are left as holes.
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(mask, contours, -1, cv::Scalar(255), cv::FILLED);
    }

    // if we enlarged the ROI at the start, the mask goes back to the real size
    if(mask.size() != roi.size())
    {
        cv::resize(mask, mask, roi.size(), 0, 0, cv::INTER_NEAREST);
    }

    return mask;
}
