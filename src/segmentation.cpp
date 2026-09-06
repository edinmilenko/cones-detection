#include "segmentation.hpp"

// below this height there are too few pixels for kmeans and the morphology to work on
static const int kMinHeight = 80;

// the white stripes cut the cone into separate blobs, closed back together before picking a
// component. Proportional because the stripes of a near cone are much thicker.
static const double kCloseFrac = 0.30;

// color threshold -> kmeans -> largest connected component
cv::Mat segmentCone(const cv::Mat& roi, int predictedClass)
{
    if(roi.empty()) return cv::Mat::zeros(1, 1, CV_8UC1);

    cv::Mat work = roi;
    if(roi.rows < kMinHeight)
    {
        const double scale = static_cast<double>(kMinHeight) / roi.rows;
        cv::resize(roi, work, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }

    cv::Mat hsvRoi;
    cv::cvtColor(work, hsvRoi, cv::COLOR_BGR2HSV);

    // color mask of the predicted class, used as a hint for kmeans
    cv::Mat colorMask = cv::Mat::zeros(hsvRoi.size(), CV_8UC1);
    switch(predictedClass)
    {
        case 1: // yellow
            cv::inRange(hsvRoi, cv::Scalar(15, 80, 50), cv::Scalar(40, 255, 255), colorMask);
            break;
        case 2: // blue
            cv::inRange(hsvRoi, cv::Scalar(95, 85, 40), cv::Scalar(130, 255, 255), colorMask);
            break;
        case 3:
        case 4: // orange
            cv::inRange(hsvRoi, cv::Scalar(0, 60, 50), cv::Scalar(15, 255, 255), colorMask);
            break;
    }

    cv::morphologyEx(colorMask, colorMask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    if(cv::countNonZero(colorMask) == 0)
    {
        return cv::Mat::zeros(roi.size(), CV_8UC1);
    }

    cv::Mat samples = hsvRoi.reshape(1, hsvRoi.total());
    samples.convertTo(samples, CV_32F);

    // K=3: background, cone color, white stripes
    const int K = 3;
    cv::Mat labels, centers;
    cv::kmeans(samples, K, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    labels = labels.reshape(1, work.rows);

    // only the cone cluster is taken, not the stripe one: the box is mostly background and lit
    // asphalt passes a white threshold too, so voting for it makes the mask fill the whole box.
    // The stripes come back with the closing below.
    std::vector<int> colorVotes(K, 0);
    for(int y = 0; y < work.rows; y++)
    {
        for(int x = 0; x < work.cols; x++)
        {
            if(colorMask.at<uchar>(y, x) > 0) colorVotes[labels.at<int>(y, x)]++;
        }
    }

    int bestCluster = 0, maxVotes = -1;
    for(int i = 0; i < K; i++)
    {
        if(colorVotes[i] > maxVotes)
        {
            maxVotes = colorVotes[i];
            bestCluster = i;
        }
    }

    cv::Mat mask = cv::Mat::zeros(work.size(), CV_8UC1);
    for(int y = 0; y < work.rows; y++)
    {
        for(int x = 0; x < work.cols; x++)
        {
            if(labels.at<int>(y, x) == bestCluster) mask.at<uchar>(y, x) = 255;
        }
    }

    const int closeH = std::max(5, static_cast<int>(work.rows * kCloseFrac));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, closeH)));

    // a tall kernel drags thin streaks below the cone (its shadow), the opening cuts them off
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

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

        mask = (labeledImage == maxLabel);
        mask.convertTo(mask, CV_8UC1, 255.0);

        // filling the outer contour closes the stripes left as holes. A convex hull would too, but
        // it also swallows the background around a cone, which is never really convex.
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(mask, contours, -1, cv::Scalar(255), cv::FILLED);
    }

    if(mask.size() != roi.size())
    {
        cv::resize(mask, mask, roi.size(), 0, 0, cv::INTER_NEAREST);
    }

    return mask;
}
