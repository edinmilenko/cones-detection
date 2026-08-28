#include "segmentation.hpp"

// K-means++ -> Largest Component -> Convex Hull
cv::Mat segmentCone(const cv::Mat& roi, int predictedClass)
{
    if (roi.empty()) return cv::Mat::zeros(1, 1, CV_8UC1);

    cv::Mat hsv_roi;
    cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);

    cv::Mat segm_roi = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);

    // 1. Color thresholds (hints for K-Means)
    switch(predictedClass)
    {
        case 1: // Yellow
            cv::inRange(hsv_roi, cv::Scalar(15, 80, 50), cv::Scalar(40, 255, 255), segm_roi);
            break;
        case 2: // Blue
            cv::inRange(hsv_roi, cv::Scalar(95, 85, 40), cv::Scalar(130, 255, 255), segm_roi);
            break;
        case 3:
        case 4: // Orange
            cv::inRange(hsv_roi, cv::Scalar(0, 60, 50), cv::Scalar(15, 255, 255), segm_roi);
            break;
    }

    cv::morphologyEx(segm_roi, segm_roi, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    
    cv::Mat maskWhite;
    cv::inRange(hsv_roi, cv::Scalar(0, 0, 150), cv::Scalar(179, 50, 255), maskWhite);

    if (cv::countNonZero(segm_roi) == 0) {
        return cv::Mat::zeros(roi.size(), CV_8UC1);
    }

    // 2. Prepare 1D data for K-Means
    cv::Mat samples = hsv_roi.reshape(1, hsv_roi.total());
    samples.convertTo(samples, CV_32F);

    // 3. K-Means clustering (K=3: Background, Cone Color, White Stripes)
    int K = 3;
    cv::Mat labels, centers;
    cv::kmeans(samples, K, labels, 
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 
               3, cv::KMEANS_PP_CENTERS, centers);

    labels = labels.reshape(1, roi.rows);

    // 4. Vote for cone and stripe clusters
    int colorVotes[3] = {0, 0, 0};
    int whiteVotes[3] = {0, 0, 0};

    for (int y = 0; y < roi.rows; y++) {
        for (int x = 0; x < roi.cols; x++) {
            int cluster_idx = labels.at<int>(y, x);
            if (segm_roi.at<uchar>(y, x) > 0) colorVotes[cluster_idx]++;
            if (maskWhite.at<uchar>(y, x) > 0) whiteVotes[cluster_idx]++;
        }
    }

    int bestColorCluster = 0, maxColorVotes = -1;
    int bestWhiteCluster = 0, maxWhiteVotes = -1;

    for (int i = 0; i < K; i++) {
        if (colorVotes[i] > maxColorVotes) {
            maxColorVotes = colorVotes[i];
            bestColorCluster = i;
        }
        if (whiteVotes[i] > maxWhiteVotes) {
            maxWhiteVotes = whiteVotes[i];
            bestWhiteCluster = i;
        }
    }

    // 5. Build initial mask
    cv::Mat final_mask = cv::Mat::zeros(roi.size(), CV_8UC1);
    for (int y = 0; y < roi.rows; y++) {
        for (int x = 0; x < roi.cols; x++) {
            int cluster_idx = labels.at<int>(y, x);
            if (cluster_idx == bestColorCluster || (cluster_idx == bestWhiteCluster && maxWhiteVotes > 15)) {
                final_mask.at<uchar>(y, x) = 255;
            }
        }
    }

    // 6. Close small gaps (vertical kernel)
    cv::morphologyEx(final_mask, final_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5)));

    // 7. Keep largest connected component (NMS equivalent)
    cv::Mat labeledImage, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(final_mask, labeledImage, stats, centroids, 8, CV_32S);
    
    if (nLabels > 1) { 
        int maxArea = 0;
        int maxLabel = 1;
        
        for (int i = 1; i < nLabels; i++) { // Ignore background (label 0)
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area > maxArea) {
                maxArea = area;
                maxLabel = i;
            }
        }
        
        // Extract largest blob
        final_mask = (labeledImage == maxLabel);
        final_mask.convertTo(final_mask, CV_8UC1, 255.0);

        // 8. Apply Convex Hull for solid shape
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(final_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            std::vector<cv::Point> hull;
            cv::convexHull(contours[0], hull);

            // Redraw convex shape
            final_mask = cv::Mat::zeros(final_mask.size(), CV_8UC1);
            cv::fillConvexPoly(final_mask, hull, cv::Scalar(255));
        }
    }

    return final_mask;
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