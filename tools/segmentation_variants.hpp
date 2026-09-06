#pragma once
// Experimental variants of segmentCone(), kept here rather than in src/segmentation.cpp until one
// wins on eval_segmentation. Variant 0 is the one in src/, called directly; the others share the
// same structure (color threshold -> kmeans -> cluster vote -> largest connected component ->
// convex hull) with the suspect steps turned into parameters.
#include "segmentation.hpp"
#include <opencv2/opencv.hpp>

namespace segvar {

struct Params {
    bool useKmeans = true;          // false = use the color mask directly
    int k = 3;                      // number of kmeans clusters
    bool useWhite = true;           // also take the cluster of the white stripes
    int whiteMinVotes = 15;
    int whiteMinValue = 150;        // lowest V of the "white" mask
    int whiteMaxSat = 50;           // highest S of the "white" mask
    bool whiteNearColor = false;    // count white votes only around the color blob
    double whiteGrow = 1.0;         // how far to widen that area, in fractions of the blob
    bool convexHull = true;
    int closeH = 5;                 // closing kernel height, joins the blobs split by the stripes
    double closeHFrac = 0.0;        // if > 0, kernel height as a fraction of the ROI height
    double closeWFrac = 0.0;        // if > 0, same for the width (default: a fixed 3 pixels)
    bool fillHoles = false;         // fill the holes inside the chosen component
    int closeHMax = 0;              // if > 0, cap on the closing kernel (small cones swell up)
    bool colorLoose = false;        // looser S/V color thresholds, for small or shadowed cones
    int minRoiHeight = 0;           // if > 0, upscale the ROI to this height before segmenting
    int openAfter = 0;              // if > 0, opening after the closing: cuts off the thin streaks
                                    // the vertical closing drags below the cone
    double openAfterFrac = 0.0;     // if > 0, same kernel as a fraction of the ROI width instead
                                    // (on a near cone the streaks are wider than 3px)
    bool keepNearestBlob = false;   // keep the component nearest the center, not the largest
    double maxColorFrac = 1.0;      // drop the color cluster if it covers more than this of the ROI
};

inline cv::Mat segment(const cv::Mat& roiIn, int predictedClass, const Params& p){
    if (roiIn.empty()) return cv::Mat::zeros(1, 1, CV_8UC1);

    // on a 15px cone kmeans and the morphology have very few pixels to work on: upscale, segment,
    // then bring the mask back to the original size
    cv::Mat roi = roiIn;
    if (p.minRoiHeight > 0 && roiIn.rows < p.minRoiHeight) {
        const double scale = (double)p.minRoiHeight / roiIn.rows;
        cv::resize(roiIn, roi, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }

    cv::Mat hsv_roi;
    cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);

    const int minS = p.colorLoose ? 45 : 0;   // 0 = keep the original per-class thresholds
    cv::Mat segm_roi = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
    switch(predictedClass)
    {
        case 1:
            cv::inRange(hsv_roi, cv::Scalar(15, p.colorLoose ? minS : 80, p.colorLoose ? 40 : 50), cv::Scalar(40, 255, 255), segm_roi);
            break;
        case 2:
            cv::inRange(hsv_roi, cv::Scalar(95, p.colorLoose ? minS : 85, p.colorLoose ? 30 : 40), cv::Scalar(130, 255, 255), segm_roi);
            break;
        case 3:
        case 4:
            cv::inRange(hsv_roi, cv::Scalar(0, p.colorLoose ? minS : 60, p.colorLoose ? 40 : 50), cv::Scalar(15, 255, 255), segm_roi);
            break;
    }

    cv::morphologyEx(segm_roi, segm_roi, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    cv::Mat maskWhite;
    cv::inRange(hsv_roi, cv::Scalar(0, 0, p.whiteMinValue), cv::Scalar(179, p.whiteMaxSat, 255), maskWhite);

    if (cv::countNonZero(segm_roi) == 0) {
        return cv::Mat::zeros(roiIn.size(), CV_8UC1);
    }

    // the stripes sit on the cone, so near the colored pixels: outside that area "white" is lit
    // asphalt or sky, which inside a padded box is almost always the majority
    if (p.whiteNearColor) {
        cv::Rect colorBox = cv::boundingRect(segm_roi);
        cv::Mat near = cv::Mat::zeros(maskWhite.size(), CV_8UC1);
        const int growX = std::max(2, (int)(colorBox.width * 0.25 * p.whiteGrow));
        const int growY = std::max(2, (int)(colorBox.height * p.whiteGrow));
        cv::Rect grown(colorBox.x - growX, colorBox.y - growY, colorBox.width + 2 * growX, colorBox.height + 2 * growY);
        near(grown & cv::Rect(0, 0, near.cols, near.rows)).setTo(255);
        cv::bitwise_and(maskWhite, near, maskWhite);
    }

    cv::Mat final_mask;
    if (!p.useKmeans) {
        final_mask = segm_roi.clone();
    } else {

    cv::Mat samples = hsv_roi.reshape(1, hsv_roi.total());
    samples.convertTo(samples, CV_32F);

    cv::Mat labels, centers;
    cv::kmeans(samples, p.k, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    labels = labels.reshape(1, roi.rows);

    std::vector<int> colorVotes(p.k, 0), whiteVotes(p.k, 0), size(p.k, 0);
    for (int y = 0; y < roi.rows; y++) {
        for (int x = 0; x < roi.cols; x++) {
            int cluster_idx = labels.at<int>(y, x);
            size[cluster_idx]++;
            if (segm_roi.at<uchar>(y, x) > 0) colorVotes[cluster_idx]++;
            if (maskWhite.at<uchar>(y, x) > 0) whiteVotes[cluster_idx]++;
        }
    }

    const int totalPixels = roi.rows * roi.cols;

    int bestColorCluster = -1, maxColorVotes = -1;
    for (int i = 0; i < p.k; i++) {
        if (size[i] > totalPixels * p.maxColorFrac) continue;   // too big to be the cone
        if (colorVotes[i] > maxColorVotes) {
            maxColorVotes = colorVotes[i];
            bestColorCluster = i;
        }
    }
    if (bestColorCluster < 0) return cv::Mat::zeros(roiIn.size(), CV_8UC1);

    int bestWhiteCluster = -1, maxWhiteVotes = -1;
    if (p.useWhite) {
        for (int i = 0; i < p.k; i++) {
            if (i == bestColorCluster) continue;
            if (size[i] > totalPixels * p.maxColorFrac) continue;
            if (whiteVotes[i] > maxWhiteVotes) {
                maxWhiteVotes = whiteVotes[i];
                bestWhiteCluster = i;
            }
        }
    }

    final_mask = cv::Mat::zeros(roi.size(), CV_8UC1);
    for (int y = 0; y < roi.rows; y++) {
        for (int x = 0; x < roi.cols; x++) {
            int cluster_idx = labels.at<int>(y, x);
            if (cluster_idx == bestColorCluster || (cluster_idx == bestWhiteCluster && maxWhiteVotes > p.whiteMinVotes)) {
                final_mask.at<uchar>(y, x) = 255;
            }
        }
    }

    } // end of the kmeans branch

    int closeH = p.closeHFrac > 0 ? std::max(5, (int)(roi.rows * p.closeHFrac)) : p.closeH;
    if (p.closeHMax > 0) closeH = std::min(closeH, p.closeHMax);
    const int closeW = p.closeWFrac > 0 ? std::max(3, (int)(roi.cols * p.closeWFrac)) : 3;
    cv::morphologyEx(final_mask, final_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(closeW, closeH)));

    const int openK = p.openAfterFrac > 0 ? std::max(3, (int)(roi.cols * p.openAfterFrac)) : p.openAfter;
    if (openK > 0) {
        cv::morphologyEx(final_mask, final_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(openK, openK)));
    }

    cv::Mat labeledImage, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(final_mask, labeledImage, stats, centroids, 8, CV_32S);

    if (nLabels > 1) {
        int best = 1;
        if (p.keepNearestBlob) {
            // the cone is centered in the box by construction, the largest component may be background
            const double cx = roi.cols / 2.0, cy = roi.rows / 2.0;
            double bestDist = 1e18;
            for (int i = 1; i < nLabels; i++) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) < 8) continue;
                const double dx = centroids.at<double>(i, 0) - cx;
                const double dy = centroids.at<double>(i, 1) - cy;
                const double d = dx * dx + dy * dy;
                if (d < bestDist) { bestDist = d; best = i; }
            }
        } else {
            int maxArea = 0;
            for (int i = 1; i < nLabels; i++) {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area > maxArea) { maxArea = area; best = i; }
            }
        }

        final_mask = (labeledImage == best);
        final_mask.convertTo(final_mask, CV_8UC1, 255.0);

        if (p.convexHull) {
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(final_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (!contours.empty()) {
                std::vector<cv::Point> hull;
                cv::convexHull(contours[0], hull);
                final_mask = cv::Mat::zeros(final_mask.size(), CV_8UC1);
                cv::fillConvexPoly(final_mask, hull, cv::Scalar(255));
            }
        }
        else if (p.fillHoles) {
            // filling the outer contour closes the stripes left as holes, without swelling the
            // mask the way a convex hull would
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(final_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(final_mask, contours, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    if (final_mask.size() != roiIn.size()) {
        cv::resize(final_mask, final_mask, roiIn.size(), 0, 0, cv::INTER_NEAREST);
    }
    return final_mask;
}

} // namespace segvar

// The numbers in the comments are the per-cone mIoU eval_segmentation gave when each variant was
// tried (padding of the time: 0.15/0.4/0.2, so they compare to each other but not to what src/
// scores now on a tighter box). They are here to avoid walking the same roads twice: the list
// covers the directions explored, not only the ones that won.
inline cv::Mat segmentVariant(const cv::Mat& roi, int predictedClass, int variant){
    segvar::Params p;
    switch(variant)
    {
        case 0:  return segmentCone(roi, predictedClass);   // the one in src/, the best so far
        case 1:  break;                                     // parametric rewrite, = variant 0 back then: 0.377

        // ablation, one lever at a time starting from the original version
        case 2:  p.useWhite = false; break;                 // 0.536  <- the biggest lever
        case 3:  p.convexHull = false; break;               // 0.356  (worse on its own)
        case 10: p.useKmeans = false; break;                // 0.436  (kmeans does earn its place)
        case 20: p.useWhite = false; p.closeHFrac = 0.25; break;   // 0.592  proportional closing

        // directions that did not pay off
        case 4:  p.whiteNearColor = true; break;                   // 0.378
        case 5:  p.k = 4; break;                                   // 0.371
        case 6:  p.maxColorFrac = 0.5; break;                      // 0.520
        case 7:  p.keepNearestBlob = true; break;                  // 0.369
        case 43: p.useWhite = false; p.convexHull = false; p.closeHFrac = 0.30; p.fillHoles = true; p.colorLoose = true; break;   // 0.610

        // the road that won, one piece at a time
        case 30: p.useWhite = false; p.convexHull = false; p.closeHFrac = 0.30; break;                       // 0.623
        case 38: p.useWhite = false; p.convexHull = false; p.closeHFrac = 0.30; p.fillHoles = true; break;   // 0.628
        case 45: p.useWhite = false; p.convexHull = false; p.closeHFrac = 0.30; p.fillHoles = true; p.minRoiHeight = 80; break;   // 0.639
        case 51: p.useWhite = false; p.convexHull = false; p.closeHFrac = 0.30; p.fillHoles = true; p.minRoiHeight = 80; p.openAfter = 3; break;   // 0.639, = src/

        // ablation of variant 51, to check every piece earns its place
        case 54: p.useWhite = false; p.convexHull = false; p.closeHFrac = 0.30; p.minRoiHeight = 80; break;   // 0.635 without fillHoles
        case 55: p.useWhite = false; p.closeHFrac = 0.30; p.fillHoles = true; p.minRoiHeight = 80; break;     // 0.606 with the convex hull
        case 56: p.convexHull = false; p.closeHFrac = 0.30; p.fillHoles = true; p.minRoiHeight = 80; break;   // 0.414 with the white cluster
        case 57: p.useWhite = false; p.convexHull = false; p.fillHoles = true; p.minRoiHeight = 80; break;    // 0.482 with the 5px closing
    }
    return segvar::segment(roi, predictedClass, p);
}
