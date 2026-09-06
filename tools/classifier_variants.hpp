#pragma once
// Experimental variants of classifier(), kept here rather than in src/classifier.cpp until one
// wins on eval_pipeline. Variant 0 is the one in src/, called directly.
//
// What is under test: in src/ orange is the else branch, the class that takes everything which is
// not clearly yellow or blue. On a false positive of the detection (asphalt, grass, a shadow) the
// answer is always "small orange", and indeed every orange FP measured is not a cone. Here orange
// has to win on pixel count like the other classes, and on top of that a box without enough pixels
// of any color can be rejected outright (label 0).
#include "classifier.hpp"
#include <opencv2/opencv.hpp>

namespace clsvar {

struct Params {
    bool orangeIsDefault = true;   // false = orange must win on pixels, not by exclusion
    double minColorFrac = 0.0;     // if > 0, a box below this fraction of colored pixels is rejected
};

inline std::vector<int> classify(const cv::Mat& image, const std::vector<cv::Rect>& bboxes, const Params& p){
    std::vector<int> labels;
    labels.reserve(bboxes.size());
    for(const auto& box : bboxes)
    {
        cv::Mat roi = image(box);

        cv::Mat hsv_roi;
        cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv_roi, hsv_channels);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(hsv_channels[2], hsv_channels[2]);
        cv::merge(hsv_channels, hsv_roi);

        cv::Scalar lowerBoundYellow(16, 100, 50);
        cv::Scalar upperBoundYellow(40, 255, 255);
        cv::Scalar lowerBoundBlack(0, 0, 0);
        cv::Scalar upperBoundBlack(179, 50, 50);
        cv::Scalar lowerBoundOrange(0, 60, 50);
        cv::Scalar upperBoundOrange(15, 255, 255);
        cv::Scalar lowerBoundBlue(95, 85, 40);
        cv::Scalar upperBoundBlue(130, 255, 255);

        cv::Mat maskYellow, maskBlack, maskOrange, maskBlue;
        cv::inRange(hsv_roi, lowerBoundYellow, upperBoundYellow, maskYellow);
        cv::inRange(hsv_roi, lowerBoundBlack, upperBoundBlack, maskBlack);
        cv::inRange(hsv_roi, lowerBoundOrange, upperBoundOrange, maskOrange);
        cv::inRange(hsv_roi, lowerBoundBlue, upperBoundBlue, maskBlue);

        int w = hsv_roi.cols;
        int h = hsv_roi.rows;
        cv::Mat coneShapeMask = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
        std::vector<cv::Point> pts = {
            cv::Point(w * 0.25, 0),
            cv::Point(w * 0.75, 0),
            cv::Point(w, h),
            cv::Point(0, h)
        };
        cv::fillPoly(coneShapeMask, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(255));

        cv::bitwise_and(maskYellow, coneShapeMask, maskYellow);
        cv::bitwise_and(maskBlack, coneShapeMask, maskBlack);
        cv::bitwise_and(maskOrange, coneShapeMask, maskOrange);
        cv::bitwise_and(maskBlue, coneShapeMask, maskBlue);

        int countYellow = cv::countNonZero(maskYellow);
        int countBlack = cv::countNonZero(maskBlack);
        int orangeConeScore = cv::countNonZero(maskOrange);
        int blueConeScore = cv::countNonZero(maskBlue);
        int yellowConeScore = countYellow;
        int totalPixels = hsv_roi.total();

        if (countYellow > orangeConeScore || countBlack > (totalPixels * 0.10))
        {
            yellowConeScore += countBlack;
        }

        const int best = std::max({yellowConeScore, blueConeScore, orangeConeScore});

        // a box with no colored pixels at all is not a cone: rejecting it drops false positives
        // from all four classes, not just orange
        if (p.minColorFrac > 0 && best < totalPixels * p.minColorFrac)
        {
            labels.push_back(0);
            continue;
        }

        if (p.orangeIsDefault)
        {
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
                labels.push_back(static_cast<int>(classifyOrangeCone(hsv_roi)));
            }
        }
        else
        {
            if (yellowConeScore == best) labels.push_back(static_cast<int>(ConeClass::YELLOW));
            else if (blueConeScore == best) labels.push_back(static_cast<int>(ConeClass::BLUE));
            else labels.push_back(static_cast<int>(classifyOrangeCone(hsv_roi)));
        }
    }
    return labels;
}

} // namespace clsvar

// No variant beat the one in src/: this file stays as a record of two dead ends. The numbers are
// the macro F1 from eval_pipeline with the padding of the time (0.15/0.4/0.2), where variant 0
// scored 0.388.
inline std::vector<int> classifierVariant(const cv::Mat& image, const std::vector<cv::Rect>& bboxes, int variant){
    clsvar::Params p;
    switch(variant)
    {
        case 0:  return classifier(image, bboxes);       // the one in src/: 0.388
        case 1:  break;                                  // parametric rewrite, identical: 0.388

        // 1) taking the else branch away from orange changes nothing (0.388): on false positives
        //    orange really does win on pixel count, because its H 0-15 band also covers soil and
        //    reddish asphalt
        case 2:  p.orangeIsDefault = false; break;

        // 2) rejecting boxes poor in color always makes it worse (0.386 / 0.361 / 0.239 / 0.158):
        //    inside a padded box even a real cone covers little area, so the fraction of colored
        //    pixels does not separate cones from false positives
        case 3:  p.minColorFrac = 0.02; break;
        case 4:  p.minColorFrac = 0.05; break;
        case 5:  p.minColorFrac = 0.10; break;
        case 6:  p.minColorFrac = 0.15; break;
    }
    return clsvar::classify(image, bboxes, p);
}
