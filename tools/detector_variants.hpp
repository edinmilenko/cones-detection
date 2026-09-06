#pragma once
// Parametric version of detectCones(), to try stage 2 threshold, NMS IoU and padding without
// rebuilding src/. Same logic as src/detector.cpp: if a combination wins, the constants there are
// the ones to change.
#include "color_proposals.hpp"
#include "hog.hpp"
#include "utils.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

inline std::vector<cv::Rect> detectConesTuned(const cv::Mat& bgr, std::vector<float>* outScores,
                                              float stage1Thr, float stage2Thr, double nmsIou,
                                              double padTop, double padBottom, double padSide){
    static cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load("../data/svm_hog_sgd_proposal.yml");
    static cv::Ptr<cv::ml::SVM> stage2 = cv::ml::SVM::load("../data/svm_stage2_rbf_balanced.yml");

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for(const cv::Rect& candidate : colorProposals(bgr))
    {
        cv::Mat patch;
        const int interp = candidate.height > kPatchSize.height ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(bgr(candidate), patch, kPatchSize, 0, 0, interp);   // colour: describePatch needs it

        if(rawScore(stage1, patch) <= stage1Thr) continue;

        const float score = rbfScore(stage2, patch);
        if(score <= stage2Thr) continue;

        boxes.push_back(candidate);
        scores.push_back(score);
    }

    std::vector<cv::Rect> kept;
    for(int idx : nonMaxSuppression(boxes, scores, nmsIou))
    {
        const cv::Rect& box = boxes[idx];
        const int extraSide = (int)std::lround(box.width * padSide);
        const int extraTop = (int)std::lround(box.height * padTop);
        const int extraBottom = (int)std::lround(box.height * padBottom);
        const int x = std::max(0, box.x - extraSide);
        const int y = std::max(0, box.y - extraTop);
        const int x2 = std::min(bgr.cols, box.x + box.width + extraSide);
        const int y2 = std::min(bgr.rows, box.y + box.height + extraBottom);
        kept.push_back(cv::Rect(cv::Point(x, y), cv::Point(x2, y2)));
        if(outScores) outScores->push_back(scores[idx]);
    }
    return kept;
}
