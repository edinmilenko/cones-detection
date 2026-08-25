#include "detector.hpp"
#include "color_proposals.hpp"
#include "hog.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <stdexcept>

namespace {
    // tuned by comparing precision/recall on the test sets
    constexpr float kStage1Thr = 15.0f;  // cheap, permissive filter
    constexpr float kStage2Thr = 0.01f;  // best-F1 point
    constexpr double kNmsIou = 0.1;      // lower = breaks up duplicate boxes in dense clusters

    // grown asymmetrically: the cascade's box often cuts off the cone's base, and the next
    // stage (segmentation) can trim excess but can't recover a cut-off cone
    constexpr double kPadTop = 0.15, kPadBottom = 0.4, kPadSide = 0.2;

    cv::Rect padBox(const cv::Rect& box, cv::Size imgSize){
        int extraSide = (int)std::lround(box.width * kPadSide);
        int extraTop = (int)std::lround(box.height * kPadTop);
        int extraBottom = (int)std::lround(box.height * kPadBottom);
        int x = std::max(0, box.x - extraSide);
        int y = std::max(0, box.y - extraTop);
        int x2 = std::min(imgSize.width, box.x + box.width + extraSide);
        int y2 = std::min(imgSize.height, box.y + box.height + extraBottom);
        return cv::Rect(cv::Point(x, y), cv::Point(x2, y2));
    }
}

std::vector<cv::Rect> detectCones(const cv::Mat& bgr){
    static cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load("../data/svm_hog_sgd_proposal.yml");
    static cv::Ptr<cv::ml::SVM> stage2 = cv::ml::SVM::load("../data/svm_stage2_rbf_balanced.yml"); // retrained with equal pos/neg patches

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for(const cv::Rect& candidate : colorProposals(bgr))
    {
        cv::Mat patch;
        int interp = candidate.height > kPatchSize.height ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(gray(candidate), patch, kPatchSize, 0, 0, interp);

        if(rawScore(stage1, patch) <= kStage1Thr) continue;

        float score = rbfScore(stage2, patch);
        if(score <= kStage2Thr) continue;

        boxes.push_back(candidate);
        scores.push_back(score);
    }

    std::vector<cv::Rect> kept;
    for(int idx : nonMaxSuppression(boxes, scores, kNmsIou))
    {
        kept.push_back(padBox(boxes[idx], bgr.size()));
    }
    return kept;
}

void detectPipeline(const std::string& imagePath, const std::string& outPath){
    cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
    if(bgr.empty())
    {
        throw std::runtime_error("detectPipeline: cannot read " + imagePath);
    }

    std::vector<cv::Rect> boxes = detectCones(bgr);
    for(const cv::Rect& box : boxes)
    {
        cv::rectangle(bgr, box, cv::Scalar(0, 0, 255), 2);
    }

    cv::imwrite(outPath, bgr);
    std::cout << "[detect] found " << boxes.size() << " cones, saved to: " << outPath << '\n';
}
