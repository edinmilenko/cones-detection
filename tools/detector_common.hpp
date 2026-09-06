#pragma once
// Code shared by the ad-hoc tools in tools/ (not wired into CMakeLists, does not touch
// patches.cpp/color_proposals.cpp/hog.cpp): candidate generation starting from the
// color blobs of colorProposals(), aspect ratio normalization to the shape of kPatchSize,
// IoU e NMS.
#include "color_proposals.hpp"
#include "hog.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <vector>

namespace detector {

inline const double kAspect = static_cast<double>(kPatchSize.width) / kPatchSize.height;

// a color blob almost always has an arbitrary shape (often just the orange or yellow band of the
// cone, not the whole cone): squashing it straight to kPatchSize introduces a distortion that the
// classifier, trained on clean crops of the GT boxes, no longer reads as a cone (checked: the
// recall downstream of the classifier, on candidates with IoU>=0.3 against a real cone, goes up if
// the box is brought back to the training aspect ratio first).
inline std::vector<cv::Rect> normalizeAspect(const std::vector<cv::Rect>& blobs, cv::Size imgSize, double scale = 1.0){
    std::vector<cv::Rect> out;
    for(const auto& b : blobs){
        double cx = b.x + b.width / 2.0;
        double cy = b.y + b.height / 2.0;
        double baseH = std::max(static_cast<double>(b.height), b.width / kAspect);
        double nh = baseH * scale;
        double nw = nh * kAspect;
        int nx = static_cast<int>(std::lround(cx - nw / 2.0));
        int ny = static_cast<int>(std::lround(cy - nh / 2.0));
        int nwi = static_cast<int>(std::lround(nw));
        int nhi = static_cast<int>(std::lround(nh));
        nx = std::max(0, nx);
        ny = std::max(0, ny);
        nwi = std::min(nwi, imgSize.width - nx);
        nhi = std::min(nhi, imgSize.height - ny);
        if(nwi >= 4 && nhi >= 4) out.push_back(cv::Rect(nx, ny, nwi, nhi));
    }
    return out;
}

inline std::vector<cv::Rect> proposeCandidates(const cv::Mat& imgBGR){
    std::vector<cv::Rect> blobs = colorProposals(imgBGR);
    return normalizeAspect(blobs, imgBGR.size(), 1.0);
}

inline std::vector<int> nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, double iouThr = 0.4){
    std::vector<int> order(boxes.size());
    for(size_t i = 0; i < order.size(); i++) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return scores[a] > scores[b]; });

    std::vector<int> kept;
    for(int idx : order){
        bool overlaps = false;
        for(int k : kept){
            if(calculateIoU(boxes[idx], boxes[k]) > iouThr){ overlaps = true; break; }
        }
        if(!overlaps) kept.push_back(idx);
    }
    return kept;
}

} // namespace detector
