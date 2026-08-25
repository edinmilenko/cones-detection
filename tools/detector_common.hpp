#pragma once
// Codice condiviso dai tool ad-hoc in tools/ (non collegato a CMakeLists, non tocca
// patches.cpp/color_proposals.cpp/hog.cpp): generazione candidati a partire dai blob
// colore di colorProposals(), normalizzazione dell'aspect ratio alla forma di kPatchSize,
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

// un blob colore ha quasi sempre una forma arbitraria (spesso solo la fascia arancione/gialla
// del cono, non tutto il cono): schiacciarlo direttamente a kPatchSize introduce una
// distorsione che il classificatore, addestrato su crop puliti dei box GT, non riconosce piu'
// come cono (verificato: la recall a valle del classificatore su candidati con IoU>=0.3 con un
// vero cono sale se il box viene prima riportato all'aspect ratio di training).
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
