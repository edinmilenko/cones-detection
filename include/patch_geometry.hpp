#pragma once
#include <opencv2/core/types.hpp>

// dimensione standard delle patch (positive, negative e finestra HOG).
// unica fonte di verita', condivisa tra patches.cpp (ridimensiona) e hog.cpp (descrive).
inline const cv::Size kPatchSize(16, 24);
