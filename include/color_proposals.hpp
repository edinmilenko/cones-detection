//Author: Alessandro Martini
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::Rect> colorProposals(const cv::Mat& imgBGR);

