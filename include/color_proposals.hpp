#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

std::vector<cv::Rect> colorProposals(const cv::Mat& imgBGR);

