#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

std::vector<cv::Rect> colorProposals(const cv::Mat& imgBGR);

double candidateRecall(const std::vector<cv::Rect>& proposals, const std::vector<cv::Rect>& groundTruth, double iouThr = 0.3);
