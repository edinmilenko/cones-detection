//Author: Nicola Ranzolin
#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

std::vector<cv::Rect> detectCones(const cv::Mat& bgr, std::vector<float>* scores = nullptr);

//one cone, as the pipeline sees it
struct Cone {
    cv::Rect roi;    
    cv::Rect box;    
    cv::Mat mask;    
    int cls;
    float score;
};

std::vector<Cone> detectAndSegment(const cv::Mat& bgr);

void detectPipeline(const std::string& imagePath, const std::string& outPath);

void detectTestSet(const std::string& testDir, const std::string& outDir);

void evaluatePipeline(const std::string& testDir);
