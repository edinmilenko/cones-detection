//Author: Nicola Ranzolin
#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

enum class AnnotationGeometry { Rectangle, Bitmap };

struct Annotation {
    AnnotationGeometry geometry = AnnotationGeometry::Rectangle;
    cv::Point topLeft;
    cv::Point bottomRight;
    cv::Mat mask;
    cv::Point origin;
    std::string label;
};


std::vector<Annotation> readAnnotations(const std::string& annotationPath);

std::string testSetDir();

void dataLoader();

void csvDatasetMaker(std::string datasetDir);

