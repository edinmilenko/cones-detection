#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

enum class AnnotationGeometry { Rectangle, Bitmap };

// either a rectangle, or a bitmap mask placed at origin
struct Annotation {
    AnnotationGeometry geometry = AnnotationGeometry::Rectangle;
    cv::Point topLeft;
    cv::Point bottomRight;
    cv::Mat mask;
    cv::Point origin;
    std::string label;
};

// reads one Supervisely annotation file. Needed outside dataset.cpp to score the segmentation,
// which is the only place the bitmap masks are used.
std::vector<Annotation> readAnnotations(const std::string& annotationPath);

// merges fsoco_bounding_boxes_train and fsoco_segmentation_train into dataset/, renumbering the
// image/annotation pairs they share (1.jpg, 1.json, 2.jpg, 2.json, ...).
void dataLoader();

// pulls the bounding boxes ("rectangle" annotations) out of dataset/ into dataset.csv
// (img,x1,y1,x2,y2).
void csvDatasetMaker(std::string datasetDir);

// draws the annotations (rectangles or bitmap masks) of dataset/ into labeled/, to look at them.
void labeler();
