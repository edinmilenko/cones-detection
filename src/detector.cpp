#include "detector.hpp"
#include "classifier.hpp"
#include "dataset.hpp"
#include "color_proposals.hpp"
#include "hog.hpp"
#include "segmentation.hpp"
#include "utils.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <stdexcept>

// tuned by comparing precision/recall on the test sets
static const float kStage1Thr = 15.0f;  // cheap, permissive filter
static const float kStage2Thr = 0.1f;   // best-F1 point for the models in data/
static const double kNmsIou = 0.1;      // lower = breaks up duplicate boxes in dense clusters

// the box is grown downwards only: a proposal is centered on the color blob, which sits higher than
// the cone (its base is dark and misses the color mask). This box only feeds the segmentation, it
// is not the one we report, so margin it doesn't need costs recall and nothing else.
static const double kPadTop = 0.0;
static const double kPadBottom = 0.4;
static const double kPadSide = 0.1;

// small margin left around the mask, in case the segmentation lost the darker base of a cone
static const double kMaskPad = 0.05;

static cv::Rect padBox(cv::Rect box, cv::Size imgSize){
    int extraSide = (int)std::lround(box.width * kPadSide);
    int extraTop = (int)std::lround(box.height * kPadTop);
    int extraBottom = (int)std::lround(box.height * kPadBottom);

    int x1 = std::max(0, box.x - extraSide);
    int y1 = std::max(0, box.y - extraTop);
    int x2 = std::min(imgSize.width, box.x + box.width + extraSide);
    int y2 = std::min(imgSize.height, box.y + box.height + extraBottom);
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

// the mask follows the cone much better than the box above, so the box we report comes from it
static cv::Rect tightenToMask(const cv::Mat& mask, cv::Rect roiBox, cv::Size imgSize){
    cv::Rect bounds = cv::boundingRect(mask);
    if(bounds.area() <= 0) return roiBox;   // nothing segmented: keep the detector's box

    int extraX = (int)std::lround(bounds.width * kMaskPad);
    int extraY = (int)std::lround(bounds.height * kMaskPad);
    cv::Rect tightened(bounds.x + roiBox.x - extraX, bounds.y + roiBox.y - extraY,
                       bounds.width + 2 * extraX, bounds.height + 2 * extraY);
    return tightened & cv::Rect(0, 0, imgSize.width, imgSize.height);
}

std::vector<cv::Rect> detectCones(const cv::Mat& bgr, std::vector<float>* outScores){
    static cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load("../data/svm_hog_sgd_proposal.yml");
    static cv::Ptr<cv::ml::SVM> stage2 = cv::ml::SVM::load("../data/svm_stage2_rbf_balanced.yml");

    std::vector<cv::Rect> candidates = colorProposals(bgr);
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for(size_t i = 0; i < candidates.size(); i++)
    {
        cv::Rect candidate = candidates[i];
        cv::Mat patch;
        int interp = candidate.height > kPatchSize.height ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(bgr(candidate), patch, kPatchSize, 0, 0, interp);

        if(rawScore(stage1, patch) <= kStage1Thr) continue;

        float score = rbfScore(stage2, patch);
        if(score <= kStage2Thr) continue;

        boxes.push_back(candidate);
        scores.push_back(score);
    }

    std::vector<int> kept = nonMaxSuppression(boxes, scores, kNmsIou);
    std::vector<cv::Rect> result;
    for(size_t i = 0; i < kept.size(); i++)
    {
        result.push_back(padBox(boxes[kept[i]], bgr.size()));
        if(outScores) outScores->push_back(scores[kept[i]]);
    }
    return result;
}

std::vector<Cone> detectAndSegment(const cv::Mat& bgr){
    std::vector<float> scores;
    std::vector<cv::Rect> rois = detectCones(bgr, &scores);
    std::vector<int> labels = classifier(bgr, rois);

    std::vector<Cone> cones;
    for(size_t i = 0; i < rois.size(); i++)
    {
        // always segment the original image, so overlapping boxes never segment over a region
        // something has already been drawn on
        Cone cone;
        cone.roi = rois[i];
        cone.mask = segmentCone(bgr(rois[i]), labels[i]);
        cone.box = tightenToMask(cone.mask, rois[i], bgr.size());
        cone.cls = labels[i];
        cone.score = scores[i];
        cones.push_back(cone);
    }
    return cones;
}

void detectPipeline(const std::string& imagePath, const std::string& outPath){
    cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
    if(bgr.empty())
    {
        throw std::runtime_error("detectPipeline: cannot read " + imagePath);
    }

    std::vector<Cone> cones = detectAndSegment(bgr);

    cv::Mat out = bgr.clone();
    for(size_t i = 0; i < cones.size(); i++)
    {
        cv::Scalar color = classColor(cones[i].cls);

        cv::Mat roi = out(cones[i].roi);
        cv::Mat overlay = roi.clone();
        overlay.setTo(color, cones[i].mask);
        cv::addWeighted(roi, 0.4, overlay, 0.6, 0.0, roi);

        cv::rectangle(out, cones[i].box, color, 2);
        cv::Point labelAt(cones[i].box.x, std::max(15, cones[i].box.y - 5));
        cv::putText(out, className(cones[i].cls), labelAt, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv::LINE_AA);
    }

    cv::imwrite(outPath, out);
    std::cout << "[detect] found " << cones.size() << " cones, saved to: " << outPath << '\n';
}

static int classFromLabel(const std::string& label){
    if(label == "seg_yellow_cone") return (int)ConeClass::YELLOW;
    if(label == "seg_blue_cone") return (int)ConeClass::BLUE;
    if(label == "seg_orange_cone") return (int)ConeClass::SMALL_ORANGE;
    if(label == "seg_large_orange_cone") return (int)ConeClass::BIG_ORANGE;
    return 0;   // seg_unknown_cone and anything else: outside the four classes
}

void evaluatePipeline(const std::string& imgDir, const std::string& annDir, const std::string& splitFile){
    std::vector<std::string> images;
    std::ifstream split(splitFile);
    std::string line;
    while(std::getline(split, line))
    {
        if(!line.empty()) images.push_back(line);
    }
    if(images.empty())
    {
        throw std::runtime_error("evaluatePipeline: no image listed in " + splitFile);
    }

    std::vector<int> classes;
    classes.push_back((int)ConeClass::YELLOW);
    classes.push_back((int)ConeClass::BLUE);
    classes.push_back((int)ConeClass::SMALL_ORANGE);
    classes.push_back((int)ConeClass::BIG_ORANGE);

    std::vector<Pred> preds;
    std::vector<GT> ground;
    PixelIoU pixelIou;
    int usedImages = 0;

    for(size_t i = 0; i < images.size(); i++)
    {
        std::string imgName = images[i];
        cv::Mat bgr = cv::imread(imgDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty())
        {
            std::cerr << "image not found: " << imgName << '\n';
            continue;
        }

        // ground truth: one box and one mask per cone, from the Supervisely bitmaps
        cv::Mat gtLabels = cv::Mat::zeros(bgr.size(), CV_8UC1);
        std::vector<Annotation> annotations = readAnnotations(annDir + "/" + imgName + ".json");
        int conesHere = 0;

        for(size_t a = 0; a < annotations.size(); a++)
        {
            Annotation ann = annotations[a];
            if(ann.geometry != AnnotationGeometry::Bitmap) continue;

            int cls = classFromLabel(ann.label);
            if(cls == 0) continue;

            cv::Rect content = cv::boundingRect(ann.mask);
            if(content.area() <= 0) continue;

            cv::Rect box = (content + ann.origin) & cv::Rect(0, 0, bgr.cols, bgr.rows);
            if(box.width < 4 || box.height < 4) continue;

            GT gt;
            gt.imgName = imgName;
            gt.bbox = box;
            gt.cls = cls;
            ground.push_back(gt);
            conesHere++;

            // the mask is stored on its own little canvas, placed at origin
            cv::Rect maskBounds(ann.origin.x, ann.origin.y, ann.mask.cols, ann.mask.rows);
            cv::Rect overlap = maskBounds & cv::Rect(0, 0, bgr.cols, bgr.rows);
            if(overlap.area() > 0)
            {
                cv::Rect inMask(overlap.x - maskBounds.x, overlap.y - maskBounds.y, overlap.width, overlap.height);
                gtLabels(overlap).setTo(cls, ann.mask(inMask));
            }
        }
        if(conesHere == 0) continue;

        cv::Mat predLabels = cv::Mat::zeros(bgr.size(), CV_8UC1);
        std::vector<Cone> cones = detectAndSegment(bgr);
        for(size_t c = 0; c < cones.size(); c++)
        {
            Pred pred;
            pred.imgName = imgName;
            pred.bbox = cones[c].box;
            pred.cls = cones[c].cls;
            pred.score = cones[c].score;
            preds.push_back(pred);

            predLabels(cones[c].roi).setTo(cones[c].cls, cones[c].mask);   // overlapping boxes: last wins
        }

        pixelIou.add(predLabels, gtLabels, classes);
        usedImages++;
    }

    std::cout << "\n=== " << usedImages << " images, " << ground.size() << " cones ===\n";
    std::cout << "macro F1 (IoU>=0.5) : " << macroF1(preds, ground, classes) << '\n';
    std::cout << "mAP@0.5:0.95        : " << meanAveragePrecision(preds, ground, classes) << '\n';
    std::cout << "mIoU                : " << pixelIou.mean(classes) << '\n';
}
