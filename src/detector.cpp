//Author: Nicola Ranzolin
#include "detector.hpp"
#include "classifier.hpp"
#include "dataset.hpp"
#include "color_proposals.hpp"
#include "hog.hpp"
#include "segmentation.hpp"
#include "utils.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <stdexcept>

//thresholds found by trial and error
static const float STAGE1_THR = 15.0f;  //cheap filter, lets a lot through
static const float STAGE2_THR = 0.1f;
static const double NMS_IOU = 0.1;      //low, otherwise close cones become one box

//the box only grows downwards: the colour blob sits higher than the cone
static const double PAD_TOP = 0.0;
static const double PAD_BOTTOM = 0.4;
static const double PAD_SIDE = 0.1;


//the two models of the cascade
static void checkModelFile(const std::string& path){
    if(!std::filesystem::exists(path))
    {
        throw std::runtime_error("model not found: " + path + "\nrun main from inside the build folder, or train the models again with: main train");
    }
}

static cv::Ptr<cv::ml::SVMSGD> loadStage1(){
    const std::string path = "../data/svm_hog_sgd_proposal.yml";
    checkModelFile(path);
    return cv::ml::SVMSGD::load(path);
}

static cv::Ptr<cv::ml::SVM> loadStage2(){
    const std::string path = "../data/svm_stage2_rbf_balanced.yml";
    checkModelFile(path);
    return cv::ml::SVM::load(path);
}

// Grow the box before segmenting it, and clip it so it stays inside the image
static cv::Rect padBox(cv::Rect box, cv::Size imgSize){
    int extraSide = (int)std::lround(box.width * PAD_SIDE);
    int extraTop = (int)std::lround(box.height * PAD_TOP);
    int extraBottom = (int)std::lround(box.height * PAD_BOTTOM);

    int x1 = std::max(0, box.x - extraSide);
    int y1 = std::max(0, box.y - extraTop);
    int x2 = std::min(imgSize.width, box.x + box.width + extraSide);
    int y2 = std::min(imgSize.height, box.y + box.height + extraBottom);
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

//the mask follows the cone better than the box above, so the final box is taken from the mask
static cv::Rect tightenToMask(const cv::Mat& mask, cv::Rect roiBox, cv::Size imgSize){
    cv::Rect bounds = cv::boundingRect(mask);
    if(bounds.area() <= 0) return roiBox;   // nothing segmented: keep the detector's box

    // the mask is in ROI coordinates, we move it back to the image
    cv::Rect tightened(bounds.x + roiBox.x, bounds.y + roiBox.y, bounds.width, bounds.height);
    return tightened & cv::Rect(0, 0, imgSize.width, imgSize.height);
}

std::vector<cv::Rect> detectCones(const cv::Mat& bgr, std::vector<float>* outScores){
    //loaded once, the first time we detect something
    static cv::Ptr<cv::ml::SVMSGD> stage1 = loadStage1();
    static cv::Ptr<cv::ml::SVM> stage2 = loadStage2();

    // every window the colour mask suggests, a few tens of thousands per image
    std::vector<cv::Rect> candidates = colorProposals(bgr);
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for(size_t i = 0; i < candidates.size(); i++)
    {
        cv::Rect candidate = candidates[i];
        // resize the candidate to the size the descriptor expects
        cv::Mat patch;
        int interp = cv::INTER_LINEAR;
        if(candidate.height > PATCH_SIZE.height)
        {
            interp = cv::INTER_AREA;
        }
        cv::resize(bgr(candidate), patch, PATCH_SIZE, 0, 0, interp);

        // first stage
        if(rawScore(stage1, patch) <= STAGE1_THR) continue;

        // second stage: only what got here is worth a real decision
        float score = rbfScore(stage2, patch);
        if(score <= STAGE2_THR) continue;

        boxes.push_back(candidate);
        scores.push_back(score);
    }

    // the same cone is found many times, we keep the best box of each group
    std::vector<int> kept = nonMaxSuppression(boxes, scores, NMS_IOU);
    std::vector<cv::Rect> result;
    for(size_t i = 0; i < kept.size(); i++)
    {
        result.push_back(padBox(boxes[kept[i]], bgr.size()));
        if(outScores) outScores->push_back(scores[kept[i]]);
    }
    return result;
}

// Detection, then a class and a mask for every cone found
std::vector<Cone> detectAndSegment(const cv::Mat& bgr){
    std::vector<float> scores;
    std::vector<cv::Rect> rois = detectCones(bgr, &scores);
    std::vector<int> labels = classifier(bgr, rois);

    std::vector<Cone> cones;
    for(size_t i = 0; i < rois.size(); i++)
    {
        Cone cone;
        cone.roi = rois[i];
        //always the original image, so a cone is never segmented on top of another one
        cone.mask = segmentCone(bgr(rois[i]), labels[i]);
        // the box we report is the one from the mask, not the padded one
        cone.box = tightenToMask(cone.mask, rois[i], bgr.size());
        cone.cls = labels[i];
        cone.score = scores[i];
        cones.push_back(cone);
    }
    return cones;
}

// The images of a folder, sorted so two runs go in the same order
static std::vector<std::string> listImages(const std::string& dir){
    std::vector<std::string> images;
    for(const auto& entry : std::filesystem::directory_iterator(dir))
    {
        // lowercase the extension, some files come with .JPG
        std::string ext = entry.path().extension().string();
        for(size_t i = 0; i < ext.size(); i++)
        {
            ext[i] = (char)std::tolower(ext[i]);
        }

        if(ext == ".jpg" || ext == ".jpeg" || ext == ".png")
        {
            images.push_back(entry.path().filename().string());
        }
    }
    std::sort(images.begin(), images.end());
    return images;
}

//draws every cone on a copy of the image: mask, box and class name, all in the class colour
static cv::Mat drawCones(const cv::Mat& bgr, const std::vector<Cone>& cones){
    cv::Mat out = bgr.clone();
    for(size_t i = 0; i < cones.size(); i++)
    {
        cv::Scalar color = classColor(cones[i].cls);

        // the mask is painted over the image, half transparent
        cv::Mat roi = out(cones[i].roi);
        cv::Mat overlay = roi.clone();
        overlay.setTo(color, cones[i].mask);
        cv::addWeighted(roi, 0.4, overlay, 0.6, 0.0, roi);

        cv::rectangle(out, cones[i].box, color, 2);
        // the name goes above the box, unless the box is already at the top
        cv::Point labelAt(cones[i].box.x, std::max(15, cones[i].box.y - 5));
        cv::putText(out, className(cones[i].cls), labelAt, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv::LINE_AA);
    }
    return out;
}

void detectPipeline(const std::string& imagePath, const std::string& outPath){
    cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
    if(bgr.empty())
    {
        throw std::runtime_error("detectPipeline: cannot read " + imagePath);
    }

    std::vector<Cone> cones = detectAndSegment(bgr);
    cv::imwrite(outPath, drawCones(bgr, cones));
    std::cout << "found " << cones.size() << " cones, image saved in " << outPath << '\n';
}

//same drawing as detect, but on every image of the test set: it only reads the trained models,
//it does not train anything
void detectTestSet(const std::string& testDir, const std::string& outDir){
    const std::string imgDir = testDir + "/img";
    std::vector<std::string> images = listImages(imgDir);
    if(images.empty())
    {
        throw std::runtime_error("detectTestSet: no image in " + imgDir);
    }

    std::filesystem::create_directories(outDir);

    int done = 0;
    for(size_t i = 0; i < images.size(); i++)
    {
        cv::Mat bgr = cv::imread(imgDir + "/" + images[i], cv::IMREAD_COLOR);
        if(bgr.empty())
        {
            std::cerr << "image not found: " << images[i] << '\n';
            continue;
        }

        std::vector<Cone> cones = detectAndSegment(bgr);
        const std::string outPath = outDir + "/" + images[i];
        cv::imwrite(outPath, drawCones(bgr, cones));
        done++;
        std::cout << "[" << done << "/" << images.size() << "] " << images[i] << ": " << cones.size() << " cones\n";
    }

    std::cout << "saved " << done << " images in " << outDir << '\n';
}

// The names Supervisely uses in the annotations, turned into our four classes
static int classFromLabel(const std::string& label){
    if(label == "seg_yellow_cone") return (int)ConeClass::YELLOW;
    if(label == "seg_blue_cone") return (int)ConeClass::BLUE;
    if(label == "seg_orange_cone") return (int)ConeClass::SMALL_ORANGE;
    if(label == "seg_large_orange_cone") return (int)ConeClass::BIG_ORANGE;
    return 0;   // seg_unknown_cone and anything else: outside the four classes
}

// The three metrics of the specification, measured on the whole test set
void evaluatePipeline(const std::string& testDir){
    const std::string imgDir = testDir + "/img";
    const std::string annDir = testDir + "/ann";
    std::vector<std::string> images = listImages(imgDir);
    if(images.empty())
    {
        throw std::runtime_error("evaluatePipeline: no image in " + imgDir);
    }

    // the four classes we are scored on, unknown cones stay out of the count
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

        // where this image starts in the two lists, so we can score it on its own
        const size_t firstGt = ground.size(), firstPred = preds.size();

        // ground truth: one box and one mask per cone, from the Supervisely bitmaps
        cv::Mat gtLabels = cv::Mat::zeros(bgr.size(), CV_8UC1);
        std::vector<Annotation> annotations = readAnnotations(annDir + "/" + imgName + ".json");
        int conesHere = 0;

        for(size_t a = 0; a < annotations.size(); a++)
        {
            Annotation ann = annotations[a];
            // only the cones annotated with a mask, the rest we cannot score
            if(ann.geometry != AnnotationGeometry::Bitmap) continue;

            int cls = classFromLabel(ann.label);
            if(cls == 0) continue;

            // the true box is the extent of the mask, it is not annotated on its own
            cv::Rect content = cv::boundingRect(ann.mask);
            if(content.area() <= 0) continue;

            cv::Rect box = (content + ann.origin) & cv::Rect(0, 0, bgr.cols, bgr.rows);
            // a cone this small cannot be matched fairly, we leave it out
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
        // an image with no annotated cone tells us nothing, we skip it
        if(conesHere == 0) continue;

        // one canvas per image with the class of every pixel, for the mIoU
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

        // the same three metrics, on this image alone
        PixelIoU imageIou;
        imageIou.add(predLabels, gtLabels, classes);
        std::vector<GT> imageGround(ground.begin() + firstGt, ground.end());
        std::vector<Pred> imagePreds(preds.begin() + firstPred, preds.end());
        std::cout << imgName << "  F1 " << macroF1(imagePreds, imageGround, classes)
                << "  mAP " << meanAveragePrecision(imagePreds, imageGround, classes)
                << "  mIoU " << imageIou.mean(classes) << '\n';

        // pixels are summed over the whole test set, not averaged image by image
        pixelIou.add(predLabels, gtLabels, classes);
        usedImages++;
    }

    std::cout << "\n=== " << usedImages << " images, " << ground.size() << " cones ===\n";
    std::cout << "macro F1 (IoU>=0.5) : " << macroF1(preds, ground, classes) << '\n';
    std::cout << "mAP@0.5:0.95        : " << meanAveragePrecision(preds, ground, classes) << '\n';
    std::cout << "mIoU                : " << pixelIou.mean(classes) << '\n';
}
