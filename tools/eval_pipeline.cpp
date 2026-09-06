// Standalone file, not wired into CMakeLists: the three metrics the assignment asks for, computed
// on the full pipeline as it stands in src/ (detectCones -> classifier -> segmentCone ->
// tightenToMask), over the real external test set.
//
//   1. Macro-averaged F1: F1 worked out per class, then averaged WITHOUT weights, so the rare
//      classes (the orange ones) count as much as yellow and blue.
//   2. mAP@0.5:0.95: AP per class with 101-point interpolation (COCO), averaged over the 10 IoU
//      thresholds from 0.50 to 0.95. It needs a confidence score per box, which detectCones()
//      hands back through its optional parameter.
//   3. mIoU (Jaccard): over mask PIXELS, TP/(TP+FP+FN) accumulated per class across every image,
//      then averaged over the classes. Not the same as the per-cone mean IoU eval_segmentation
//      reports.
//
// It includes detector.cpp because tightenToMask() lives in an anonymous namespace in there.
//
// Ad-hoc build (from the build/ folder):
//   g++ -std=gnu++17 -O2 -I../include -I../tools -I/usr/include/opencv4 \
//     ../tools/eval_pipeline.cpp ../src/color_proposals.cpp ../src/hog.cpp ../src/utils.cpp \
//     ../src/classifier.cpp ../src/segmentation.cpp ../src/dataset.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml -lz \
//     -o eval_pipeline
//   ./eval_pipeline

#include "dataset.hpp"
#include "../src/detector.cpp"
#include "classifier.hpp"
#include "classifier_variants.hpp"
#include "detector_variants.hpp"
#include "segmentation_variants.hpp"
#include "segmentation.hpp"
#include "utils.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {

// classFromLabel comes from src/detector.cpp, included above
const std::vector<int> kClasses = {
    (int)ConeClass::YELLOW, (int)ConeClass::BLUE,
    (int)ConeClass::SMALL_ORANGE, (int)ConeClass::BIG_ORANGE
};

// a detection, plus the image it came from: AP needs them sorted by score across the whole test set
struct Det {
    int image;
    cv::Rect box;
    int cls;
    float score;
};

struct GtBox {
    int image;
    cv::Rect box;
    int cls;
};

// AP of one class at one IoU threshold, with the 101-point interpolation COCO uses
double averagePrecision(std::vector<Det> dets, const std::vector<GtBox>& gts, int cls, double iouThr){
    std::vector<const GtBox*> classGts;
    for(const GtBox& g : gts) if(g.cls == cls) classGts.push_back(&g);
    if(classGts.empty()) return -1.0;   // class not present: left out of the average

    dets.erase(std::remove_if(dets.begin(), dets.end(), [&](const Det& d){ return d.cls != cls; }), dets.end());
    std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b){ return a.score > b.score; });

    std::vector<bool> matched(classGts.size(), false);
    std::vector<double> precisions, recalls;
    long long tp = 0, fp = 0;

    for(const Det& d : dets)
    {
        double bestIou = 0.0;
        int bestJ = -1;
        for(size_t j = 0; j < classGts.size(); j++)
        {
            if(matched[j] || classGts[j]->image != d.image) continue;
            const double iou = calculateIoU(d.box, classGts[j]->box);
            if(iou > bestIou){ bestIou = iou; bestJ = (int)j; }
        }
        if(bestIou >= iouThr){ tp++; matched[bestJ] = true; }
        else fp++;

        precisions.push_back((double)tp / (tp + fp));
        recalls.push_back((double)tp / classGts.size());
    }

    // precision made monotonically decreasing, then sampled at 101 recall levels
    for(int i = (int)precisions.size() - 2; i >= 0; i--)
    {
        precisions[i] = std::max(precisions[i], precisions[i + 1]);
    }

    double sum = 0.0;
    for(int i = 0; i <= 100; i++)
    {
        const double r = i / 100.0;
        double p = 0.0;
        for(size_t k = 0; k < recalls.size(); k++)
        {
            if(recalls[k] >= r){ p = precisions[k]; break; }
        }
        sum += p;
    }
    return sum / 101.0;
}

} // namespace

int main(int argc, char** argv){
    const int clsVariant = argc > 1 ? std::atoi(argv[1]) : 0;
    const double nmsIou = argc > 2 ? std::atof(argv[2]) : kNmsIou;
    const float stage2Thr = argc > 3 ? (float)std::atof(argv[3]) : kStage2Thr;
    const bool showFunnel = argc > 4 && std::atoi(argv[4]) != 0;
    const double padTop = argc > 5 ? std::atof(argv[5]) : kPadTop;
    const double padBottom = argc > 6 ? std::atof(argv[6]) : kPadBottom;
    const double padSide = argc > 7 ? std::atof(argv[7]) : kPadSide;
    const int segVariant = argc > 8 ? std::atoi(argv[8]) : 0;
    const std::string imgDir = "../test_set/segmentation_test/img";
    const std::string annDir = "../test_set/segmentation_test/ann";

    std::vector<std::string> testImgs;
    {
        std::ifstream f("../data/test_real.txt");
        std::string line;
        while(std::getline(f, line)) if(!line.empty()) testImgs.push_back(line);
    }

    std::vector<Det> dets;
    std::vector<GtBox> gts;
    std::map<int, long long> pixTp, pixFp, pixFn;   // for the pixel mIoU
    int nImages = 0;
    long long nRejected = 0;
    std::map<int, long long> propCovered, propTotal, stage1Ok, stage2Ok;

    for(const std::string& imgName : testImgs)
    {
        cv::Mat bgr = cv::imread(imgDir + "/" + imgName, cv::IMREAD_COLOR);
        if(bgr.empty()){ std::cerr << "image not found: " << imgName << "\n"; continue; }

        // ground truth: box and mask per cone, from the Supervisely bitmaps
        cv::Mat gtLabelMap = cv::Mat::zeros(bgr.size(), CV_8UC1);
        int nGtHere = 0;
        for(const Annotation& ann : readAnnotations(annDir + "/" + imgName + ".json"))
        {
            if(ann.geometry != AnnotationGeometry::Bitmap) continue;
            const int cls = classFromLabel(ann.label);
            if(cls == 0) continue;
            const cv::Rect content = cv::boundingRect(ann.mask);
            if(content.area() <= 0) continue;
            const cv::Rect box = (content + ann.origin) & cv::Rect(0, 0, bgr.cols, bgr.rows);
            if(box.width < 4 || box.height < 4) continue;

            gts.push_back({nImages, box, cls});
            nGtHere++;

            const cv::Rect maskBounds(ann.origin.x, ann.origin.y, ann.mask.cols, ann.mask.rows);
            const cv::Rect overlap = maskBounds & cv::Rect(0, 0, bgr.cols, bgr.rows);
            if(overlap.area() > 0)
            {
                const cv::Rect src(overlap.x - maskBounds.x, overlap.y - maskBounds.y, overlap.width, overlap.height);
                gtLabelMap(overlap).setTo(cls, ann.mask(src));
            }
        }
        if(nGtHere == 0) continue;

        // coverage of the color proposals alone, before the cascade: tells "the candidate is never
        // even generated" apart from "the cascade throws it away"
        if(showFunnel){
            const std::vector<cv::Rect> props = colorProposals(bgr);
            static cv::Ptr<cv::ml::SVMSGD> s1 = cv::ml::SVMSGD::load("../data/svm_hog_sgd_proposal.yml");
            static cv::Ptr<cv::ml::SVM> s2 = cv::ml::SVM::load("../data/svm_stage2_rbf_balanced.yml");

            for(const GtBox& g : gts)
            {
                if(g.image != nImages) continue;
                propTotal[g.cls]++;

                bool covered = false, passed1 = false, passed2 = false;
                for(const cv::Rect& p : props)
                {
                    if(calculateIoU(p, g.box) < 0.3) continue;
                    covered = true;

                    cv::Mat patch;
                    const int interp = p.height > kPatchSize.height ? cv::INTER_AREA : cv::INTER_LINEAR;
                    cv::resize(bgr(p), patch, kPatchSize, 0, 0, interp);
                    if(rawScore(s1, patch) <= kStage1Thr) continue;
                    passed1 = true;
                    if(rbfScore(s2, patch) > kStage2Thr){ passed2 = true; break; }
                }
                if(covered) propCovered[g.cls]++;
                if(passed1) stage1Ok[g.cls]++;
                if(passed2) stage2Ok[g.cls]++;
            }
        }

        // pipeline
        std::vector<float> scores;
        const std::vector<cv::Rect> loose = detectConesTuned(bgr, &scores, kStage1Thr, stage2Thr, nmsIou, padTop, padBottom, padSide);
        const std::vector<int> labels = classifierVariant(bgr, loose, clsVariant);

        cv::Mat predLabelMap = cv::Mat::zeros(bgr.size(), CV_8UC1);
        for(size_t i = 0; i < loose.size(); i++)
        {
            if(labels[i] == 0){ nRejected++; continue; }   // rejected by the classifier
            const cv::Mat mask = segmentVariant(bgr(loose[i]), labels[i], segVariant);
            dets.push_back({nImages, tightenToMask(mask, loose[i], bgr.size()), labels[i], scores[i]});
            predLabelMap(loose[i]).setTo(labels[i], mask);   // on overlapping boxes the last wins
        }

        for(int cls : kClasses)
        {
            const cv::Mat g = (gtLabelMap == cls);
            const cv::Mat p = (predLabelMap == cls);
            cv::Mat inter, uni;
            cv::bitwise_and(g, p, inter);
            cv::bitwise_or(g, p, uni);
            const long long i = cv::countNonZero(inter);
            pixTp[cls] += i;
            pixFp[cls] += cv::countNonZero(p) - i;
            pixFn[cls] += cv::countNonZero(g) - i;
        }

        nImages++;
    }

    std::cout << "=== full pipeline, real test set (" << nImages << " images, "
              << gts.size() << " cones), classifier variant " << clsVariant << " ===\n";
    std::cout << "nms=" << nmsIou << " stage2_thr=" << stage2Thr << " pad=" << padTop << "/" << padBottom << "/" << padSide << "  boxes rejected by the classifier: " << nRejected << "\n\n";

    // --- 1. Macro-averaged F1, at IoU 0.5 ---
    std::cout << "Macro-averaged F1 (IoU>=0.5)\n";
    double macroF1 = 0.0;
    int nClassesF1 = 0;
    for(int cls : kClasses)
    {
        std::vector<const GtBox*> classGts;
        for(const GtBox& g : gts) if(g.cls == cls) classGts.push_back(&g);
        if(classGts.empty()) continue;

        std::vector<Det> classDets;
        for(const Det& d : dets) if(d.cls == cls) classDets.push_back(d);
        std::sort(classDets.begin(), classDets.end(), [](const Det& a, const Det& b){ return a.score > b.score; });

        std::vector<bool> matched(classGts.size(), false);
        long long tp = 0, fp = 0, fpWrongClass = 0, fpNotACone = 0;
        for(const Det& d : classDets)
        {
            double bestIou = 0.0;
            int bestJ = -1;
            for(size_t j = 0; j < classGts.size(); j++)
            {
                if(matched[j] || classGts[j]->image != d.image) continue;
                const double iou = calculateIoU(d.box, classGts[j]->box);
                if(iou > bestIou){ bestIou = iou; bestJ = (int)j; }
            }
            if(bestIou >= 0.5){ tp++; matched[bestJ] = true; }
            else
            {
                fp++;
                // a FP can be a real cone with the wrong class, or something that is not a cone at
                // all: two different problems, fixed in two different ways
                double bestAny = 0.0;
                for(const GtBox& g : gts)
                {
                    if(g.image != d.image) continue;
                    bestAny = std::max(bestAny, calculateIoU(d.box, g.box));
                }
                if(bestAny >= 0.5) fpWrongClass++;
                else fpNotACone++;
            }
        }
        const long long fn = (long long)classGts.size() - tp;

        const double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
        const double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
        const double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

        std::cout << "  " << std::left << std::setw(10) << className(cls) << std::fixed << std::setprecision(3)
                  << " P=" << precision << " R=" << recall << " F1=" << f1
                  << "   (TP=" << tp << " FP=" << fp << " FN=" << fn << ")"
                  << "   of the FP: " << fpWrongClass << " cones with the wrong class, "
                  << fpNotACone << " not cones\n";
        macroF1 += f1;
        nClassesF1++;
    }
    std::cout << "  --> macro F1 = " << (nClassesF1 ? macroF1 / nClassesF1 : 0.0) << "\n\n";

    // how many cones the detection finds, ignoring the class: separates "we never find it" from
    // "we find it but label it wrong"
    std::cout << "Recall of the detection alone (class ignored, IoU>=0.5)\n";
    for(int cls : kClasses)
    {
        long long found = 0, total = 0;
        for(const GtBox& g : gts)
        {
            if(g.cls != cls) continue;
            total++;
            for(const Det& d : dets)
            {
                if(d.image != g.image) continue;
                if(calculateIoU(d.box, g.box) >= 0.5){ found++; break; }
            }
        }
        if(total == 0) continue;
        std::cout << "  " << std::left << std::setw(10) << className(cls)
                  << " found " << found << "/" << total << " = " << (double)found / total << "\n";
    }
    if(showFunnel) std::cout << "Detection funnel per class (candidates with IoU>=0.3 on the true cone)\n";
    for(int cls : kClasses)
    {
        if(!showFunnel || !propTotal[cls]) continue;
        std::cout << "  " << std::left << std::setw(10) << className(cls)
                  << " proposals " << propCovered[cls] << "/" << propTotal[cls]
                  << "  -> after stage1 " << stage1Ok[cls]
                  << "  -> after stage2 " << stage2Ok[cls] << "\n";
    }
    std::cout << "\n";

    // --- 2. mAP@0.5:0.95 ---
    std::cout << "mAP@0.5:0.95\n";
    std::map<int, double> apAt50;
    double mapSum = 0.0;
    int nThr = 0;
    for(int t = 0; t < 10; t++)
    {
        const double thr = 0.50 + 0.05 * t;
        double sum = 0.0;
        int n = 0;
        for(int cls : kClasses)
        {
            const double ap = averagePrecision(dets, gts, cls, thr);
            if(ap < 0) continue;
            if(t == 0) apAt50[cls] = ap;
            sum += ap;
            n++;
        }
        if(n == 0) continue;
        const double mAP = sum / n;
        mapSum += mAP;
        nThr++;
        if(t == 0 || t == 9) std::cout << "  mAP@" << std::setprecision(2) << thr << std::setprecision(3) << " = " << mAP << "\n";
    }
    for(int cls : kClasses)
    {
        if(apAt50.count(cls)) std::cout << "    AP@0.5 " << std::left << std::setw(10) << className(cls) << " = " << apAt50[cls] << "\n";
    }
    std::cout << "  --> mAP@0.5:0.95 = " << (nThr ? mapSum / nThr : 0.0) << "\n\n";

    // --- 3. pixel mIoU ---
    std::cout << "mIoU (Jaccard over pixels)\n";
    double miou = 0.0;
    int nClassesIou = 0;
    for(int cls : kClasses)
    {
        const long long den = pixTp[cls] + pixFp[cls] + pixFn[cls];
        if(den == 0) continue;
        const double iou = (double)pixTp[cls] / den;
        std::cout << "  " << std::left << std::setw(10) << className(cls) << " = " << iou << "\n";
        miou += iou;
        nClassesIou++;
    }
    std::cout << "  --> mIoU = " << (nClassesIou ? miou / nClassesIou : 0.0) << "\n";

    return 0;
}
