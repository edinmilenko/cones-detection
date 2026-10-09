//Author: Edin Milenko
#include "utils.hpp"

// How much two boxes overlap: shared area over the area they cover together
double calculateIoU(cv::Rect a, cv::Rect b){
    int intersectionArea = (a & b).area();
    if(intersectionArea == 0)
    {
        return 0.0;
    }
    return double(intersectionArea) / (a.area() + b.area() - intersectionArea);
}

//sort by score, then greedily keep a box only if it doesn't overlap any box already kept
std::vector<int> nonMaxSuppression(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, double iouThr){
    std::vector<int> order(boxes.size());
    for(size_t i = 0; i < order.size(); i++)
    {
        order[i] = (int)i;
    }
    // best score first
    std::sort(order.begin(), order.end(), [&](int a, int b){ return scores[a] > scores[b]; });

    std::vector<int> kept;
    for(int idx : order)
    {
        // a box survives only if it does not step on one we already kept
        bool overlaps = false;
        for(int k : kept)
        {
            if(calculateIoU(boxes[idx], boxes[k]) > iouThr)
            {
                overlaps = true;
                break;
            }
        }
        if(!overlaps)
        {
            kept.push_back(idx);
        }
    }
    return kept;
}

//each detection takes the free GT it overlaps most, above iouThr
struct ClassMatch {
    int tp = 0, fp = 0, fn = 0;
    std::vector<double> precisions, recalls;
};

// Matches detections and real cones of one class, the step every metric is built on
static ClassMatch matchOneClass(const std::vector<Pred>& preds, const std::vector<GT>& ground, int cls, double iouThr){
    ClassMatch m;
    // the real cones of this class
    std::vector<const GT*> classGts;
    for(const GT& g : ground)
    {
        if(g.cls == cls) classGts.push_back(&g);
    }

    // and our detections of this class
    std::vector<Pred> classPreds;
    for(const Pred& p : preds)
    {
        if(p.cls == cls) classPreds.push_back(p);
    }
    // from the most confident down, so the best detection gets the first pick
    std::sort(classPreds.begin(), classPreds.end(), [](const Pred& a, const Pred& b){ return a.score > b.score; });

    // a cone already taken cannot be matched again, two boxes on it are one hit and one mistake
    std::vector<bool> matched(classGts.size(), false);
    for(const Pred& p : classPreds)
    {
        double bestIou = 0.0;
        int bestIdx = -1;
        for(size_t j = 0; j < classGts.size(); j++)
        {
            // only free cones, and only in the same image
            if(matched[j] || classGts[j]->imgName != p.imgName) continue;
            double iou = calculateIoU(p.bbox, classGts[j]->bbox);
            if(iou > bestIou)
            {
                bestIou = iou;
                bestIdx = (int)j;
            }
        }

        // enough overlap and it is a hit, otherwise it is a box on nothing
        if(bestIou >= iouThr)
        {
            m.tp++;
            matched[bestIdx] = true;
        }
        else
        {
            m.fp++;
        }

        // precision and recall as they stand after this detection, the curve the AP needs
        m.precisions.push_back(double(m.tp) / (m.tp + m.fp));
        if(classGts.empty())
        {
            m.recalls.push_back(0.0);
        }
        else
        {
            m.recalls.push_back(double(m.tp) / classGts.size());
        }
    }
    // the cones nobody matched are the ones we missed
    m.fn = (int)classGts.size() - m.tp;
    return m;
}

// The F1 of each class, averaged without weights: a rare class counts as much as a common one
double macroF1(const std::vector<Pred>& preds, const std::vector<GT>& ground, const std::vector<int>& classes, double iouThr){
    double sum = 0.0;
    int counted = 0;

    for(int cls : classes)
    {
        const ClassMatch m = matchOneClass(preds, ground, cls, iouThr);
        if(m.tp + m.fn == 0) continue; //class absent from the ground truth, left out of the average

        double precision = 0.0;
        if(m.tp + m.fp > 0)
        {
            precision = double(m.tp) / (m.tp + m.fp);
        }
        double recall = double(m.tp) / (m.tp + m.fn);

        double f1 = 0.0;
        if(precision + recall > 0)
        {
            f1 = 2 * precision * recall / (precision + recall);
        }

        sum += f1;
        counted++;
    }
    if(counted == 0) return 0.0;
    return sum / counted;
}

static double averagePrecision(const std::vector<Pred>& preds, const std::vector<GT>& ground, int cls, double iouThr){
    ClassMatch m = matchOneClass(preds, ground, cls, iouThr);
    if(m.tp + m.fn == 0) return -1.0;

    std::vector<double>& precisions = m.precisions;
    const std::vector<double>& recalls = m.recalls;
    for(int i = (int)precisions.size() - 2; i >= 0; i--)
    {
        precisions[i] = std::max(precisions[i], precisions[i + 1]);
    }

    double sum = 0.0;
    for(int i = 0; i <= 100; i++)
    {
        double level = i / 100.0;
        double best = 0.0;
        for(size_t k = 0; k < recalls.size(); k++)
        {
            if(recalls[k] >= level)
            {
                best = precisions[k];
                break;
            }
        }
        sum += best;
    }
    return sum / 101.0;
}

// The AP averaged over the classes and over ten thresholds, from 0.50 to 0.95 in steps of 0.05
double meanAveragePrecision(const std::vector<Pred>& preds, const std::vector<GT>& ground, const std::vector<int>& classes){
    double sum = 0.0;
    int counted = 0;

    for(int step = 0; step < 10; step++)
    {
        double iouThr = 0.50 + 0.05 * step;
        double classSum = 0.0;
        int classCount = 0;
        for(int cls : classes)
        {
            double ap = averagePrecision(preds, ground, cls, iouThr);
            if(ap < 0) continue;
            classSum += ap;
            classCount++;
        }
        if(classCount == 0) continue;

        sum += classSum / classCount;
        counted++;
    }
    if(counted == 0) return 0.0;
    return sum / counted;
}

// Counts the pixels of one image into the running totals, class by class
void PixelIoU::add(const cv::Mat& predLabels, const cv::Mat& gtLabels, const std::vector<int>& classes){
    for(int cls : classes)
    {
        cv::Mat predMask = (predLabels == cls);
        cv::Mat gtMask = (gtLabels == cls);
        cv::Mat intersection;
        cv::bitwise_and(predMask, gtMask, intersection);

        // pixels where we said the right class, and the two kinds of mistake around them
        long long shared = cv::countNonZero(intersection);
        tp[cls] += shared;
        fp[cls] += cv::countNonZero(predMask) - shared;
        fn[cls] += cv::countNonZero(gtMask) - shared;
    }
}

// One IoU per class over the whole test set, then the average: a missed cone weighs here too
double PixelIoU::mean(const std::vector<int>& classes) const{
    double sum = 0.0;
    int counted = 0;

    for(int cls : classes)
    {
        long long unionArea = tp.at(cls) + fp.at(cls) + fn.at(cls);
        if(unionArea == 0) continue;

        sum += double(tp.at(cls)) / unionArea;
        counted++;
    }
    if(counted == 0) return 0.0;
    return sum / counted;
}

