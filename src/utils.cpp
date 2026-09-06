#include "utils.hpp"

//returns iou
double calculateIoU(cv::Rect a, cv::Rect b){
    int intersectionArea = (a & b).area();
    if(intersectionArea == 0)
    {
        return 0.0;
    }
    return double(intersectionArea) / (a.area() + b.area() - intersectionArea);
}

//non-maximum suppression: sort by score descending, then greedily keep a box only if it
//doesn't overlap (IoU > iouThr) any box already kept
std::vector<int> nonMaxSuppression(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, double iouThr){
    std::vector<int> order(boxes.size());
    for(size_t i = 0; i < order.size(); i++)
    {
        order[i] = (int)i;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b){ return scores[a] > scores[b]; });

    std::vector<int> kept;
    for(int idx : order)
    {
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

//what the greedy matching of one class produces: each detection takes the free GT it overlaps
//most, above iouThr. precisions/recalls are the running curve, which only the AP looks at.
struct ClassMatch {
    long long tp = 0, fp = 0, fn = 0;
    std::vector<double> precisions, recalls;
};

static ClassMatch matchOneClass(const std::vector<Pred>& preds, const std::vector<GT>& ground, int cls, double iouThr){
    ClassMatch m;
    std::vector<const GT*> classGts;
    for(const GT& g : ground)
    {
        if(g.cls == cls) classGts.push_back(&g);
    }

    std::vector<Pred> classPreds;
    for(const Pred& p : preds)
    {
        if(p.cls == cls) classPreds.push_back(p);
    }
    std::sort(classPreds.begin(), classPreds.end(), [](const Pred& a, const Pred& b){ return a.score > b.score; });

    std::vector<bool> matched(classGts.size(), false);
    for(const Pred& p : classPreds)
    {
        double bestIou = 0.0;
        int bestIdx = -1;
        for(size_t j = 0; j < classGts.size(); j++)
        {
            if(matched[j] || classGts[j]->imgName != p.imgName) continue;
            double iou = calculateIoU(p.bbox, classGts[j]->bbox);
            if(iou > bestIou)
            {
                bestIou = iou;
                bestIdx = (int)j;
            }
        }

        if(bestIou >= iouThr)
        {
            m.tp++;
            matched[bestIdx] = true;
        }
        else
        {
            m.fp++;
        }

        m.precisions.push_back(double(m.tp) / (m.tp + m.fp));
        m.recalls.push_back(classGts.empty() ? 0.0 : double(m.tp) / classGts.size());
    }
    m.fn = (long long)classGts.size() - m.tp;
    return m;
}

double macroF1(const std::vector<Pred>& preds, const std::vector<GT>& ground, const std::vector<int>& classes, double iouThr){
    double sum = 0.0;
    int counted = 0;

    for(int cls : classes)
    {
        const ClassMatch m = matchOneClass(preds, ground, cls, iouThr);
        if(m.tp + m.fn == 0) continue; //class absent from the ground truth, left out of the average

        double precision = (m.tp + m.fp) > 0 ? double(m.tp) / (m.tp + m.fp) : 0.0;
        double recall = double(m.tp) / (m.tp + m.fn);
        double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

        sum += f1;
        counted++;
    }
    return counted ? sum / counted : 0.0;
}

//area under the precision/recall curve, sampled at 101 recall levels after making the precision
//monotonically decreasing (the COCO convention)
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
    return counted ? sum / counted : 0.0;
}

void PixelIoU::add(const cv::Mat& predLabels, const cv::Mat& gtLabels, const std::vector<int>& classes){
    for(int cls : classes)
    {
        cv::Mat predMask = (predLabels == cls);
        cv::Mat gtMask = (gtLabels == cls);
        cv::Mat intersection;
        cv::bitwise_and(predMask, gtMask, intersection);

        long long shared = cv::countNonZero(intersection);
        tp[cls] += shared;
        fp[cls] += cv::countNonZero(predMask) - shared;
        fn[cls] += cv::countNonZero(gtMask) - shared;
    }
}

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
    return counted ? sum / counted : 0.0;
}

double maskIoU(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask)
{
    cv::Mat intersectionMask, unionMask;
    cv::bitwise_and(predictedMask, groundTruthMask, intersectionMask);
    cv::bitwise_or(predictedMask, groundTruthMask, unionMask);

    const double intersectionArea = cv::countNonZero(intersectionMask);
    const double unionArea = cv::countNonZero(unionMask);

    if(unionArea == 0.0)
    {
        return 0.0;
    }
    return intersectionArea / unionArea;
}

//calculates the recall of the chosen candidates
double candidateRecall(const std::vector<cv::Rect>& proposals, const std::vector<cv::Rect>& groundTruth, double iouThr){
    if(groundTruth.empty())
    {
        return -1.0; // nothing to measure
    }
    int covered = 0;
    for(const cv::Rect& gt : groundTruth)
    {
        for(const cv::Rect& p : proposals)
        {
            if(calculateIoU(p, gt) >= iouThr)
            {
                ++covered;
                break;
            }
        }
    }
    return double(covered) / groundTruth.size();
}
