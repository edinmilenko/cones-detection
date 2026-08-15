#include "utils.hpp"

//returns iou
static double calculateIoU(cv::Rect a, cv::Rect b){
    int intersectionArea = (a & b).area();
    if (intersectionArea == 0){
        return 0.0;
    }
    return double(intersectionArea) / (a.area() + b.area() - intersectionArea);
}

void evaluatePredictions(std::vector<Pred> predictions, std::vector<GT> ground, double iouThreshold){

    std::sort(predictions.begin(), predictions.end(), [](const Pred& a, const Pred& b){return a.score > b.score;});
    std::unordered_map<std::string, std::vector<int>> gtByImg;

    //map GT to the corresponding image to avoid comparing predictions to all bboxes of all images
    for(int i = 0; i < ground.size(); i ++){
        gtByImg[ground[i].imgName].push_back(i);
    }
    
    //greedy matching
    int tp = 0, fp = 0;
    for(const auto& pred : predictions){
        int bestIdx = -1;
        double bestIoU = 0;
        for(int gt : gtByImg[pred.imgName]){
            if(ground[gt].matched){
                continue;
            }
            double currIoU = calculateIoU(pred.bbox, ground[gt].bbox);
            if (currIoU > bestIoU){
                bestIoU = currIoU;
                bestIdx = gt;
            }
        }
        //false positive, no matching
        if(bestIdx == -1 || bestIoU < iouThreshold){
            fp++;
        }
        else{
            tp++;
            ground[bestIdx].matched = true;
        }
    }

    //now calculate false negatives
    int fn = 0;
    for(const auto& gt : ground){
        if(!gt.matched){fn++;}
    }

    double precision = 0.0;
    if (tp + fp > 0) {
        precision = double(tp) / (tp + fp);
    }

    double recall = 0.0;
    if (tp + fn > 0) {
        recall = double(tp) / (tp + fn);
    }

    double f1 = 0.0;
    if (precision + recall > 0) {
        f1 = 2 * precision * recall / (precision + recall);
    }
}
