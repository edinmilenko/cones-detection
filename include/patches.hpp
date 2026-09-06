#pragma once
#include <string>

// patches for the cascade: every colorProposals() candidate on every annotated image of dataset/,
// labelled by its IoU with the real boxes. This is what the two stages are trained on.
// The two caps are per image; 0 means no cap. What matters is their ratio, around four positives
// per negative: with the two classes about even the linear stage scores everything low and the
// cascade stops detecting anything at all (measured: macro F1 0.28 instead of 0.42).
void extractProposalPatches(const std::string& datasetDir, const std::string& csvPath, const std::string& posOutDir, const std::string& negOutDir, int maxPosPerImage = 0, int maxNegPerImage = 50);

// hard negative mining: the candidates stage 1 wrongly accepts are added to the negatives, and
// stage 1 is retrained with them, which cuts the false positives it feeds to stage 2.
void mineHardNegatives(const std::string& datasetDir, const std::string& csvPath, const std::string& stage1Path, const std::string& negOutDir, float stage1Thr = 15.0f, int maxImages = 400, int targetCount = 40000);
