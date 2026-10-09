//Author: Nicola Ranzolin
#pragma once
#include <string>

void extractProposalPatches(const std::string& datasetDir, const std::string& csvPath, const std::string& posOutDir, const std::string& negOutDir, int maxPosPerImage = 0, int maxNegPerImage = 50);

void mineHardNegatives(const std::string& datasetDir, const std::string& csvPath, const std::string& stage1Path, const std::string& negOutDir, float stage1Thr = 15.0f, int maxImages = 400, int targetCount = 40000);
