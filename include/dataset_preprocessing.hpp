#pragma once
#include <string>

void csvDatasetMaker(std::string datasetDir);

void makeSplit(std::string csvPath, std::string outDir, int devSize = 2000, double trainFrac = 0.7, int seed = 42);