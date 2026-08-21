#pragma once
#include <string>

void extractPositives(const std::string& datasetDir, const std::string& csvPath, const std::string& splitFile, const std::string& outDir, int minHeight = 20);
void extractNegatives(const std::string& datasetDir, const std::string& csvPath, const std::string& splitFile, const std::string& outDir, int numPatches = 10000);
