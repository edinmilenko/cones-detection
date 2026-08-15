#include <opencv2/opencv.hpp>
#include <iostream>
#include "nlohmann/json.hpp"
#include <fstream>
#include <filesystem>
#include <vector>
#include <random>

using json = nlohmann::json;

//makes a csv file with each image name and the corresponding bounding boxes
void csvDatasetMaker(std::string datasetDir){
    std::cout << "Creating CSV dataset from: " << datasetDir << std::endl;
    std::ofstream csvFile("dataset.csv"); // file to write the dataset

    for(auto entry : std::filesystem::directory_iterator{datasetDir}){
        std::string ext = entry.path().extension().string();
        if(ext != ".json"){
            continue;
        }

        //read height and width of the image
        std::ifstream jsonFile(entry.path());
        json d = json::parse(jsonFile);
        int W = d["size"]["width"];
        int H = d["size"]["height"];
        std::string imageName;

        //find the corresponding extension for the image
        const std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"};
        for(auto extension : extensions){
            if(std::filesystem::exists(entry.path().parent_path() / (entry.path().stem().string() + extension))){
                imageName = entry.path().stem().string() + extension;
                break;
            }
        }

        if(imageName.empty()){
            std::cerr << "No image found for the given json file: " << entry.path() << std::endl;
            continue;
        }

        for(auto object : d["objects"]){
            //ignore non boxes
            if(object["geometryType"] != "rectangle"){
                continue;
            }
            auto pts = object["points"]["exterior"];
            int p0x = pts[0][0].get<int>();
            int p0y = pts[0][1].get<int>();
            int p1x = pts[1][0].get<int>();
            int p1y = pts[1][1].get<int>();

            int x1 = std::min(p0x, p1x);
            int y1 = std::min(p0y, p1y);
            int x2 = std::max(p0x, p1x);
            int y2 = std::max(p0y, p1y);

            //bboxes are too close to the border, ignore them
            if(x1 < 2 || y1 < 2 || x2 > W - 2|| y2 > H - 2){
                continue;
            }
            csvFile << imageName << "," << x1 << "," << y1 << "," << x2 << "," << y2 << "\n";
        }
    }   
}

//splits the dataset in two, training and test, saving two different files with the indexes
void makeSplit(std::string csvPath, std::string outDir, int devSize, double trainFrac, int seed){
    
    std::set<std::string> imgNames;
    std::ifstream csv(csvPath);
    std::string line;

    while(std::getline(csv, line)){
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string name;
        std::getline(ss, name, ',');
        imgNames.insert(name);
    }

    //deterministic ordering followed by shuffling
    std::vector<std::string> names(imgNames.begin(), imgNames.end());
    std::mt19937 rng(seed);
    std::shuffle(names.begin(), names.end(), rng);

    int n = std::min<int>(devSize, names.size());
    int nTrain = static_cast<int>(n * trainFrac);

    std::ofstream trainFile(outDir + "/train.txt");
    std::ofstream testFile(outDir + "/test.txt");
    for (int i = 0; i < n; ++i) {
        (i < nTrain ? trainFile : testFile) << names[i] << "\n";
    }

    std::cout << "train: " << nTrain << "   test: " << (n - nTrain) << std::endl;
}