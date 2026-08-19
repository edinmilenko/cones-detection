#include "hog.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <set>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <iostream>

void extractPositives(const std::string& datasetDir, const std::string& csvPath, const std::string& splitFile, const std::string& outDir, int minHeight){
    //add all the filenames of the training to a set, used later to check if an image should be used for the training
    std::set<std::string> set; 
    std::ifstream split(splitFile);
    std::string line;
    while(std::getline(split, line)){
        set.insert(line);
    }
    
    //parse the dataset csv, adding the training img boxes to the map
    std::ifstream csvFile(csvPath);
    std::string line1;
    std::unordered_map<std::string, std::vector<cv::Rect>> mp; //map to store the boxes foreach img
    int counter = 0; //counter to see how many boxes are skipped
    while(std::getline(csvFile, line1)){

        std::stringstream ss(line1);
        std::string imgName;
        std::getline(ss, imgName, ',');
        std::string x1s, y1s, x2s, y2s;

        std::getline(ss, x1s, ',');
        std::getline(ss, y1s, ',');
        std::getline(ss, x2s, ',');
        std::getline(ss, y2s, ',');

        int x1 = std::stoi(x1s);
        int y1 = std::stoi(y1s);
        int x2 = std::stoi(x2s);
        int y2 = std::stoi(y2s);

        if(set.count(imgName) != 0){ //skip if the height is too small
            if((y2 - y1) < minHeight) counter++;
            else mp[imgName].push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2,y2)));
        }
    }
    std::cout << "discarded boxes: " << counter << '\n';
    
    //save the positive samples. Open one img at a time
    std::unordered_map<std::string, std::vector<cv::Rect>>::iterator it = mp.begin();
    std::filesystem::create_directories(outDir);
    int positiveCounter = 0;
    while(it != mp.end()){
        std::string imgPath = it->first;
        const std::vector<cv::Rect>& boxes = it->second;
        int boxCount = 0;
        cv::Mat img = cv::imread(datasetDir + "/" + imgPath, cv::IMREAD_GRAYSCALE);
        if(img.empty()){it++; continue;}
        std::string stem = imgPath.substr(0, imgPath.find_last_of('.')); 

        for(auto& box : boxes){

            cv::Rect safe = box & cv::Rect(0, 0, img.cols, img.rows);
            if (safe.width < 4 || safe.height < 4) continue; //if the box is out of the img it will fail so skip it.
            cv::Mat positive = img(safe);

            //resize to standard 16 x 24 size;
            int interp = (box.height > 24) ? cv::INTER_AREA : cv::INTER_LINEAR; //use different interpolations for enlarging and shrinking
            cv::resize(positive, positive, cv::Size(16, 24), 0, 0, interp);            
            cv::imwrite(outDir + "/" + stem + "_" + std::to_string(boxCount) + ".png", positive);
            cv::Mat flipped;
            cv::flip(positive, flipped, 1); //data augmentation keep flipped version also.
            cv::imwrite(outDir + "/" + stem + "_" + std::to_string(boxCount) + "_flip" + ".png", flipped);
            boxCount++;
            positiveCounter++;
        }
        it++;
    }
}
