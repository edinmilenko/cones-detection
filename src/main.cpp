//Author: Alessandro Martini
#include "dataset.hpp"
#include "detector.hpp"
#include "hog.hpp"
#include "patches.hpp"
#include <iostream>

static std::string findTestSet(){
    const std::string dir = testSetDir();
    if(dir.empty())
    {
        throw std::runtime_error("test set not found: unzip test_set.zip in the project root, it makes a segmentation_test/ folder with img/ and ann/ inside");
    }
    return dir;
}

static void printUsage(){
    std::cerr << "usage:\n"
              << "  main detect <image> [output.png]   full pipeline on one image\n"
              << "  main dataset                       merges the two FSOCO folders into dataset/\n"
              << "  main csv <dataset_dir>             extracts the bounding boxes into dataset.csv\n"
              << "  main train                         extracts the patches and trains both stages\n"
              << "  main eval                          macro F1, mAP@0.5:0.95 and mIoU on the test set\n"
              << "  main testset [out_dir]             runs the pipeline on the whole test set and saves the images\n";
}

int main(int argc, char* argv[]) {

    try {
        std::string mode = "";
        if(argc > 1)
        {
            mode = argv[1];
        }

        if(mode == "detect")
        {
            if(argc < 3){ printUsage(); return 1; }
            std::string outPath = "detection.png";
            if(argc > 3)
            {
                outPath = argv[3];
            }
            detectPipeline(argv[2], outPath);
        }
        else if(mode == "dataset")
        {
            // run once, with the two FSOCO archives already unpacked next to the project
            dataLoader();
        }
        else if(mode == "csv")
        {
            if(argc < 3){ printUsage(); return 1; }
            csvDatasetMaker(argv[2]);
        }
        else if(mode == "train")
        {
            //in order: stage 2 trains on what stage 1 lets through, so stage 1 must exist first
            extractProposalPatches("../dataset", "../data/dataset.csv",
                                   "../data/patches/positives", "../data/patches/negatives");
            trainSvm("../data/patches/positives", "../data/patches/negatives",
                     "../data/svm_hog_sgd_proposal.yml");

            //one round of hard negative mining: the candidates stage 1 gets wrong go back in the
            //negatives and we train stage 1 again
            mineHardNegatives("../dataset", "../data/dataset.csv", "../data/svm_hog_sgd_proposal.yml",
                              "../data/patches/negatives");
            trainSvm("../data/patches/positives", "../data/patches/negatives",
                     "../data/svm_hog_sgd_proposal.yml");

            trainStage2("../data/patches/positives", "../data/patches/negatives",
                        "../data/svm_hog_sgd_proposal.yml", "../data/svm_stage2_rbf_balanced.yml");
        }
        else if(mode == "testset")
        {
            //only detection: it loads the models already in data/, it does not train
            std::string outDir = "../data/detect_test";
            if(argc > 2)
            {
                outDir = argv[2];
            }
            detectTestSet(findTestSet(), outDir);
        }
        else if(mode == "eval")
        {
            evaluatePipeline(findTestSet());
        }
        else
        {
            printUsage();
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
