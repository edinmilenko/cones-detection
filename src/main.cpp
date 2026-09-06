#include "dataset.hpp"
#include "detector.hpp"
#include "hog.hpp"
#include "patches.hpp"
#include <iostream>

static void printUsage(){
    std::cerr << "usage:\n"
              << "  main detect <image> [output.png]   full pipeline on one image\n"
              << "  main dataset                       merges the two FSOCO folders into dataset/\n"
              << "  main label                         redraws the annotations into labeled/\n"
              << "  main csv <dataset_dir>             extracts the bounding boxes into dataset.csv\n"
              << "  main train                         extracts the patches and trains both stages\n"
              << "  main eval                          macro F1, mAP@0.5:0.95 and mIoU on the test set\n";
}

int main(int argc, char* argv[]) {

    try {
        const std::string mode = (argc > 1) ? argv[1] : "";

        if(mode == "detect")
        {
            if(argc < 3){ printUsage(); return 1; }
            detectPipeline(argv[2], argc > 3 ? argv[3] : "detection.png");
        }
        else if(mode == "dataset")
        {
            // run once, with the two FSOCO archives already unpacked next to the project
            dataLoader();
        }
        else if(mode == "label")
        {
            labeler();
        }
        else if(mode == "csv")
        {
            if(argc < 3){ printUsage(); return 1; }
            csvDatasetMaker(argv[2]);
        }
        else if(mode == "train")
        {
            // the two stages in order: stage 2 is trained on what stage 1 lets through, so the
            // stage 1 model has to exist first
            extractProposalPatches("../dataset", "../data/dataset.csv",
                                   "../data/patches/positives", "../data/patches/negatives");
            trainSvm("../data/patches/positives", "../data/patches/negatives",
                     "../data/svm_hog_sgd_proposal.yml");

            // one round of hard negative mining: the candidates stage 1 gets wrong go back into the
            // negatives and stage 1 is trained again with them in the pile
            mineHardNegatives("../dataset", "../data/dataset.csv", "../data/svm_hog_sgd_proposal.yml",
                              "../data/patches/negatives");
            trainSvm("../data/patches/positives", "../data/patches/negatives",
                     "../data/svm_hog_sgd_proposal.yml");

            trainStage2("../data/patches/positives", "../data/patches/negatives",
                        "../data/svm_hog_sgd_proposal.yml", "../data/svm_stage2_rbf_balanced.yml");
        }
        else if(mode == "eval")
        {
            evaluatePipeline("../test_set/segmentation_test/img", "../test_set/segmentation_test/ann",
                             "../data/test_real.txt");
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
