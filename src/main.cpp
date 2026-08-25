#include "dataset.hpp"
#include "detector.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {

    try {
        std::string mode = (argc > 1) ? argv[1] : "";

        if(mode == "detect")
        {
            if(argc < 3)
            {
                std::cerr << "usage: main detect <image> [output.png]\n";
                return 1;
            }
            std::string outPath = (argc > 3) ? argv[3] : "detection.png";
            detectPipeline(argv[2], outPath);
            return 0;
        }

        //  LEAVE DISABLED for first run only
        //  otherwise it redoes it every time; either add a check or document where to put
        //  the three zips before the first run
        //  consider a hardcoded check?
        //dataLoader();

        // labeler should be removed before submission, probably only useful for part 2
        //labeler();
        //std::string datasetDir = argv[1];
        //csvDatasetMaker(datasetDir);
        makeSplit("../data/dataset.csv", "../data");

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
