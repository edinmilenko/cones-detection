//Author: Alessandro Martini
#include "hog.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <stdexcept>

//HOG parameters in one place, so training and detection build the same descriptor
cv::HOGDescriptor makeHog(){
    return cv::HOGDescriptor(PATCH_SIZE, cv::Size(8, 8), cv::Size(4, 4), cv::Size(4, 4), 9);
}

//how much the colour histogram weighs next to the HOG values
static const float COLOUR_WEIGHT = 1.5f;

// The descriptor of one patch: HOG and the colour from a histogram, one after the other
std::vector<float> describePatch(const cv::Mat& patch){
    // HOG works on gradients, so we use the greyspace
    cv::Mat grey;
    cv::cvtColor(patch, grey, cv::COLOR_BGR2GRAY);

    // built once and kept, this runs thousands of times per image
    static const cv::HOGDescriptor hog = makeHog();
    std::vector<float> descriptor;
    hog.compute(grey, descriptor);

    // the colour we just threw away comes back now, a cone and a piece of asphalt have the same shape
    cv::Mat hsv;
    cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);

    // 12 hue bins x 4 saturation bins, counted over the whole patch
    const int hueBins = 12;
    const int satBins = 4;
    std::vector<float> histogram(hueBins * satBins, 0.0f);
    for(int y = 0; y < hsv.rows; y++)
    {
        for(int x = 0; x < hsv.cols; x++)
        {
            // hue says which colour, saturation how washed out it is, the brightness we ignore
            cv::Vec3b pixel = hsv.at<cv::Vec3b>(y, x);
            // OpenCV keeps the hue on 0-179 and the saturation on 0-255
            int hueBin = pixel[0] * hueBins / 180;
            int satBin = pixel[1] * satBins / 256;
            // the division can land one past the last bin, we pull it back
            if(hueBin >= hueBins) hueBin = hueBins - 1;
            if(satBin >= satBins) satBin = satBins - 1;
            histogram[hueBin * satBins + satBin]++;
        }
    }

    // turned into fractions of the patch, so the size of the patch doesn't matter, then weighted
    const float pixels = (float)(hsv.rows * hsv.cols);
    for(size_t i = 0; i < histogram.size(); i++)
    {
        // appended after the HOG values, so the descriptor is a single vector
        descriptor.push_back(COLOUR_WEIGHT * histogram[i] / pixels);
    }
    return descriptor;
}

// The files of a patch folder, sorted so two trainings see them in the same order
static std::vector<std::string> listPatchFiles(const std::string& dir){
    std::vector<std::string> files;
    for(const auto& entry : std::filesystem::directory_iterator(dir))
    {
        if(entry.is_regular_file())
        {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// From patch files to a matrix of descriptors, ready for the SVM
static cv::Mat computeFeatures(const std::vector<std::string>& files){
    // one row per patch, allocated on the first descriptor when we know how long it is
    cv::Mat features;

    for(size_t i = 0; i < files.size(); i++)
    {
        cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);
        if(img.empty())
        {
            throw std::runtime_error("computeFeatures: cannot read " + files[i]);
        }
        std::vector<float> descriptor = describePatch(img);
        if(features.empty())
        {
            features.create((int)files.size(), (int)descriptor.size(), CV_32F);
        }
        // the descriptor is a column, we lay it down as the i-th row
        cv::Mat(descriptor).reshape(1, 1).copyTo(features.row(static_cast<int>(i)));
    }
    return features;
}

// The linear stage, trained on all the patches we extracted from the training images
void trainSvm(const std::string& posDir, const std::string& negDir, const std::string& modelPath){
    std::vector<std::string> posFiles = listPatchFiles(posDir);
    std::vector<std::string> negFiles = listPatchFiles(negDir);

    std::cout << "training stage 1 on " << posFiles.size() << " positives and " << negFiles.size() << " negatives\n";

    //every patch is read from disk and described
    cv::Mat posFeatures = computeFeatures(posFiles);
    cv::Mat negFeatures = computeFeatures(negFiles);

    // positives first, negatives below, so the labels are two blocks
    cv::Mat features;
    cv::vconcat(posFeatures, negFeatures, features);

    // 1 for the cones, -1 for everything else
    cv::Mat labels(features.rows, 1, CV_32F); 
    labels.rowRange(0, posFeatures.rows).setTo(1);
    labels.rowRange(posFeatures.rows, features.rows).setTo(-1);

    cv::Ptr<cv::ml::SVMSGD> svm = cv::ml::SVMSGD::create();
    svm->setSvmsgdType(cv::ml::SVMSGD::ASGD);        // averaged SGD, more stable than plain SGD
    svm->setMarginType(cv::ml::SVMSGD::SOFT_MARGIN); 
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100000, 0));

    svm->train(features, cv::ml::ROW_SAMPLE, labels);

    // save model on disk
    svm->save(modelPath);
    std::cout << "stage 1 saved in " << modelPath << '\n';
}

// How far the patch is from the hyperplane: positive means it looks like a cone
float rawScore(const cv::Ptr<cv::ml::SVMSGD>& svm, const cv::Mat& patch){
    std::vector<float> descriptor = describePatch(patch);
    // SVMSGD does not give the raw score, so we redo the product by hand (still don't understand why It does not provide it directly)
    cv::Mat w = svm->getWeights();
    float score = svm->getShift();
    for(size_t i = 0; i < descriptor.size(); i++)
    {
        score += w.at<float>(0, (int)i) * descriptor[i];
    }
    return score;
}

// The score of the RBF stage, higher means more likely a cone
float rbfScore(const cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& patch){
    std::vector<float> descriptor = describePatch(patch);
    cv::Mat sample = cv::Mat(descriptor).reshape(1, 1);
    cv::Mat raw;
    svm->predict(sample, raw, cv::ml::StatModel::RAW_OUTPUT); // unlike SVMSGD, this works
    return -raw.at<float>(0, 0); //negative = positive class (weird I know but yeah)
}

//stage 2 is trained only on the patches stage 1 accepts, which are the ones it sees when detecting
static cv::Mat keepSurvivors(const std::vector<std::string>& files, const cv::Mat& weights, float shift, float stage1Thr){
    const int descriptorSize = weights.cols;
    std::vector<cv::Mat> kept;

    for(size_t i = 0; i < files.size(); i++)
    {
        cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);
        if(img.empty()) continue;

        // the same score as rawScore, rewritten here to avoid reloading the model every patch
        std::vector<float> descriptor = describePatch(img);
        float score = shift;
        for(int j = 0; j < descriptorSize; j++)
        {
            score += weights.at<float>(0, j) * descriptor[j];
        }
        // stage 1 would have thrown it away, so stage 2 will never see it
        if(score <= stage1Thr) continue;

        kept.push_back(cv::Mat(1, descriptorSize, CV_32F, descriptor.data()).clone());
    }

    // the survivors, stacked into one matrix
    cv::Mat features((int)kept.size(), descriptorSize, CV_32F);
    for(size_t i = 0; i < kept.size(); i++)
    {
        kept[i].copyTo(features.row((int)i));
    }
    return features;
}

// keeps at most cap rows, picked at random with a fixed seed
static cv::Mat subsample(const cv::Mat& features, int cap){
    if(features.rows <= cap) return features;

    std::vector<int> indexes(features.rows);
    for(int i = 0; i < features.rows; i++)
    {
        indexes[i] = i;
    }
    // fixed seed, so two trainings pick the same patches
    std::mt19937 rng(42);
    std::shuffle(indexes.begin(), indexes.end(), rng);

    cv::Mat reduced(cap, features.cols, CV_32F);
    for(int i = 0; i < cap; i++)
    {
        features.row(indexes[i]).copyTo(reduced.row(i));
    }
    return reduced;
}

void trainStage2(const std::string& posDir, const std::string& negDir, const std::string& stage1Path, const std::string& modelPath, float stage1Thr, int maxPerClass){
    cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load(stage1Path);
    if(stage1.empty())
    {
        throw std::runtime_error("trainStage2: cannot load " + stage1Path);
    }

    // we need its hyperplane to know what it would accept
    cv::Mat weights = stage1->getWeights();
    const float shift = stage1->getShift();

    cv::Mat posFeatures = keepSurvivors(listPatchFiles(posDir), weights, shift, stage1Thr);
    cv::Mat negFeatures = keepSurvivors(listPatchFiles(negDir), weights, shift, stage1Thr);
    std::cout << "patches accepted by stage 1 (threshold " << stage1Thr << "): " << posFeatures.rows << " positives, " << negFeatures.rows << " negatives\n";

    if(posFeatures.rows == 0 || negFeatures.rows == 0)
    {
        throw std::runtime_error("trainStage2: one of the two classes is empty, lower the stage 1 threshold");
    }

    //an RBF SVM gets slow with many samples, so both classes are capped
    posFeatures = subsample(posFeatures, maxPerClass);
    negFeatures = subsample(negFeatures, maxPerClass);

    //same number of positives and negatives
    const int perClass = std::min(posFeatures.rows, negFeatures.rows);
    posFeatures = posFeatures.rowRange(0, perClass);
    negFeatures = negFeatures.rowRange(0, perClass);
    std::cout << "training stage 2 on " << perClass << " patches per class\n";

    cv::Mat features;
    cv::vconcat(posFeatures, negFeatures, features);
    cv::Mat labels(features.rows, 1, CV_32S); // cv::ml::SVM wants CV_32S, unlike SVMSGD
    labels.rowRange(0, perClass).setTo(1);
    labels.rowRange(perClass, features.rows).setTo(-1);

    // an RBF kernel, the two classes are not separable by a plane
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-3));
    
    //Parameters for the model tweaked by hand
    svm->setC(2.5);
    svm->setGamma(0.50625);
    svm->train(features, cv::ml::ROW_SAMPLE, labels);
    std::cout << "C=" << svm->getC() << " gamma=" << svm->getGamma() << " support vectors=" << svm->getSupportVectors().rows << '\n';

    svm->save(modelPath);
    std::cout << "stage 2 saved in " << modelPath << '\n';
}
