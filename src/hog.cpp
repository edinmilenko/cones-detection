#include "hog.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <stdexcept>

// single definition of the HOG parameters: training and detection both read from here, so
// a wrong-size descriptor can't arise from two places in the code drifting out of sync.
cv::HOGDescriptor makeHog(){
    return cv::HOGDescriptor(kPatchSize, cv::Size(8, 8), cv::Size(4, 4), cv::Size(4, 4), 9);
}

// how much the colour histogram counts next to the HOG values. Tuned by hand: 1.0 and 2.0 are both
// clearly worse, above 2 the histogram drowns the shape out completely.
static const float kColourWeight = 1.5f;

std::vector<float> describePatch(const cv::Mat& patch){
    cv::Mat grey;
    cv::cvtColor(patch, grey, cv::COLOR_BGR2GRAY);

    cv::HOGDescriptor hog = makeHog();
    std::vector<float> descriptor;
    hog.compute(grey, descriptor);

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
            cv::Vec3b pixel = hsv.at<cv::Vec3b>(y, x);
            int hueBin = pixel[0] * hueBins / 180;
            int satBin = pixel[1] * satBins / 256;
            if(hueBin >= hueBins) hueBin = hueBins - 1;
            if(satBin >= satBins) satBin = satBins - 1;
            histogram[hueBin * satBins + satBin]++;
        }
    }

    // turned into fractions of the patch, so the size of the patch doesn't matter, then weighted
    const float pixels = (float)(hsv.rows * hsv.cols);
    for(size_t i = 0; i < histogram.size(); i++)
    {
        descriptor.push_back(kColourWeight * histogram[i] / pixels);
    }
    return descriptor;
}

// sorted (for reproducibility) list of regular files in a patch folder.
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

// patches (already resized to kPatchSize) -> matrix of HOG descriptors, one row per patch
static cv::Mat computeFeatures(const std::vector<std::string>& files){
    cv::Mat features;
    int descriptorSize = 0;

    for(size_t i = 0; i < files.size(); i++)
    {
        cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);
        if(img.empty())
        {
            throw std::runtime_error("computeFeatures: cannot read " + files[i]);
        }
        if(img.size() != kPatchSize)
        {
            throw std::runtime_error("computeFeatures: " + files[i] + " has size " + std::to_string(img.cols) + "x" + std::to_string(img.rows) + ", expected " + std::to_string(kPatchSize.width) + "x" + std::to_string(kPatchSize.height));
        }

        std::vector<float> descriptor = describePatch(img);
        if(features.empty())
        {
            descriptorSize = (int)descriptor.size();
            features.create((int)files.size(), descriptorSize, CV_32F);
        }
        if((int)descriptor.size() != descriptorSize)
        {
            throw std::runtime_error("computeFeatures: unexpected descriptor size for " + files[i]);
        }
        cv::Mat(descriptor).reshape(1, 1).copyTo(features.row(static_cast<int>(i)));
    }
    return features;
}

void trainSvm(const std::string& posDir, const std::string& negDir, const std::string& modelPath){
    std::vector<std::string> posFiles = listPatchFiles(posDir);
    std::vector<std::string> negFiles = listPatchFiles(negDir);

    std::cout << "[trainSvm] positive samples: " << posFiles.size() << '\n';
    std::cout << "[trainSvm] negative samples: " << negFiles.size() << '\n';

    cv::Mat posFeatures = computeFeatures(posFiles);
    cv::Mat negFeatures = computeFeatures(negFiles);

    cv::Mat features;
    cv::vconcat(posFeatures, negFeatures, features);

    cv::Mat labels(features.rows, 1, CV_32F); // SVMSGD wants CV_32F, not CV_32S like cv::ml::SVM
    labels.rowRange(0, posFeatures.rows).setTo(1);
    labels.rowRange(posFeatures.rows, features.rows).setTo(-1);

    cv::Ptr<cv::ml::SVMSGD> svm = cv::ml::SVMSGD::create();
    svm->setSvmsgdType(cv::ml::SVMSGD::ASGD);        // averaged SGD, more stable than plain SGD
    svm->setMarginType(cv::ml::SVMSGD::SOFT_MARGIN); // classes aren't linearly separable
    // MAX_ITER only: with the EPS component active, convergence kicks in too early on
    // harder datasets and the model ends up untrained.
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100000, 0));

    auto t0 = std::chrono::steady_clock::now();
    svm->train(features, cv::ml::ROW_SAMPLE, labels);
    auto t1 = std::chrono::steady_clock::now();
    double trainSeconds = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[trainSvm] training time: " << trainSeconds << " s\n";

    std::filesystem::path modelFsPath(modelPath);
    if(modelFsPath.has_parent_path())
    {
        std::filesystem::create_directories(modelFsPath.parent_path());
    }
    svm->save(modelPath);
    std::cout << "[trainSvm] model saved to: " << modelPath << '\n';
}

float rawScore(const cv::Ptr<cv::ml::SVMSGD>& svm, const cv::Mat& patch){
    if(patch.size() != kPatchSize)
    {
        throw std::runtime_error("rawScore: patch does not have the expected size " + std::to_string(kPatchSize.width) + "x" + std::to_string(kPatchSize.height));
    }
    std::vector<float> descriptor = describePatch(patch);
    // predict(..., RAW_OUTPUT) is broken for SVMSGD (returns the same discrete label as a
    // plain predict()), so the continuous score is rebuilt by hand: weights . descriptor + shift
    cv::Mat w = svm->getWeights();
    float score = svm->getShift();
    for(size_t i = 0; i < descriptor.size(); i++)
    {
        score += w.at<float>(0, (int)i) * descriptor[i];
    }
    return score;
}

float rbfScore(const cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& patch){
    if(patch.size() != kPatchSize)
    {
        throw std::runtime_error("rbfScore: patch does not have the expected size " + std::to_string(kPatchSize.width) + "x" + std::to_string(kPatchSize.height));
    }
    std::vector<float> descriptor = describePatch(patch);
    cv::Mat sample = cv::Mat(descriptor).reshape(1, 1);
    cv::Mat raw;
    svm->predict(sample, raw, cv::ml::StatModel::RAW_OUTPUT); // unlike SVMSGD, this works for cv::ml::SVM
    return -raw.at<float>(0, 0); // sign is flipped: negative = positive class, so negate it
}

// stage 2 must see in training exactly what it will see at inference: the patches stage 1 lets
// through, not all of them. The stage 1 score is rebuilt by hand here for the same reason as in
// rawScore().
static cv::Mat keepSurvivors(const std::vector<std::string>& files, const cv::Mat& weights, float shift, float stage1Thr){
    const int descriptorSize = weights.cols;
    std::vector<cv::Mat> kept;

    for(size_t i = 0; i < files.size(); i++)
    {
        cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);
        if(img.empty() || img.size() != kPatchSize) continue;

        std::vector<float> descriptor = describePatch(img);
        float score = shift;
        for(int j = 0; j < descriptorSize; j++)
        {
            score += weights.at<float>(0, j) * descriptor[j];
        }
        if(score <= stage1Thr) continue;

        kept.push_back(cv::Mat(1, descriptorSize, CV_32F, descriptor.data()).clone());
    }

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
    std::mt19937 rng(42);
    std::shuffle(indexes.begin(), indexes.end(), rng);

    cv::Mat reduced(cap, features.cols, CV_32F);
    for(int i = 0; i < cap; i++)
    {
        features.row(indexes[i]).copyTo(reduced.row(i));
    }
    return reduced;
}

void trainStage2(const std::string& posDir, const std::string& negDir, const std::string& stage1Path, const std::string& modelPath, float stage1Thr, int maxPerClass, double fixedC){
    cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load(stage1Path);
    if(stage1.empty())
    {
        throw std::runtime_error("trainStage2: cannot load " + stage1Path);
    }

    cv::Mat weights = stage1->getWeights();
    const float shift = stage1->getShift();

    cv::Mat posFeatures = keepSurvivors(listPatchFiles(posDir), weights, shift, stage1Thr);
    cv::Mat negFeatures = keepSurvivors(listPatchFiles(negDir), weights, shift, stage1Thr);
    std::cout << "[trainStage2] survivors of stage 1 (thr=" << stage1Thr << "): pos=" << posFeatures.rows << " neg=" << negFeatures.rows << '\n';

    if(posFeatures.rows == 0 || negFeatures.rows == 0)
    {
        throw std::runtime_error("trainStage2: one of the two classes is empty, lower the stage 1 threshold");
    }

    // an RBF SVM scales badly with the number of samples, so both classes are capped
    posFeatures = subsample(posFeatures, maxPerClass);
    negFeatures = subsample(negFeatures, maxPerClass);

    // balanced classes: measured slightly better than leaving more positives than negatives
    const int perClass = std::min(posFeatures.rows, negFeatures.rows);
    posFeatures = posFeatures.rowRange(0, perClass);
    negFeatures = negFeatures.rowRange(0, perClass);
    std::cout << "[trainStage2] training on " << perClass << " samples per class\n";

    cv::Mat features;
    cv::vconcat(posFeatures, negFeatures, features);
    cv::Mat labels(features.rows, 1, CV_32S); // cv::ml::SVM wants CV_32S, unlike SVMSGD
    labels.rowRange(0, perClass).setTo(1);
    labels.rowRange(perClass, features.rows).setTo(-1);

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-3));

    auto t0 = std::chrono::steady_clock::now();
    if(fixedC > 0)
    {
        std::cout << "[trainStage2] training with C fixed at " << fixedC << "\n";
        svm->setC(fixedC);
        svm->setGamma(0.50625);
        svm->train(features, cv::ml::ROW_SAMPLE, labels);
    }
    else
    {
        std::cout << "[trainStage2] trainAuto with 5-fold cross validation, this takes a while...\n";
        svm->trainAuto(cv::ml::TrainData::create(features, cv::ml::ROW_SAMPLE, labels), 5);
    }
    auto t1 = std::chrono::steady_clock::now();

    std::cout << "[trainStage2] training time: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    std::cout << "[trainStage2] C=" << svm->getC() << " gamma=" << svm->getGamma() << " #SV=" << svm->getSupportVectors().rows << '\n';

    svm->save(modelPath);
    std::cout << "[trainStage2] model saved to: " << modelPath << '\n';

}
