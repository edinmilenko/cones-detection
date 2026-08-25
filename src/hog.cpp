#include "hog.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

// single definition of the HOG parameters: training and detection both read from here, so
// a wrong-size descriptor can't arise from two places in the code drifting out of sync.
cv::HOGDescriptor makeHog(){
    return cv::HOGDescriptor(kPatchSize, cv::Size(8, 8), cv::Size(4, 4), cv::Size(4, 4), 9);
}

namespace {

// sorted (for reproducibility) list of regular files in a patch folder.
std::vector<std::string> listPatchFiles(const std::string& dir){
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

} // namespace

cv::Mat computeFeatures(const std::vector<std::string>& files){
    cv::HOGDescriptor hog = makeHog();
    const int descriptorSize = static_cast<int>(hog.getDescriptorSize());
    cv::Mat features(static_cast<int>(files.size()), descriptorSize, CV_32F);

    for(size_t i = 0; i < files.size(); i++)
    {
        cv::Mat img = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
        if(img.empty())
        {
            throw std::runtime_error("computeFeatures: cannot read " + files[i]);
        }
        if(img.size() != kPatchSize)
        {
            throw std::runtime_error("computeFeatures: " + files[i] + " has size " + std::to_string(img.cols) + "x" + std::to_string(img.rows) + ", expected " + std::to_string(kPatchSize.width) + "x" + std::to_string(kPatchSize.height));
        }

        std::vector<float> descriptor;
        hog.compute(img, descriptor);
        if(static_cast<int>(descriptor.size()) != descriptorSize)
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

float classify(const cv::Ptr<cv::ml::SVMSGD>& svm, const cv::Mat& patch){
    if(patch.size() != kPatchSize)
    {
        throw std::runtime_error("classify: patch does not have the expected size " + std::to_string(kPatchSize.width) + "x" + std::to_string(kPatchSize.height));
    }
    cv::HOGDescriptor hog = makeHog();
    std::vector<float> descriptor;
    hog.compute(patch, descriptor);
    cv::Mat sample = cv::Mat(descriptor).reshape(1, 1);
    return svm->predict(sample); // discrete +1/-1 label, not a score (see rawScore)
}

float rawScore(const cv::Ptr<cv::ml::SVMSGD>& svm, const cv::Mat& patch){
    if(patch.size() != kPatchSize)
    {
        throw std::runtime_error("rawScore: patch does not have the expected size " + std::to_string(kPatchSize.width) + "x" + std::to_string(kPatchSize.height));
    }
    cv::HOGDescriptor hog = makeHog();
    std::vector<float> descriptor;
    hog.compute(patch, descriptor);
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
    cv::HOGDescriptor hog = makeHog();
    std::vector<float> descriptor;
    hog.compute(patch, descriptor);
    cv::Mat sample = cv::Mat(descriptor).reshape(1, 1);
    cv::Mat raw;
    svm->predict(sample, raw, cv::ml::StatModel::RAW_OUTPUT); // unlike SVMSGD, this works for cv::ml::SVM
    return -raw.at<float>(0, 0); // sign is flipped: negative = positive class, so negate it
}
