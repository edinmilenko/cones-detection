// File a parte, non collegato a CMakeLists: secondo stadio della cascata. Carica le patch gia'
// estratte da extract_proposal_patches.cpp (pos_proposal/neg_proposal, candidati reali del
// detector), calcola per ognuna lo score dello stadio 1 (il lineare gia' allenato,
// data/svm_hog_sgd_proposal.yml) e tiene SOLO quelle che lo stadio 1 lascerebbe passare
// (score > STAGE1_THR) — lo stadio 2 deve vedere in training esattamente la distribuzione che
// vedra' in inferenza (i sopravvissuti allo stadio 1), non tutti i candidati. Allena un
// cv::ml::SVM con kernel RBF (non lineare, piu' capacita' del lineare) su quel sottoinsieme.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/train_stage2.cpp ../src/hog.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_ml \
//     -o train_stage2
//   ./train_stage2 [stage1_thr] [max_per_class] [output.yml]

#include "hog.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
#include <random>

namespace fs = std::filesystem;

namespace {
std::vector<std::string> listFiles(const std::string& dir){
    std::vector<std::string> files;
    for(const auto& e : fs::directory_iterator(dir)) if(e.is_regular_file()) files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    return files;
}
}

int main(int argc, char** argv){
    float stage1Thr = argc > 1 ? std::atof(argv[1]) : 10.0f;
    // limite di sicurezza sul numero di campioni passati a trainAuto (RBF e' O(n^2)-ish):
    // se dopo il filtro dello stadio 1 restano troppi campioni, sottocampiona.
    size_t maxPerClass = argc > 2 ? (size_t)std::atoi(argv[2]) : 15000;
    std::string outPath = argc > 3 ? argv[3] : "../data/svm_stage2_rbf.yml";

    std::vector<std::string> posFiles = listFiles("../data/patches/pos_proposal");
    std::vector<std::string> negFiles = listFiles("../data/patches/neg_proposal");
    std::cout << "patch disponibili: pos=" << posFiles.size() << " neg=" << negFiles.size() << "\n";

    cv::Ptr<cv::ml::SVMSGD> stage1 = cv::ml::SVMSGD::load("../data/svm_hog_sgd_proposal.yml");
    cv::Mat W = stage1->getWeights();
    float shift = stage1->getShift();
    cv::HOGDescriptor hog = makeHog();
    const int descSize = static_cast<int>(hog.getDescriptorSize());

    auto scoreAndFilter = [&](const std::vector<std::string>& files, cv::Mat& keptFeatures) -> int {
        std::vector<cv::Mat> kept;
        for(const auto& f : files){
            cv::Mat img = cv::imread(f, cv::IMREAD_GRAYSCALE);
            if(img.empty()) continue;
            std::vector<float> descriptor;
            hog.compute(img, descriptor);
            float s = shift;
            for(int j = 0; j < descSize; j++) s += W.at<float>(0, j) * descriptor[j];
            if(s <= stage1Thr) continue; // lo stadio 1 lo scarterebbe: lo stadio 2 non deve vederlo
            kept.push_back(cv::Mat(1, descSize, CV_32F, descriptor.data()).clone());
        }
        int n = (int)kept.size();
        keptFeatures.create(n, descSize, CV_32F);
        for(int i = 0; i < n; i++) kept[i].copyTo(keptFeatures.row(i));
        return n;
    };

    cv::Mat posFeat, negFeat;
    int nPos = scoreAndFilter(posFiles, posFeat);
    int nNeg = scoreAndFilter(negFiles, negFeat);
    std::cout << "sopravvissuti allo stadio 1 (thr=" << stage1Thr << "): pos=" << nPos << " neg=" << nNeg << "\n";

    if(nPos == 0 || nNeg == 0){
        std::cerr << "nessun campione sopravvissuto in una delle due classi, abbassa la soglia\n";
        return 1;
    }

    // sottocampiona se troppi (RBF SVM training e' costoso in campioni)
    auto subsample = [&](cv::Mat& feat, int cap) {
        if(feat.rows <= cap) return;
        std::vector<int> idx(feat.rows);
        for(int i = 0; i < feat.rows; i++) idx[i] = i;
        std::mt19937 rng(42);
        std::shuffle(idx.begin(), idx.end(), rng);
        cv::Mat sub(cap, feat.cols, CV_32F);
        for(int i = 0; i < cap; i++) feat.row(idx[i]).copyTo(sub.row(i));
        feat = sub;
    };
    subsample(posFeat, (int)maxPerClass);
    subsample(negFeat, (int)maxPerClass);
    std::cout << "dopo sottocampionamento (cap=" << maxPerClass << "): pos=" << posFeat.rows
               << " neg=" << negFeat.rows << "\n";

    cv::Mat features, labels;
    cv::vconcat(posFeat, negFeat, features);
    labels.create(features.rows, 1, CV_32S);
    labels.rowRange(0, posFeat.rows).setTo(1);
    labels.rowRange(posFeat.rows, features.rows).setTo(-1);

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-3));

    std::cout << "training RBF SVM (trainAuto, puo' volerci qualche minuto)...\n";
    cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(features, cv::ml::ROW_SAMPLE, labels);
    auto t0 = std::chrono::steady_clock::now();
    svm->trainAuto(data, 5); // 5-fold CV per C e gamma
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "tempo di training: " << std::chrono::duration<double>(t1-t0).count() << " s\n";
    std::cout << "C=" << svm->getC() << " gamma=" << svm->getGamma() << " #SV=" << svm->getSupportVectors().rows << "\n";

    svm->save(outPath);
    std::cout << "modello salvato in " << outPath << "\n";
    return 0;
}
