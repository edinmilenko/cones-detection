// Standalone file, not wired into CMakeLists: measures the segmentation ALONE, away from detection
// and classification. It takes the bitmap masks of the real external test set
// (test_set/segmentation_test, 398 cones annotated with a mask), derives the tight box, pads it the
// way the detector does, and compares the mask segmentCone() produces with the annotated one. Good
// for seeing what the detector padding costs the segmentation, and for comparing variants of the
// algorithm without running the whole pipeline.
//
// Ad-hoc build (from the build/ folder):
//   g++ -std=gnu++17 -O2 -I../include -I../tools -I/usr/include/opencv4 \
//     ../tools/eval_segmentation.cpp ../src/segmentation.cpp ../src/classifier.cpp \
//     ../src/utils.cpp ../src/dataset.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lz \
//     -o eval_segmentation
//   ./eval_segmentation [variant] [output_folder] [pad]

#include "dataset.hpp"
#include "classifier.hpp"
#include "segmentation.hpp"
#include "segmentation_variants.hpp"
#include "utils.hpp"
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <map>

namespace fs = std::filesystem;

namespace {

// copy of the constants in src/detector.cpp: in production the segmentation sees this box, not the
// tight one. If they change over there, they have to change here too.
constexpr double kPadTop = 0.0, kPadBottom = 0.4, kPadSide = 0.1;

cv::Rect padBox(const cv::Rect& box, cv::Size imgSize){
    int extraSide = (int)std::lround(box.width * kPadSide);
    int extraTop = (int)std::lround(box.height * kPadTop);
    int extraBottom = (int)std::lround(box.height * kPadBottom);
    int x = std::max(0, box.x - extraSide);
    int y = std::max(0, box.y - extraTop);
    int x2 = std::min(imgSize.width, box.x + box.width + extraSide);
    int y2 = std::min(imgSize.height, box.y + box.height + extraBottom);
    return cv::Rect(cv::Point(x, y), cv::Point(x2, y2));
}

int classFromLabel(const std::string& label){
    if(label == "seg_yellow_cone") return (int)ConeClass::YELLOW;
    if(label == "seg_blue_cone") return (int)ConeClass::BLUE;
    if(label == "seg_orange_cone") return (int)ConeClass::SMALL_ORANGE;
    if(label == "seg_large_orange_cone") return (int)ConeClass::BIG_ORANGE;
    return 0; // seg_unknown_cone and anything else: outside the 4 classes, skipped
}

struct Bucket {
    double sumIou = 0;
    int n = 0;
    void add(double iou){ sumIou += iou; n++; }
    double mean() const { return n ? sumIou / n : 0.0; }
};

} // namespace

int main(int argc, char** argv){
    const int variant = argc > 1 ? std::atoi(argv[1]) : 0;
    const std::string outDir = argc > 2 ? argv[2] : "";
    const double maskPad = argc > 3 ? std::atof(argv[3]) : 0.05;   // padding of the box taken from the mask

    const fs::path annDir = "../test_set/segmentation_test/ann";
    const fs::path imgDir = "../test_set/segmentation_test/img";

    if(!outDir.empty()) fs::create_directories(outDir);

    Bucket all, tight;
    Bucket boxPadded, boxFromMask;          // IoU of both boxes with the true one
    Bucket classOk;                         // classifier accuracy on the true boxes
    std::map<int, std::map<int, int>> confusion;
    std::map<std::string, Bucket> bySize;   // bucketed by cone height in pixels
    std::map<int, Bucket> byClass;

    std::vector<fs::path> annFiles;
    for(const auto& e : fs::directory_iterator(annDir))
    {
        if(e.path().extension() == ".json") annFiles.push_back(e.path());
    }
    std::sort(annFiles.begin(), annFiles.end());

    for(const fs::path& annPath : annFiles)
    {
        const fs::path imgPath = imgDir / annPath.stem();   // "amz_01260.png.json" -> "amz_01260.png"
        cv::Mat image = cv::imread(imgPath.string(), cv::IMREAD_COLOR);
        if(image.empty())
        {
            std::cerr << "missing image: " << imgPath << "\n";
            continue;
        }

        cv::Mat vis;
        if(!outDir.empty()) vis = image.clone();

        for(const Annotation& ann : readAnnotations(annPath))
        {
            if(ann.geometry != AnnotationGeometry::Bitmap) continue;
            const int cls = classFromLabel(ann.label);
            if(cls == 0) continue;

            // the PNG mask can carry zero padding around it: the tight box is the bbox of the
            // content, not the size of the mask
            const cv::Rect content = cv::boundingRect(ann.mask);
            if(content.area() <= 0) continue;
            const cv::Rect tightBox = (content + ann.origin) & cv::Rect(0, 0, image.cols, image.rows);
            if(tightBox.width < 4 || tightBox.height < 4) continue;

            const cv::Rect box = padBox(tightBox, image.size());

            // annotated mask moved into the frame of the padded box
            cv::Mat gt = cv::Mat::zeros(box.size(), CV_8UC1);
            const cv::Rect maskBounds(ann.origin.x, ann.origin.y, ann.mask.cols, ann.mask.rows);
            const cv::Rect overlap = maskBounds & box;
            if(overlap.area() > 0)
            {
                const cv::Rect src(overlap.x - maskBounds.x, overlap.y - maskBounds.y, overlap.width, overlap.height);
                const cv::Rect dst(overlap.x - box.x, overlap.y - box.y, overlap.width, overlap.height);
                ann.mask(src).copyTo(gt(dst));
            }

            // in the pipeline the classifier runs on the padded box, not on the tight one
            const int predictedCls = classifier(image, {box})[0];
            classOk.add(predictedCls == cls ? 1.0 : 0.0);
            confusion[cls][predictedCls]++;

            const cv::Mat pred = segmentVariant(image(box), cls, variant);
            const double iou = maskIoU(pred, gt);

            all.add(iou);
            byClass[cls].add(iou);

            // how much tighter the box from the mask is than the padded one, measured against the
            // true box of the cone
            boxPadded.add(calculateIoU(box, tightBox));
            const cv::Rect maskBox = cv::boundingRect(pred);
            if(maskBox.area() > 0)
            {
                const int padX = (int)std::lround(maskBox.width * maskPad);
                const int padY = (int)std::lround(maskBox.height * maskPad);
                cv::Rect refined(maskBox.x + box.x - padX, maskBox.y + box.y - padY,
                                 maskBox.width + 2 * padX, maskBox.height + 2 * padY);
                boxFromMask.add(calculateIoU(refined & cv::Rect(0, 0, image.cols, image.rows), tightBox));
            }
            else
            {
                boxFromMask.add(calculateIoU(box, tightBox));   // empty mask: keep the original box
            }
            const int h = tightBox.height;
            bySize[h < 25 ? "  <25px" : (h < 60 ? " 25-60px" : "  >60px")].add(iou);

            // same thing on the tight box, as a reference for what the padding costs
            cv::Mat gtTight = cv::Mat::zeros(tightBox.size(), CV_8UC1);
            const cv::Rect overlapT = maskBounds & tightBox;
            if(overlapT.area() > 0)
            {
                const cv::Rect src(overlapT.x - maskBounds.x, overlapT.y - maskBounds.y, overlapT.width, overlapT.height);
                const cv::Rect dst(overlapT.x - tightBox.x, overlapT.y - tightBox.y, overlapT.width, overlapT.height);
                ann.mask(src).copyTo(gtTight(dst));
            }
            tight.add(maskIoU(segmentVariant(image(tightBox), cls, variant), gtTight));

            if(!vis.empty())
            {
                const cv::Scalar color = classColor(cls);
                cv::Mat roi = vis(box);
                cv::Mat overlay = roi.clone();
                overlay.setTo(color, pred);
                cv::addWeighted(roi, 0.4, overlay, 0.6, 0.0, roi);
                cv::rectangle(vis, box, color, 1);
                cv::rectangle(vis, tightBox, cv::Scalar(0, 255, 0), 1);   // green = ground truth
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << iou;
                cv::putText(vis, ss.str(), {box.x, std::max(12, box.y - 4)}, cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
            }
        }

        if(!vis.empty()) cv::imwrite(outDir + "/" + annPath.stem().string(), vis);
    }

    std::cout << "variant " << variant << "   cones: " << all.n << "\n";
    std::cout << "  mIoU on padded box : " << std::fixed << std::setprecision(4) << all.mean() << "\n";
    std::cout << "  mIoU on tight box  : " << tight.mean() << "   (reference only, not what the pipeline sees)\n";
    std::cout << "  box: IoU with the true box -> padded " << boxPadded.mean()
              << " | taken from the mask (pad " << maskPad << ") " << boxFromMask.mean() << "\n";
    for(const auto& [k, b] : bySize)
    {
        std::cout << "    " << k << " : " << b.mean() << "  (" << b.n << " cones)\n";
    }
    for(const auto& [k, b] : byClass)
    {
        std::cout << "    " << className(k) << " : " << b.mean() << "  (" << b.n << " cones)\n";
    }

    std::cout << "  classification accuracy : " << classOk.mean() << "\n";
    for(const auto& [trueCls, row] : confusion)
    {
        std::cout << "    " << className(trueCls) << " ->";
        for(const auto& [predictedCls, n] : row)
        {
            std::cout << "  " << className(predictedCls) << ":" << n;
        }
        std::cout << "\n";
    }
    return 0;
}
