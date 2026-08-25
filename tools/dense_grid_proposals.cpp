// File a parte, non collegato a CMakeLists: approccio radicalmente diverso dal blob-bbox
// puro di color_proposals.cpp. Idea: invece di prendere UN box = UN blob (fragile per coni
// piccoli/lontani la cui maschera e' frammentata o copre solo una parte del cono, es. solo la
// punta), si usa la maschera colore (permissiva, niente filtro area, niente opening aggressivo)
// solo per delimitare REGIONI di interesse, e dentro ogni regione si genera una GRIGLIA di
// candidati aspect-normalizzati a piu' scale, tassellando la bbox del blob invece di fidarsi
// della sua forma esatta. Il costo e' candidati/immagine molto piu' alti (il classificatore a
// valle, gia' riallenato sui candidati reali, deve filtrare di piu'), il beneficio atteso e'
// una coverage molto piu' alta perche' basta che il blob esista da qualche parte vicino al cono,
// non che la sua bbox combaci.
//
// Compilazione ad-hoc (dalla cartella build/):
//   g++ -std=gnu++17 -O2 -I../include -I/usr/include/opencv4 \
//     ../tools/dense_grid_proposals.cpp ../src/utils.cpp \
//     -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
//     -o dense_grid_proposals
//   ./dense_grid_proposals [numero_immagini] [seed]

#include "hog.hpp"
#include "utils.hpp"
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

const double kAspect = static_cast<double>(kPatchSize.width) / kPatchSize.height; // w/h

cv::Mat whiteBalance(const cv::Mat& imgBGR, double p){
    cv::Mat imgF; imgBGR.convertTo(imgF, CV_32F);
    cv::Mat expImg; cv::pow(imgF, p, expImg);
    cv::Scalar m = cv::mean(expImg);
    double e[3];
    for(int i = 0; i < 3; i++){ e[i] = std::pow(m[i], 1.0/p); if(e[i] < 1e-6) e[i] = 1e-6; }
    double gainMean = (e[0]+e[1]+e[2])/3;
    cv::Scalar gain(gainMean/e[0], gainMean/e[1], gainMean/e[2]);
    cv::multiply(imgF, gain, imgF);
    cv::Mat out; imgF.convertTo(out, CV_8U);
    return out;
}

cv::Mat colorMask(const cv::Mat& imgBGR){
    cv::Mat hsv; cv::cvtColor(imgBGR, hsv, cv::COLOR_BGR2HSV);
    cv::Mat blue, orange, yellow, mask;
    cv::inRange(hsv, cv::Scalar(100, 80, 60), cv::Scalar(130, 255, 255), blue);
    cv::inRange(hsv, cv::Scalar(  5, 80, 60), cv::Scalar( 20, 255, 255), orange);
    cv::inRange(hsv, cv::Scalar( 20, 80, 60), cv::Scalar( 35, 255, 255), yellow);
    cv::bitwise_or(blue, orange, mask);
    cv::bitwise_or(mask, yellow, mask);
    return mask;
}

// tassella una regione (bbox di un blob, magari grande e irregolare) con finestre
// aspect-normalizzate ad altezze ASSOLUTE (non relative alla dimensione del blob: un blob
// puo' essere un frammento minuscolo di un cono molto piu' grande, usare la sua altezza come
// riferimento sottostima sistematicamente la finestra). stride = meta' finestra.
// tassella SOLO dove la maschera originale (non dilatata) ha davvero pixel di colore: la bbox
// del blob dilatato serve solo a delimitare l'area di ricerca, ma tassellarla per intero (specie
// se il dilate ha fuso mezza immagine) genera candidati quasi tutti su sfondo puro. Si scarta
// una finestra se non contiene nessun pixel della maschera originale.
std::vector<cv::Rect> tileRegion(const cv::Rect& region, cv::Size imgSize,
                                   const std::vector<double>& heights, const cv::Mat& rawMask,
                                   double minCoverageFrac){
    std::vector<cv::Rect> out;
    for(double h : heights){
        double w = h * kAspect;
        double strideY = std::max(4.0, h * 0.5);
        double strideX = std::max(4.0, w * 0.5);

        double y0 = region.y - h * 0.25; // un po' di margine oltre i bordi della regione
        double y1 = region.y + region.height + h * 0.25;
        double x0 = region.x - w * 0.25;
        double x1 = region.x + region.width + w * 0.25;

        for(double cy = y0 + h/2; cy <= y1; cy += strideY){
            for(double cx = x0 + w/2; cx <= x1; cx += strideX){
                int nx = static_cast<int>(std::lround(cx - w/2));
                int ny = static_cast<int>(std::lround(cy - h/2));
                int nw = static_cast<int>(std::lround(w));
                int nh = static_cast<int>(std::lround(h));
                nx = std::max(0, nx); ny = std::max(0, ny);
                nw = std::min(nw, imgSize.width - nx);
                nh = std::min(nh, imgSize.height - ny);
                if(nw < 4 || nh < 4) continue;
                cv::Rect box(nx, ny, nw, nh);
                double frac = cv::countNonZero(rawMask(box)) / double(box.area());
                if(frac < minCoverageFrac) continue;
                out.push_back(box);
            }
        }
    }
    return out;
}

// un box per blob per ogni altezza assoluta, centrato sul centroide del blob: niente tiling
// spaziale (niente stride), molto piu' economico del grid completo. Assume che il centro del
// blob (anche se frammento) sia una stima ragionevole del centro del cono vero.
std::vector<cv::Rect> centeredMultiScale(const cv::Rect& region, cv::Size imgSize,
                                          const std::vector<double>& heights, double snapGrid){
    std::vector<cv::Rect> out;
    double cx = region.x + region.width / 2.0;
    double cy = region.y + region.height / 2.0;
    if(snapGrid > 0){
        cx = std::round(cx / snapGrid) * snapGrid;
        cy = std::round(cy / snapGrid) * snapGrid;
    }
    for(double h : heights){
        double w = h * kAspect;
        int nx = static_cast<int>(std::lround(cx - w/2));
        int ny = static_cast<int>(std::lround(cy - h/2));
        int nw = static_cast<int>(std::lround(w));
        int nh = static_cast<int>(std::lround(h));
        nx = std::max(0, nx); ny = std::max(0, ny);
        nw = std::min(nw, imgSize.width - nx);
        nh = std::min(nh, imgSize.height - ny);
        if(nw >= 4 && nh >= 4) out.push_back(cv::Rect(nx, ny, nw, nh));
    }
    return out;
}

enum class Mode { GridTile, CenteredMultiScale };

struct Config {
    std::string name;
    Mode mode;
    cv::Size dilateKernel; // fonde frammenti vicini in un'unica regione prima di trovare i contorni
    double maxAreaFrac;    // per scartare le regioni enormi (cielo/erba interi), non i blob normali
    std::vector<double> heights; // altezze ASSOLUTE (px) delle finestre generate
    double minCoverageFrac; // (solo GridTile) frazione minima di pixel-maschera dentro la finestra
    double snapGrid = 0;    // (solo CenteredMultiScale) arrotonda il centroide a una griglia, cosi'
                             // frammenti vicini generano lo stesso box (deduplicato) invece di box diversi
    int minBlobAreaPx = 1;  // scarta i blob (contorni della maschera grezza) piu' piccoli di questo
};

// molti frammenti vicini generano candidati quasi identici (stesso box arrotondato alla
// stessa posizione/scala): deduplica per tenere il conteggio sotto controllo senza perdere
// copertura (un duplicato non aggiunge mai un IoU migliore).
std::vector<cv::Rect> dedup(std::vector<cv::Rect> boxes){
    std::sort(boxes.begin(), boxes.end(), [](const cv::Rect& a, const cv::Rect& b){
        if(a.x != b.x) return a.x < b.x;
        if(a.y != b.y) return a.y < b.y;
        if(a.width != b.width) return a.width < b.width;
        return a.height < b.height;
    });
    boxes.erase(std::unique(boxes.begin(), boxes.end(), [](const cv::Rect& a, const cv::Rect& b){
        return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
    }), boxes.end());
    return boxes;
}

std::vector<cv::Rect> runConfig(const cv::Mat& imgBGR, const Config& cfg){
    cv::Mat corrected = whiteBalance(imgBGR, 6.0);
    cv::Mat mask = colorMask(corrected);

    cv::Mat dilated;
    cv::Mat elemDilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cfg.dilateKernel);
    cv::dilate(mask, dilated, elemDilate);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double maxArea = cfg.maxAreaFrac * mask.rows * mask.cols;

    std::vector<cv::Rect> candidates;
    for(auto& c : contours){
        cv::Rect r = cv::boundingRect(c);
        if(r.area() > maxArea) continue;
        if(r.area() < cfg.minBlobAreaPx) continue;
        std::vector<cv::Rect> generated = (cfg.mode == Mode::GridTile)
            ? tileRegion(r, imgBGR.size(), cfg.heights, mask, cfg.minCoverageFrac)
            : centeredMultiScale(r, imgBGR.size(), cfg.heights, cfg.snapGrid);
        candidates.insert(candidates.end(), generated.begin(), generated.end());
    }
    return dedup(std::move(candidates));
}

} // namespace

int main(int argc, char** argv){
    int nSample = argc > 1 ? std::atoi(argv[1]) : 150;
    unsigned seed = argc > 2 ? (unsigned)std::atoi(argv[2]) : 7;

    const std::string datasetDir = "../dataset";
    const std::string csvPath = "../data/dataset.csv";
    const std::string splitFile = "../data/test.txt";

    std::unordered_map<std::string, std::vector<cv::Rect>> gt;
    {
        std::ifstream f(csvPath); std::string l;
        while(std::getline(f, l)){
            std::stringstream ss(l);
            std::string img, x1s, y1s, x2s, y2s;
            std::getline(ss, img, ','); std::getline(ss, x1s, ',');
            std::getline(ss, y1s, ','); std::getline(ss, x2s, ',');
            std::getline(ss, y2s, ',');
            int x1 = std::stoi(x1s), y1 = std::stoi(y1s), x2 = std::stoi(x2s), y2 = std::stoi(y2s);
            gt[img].push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        }
    }
    std::vector<std::string> testImgs;
    { std::ifstream f(splitFile); std::string l; while(std::getline(f, l)) if(!l.empty()) testImgs.push_back(l); }
    std::mt19937 rng(seed);
    std::shuffle(testImgs.begin(), testImgs.end(), rng);
    if((int)testImgs.size() > nSample) testImgs.resize(nSample);

    std::vector<double> heightsFull = {8,12,18,26,38,55,80,120,180,260};
    std::vector<double> heightsCoarse = {10,18,30,50,85,140,220};
    std::vector<double> heightsVeryCoarse = {12,22,40,75,140};
    using M = Mode;
    std::vector<Config> configs = {
        {"HH: minArea3, heightsCoarse", M::CenteredMultiScale, {1,1}, 0.50, heightsCoarse, 0, 0, 3},
    };

    for(auto& cfg : configs){
        long long nGt = 0, nCovered = 0, nCands = 0;
        int nImg = 0;
        for(const auto& imgName : testImgs){
            auto it = gt.find(imgName);
            if(it == gt.end() || it->second.empty()) continue;
            cv::Mat bgr = cv::imread(datasetDir + "/" + imgName, cv::IMREAD_COLOR);
            if(bgr.empty()) continue;
            nImg++;

            std::vector<cv::Rect> cands = runConfig(bgr, cfg);
            nCands += (long long)cands.size();

            for(const auto& gb : it->second){
                nGt++;
                double bestIou = 0.0;
                for(const auto& c : cands) bestIou = std::max(bestIou, calculateIoU(c, gb));
                if(bestIou >= 0.3) nCovered++;
            }
        }
        std::cout << cfg.name << ":  coverage=" << (100.0*nCovered/nGt) << "%  ("
                   << nCovered << "/" << nGt << ")   candidati/immagine=" << (double(nCands)/nImg) << "\n";
    }

    return 0;
}
