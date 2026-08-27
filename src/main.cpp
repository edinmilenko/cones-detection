#include "labeler.hpp"
#include "dataloader.hpp"
#include "classifier.hpp"
#include "segmentation.hpp" 
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <filesystem>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <iomanip> 

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    
    try {
        std::cout << "\n--- Avvio test su 200 immagini casuali ---" << std::endl;
        
        const fs::path datasetDir = "../dataset";
        std::vector<fs::path> imageFiles;

        if (fs::exists(datasetDir)) {
            for (const auto& entry : fs::directory_iterator(datasetDir)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                        imageFiles.push_back(entry.path());
                    }
                }
            }
        } else {
             std::cerr << "Attenzione: Cartella dataset non trovata!" << std::endl;
             return 1;
        }

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(imageFiles.begin(), imageFiles.end(), g);

        size_t numImagesToTest = std::min<size_t>(200, imageFiles.size());
        std::cout << "Selezionate " << numImagesToTest << " immagini per il test.\n" << std::endl;

        std::string winName = "Test In Corso (Premi un tasto per la prossima, 'ESC' per uscire)";
        cv::namedWindow(winName, cv::WINDOW_NORMAL); 
        cv::resizeWindow(winName, 1600, 600); 

        // ==========================================
        // VARIABILI PER IL REPORT FINALE GLOBALE
        // ==========================================
        int globalCorrectCones = 0;
        int globalTotalCones = 0;
        double globalTotalIoU = 0.0;
        int globalValidIoUCount = 0;

        for (size_t i = 0; i < numImagesToTest; ++i) {
            const fs::path& imagePath = imageFiles[i];
            cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
            
            if (image.empty()) continue;

            const fs::path jsonPath = datasetDir / (imagePath.stem().string() + ".json");
            
            std::vector<cv::Rect> bboxesToTest;
            std::vector<int> groundTruthLabels; 
            std::vector<cv::Mat> groundTruthMasks; // Salviamo le maschere perfette del dataset

            // 1. LETTURA JSON E GROUND TRUTH
            if (fs::exists(jsonPath)) {
                std::vector<Annotation> annotations = readAnnotations(jsonPath);
                for (const Annotation& ann : annotations) {
                    
                    int trueClass = -1;
                    std::string rawLabel = ann.label;
                    
                    std::transform(rawLabel.begin(), rawLabel.end(), rawLabel.begin(), ::tolower);

                    if (rawLabel == "yellow_cone" || rawLabel == "seg_yellow_cone") {
                        trueClass = static_cast<int>(ConeClass::YELLOW);
                    } 
                    else if (rawLabel == "blue_cone" || rawLabel == "seg_blue_cone") {
                        trueClass = static_cast<int>(ConeClass::BLUE);
                    } 
                    else if (rawLabel == "small_orange_cone" || rawLabel == "seg_small_orange_cone") {
                        trueClass = static_cast<int>(ConeClass::SMALL_ORANGE);
                    } 
                    else if (rawLabel == "large_orange_cone" || rawLabel == "seg_large_orange_cone") {
                        trueClass = static_cast<int>(ConeClass::BIG_ORANGE);
                    } 
                    else if (rawLabel == "unknown_cone" || rawLabel == "seg_unknown_cone") {
                        continue; 
                    } 
                    else {
                        continue; 
                    }

                    cv::Rect box;
                    cv::Mat gtMask;
                    bool isValidBox = false;
                    
                    if (ann.geometry == AnnotationGeometry::Rectangle) {
                        box = cv::Rect(std::min(ann.topLeft.x, ann.bottomRight.x), 
                                     std::min(ann.topLeft.y, ann.bottomRight.y),
                                     std::abs(ann.topLeft.x - ann.bottomRight.x) + 1, 
                                     std::abs(ann.topLeft.y - ann.bottomRight.y) + 1);
                        // Se è solo un rettangolo, la maschera GT è un blocco bianco
                        gtMask = cv::Mat(box.height, box.width, CV_8UC1, cv::Scalar(255));
                        isValidBox = true;
                    } 
                    else if (ann.geometry == AnnotationGeometry::Bitmap) {
                        box = cv::Rect(ann.origin.x, ann.origin.y, ann.mask.cols, ann.mask.rows);
                        gtMask = ann.mask.clone(); // Usiamo la maschera perfetta del dataset
                        isValidBox = true;
                    }

                    if (isValidBox) {
                        bboxesToTest.push_back(box);
                        groundTruthLabels.push_back(trueClass);
                        groundTruthMasks.push_back(gtMask);
                    }
                }
            }

            // 2. CLASSIFICAZIONE
            std::vector<int> predictedLabels = classifier(image, bboxesToTest);

            cv::Mat img_class = image.clone();
            cv::Mat img_segm = image.clone();

            int correctConesCount = 0; 
            double imageTotalIoU = 0.0;
            int imageValidIoUCount = 0;

            // 3. CONFRONTO, SEGMENTAZIONE E CALCOLO IoU
            for (size_t j = 0; j < bboxesToTest.size(); ++j) {
                cv::Rect box = bboxesToTest[j];
                
                // --- SAFETY CHECK ---
                cv::Rect safeBox = box & cv::Rect(0, 0, image.cols, image.rows);
                if (safeBox.width <= 0 || safeBox.height <= 0) continue;

                // Ritaglio della maschera GroundTruth nel caso in cui il BBox uscisse dall'immagine
                cv::Mat gtMask = groundTruthMasks[j];
                if (safeBox != box) {
                    cv::Rect gtCrop(safeBox.x - box.x, safeBox.y - box.y, safeBox.width, safeBox.height);
                    gtMask = gtMask(gtCrop);
                }

                int labelInt = predictedLabels[j];
                if (labelInt == groundTruthLabels[j]) {
                    correctConesCount++;
                }

                std::string labelText;
                cv::Scalar color; 

                if (labelInt == static_cast<int>(ConeClass::YELLOW)) {
                    labelText = "YELLOW"; color = cv::Scalar(0, 255, 255);
                } else if (labelInt == static_cast<int>(ConeClass::BLUE)) {
                    labelText = "BLUE"; color = cv::Scalar(255, 0, 0);
                } else if (labelInt == static_cast<int>(ConeClass::SMALL_ORANGE)) {
                    labelText = "S_ORANGE"; color = cv::Scalar(0, 165, 255);
                } else if (labelInt == static_cast<int>(ConeClass::BIG_ORANGE)) {
                    labelText = "B_ORANGE"; color = cv::Scalar(0, 100, 255);
                } else {
                    labelText = "UNKNOWN"; color = cv::Scalar(255, 255, 255);
                }

                // ==========================================
                // CLASSIFICAZIONE E SEGMENTAZIONE
                // ==========================================
                cv::rectangle(img_class, safeBox, color, 2);
                cv::putText(img_class, labelText, cv::Point(safeBox.x, std::max(15, safeBox.y - 5)), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);

                cv::Mat roi = img_segm(safeBox);
                cv::Mat mask = segmentCone(roi, labelInt);

                // Calcolo IoU per questo specifico cono
                double currentIoU = calculateIoU(mask, gtMask);
                imageTotalIoU += currentIoU;
                imageValidIoUCount++;

                cv::Mat coloredOverlay = roi.clone();
                coloredOverlay.setTo(color, mask);
                cv::addWeighted(roi, 0.4, coloredOverlay, 0.6, 0.0, roi);

                cv::rectangle(img_segm, safeBox, color, 2);
                
                // Formatta il testo per includere l'IoU del singolo cono
                std::stringstream ss;
                ss << labelText << " (IoU:" << std::fixed << std::setprecision(2) << currentIoU << ")";
                cv::putText(img_segm, ss.str(), cv::Point(safeBox.x, std::max(15, safeBox.y - 5)), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv::LINE_AA);
            }

            globalCorrectCones += correctConesCount;
            globalTotalCones += bboxesToTest.size();
            globalTotalIoU += imageTotalIoU;
            globalValidIoUCount += imageValidIoUCount;

            double imageAccuracy = (bboxesToTest.empty()) ? 0.0 : (static_cast<double>(correctConesCount) / bboxesToTest.size()) * 100.0;
            double imageMeanIoU = (imageValidIoUCount == 0) ? 0.0 : (imageTotalIoU / imageValidIoUCount) * 100.0;

            std::cout << "[Test " << i + 1 << "/" << numImagesToTest << "] " << imagePath.filename() << "\n"
                      << "  -> Classificazione: " << correctConesCount << " / " << bboxesToTest.size() 
                      << " (" << std::fixed << std::setprecision(1) << imageAccuracy << "%)\n"
                      << "  -> mIoU Immagine  : " << std::fixed << std::setprecision(1) << imageMeanIoU << "%\n"
                      << "----------------------------------------\n";
            
            // 4. DISPLAY SIDE-BY-SIDE
            cv::putText(img_class, "CLASSIFICATION ONLY", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);
            cv::putText(img_segm, "CLASSIFICATION + SEGMENTATION", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);

            cv::Mat combined;
            cv::hconcat(img_class, img_segm, combined);

            cv::imshow(winName, combined);

            int key = cv::waitKey(0);
            if (key == 27 || key == 'q') {
                std::cout << "\nTest interrotto manualmente." << std::endl;
                break; 
            }
        }
        
        cv::destroyAllWindows();

        // ==========================================
        // STAMPA DEL REPORT FINALE GLOBALE
        // ==========================================
        std::cout << "\n========================================" << std::endl;
        std::cout << "         RISULTATO FINALE TEST          " << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Coni totali analizzati : " << globalTotalCones << std::endl;
        
        if (globalTotalCones > 0) {
            float accuracy = (static_cast<float>(globalCorrectCones) / globalTotalCones) * 100.0f;
            std::cout << "Accuratezza Globale    : " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
        }

        if (globalValidIoUCount > 0) {
            float global_mIoU = (globalTotalIoU / globalValidIoUCount) * 100.0f;
            std::cout << "mIoU Globale           : " << std::fixed << std::setprecision(2) << global_mIoU << "%" << std::endl;
        }
        std::cout << "========================================\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}