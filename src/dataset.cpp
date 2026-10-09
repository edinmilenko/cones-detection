//Author: Nicola Ranzolin
#include "dataset.hpp"
#include <opencv2/opencv.hpp>
#include <zlib.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Supervisely stores a mask as text, so the first step is to turn that text back into bytes
static std::vector<uchar> decodeBase64(const std::string& encoded){
    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    // keep only the real characters, throwing away newlines and the '=' padding at the end
    std::string clean;
    for(size_t i = 0; i < encoded.size(); i++)
    {
        if(alphabet.find(encoded[i]) != std::string::npos)
        {
            clean.push_back(encoded[i]);
        }
    }

    // each character carries 6 bits, so every group of 4 characters gives 3 bytes
    std::vector<uchar> decoded;
    for(size_t i = 0; i + 1 < clean.size(); i += 4)
    {
        int value[4] = {0, 0, 0, 0};
        int howMany = 0;
        for(int k = 0; k < 4 && i + k < clean.size(); k++)
        {
            value[k] = (int)alphabet.find(clean[i + k]);
            howMany++;
        }

        decoded.push_back((uchar)(value[0] * 4 + value[1] / 16));
        if(howMany > 2)
        {
            decoded.push_back((uchar)((value[1] % 16) * 16 + value[2] / 4));
        }
        if(howMany > 3)
        {
            decoded.push_back((uchar)((value[2] % 4) * 64 + value[3]));
        }
    }
    return decoded;
}

// Those bytes are still compressed, zlib gives us back the PNG that was packed inside
static std::vector<uchar> decompressZlib(const std::vector<uchar>& compressed){
    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(compressed.data()));
    stream.avail_in = static_cast<uInt>(compressed.size());

    if(inflateInit(&stream) != Z_OK)
    {
        throw std::runtime_error("Cannot initialize bitmap decompressor");
    }

    // we do not know the final size, so we grow the buffer 16 KB at a time until zlib says it is done
    std::vector<uchar> decompressed;
    const size_t chunkSize = 16 * 1024;
    int result = Z_OK;
    do
    {
        const size_t previousSize = decompressed.size();
        decompressed.resize(previousSize + chunkSize);
        stream.next_out = reinterpret_cast<Bytef*>(decompressed.data() + previousSize);
        stream.avail_out = static_cast<uInt>(chunkSize);
        result = inflate(&stream, Z_NO_FLUSH);
        decompressed.resize(previousSize + chunkSize - stream.avail_out);
    }
    while(result == Z_OK);

    inflateEnd(&stream);
    if(result != Z_STREAM_END)
    {
        throw std::runtime_error("Cannot decompress bitmap data");
    }
    return decompressed;
}

// The three steps together: text, then bytes, then a PNG, and out comes the mask of one cone
static cv::Mat decodeBitmapMask(const std::string& base64Data){
    const std::vector<uchar> compressed = decodeBase64(base64Data);
    const std::vector<uchar> png = decompressZlib(compressed);
    const cv::Mat decoded = cv::imdecode(png, cv::IMREAD_UNCHANGED);
    if(decoded.empty())
    {
        throw std::runtime_error("Bitmap data does not contain a valid PNG mask");
    }

    cv::Mat mask;
    cv::extractChannel(decoded, mask, 3);
    // whatever it was, we want it black or white and nothing in between
    cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
    return mask;
}

//where the value of "key" starts, skipping the colon and the spaces.
static size_t valueStart(const std::string& text, const std::string& key){
    size_t at = text.find("\"" + key + "\"");
    if(at == std::string::npos) return std::string::npos;

    at = text.find(':', at);
    if(at == std::string::npos) return std::string::npos;

    at++;
    while(at < text.size() && std::isspace((unsigned char)text[at])) at++;
    return at;
}

//the text between the quotes of a "key": "value" pair, empty if there is no such key
static std::string stringField(const std::string& text, const std::string& key){
    const size_t at = valueStart(text, key);
    if(at == std::string::npos || text[at] != '"') return "";

    const size_t end = text.find('"', at + 1);
    if(end == std::string::npos) return "";
    return text.substr(at + 1, end - at - 1);
}

static std::vector<int> intField(const std::string& text, const std::string& key, size_t howMany){
    std::vector<int> numbers;
    size_t at = valueStart(text, key);
    if(at == std::string::npos) return numbers;

    while(at < text.size() && numbers.size() < howMany)
    {
        if(text[at] == '-' || std::isdigit((unsigned char)text[at]))
        {
            numbers.push_back(std::atoi(text.c_str() + at));
            at++;
            while(at < text.size() && std::isdigit((unsigned char)text[at])) at++;
        }
        else if(text[at] == '"')
        {
            break;      //the list is over, this is already the next key
        }
        else
        {
            at++;
        }
    }
    return numbers;
}

//the text of each annotation inside the "objects" list, one string per annotation
static std::vector<std::string> splitObjects(const std::string& text){
    std::vector<std::string> objects;
    size_t at = valueStart(text, "objects");
    if(at == std::string::npos || text[at] != '[') return objects;

    int depth = 0;
    size_t start = 0;
    for(size_t i = at; i < text.size(); i++)
    {
        if(text[i] == '{')
        {
            if(depth == 0) start = i;
            depth++;
        }
        else if(text[i] == '}')
        {
            depth--;
            if(depth == 0) objects.push_back(text.substr(start, i - start + 1));
        }
        else if(text[i] == ']' && depth == 0)
        {
            break;      //end of the list
        }
    }
    return objects;
}

// All the cones annotated in one json, each with its class and either a box or a mask
std::vector<Annotation> readAnnotations(const std::string& annotationPath){
    std::ifstream input(annotationPath);
    if(!input)
    {
        throw std::runtime_error("Cannot open annotation file");
    }

    std::stringstream buffer;
    buffer << input.rdbuf();
    const std::string text = buffer.str();

    std::vector<Annotation> annotations;
    const std::vector<std::string> objects = splitObjects(text);

    for(size_t i = 0; i < objects.size(); i++)
    {
        const std::string& object = objects[i];
        const std::string geometryType = stringField(object, "geometryType");
        const std::string label = stringField(object, "classTitle");

        if(geometryType == "rectangle")
        {
            //exterior is [[x1, y1], [x2, y2]]
            const std::vector<int> corners = intField(object, "exterior", 4);
            if(corners.size() < 4) continue;

            annotations.push_back({AnnotationGeometry::Rectangle, {corners[0], corners[1]}, {corners[2], corners[3]}, {}, {}, label});
            continue;
        }

        if(geometryType == "bitmap")
        {
            const std::vector<int> origin = intField(object, "origin", 2);
            const std::string data = stringField(object, "data");
            if(origin.size() < 2 || data.empty()) continue;

            annotations.push_back({AnnotationGeometry::Bitmap, {}, {}, decodeBitmapMask(data), {origin[0], origin[1]}, label});
        }
    }

    return annotations;
}

// Where the test set is: the zip unpacks in two different ways depending on how it was made
std::string testSetDir(){
    const std::string places[] = {"../test_set/segmentation_test", "../segmentation_test"};
    for(const std::string& place : places)
    {
        if(std::filesystem::exists(place + "/img") && std::filesystem::exists(place + "/ann"))
        {
            return place;
        }
    }
    return "";
}

// Merges the imgs folders into a single dataset/, ready for the training
void dataLoader(){
    const std::filesystem::path destination = "../dataset";
    if(std::filesystem::exists(destination))
    {
        std::cout << "The folder already exists, deleting..." << std::endl;
        std::filesystem::remove_all(destination);
    }
    if(!std::filesystem::create_directory(destination))
    {
        throw std::runtime_error("Failed to create directory");
    }

    //the test set images come from these same FSOCO folders with the same names, so we skip them
    //here: otherwise they end up in dataset/ and we train on the test set
    std::set<std::string> testImages;
    const std::string testDir = testSetDir();
    if(!testDir.empty())
    {
        for(const auto& entry : std::filesystem::directory_iterator(testDir + "/img"))
        {
            testImages.insert(entry.path().filename().string());
        }
        std::cout << "keeping " << testImages.size() << " test set images out of dataset/" << std::endl;
    }
    else
    {
        std::cout << "warning: segmentation_test not found, cannot tell the test images apart" << std::endl;
    }

    //the two folders overlap in part, on a repeated name the first one wins (emplace does not overwrite)
    const std::filesystem::path sources[] = {"../fsoco_segmentation_train", "../fsoco_bounding_boxes_train"};

    std::unordered_map<std::string, std::filesystem::path> imageFiles;
    std::unordered_map<std::string, std::filesystem::path> jsonFiles;
    for(const std::filesystem::path& source : sources)
    {
        if(!std::filesystem::exists(source))
        {
            std::cout << "warning: " << source << " not found, skipped" << std::endl;
            continue;
        }

        for(const auto& entry : std::filesystem::recursive_directory_iterator(source))
        {
            if(entry.is_directory()) continue;
            const std::string name = entry.path().filename().string();
            const std::string ext = entry.path().extension().string();
            if(ext == ".jpg" || ext == ".jpeg" || ext == ".png")
            {
                if(testImages.count(name)) continue;
                imageFiles.insert(std::make_pair(name, entry.path()));
            }
            else if(ext == ".json" && entry.path().filename() != "meta.json")
            {
                if(testImages.count(entry.path().stem().string())) continue;
                jsonFiles.insert(std::make_pair(entry.path().stem().string(), entry.path()));
            }
        }
    }

    std::vector<std::string> sharedKeys;
    std::unordered_map<std::string, std::filesystem::path>::const_iterator it;
    for(it = imageFiles.begin(); it != imageFiles.end(); ++it)
    {
        if(jsonFiles.count(it->first))
        {
            sharedKeys.push_back(it->first);
        }
        else
        {
            std::cout << "Missing JSON for source image: " << it->second << std::endl;
        }
    }
    std::sort(sharedKeys.begin(), sharedKeys.end());

    //renumber the pairs as 1.jpg, 1.json, 2.jpg, 2.json, ... We make links and not copies: copying would put a second copy of the whole FSOCO data on the disk
    int fileNumber = 1;
    for(const std::string& key : sharedKeys)
    {
        const std::filesystem::path image = std::filesystem::absolute(imageFiles.at(key));
        const std::filesystem::path json = std::filesystem::absolute(jsonFiles.at(key));
        std::filesystem::create_symlink(image, destination / (std::to_string(fileNumber) + image.extension().string()));
        std::filesystem::create_symlink(json, destination / (std::to_string(fileNumber) + ".json"));
        fileNumber++;
    }

    std::cout << "Dataset prepared with " << (fileNumber - 1) << " matched image/json pair(s)." << std::endl;
}

void csvDatasetMaker(std::string datasetDir){
    std::cout << "Creating ../data/dataset.csv from: " << datasetDir << std::endl;
    std::ofstream csvFile("../data/dataset.csv");

    for(const auto& entry : std::filesystem::directory_iterator{datasetDir})
    {
        if(entry.path().extension() != ".json") continue;

        std::ifstream jsonFile(entry.path());
        std::stringstream jsonBuffer;
        jsonBuffer << jsonFile.rdbuf();
        const std::string jsonText = jsonBuffer.str();

        //"size": { "height": H, "width": W }, read by name because the two are not always in this order
        const std::vector<int> width = intField(jsonText, "width", 1);
        const std::vector<int> height = intField(jsonText, "height", 1);
        if(width.empty() || height.empty()) continue;

        const int W = width[0];
        const int H = height[0];

        //find the corresponding extension for the image
        std::string imageName;
        const std::vector<std::string> extensions = {".jpg", ".jpeg", ".png"};
        for(const auto& extension : extensions)
        {
            if(std::filesystem::exists(entry.path().parent_path() / (entry.path().stem().string() + extension)))
            {
                imageName = entry.path().stem().string() + extension;
                break;
            }
        }
        if(imageName.empty())
        {
            std::cerr << "No image found for the given json file: " << entry.path() << std::endl;
            continue;
        }

        const std::vector<Annotation> annotations = readAnnotations(entry.path().string());

        for(const Annotation& a : annotations)
        {
            if(a.geometry != AnnotationGeometry::Rectangle) continue;

            int x1 = std::min(a.topLeft.x, a.bottomRight.x);
            int y1 = std::min(a.topLeft.y, a.bottomRight.y);
            int x2 = std::max(a.topLeft.x, a.bottomRight.x);
            int y2 = std::max(a.topLeft.y, a.bottomRight.y);

            //bboxes are too close to the border, ignore them
            if(x1 < 2 || y1 < 2 || x2 > W - 2 || y2 > H - 2) continue;
            csvFile << imageName << "," << x1 << "," << y1 << "," << x2 << "," << y2 << "\n";
        }
    }
}
