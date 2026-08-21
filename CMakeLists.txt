cmake_minimum_required(VERSION 3.16)
project(cones_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(nlohmann_json REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_executable(main
    src/main.cpp
    src/labeler.cpp
    src/dataloader.cpp
    src/dataset_preprocessing.cpp
    src/utils.cpp
    src/color_proposals.cpp
    src/patches.cpp
    src/hog.cpp
)

target_link_libraries(main ${OpenCV_LIBS} nlohmann_json::nlohmann_json ZLIB::ZLIB)

