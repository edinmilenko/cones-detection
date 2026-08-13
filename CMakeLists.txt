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
    src/viola-jones.cpp
)

add_executable(prepare_vj_dataset
    src/prepare_vj_dataset.cpp
)

add_executable(analyze_vj_dataset
    src/analyze_vj_dataset.cpp
)

add_executable(split_vj_dataset
    src/split_vj_dataset.cpp
)

target_link_libraries(main ${OpenCV_LIBS} nlohmann_json::nlohmann_json ZLIB::ZLIB)
target_link_libraries(prepare_vj_dataset ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_link_libraries(analyze_vj_dataset ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_link_libraries(split_vj_dataset ${OpenCV_LIBS})
