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
    src/viola-jones.cpp
)

add_executable(dataloader
    src/dataloader.cpp
)

add_executable(labeler
    src/labeler.cpp
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

add_executable(hard_negative_miner
    src/hard_negative_miner.cpp
    src/viola-jones.cpp
)

add_executable(evaluate_model
    src/evaluate_model.cpp
    src/viola-jones.cpp
)

target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(dataloader)
target_link_libraries(labeler ${OpenCV_LIBS} nlohmann_json::nlohmann_json ZLIB::ZLIB)
target_link_libraries(prepare_vj_dataset ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_link_libraries(analyze_vj_dataset ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_link_libraries(split_vj_dataset ${OpenCV_LIBS})
target_link_libraries(evaluate_model ${OpenCV_LIBS} nlohmann_json::nlohmann_json ZLIB::ZLIB)
target_link_libraries(hard_negative_miner ${OpenCV_LIBS} nlohmann_json::nlohmann_json ZLIB::ZLIB)