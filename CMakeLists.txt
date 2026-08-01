cmake_minimum_required(VERSION 3.16)
project(midcoursecv)

find_package(nlohmann_json REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_executable(main
	src/main.cpp
	src/labeler.cpp
)
target_link_libraries(main ${OpenCV_LIBS} nlohmann_json::nlohmann_json)