cmake_minimum_required(VERSION 3.20)
project(Orthomosaic)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SRC_FILES *.cpp *.h)
message(${SRC_FILES})
add_executable(Orthomosaic ${SRC_FILES})

target_link_libraries(Orthomosaic ${OpenCV_LIBS} Eigen3::Eigen)
