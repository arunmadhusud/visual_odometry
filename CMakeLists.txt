cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")

project(vo_project)

# Find OpenCV
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)
include_directories(${yaml-cpp_INCLUDE_DIRS})

# Executables for exercise
add_executable(vo src/vo_main.cpp src/LatLong-UTMconversion.cpp)

# Link libraries
target_link_libraries(vo ${OpenCV_LIBRARIES} yaml-cpp)




