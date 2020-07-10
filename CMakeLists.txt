cmake_minimum_required(VERSION 3.5)

project(prepare_data_set LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.72.0 REQUIRED filesystem)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(prepare_data_set main.cpp)
target_link_libraries(prepare_data_set dlib::dlib ${OpenCV_LIBS} ${Boost_LIBRARIES})

