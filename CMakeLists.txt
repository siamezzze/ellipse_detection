cmake_minimum_required(VERSION 2.8.4)
project(Hierarchical)
find_package( OpenCV REQUIRED )

set(SOURCE_FILES main.cpp)
add_executable(Hierarchical ${SOURCE_FILES})
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries( Hierarchical ${OpenCV_LIBS} )