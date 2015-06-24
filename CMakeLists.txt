cmake_minimum_required(VERSION 2.8.4)
project(Ellipse_detection)
find_package( OpenCV REQUIRED )

set(SOURCE_FILES main.cpp)
add_executable(Ellipse_detection ${SOURCE_FILES})
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries( Ellipse_detection ${OpenCV_LIBS} )
