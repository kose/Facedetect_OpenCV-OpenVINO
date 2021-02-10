##

set(EXE facedetect)
project(${EXE})

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

find_package(OpenCV 4.1 REQUIRED)

# for TBB
if(APPLE)
  include_directories(/opt/local/include)
  link_directories(/opt/local/lib)
endif(APPLE)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

file (GLOB_RECURSE SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/facedetect.cpp
  )

add_executable (${EXE} ${SOURCE_FILES})

target_link_libraries (${EXE} PRIVATE
  ${OpenCV_LIBS}
  gflags
)

# end
