project(L0-gradient-minimization)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

######################
# Compiler flags
######################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -pedantic")

##########
#Eigen
##########
#set(EIGEN_DIR /usr/local/include/eigen3/)
#include_directories(${EIGEN_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

######################
# boost
######################
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
set(Boost_INCLUDE_DIR /usr/include)
set(Boost_LIBRARY_DIR /usr/lib)
set(BOOST_LIBRARYDIR /usr/lib)
link_directories(/usr/lib)
get_directory_property(OUT_VAR LINK_DIRECTORIES)
message(STATUS "LINK DIR: ${OUT_VAR}")
endif()

FIND_PACKAGE(Boost COMPONENTS filesystem program_options system REQUIRED)
if (Boost_FOUND)
  message(STATUS "Found 'boost library'")
  include_directories(${Boost_INCLUDE_DIR})
  link_directories(${Boost_LIBRARY_DIRS})
  message(STATUS " boost lib dir:${Boost_LIBRARY_DIRS}")
  message(STATUS " boost include dir:${Boost_INCLUDE_DIR}")
  message(STATUS " libs: ${Boost_LIBRARIES}" )
  target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
else()
  message(FATAL_ERROR "Couldn't find Boost library.")
endif()

##########################
# opencv
##########################
set(OpenCV_DIR "/usr/local/opencv/share/OpenCV/OpenCVConfig.cmake"
    CACHE PATH "The path where OpenCVConfig.cmake is placed")
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
  message(STATUS "Found 'opencv library' ${OpenCV_VERSION}")
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
else()
  message(SEND_ERROR "Couldn't find OpenCV")
endif()
