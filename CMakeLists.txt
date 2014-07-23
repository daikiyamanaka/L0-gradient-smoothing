project(L0Minimization)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
aux_source_directory(src SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

######################
# Compiler flags
######################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -pedantic")

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
  if(APPLE)
  target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
  elseif(UNIX)
  target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
  endif()
else()
  message(FATAL_ERROR "Couldn't find Boost library.")
endif()

##########################
# opencv
##########################
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/local/opencv/share/OpenCV/OpenCVConfig.cmake")
set(OpenCV_DIR "/usr/local/opencv/share/OpenCV/OpenCVConfig.cmake"
    CACHE PATH "The path where OpenCVConfig.cmake is placed")
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
  message(STATUS "Found 'opencv library' ${OpenCV_VERSION}")
  #include_directories(${OpenCV_INCLUDE_DIRS}) #要らない？
  #link_directories(${OpenCV_LIB_DIR})
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
else()
  message(SEND_ERROR "Couldn't find OpenCV")
endif()


##########
#GeomProc
##########
#include_directories("/Users/daikiyamanaka/Dropbox/research/WeightAwareModeling/program/C++/OpenGeomProc/include")
#target_link_libraries(${PROJECT_NAME} /Users/daikiyamanaka/Dropbox/research/WeightAwareModeling/program/C++/OpenGeomProc/build/lib/libGeomCore.dylib)
#target_link_libraries(${PROJECT_NAME} /Users/daikiyamanaka/Dropbox/research/WeightAwareModeling/program/C++/OpenGeomProc/build/lib/libGeomIO.dylib)
#target_link_libraries(${PROJECT_NAME} /Users/daikiyamanaka/Dropbox/research/WeightAwareModeling/program/C++/OpenGeomProc/build/lib/libGeomProc.dylib)

##########
#Eigen
##########
include_directories("/Users/daikiyamanaka/Program/eigen-3.1.3")


##########
#CGAL
##########
#target_link_libraries(${RROJECT_NAME} /usr/local/lib/libCGAL.dylib)
#target_link_libraries(${RROJECT_NAME} /usr/local/lib/libCGAL_Core.dylib)
#set(CGAL_DIR "/usr/local/lib/CGAL/CGALConfig.cmake" CACHE PATH "The path where OpenCVConfig.cmake is placed")
#find_package(CGAL)
#include(${CGAL_USE_FILE})
#target_link_libraries(${PROJECT_NAME} ${CGAL_LIBRARIES}
#                              ${CGAL_3RD_PARTY_LIBRARIES} gmp)
