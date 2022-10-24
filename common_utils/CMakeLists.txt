cmake_minimum_required (VERSION 3.0.0)

project(common_utils)
## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

SET(OpenCV_DIR "~/software/opencv/build"  CACHE PATH "OpenCV directory")

set(Python_ADDITIONAL_VERSIONS "3.8")
find_package(PythonLibs REQUIRED)
message("---PYTHON--- ")
message("Python Libraries: " ${PYTHON_LIBRARIES})
message("Python Include Dirs: " ${PYTHON_INCLUDE_DIRS})
if (NOT PYTHONLIBS_FOUND)
  message("Python not found! Python components will not be built.")
endif()

#catkin
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
)

# Find OpenCV
find_package(OpenCV REQUIRED)
message("---OpenCV----")
message("OpenCV Libraries: " ${OpenCV_LIBS})
message("OpenCV Include Dirs: " ${OpenCV_INCLUDE_DIRS})
if(NOT OpenCV_FOUND)
  message("OpenCV not found!")
endif()
# Boost Libraries
find_package(Boost)
if(Boost_FOUND)
  find_package ( Boost COMPONENTS program_options filesystem system serialization chrono REQUIRED )
  set(Boost_GENERAL ${Boost_LIBRARIES})
elseif(NOT Boost_FOUND)
  message("Boost not found!")
endif()

# FIND REQUIRED PACKAGES
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

find_package(Eigen REQUIRED)
message("EIGEN_INCLUDE_DIR :" ${EIGEN_INCLUDE_DIR})
set(ADDITIONAL_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIR}/unsupported)

#find_package(TIFF REQUIRED)
#if(NOT TIFF_FOUND)
# message("libtiff not found!")
#endif()


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp rospy std_msgs
  DEPENDS OpenCV
)

set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /usr/local/lib/python3/dist-packages/numpy/core/include/numpy)

include_directories(include)
include_directories( ${PYTHON_INCLUDE_DIRS}  /usr/include/python3.8/)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${catkin_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ${ADDITIONAL_INCLUDE_DIRS} )


#${TIFF_LIBRARIES}
set (LIBS ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${Boost_GENERAL}   glog /usr/lib/x86_64-linux-gnu/libpython3.8.so)

FILE(GLOB ${PROJECT_NAME}_SRC "src/*.cpp")

add_library (${PROJECT_NAME} ${${PROJECT_NAME}_SRC})
add_dependencies(${PROJECT_NAME}  ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${LIBS} ${catkin_LIBRARIES})


install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
