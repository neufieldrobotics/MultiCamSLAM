cmake_minimum_required (VERSION 3.0.0)

project(LFApps)
## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
SET(OpenCV_DIR "~/software/opencv/build"  CACHE PATH "OpenCV directory")

#catkin
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy rosbag message_filters
  std_msgs
  cv_bridge
  image_transport
  sensor_msgs
  common_utils
  LFDataUtils
  LFReconstruct
  LFSlam
)
# fPIC flags, needed?
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
add_definitions (-march=native) #This is what made openGV work for me.
# CUDA
find_package(CUDA)
message("---CUDA---")
message("CUDA Libraries: " ${CUDA_LIBRARIES})
message("CUDA Include Dirs: " ${CUDA_INCLUDE_DIRS})
if(NOT CUDA_FOUND)
  message("CUDA not found!")
endif()

set(Python_ADDITIONAL_VERSIONS "2.7")
find_package(PythonLibs REQUIRED)
message("---PYTHON--- ")
message("Python Libraries: " ${PYTHON_LIBRARIES})
message("Python Include Dirs: " ${PYTHON_INCLUDE_DIRS})
if (NOT PYTHONLIBS_FOUND)
  message("Python not found! Python components will not be built.")
endif()

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
#FIND EIGEN
find_package(Eigen REQUIRED)
message("EIGEN_INCLUDE_DIR :" ${EIGEN_INCLUDE_DIR})
set(ADDITIONAL_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIR}/unsupported)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp rospy std_msgs common_utils LFDataUtils LFReconstruct LFSlam
  DEPENDS OpenCV
)

set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy)

include_directories(include)
include_directories( ${PYTHON_INCLUDE_DIRS}  /usr/include/python2.7/)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${catkin_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ${ADDITIONAL_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} /usr/lib/x86_64-linux-gnu/libpython2.7.so)


set (LIBS ${OpenCV_LIBS} ${CUDA_LIBRARIES}  ${Boost_GENERAL} gflags glog)
FILE(GLOB ${PROJECT_NAME}_SRC src/ParseSettings.cpp src/LiveViewer.cpp)
add_executable (LFapp  src/app.cpp ${${PROJECT_NAME}_SRC})
add_dependencies(LFapp  ${catkin_EXPORTED_TARGETS})
target_link_libraries(LFapp  ${LIBS} ${catkin_LIBRARIES} )

add_executable (LFSlamapp  src/lf_slam_app.cpp ${${PROJECT_NAME}_SRC})
add_dependencies(LFSlamapp  ${catkin_EXPORTED_TARGETS})
target_link_libraries(LFSlamapp  ${LIBS} ${catkin_LIBRARIES} )

add_executable (MonoSlamApp  src/MonoSlamApp.cpp ${${PROJECT_NAME}_SRC})
add_dependencies(MonoSlamApp  ${catkin_EXPORTED_TARGETS})
target_link_libraries(MonoSlamApp  ${LIBS} ${catkin_LIBRARIES} )
        

