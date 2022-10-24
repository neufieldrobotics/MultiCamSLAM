
//ROS headers
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/String.h"
#include <image_transport/image_transport.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/message_instance.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/simple_filter.h>


// OpenCV headers
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/features2d.hpp>

// Standard headers
#include <iostream>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <sys/stat.h>
#include <assert.h>
#include <thread>
#include <chrono>

// Tiff library (included in namespace because of typedef conflict with some OpenCV versions)
namespace libtiff {
    #include <tiffio.h>
}

// Ceres Solver headers
// #include <ceres/ceres.h>
// #include <ceres/rotation.h>

// glog and gflags
#include <glog/logging.h>
#include <gflags/gflags.h>

// Profiler header
// #include <gperftools/profiler.h>

// Boost libraries
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#ifdef WITH_PYTHON
#include <boost/python.hpp>
#endif

#include <boost/chrono.hpp>

// Python library
//#include <Python.h>

