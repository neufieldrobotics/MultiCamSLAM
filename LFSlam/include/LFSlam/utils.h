//
// Created by auv on 1/27/21.
//

#ifndef SRC_UTILS_H
#define SRC_UTILS_H

#include <algorithm>
#include <numeric>
#include <vector>
#include <opencv2/opencv.hpp>
#include "gtsam/geometry/Rot3.h"
#include "gtsam/geometry/Point3.h"
#include "gtsam/geometry/Pose3.h"
#include <math.h>

using namespace std;


template<typename T>
vector<int> argsorte(vector<T>& data, bool ascen = true) {
    int n = data.size();
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    if(ascen)
        sort(indices.begin(), indices.end(), [&data](int i, int j) -> bool {return data[i] < data[j];});
    else
        sort(indices.begin(), indices.end(), [&data](int i, int j) -> bool {return data[i] > data[j];});
    return indices;
}


static gtsam::Pose3 convertPose3_CV2GTSAM(cv::Mat &pose){

    gtsam::Rot3 R(pose.at<double>(0,0), pose.at<double>(0,1), pose.at<double>(0,2),
                  pose.at<double>(1,0), pose.at<double>(1,1), pose.at<double>(1,2),
                  pose.at<double>(2,0), pose.at<double>(2,1), pose.at<double>(2,2));

    gtsam::Point3 t(pose.at<double>(0,3), pose.at<double>(1,3), pose.at<double>(2,3));

    return gtsam::Pose3(R, t);
}

static gtsam::Point2 convertPoint2_CV2GTSAM(cv::KeyPoint &kp){
    gtsam::Point2 pt(kp.pt.x, kp.pt.y);
    return pt;
}

static gtsam::Point3 convertPoint3_CV2GTSAM(cv::Mat &landmark){
    gtsam::Point3 pt(landmark.at<double>(0,0), landmark.at<double>(1,0), landmark.at<double>(2,0));
    return pt;
}

static void convertCamConfig_CV2GTSAM(CamArrayConfig& camconfig, vector<gtsam::Pose3>& RT_Mats){
    RT_Mats.clear();
    for( int i =0; i <camconfig.num_cams_ ; i++){
        Mat R2 = camconfig.R_mats_[i];/// store the pose of the cam chain
        Mat t2 = camconfig.t_mats_[i];
        Mat camPose = Mat::eye(4,4, CV_64F);
        camPose.rowRange(0,3).colRange(0,3) = R2.t();
        camPose.rowRange(0,3).colRange(3,4) = -1* R2.t()*t2;
        RT_Mats.push_back(convertPose3_CV2GTSAM(camPose));
    }
}



#endif //SRC_UTILS_H
