//
// Created by Pushyami Kaveti on 4/4/21.
//

#ifndef SRC_UTILITIES_H
#define SRC_UTILITIES_H

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
vector<float> RotMatToQuat(const cv::Mat &M);

#endif //SRC_UTILITIES_H

