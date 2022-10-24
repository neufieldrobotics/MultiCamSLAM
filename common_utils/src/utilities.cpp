//
// Created by Pushyami Kaveti on 4/4/21.
//

#include <iostream>
#include "common_utils/utilities.h"

using namespace std;

vector<float> RotMatToQuat(const cv::Mat &M)
{

    Eigen::Matrix3f eigMat;
    cv::Mat MM;
    M.convertTo(MM, CV_32FC1);

    cv::cv2eigen(MM, eigMat);

    Eigen::Quaternionf q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}
