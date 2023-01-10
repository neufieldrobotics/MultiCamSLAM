//
// Created by auv on 6/8/20.
//

#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

class CamArrayConfig {

public:
    CamArrayConfig(std::vector<cv::Mat> Kmats, std::vector<cv::Mat> Dmats, std::vector<cv::Mat> Rmats,\
                            std::vector<cv::Mat> Tmats, Size im_size , int no_cams);
    CamArrayConfig(std::vector<cv::Mat> Kmats, std::vector<cv::Mat> Dmats, std::vector<cv::Mat> Rmats,\
                            std::vector<cv::Mat> Tmats, std::vector<cv::Mat> KalibrRmats,\
                            std::vector<cv::Mat> KalibrTmats, Size im_size, int no_cams);
    CamArrayConfig(std::vector<cv::Mat> Kmats, std::vector<cv::Mat> Dmats, std::vector<cv::Mat> Rmats,\
                            std::vector<cv::Mat> Tmats, std::vector<cv::Mat> Rect_mats_, std::vector<float> baselines, Size im_size , int no_cams, bool rectify);
    CamArrayConfig(){}
    ~CamArrayConfig(){}

    void make_ref_cam(int cam_ind);
    std::vector<cv::Mat> K_mats_, dist_coeffs_, R_mats_, t_mats_, Kalibr_R_mats_, Kalibr_t_mats_, Rect_mats_ ;
    std::vector<float> baselines_;
    int num_cams_;
    Size im_size_;
    bool RECTIFY;

};

