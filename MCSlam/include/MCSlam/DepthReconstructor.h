//
// Created by Pushyami Kaveti on 7/23/19.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "sensor_msgs/PointCloud2.h"
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/ChannelFloat32.h>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include "MCSlam/elas/elas.h"
#include <glog/logging.h>

using namespace std;
using namespace cv;

enum DepthAlgo {
    ELAS = 1,
    BLOCK_MATCH = 2,

};

class DepthReconstructor {

public:
    DepthReconstructor(){ }

    DepthReconstructor(int algo, Size im_size, bool use_q, bool debug_m){
        depthAlgo = (DepthAlgo)algo;
        with_q = use_q;
        debug_mode = debug_m;
        img_size = im_size;
    }


    void init( Mat K1, Mat D1, Mat K2, Mat D2, Mat Rot, Mat trans);
    void calcDisparity(Mat &img1 , Mat &img2, Mat &disp, Mat &depthMap);
    void calcDisparity(Mat &disp);
    void stereo_calibration(const char *imageList, int nx, int ny, vector<vector<cv::Point2f> >* points_final\
                             , string cam1 , string cam2, cv::Mat &M1, cv::Mat &M2, cv::Mat &D1, cv::Mat &D2,\
                             cv::Mat &R,cv::Mat &T,cv::Mat &E,cv::Mat &F , cv::Size &imageSize ,  int &nframes , vector<string> *imageNames);
    void convertToDepthMap(cv::Mat &disp, cv::Mat &depthMap);
    void updateImgs(cv::Mat &imgl , cv::Mat &imgr);

    Mat rectLeft, rectRight, projLeft, projRight, Q, img1_rect, img2_rect;
    Mat rectMapLeft_x, rectMapLeft_y, rectMapRight_x, rectMapRight_y;
    Mat lcam_img, rcam_img;
    cv::Ptr<StereoSGBM> stereo_proc;
    Elas* elas_proc;
    DepthAlgo depthAlgo;
    bool with_q ;
    int lcam, rcam, debug_mode;

private:
    vector<Mat> K_vec;
    vector<Mat> R_vec;
    vector<Mat> t_vec;
    vector<Mat> d_vec;
    Size img_size;
    Rect validRoi[2];

};


