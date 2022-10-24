//
// Created by Pushyami Kaveti on 6/8/20.
//

#include "LFDataUtils/CamArrayConfig.h"
using namespace cv;

CamArrayConfig::CamArrayConfig(std::vector<cv::Mat> Kmats, std::vector<cv::Mat> Dmats, std::vector<cv::Mat> Rmats,\
                            std::vector<cv::Mat> Tmats, Size im_size, int no_cams){
    num_cams_ = no_cams;
    im_size_ = im_size;
    RECTIFY = true;
    copy(Kmats.begin(), Kmats.end(), back_inserter(K_mats_));
    copy(Dmats.begin(), Dmats.end(), back_inserter(dist_coeffs_));
    copy(Rmats.begin(), Rmats.end(), back_inserter(R_mats_));
    copy(Tmats.begin(), Tmats.end(), back_inserter(t_mats_));
}


CamArrayConfig::CamArrayConfig(std::vector<cv::Mat> Kmats, std::vector<cv::Mat> Dmats, std::vector<cv::Mat> Rmats,\
                            std::vector<cv::Mat> Tmats, std::vector<cv::Mat> KalibrRmats,\
                            std::vector<cv::Mat> KalibrTmats, Size im_size, int no_cams){
    num_cams_ = no_cams;
    im_size_ = im_size;
    RECTIFY = true;
    copy(Kmats.begin(), Kmats.end(), back_inserter(K_mats_));
    copy(Dmats.begin(), Dmats.end(), back_inserter(dist_coeffs_));
    copy(Rmats.begin(), Rmats.end(), back_inserter(R_mats_));
    copy(Tmats.begin(), Tmats.end(), back_inserter(t_mats_));
    copy(KalibrRmats.begin(), KalibrRmats.end(), back_inserter(Kalibr_R_mats_));
    copy(KalibrTmats.begin(), KalibrTmats.end(), back_inserter(Kalibr_t_mats_));
}


CamArrayConfig::CamArrayConfig(std::vector<cv::Mat> Kmats, std::vector<cv::Mat> Dmats, std::vector<cv::Mat> Rmats,\
                            std::vector<cv::Mat> Tmats, std::vector<cv::Mat> Rectmats, std::vector<float> baselines, Size im_size , int no_cams, bool rectify){
    num_cams_ = no_cams;
    im_size_ = im_size;
    RECTIFY = rectify;
    copy(Rectmats.begin(), Rectmats.end(), back_inserter(Rect_mats_));
    copy(baselines.begin(), baselines.end(), back_inserter(baselines_));
    copy(Kmats.begin(), Kmats.end(), back_inserter(K_mats_));
    copy(Dmats.begin(), Dmats.end(), back_inserter(dist_coeffs_));
    copy(Rmats.begin(), Rmats.end(), back_inserter(R_mats_));
    copy(Tmats.begin(), Tmats.end(), back_inserter(t_mats_));
}

void CamArrayConfig::make_ref_cam(int cam_ind){
    Mat_<double> R = Mat_<double>::zeros(3,3);
    Mat_<double> t = Mat_<double>::zeros(3,1);

    int dist = abs(cam_ind-0);
    for(int j = 0; j < dist; j++){
        R = Kalibr_R_mats_[cam_ind - j].inv();
        t = -1 * R * Kalibr_t_mats_[cam_ind-j];
        if(j > 0){
            Mat R3 = R.clone()*R_mats_[cam_ind - j].clone();
            Mat t3 = R.clone()*t_mats_[cam_ind - j].clone() + t.clone();
            R = R3.clone(); t = t3.clone();
        }

       R_mats_[cam_ind - j - 1] = R.clone();
       t_mats_[cam_ind-j-1] = t.clone();

    }
    R = Mat_<double>::zeros(3,3);
    t = Mat_<double>::zeros(3,1);
    //cout<<"it is the camera itself. its R is identity and t is 000\n";
    R(0,0) = 1.0; R(1,1) = 1.0; R(2,2) = 1.0;
    t(0,0) = 0.0; t(1,0) = 0.0; t(2,0) = 0.0;

    R_mats_[cam_ind] = R.clone();
    t_mats_[cam_ind] = t.clone();

    for (int j = cam_ind+1  ; j < num_cams_ ; j++){
        Mat R3 = Kalibr_R_mats_[j].clone()*R_mats_[j-1].clone();
        Mat t3 = Kalibr_R_mats_[j].clone()*t_mats_[j-1].clone() + Kalibr_t_mats_[j].clone();
        R = R3.clone(); t = t3.clone();
        R_mats_[j] = R.clone();
        t_mats_[j] = t.clone();
    }
    // uploadMatsToGPU_new(cam_ind);

}



