//
// Created by Pushyami Kaveti on 1/24/22.
//

#include "LFSlam/MonoFrame.h"

MonoFrame::MonoFrame(Mat img, Mat segmap, ORBVocabulary *vocab, ORBextractor *orb_obj,
        Mat &K, Mat & dist, int id,double tStamp, bool debug):orb_vocabulary(vocab),
        orBextractor(orb_obj), Kmat_(K), distmat_(dist),frameId(id), timeStamp(tStamp), DEBUG_MODE(debug){

    numTrackedLMs =0;
    //orbCuda = cuda::ORB::create(2000,1.2f, 8, 31, 0, 2, 0, 31, 20, true);

    img_size = cv::Size(img.cols, img.rows);
    img_ = img.clone();
    multiply(img_, 255, img_);
    img_.convertTo(img_,CV_8U);
    if (img_.channels() == 3){
        Mat imgGray;
        cvtColor(img_,imgGray , COLOR_BGR2GRAY);
        img_ = imgGray;
    }
    segMask_ = segmap.clone(); // no need to convert this into 8U
    if (segMask_.channels() == 3){
        vector<Mat> channels(3);
        split(segMask_, channels);
        segMask_ = channels[0];
    }
    Mat undistImg;
    cv::undistort(img_, undistImg, K, dist );
    //cv::imshow("image rectified", undistImg);
    //waitKey(0);
    // undistImg.copyTo(all.colRange(siz.width*im, siz.width*(im+1)));
    img_ = undistImg.clone();

    // do the same for segmaps
    Mat undistSegmap;
    cv::undistort(segMask_, undistSegmap, K, dist);
    // cv::warpPerspective(undistSegmap, undistSegmap, camconfig_.Rect_mats_[im], img.size() );
    segMask_ = undistSegmap.clone();

}

void MonoFrame::extractFeatures() {
    vector<cv::KeyPoint> kps;
    cv::Mat descs;

    (*orBextractor)(img_, cv::Mat(), image_kps, image_descriptors );

    image_kps_undist = image_kps;

    /// Below code is required only if we were to do BoW matching for loop closure in future
    //std::vector<cv::Mat> vec_desc;
    //vec_desc.reserve(descs.rows);
    //for (int j=0;j<descs.rows;j++)
    //    vec_desc.push_back(descs.row(j));
    //image_descriptors[cam_ind]= vec_desc;
}

int MonoFrame::countLandmarks (){
    int c=0;
    for(auto& l : lIds){
        if(l != -1)
            c++;
    }

    return c;
}

MonoFrame::~MonoFrame() {
    img_.release();
    segMask_.release();
}
