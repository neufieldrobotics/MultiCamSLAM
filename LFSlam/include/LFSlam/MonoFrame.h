//
// Created by Pushyami Kaveti on 1/24/22.
//

#ifndef SRC_MONOFRAME_H
#define SRC_MONOFRAME_H

#include <thread>
#include "LFDataUtils/CamArrayConfig.h"
#include "LFSlam/ORBVocabulary.h"
#include "LFSlam/ORBextractor.h"
#include "common_utils/tools.h"
#include "LFSlam/utils.h"
#include <opencv2/cudafeatures2d.hpp>

// OPENGV ////
#include <Eigen/Eigen>
#include <memory>
#include "time_measurement.hpp"
#include <opencv2/core/eigen.hpp>
#include <mutex>

using namespace std;
using namespace cv;


class MonoFrame {

public:

    MonoFrame(Mat img, Mat segmap, ORBVocabulary *vocab, ORBextractor *orb_obj,
    Mat &K, Mat & dist, int id,double tStamp, bool debug);

    MonoFrame( CamArrayConfig &camconfig);

    ~MonoFrame();

    //Methods for setting, updating the LightFieldFrame data
    void setData(Mat img, Mat segmap);

    //Methods for data processing
    void extractFeatures();
    void extractORBCuda();
    void UndistortKeyPoints(int cam);
    bool checkItersEnd(DBoW2::FeatureVector::const_iterator* featVecIters, DBoW2::FeatureVector::const_iterator* featVecEnds);

    void BowMatching(int cam1_ind, int cam2_ind,vector<unsigned int>& indices_1, vector<unsigned int>& indices_2,set<DBoW2::NodeId>& words  );

    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    //Utility functions
    int countLandmarks ();
    //cv::Mat getPose();
    void getMapPoints(vector<Point3f>& pts);

    //class variables
    int frameId;
    double timeStamp;
    Size img_size;
    bool DEBUG_MODE;
    Mat img_;
    Mat segMask_;
    Mat Kmat_, distmat_;
    ORBextractor* orBextractor;
    ORBVocabulary* orb_vocabulary;

    /// KeyPoints of images
    vector<cv::KeyPoint> image_kps, image_kps_undist;

    /// Descriptors of images
    cv::Mat image_descriptors;


    //Landmarks
    int numTrackedLMs;
    vector<int> lIds;

    //paramsmax_neighbor_ratio
    double max_neighbor_ratio = 0.7;

    cv::Mat pose;
    std::mutex mMutexPose, mMutexPts;

};


#endif //SRC_MONOFRAME_H
