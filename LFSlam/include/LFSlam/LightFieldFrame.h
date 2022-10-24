//
// Created by auv on 6/29/20.
//

#ifndef LIGHT_FIELDS_ROS_LIGHTFIELDFRAME_H
#define LIGHT_FIELDS_ROS_LIGHTFIELDFRAME_H

#include <thread>
#include "LFDataUtils/CamArrayConfig.h"
#include "LFSlam/ORBVocabulary.h"
#include "LFSlam/ORBextractor.h"
#include "common_utils/tools.h"
#include "LFSlam/utils.h"
#include <opencv2/cudafeatures2d.hpp>

// OPENGV ////
#include <Eigen/Eigen>
#include <opengv/types.hpp>
#include <memory>
#include "time_measurement.hpp"
#include <opencv2/core/eigen.hpp>
#include <mutex>

using namespace std;
using namespace cv;
using namespace opengv;


struct support_pt {
    int32_t u;
    int32_t v;
    int32_t d;
    support_pt(int32_t u,int32_t v,int32_t d):u(u),v(v),d(d){}

    bool operator<(const support_pt &ob) const
    {
        return u < ob.u || (u == ob.u && v < ob.v);
    }
};
// todo: generalize for n cameras
class IntraMatch{
public:
    array<int, 5> matchIndex;
    cv::Point2f uv_ref;
    bool mono;
    int n_rays;

    cv::Mat matchDesc;
    cv::Mat point3D;

    IntraMatch() : matchIndex{-1, -1, -1, -1, -1}, mono(true){}
    ~IntraMatch(){
        matchDesc.release();
        point3D.release();
    }
};

class LightFieldFrame {
    public:
        LightFieldFrame(vector<Mat> img_set, vector<Mat> segmap_set, ORBVocabulary *vocab, ORBextractor *orb_obj, vector<ORBextractor*> orbs,
                        CamArrayConfig &camconfig, int id,double tStamp, cv::Ptr<cv::CLAHE> clahe, bool debug);
        LightFieldFrame( CamArrayConfig &camconfig);
        ~LightFieldFrame();

        //Methods for setting, updating the LightFieldFrame data
        void setData(vector<Mat> img_set, vector<Mat> segmap_set);

        //Methods for data processing
        void extractFeatures();
        void extractFeaturesParallel();
        void extractORBCuda();
        void extractFeatureSingle(int cam_ind);
        void UndistortKeyPoints(int cam);
        void parseandadd_BoW();
        void parseIntraMatchBoW();
        bool checkItersEnd(DBoW2::FeatureVector::const_iterator* featVecIters, DBoW2::FeatureVector::const_iterator* featVecEnds);
        void computeDisparity();
        void computeIntraMatches(std::vector<IntraMatch>& matches,vector<DBoW2::NodeId >& words_);
        void computeIntraMatches(std::vector<IntraMatch>& matches, bool old);
        void BowMatching(int cam1_ind, int cam2_ind,vector<unsigned int>& indices_1, vector<unsigned int>& indices_2,vector<Point2f>& kps1, vector<Point2f>& kps2, set<DBoW2::NodeId>& words  );
        void computeIntraMatches(vector<vector<vector<cv::DMatch>>>& matches, \
                                   vector<vector<vector<Point2f>>>& kps_1,vector<vector<vector<Point2f>>>& kps_2);

        void computeRepresentativeDesc( vector<cv::Mat> descs, cv::Mat& desc_out );

        void reconstruct3D();

        void triangulate(vector<Point2f>& kps_1, vector<Point2f>& kps_2, vector<Point3f>& pts3D , int cam1, int cam2);
        void testTriangulateIntraMatches();
        void triangulateIntraMatches(vector<vector<vector<Point2f>>>& kps_1,vector<vector<vector<Point2f>>>& kps_2);
        void triangulateIntraMatches(const Mat_<double> &x, vector<int> view_inds, Eigen::Vector3d& pt);

        //Utility functions
        void drawIntraMatches();
        void drawIntraMatches(bool all_matches);
        int countLandmarks (bool mono);


        //cv::Mat getPose();
        void getMapPoints(vector<Point3f>& pts);

        //class variables
        int frameId;
        double timeStamp;
        int num_cams_;
        Size img_size;
        bool DEBUG_MODE;
        vector<Mat> imgs;
        vector<Mat> segMasks;
        CamArrayConfig& camconfig_;
        ORBextractor* orBextractor;
        Ptr<cv::cuda::ORB> orbCuda;
        vector<ORBextractor*> orBextractors;
        ORBVocabulary* orb_vocabulary;
        cv::Ptr<cv::CLAHE> clahe;
        //ORBDatabase* orb_database;
        vector<DBoW2::FeatureVector> BoW_feats;
        vector<DBoW2::BowVector> BoW_vecs;
        /// KeyPoints of images
        vector<vector<cv::KeyPoint> > image_kps, image_kps_undist;

        /// Descriptors of images
        vector<vector<cv::Mat> > image_descriptors;

        int intramatch_size, mono_size;


        //Landmarks

        int numTrackedLMs;
        vector<int> lIds;

        std::vector<IntraMatch> intraMatches;
        DBoW2::BowVector lfBoW;
        DBoW2::FeatureVector lfFeatVec;

        //This to test feature tracks in monocular case
        vector<int> lIds_mono;

        cv::Mat pose;
        cv::Mat cov;
        std::mutex mMutexPose, mMutexPts;

        // opengv pose estimation stuff
        vector<int> correspondences;
        opengv::bearingVectors_t bearings;
        opengv::rotations_t rotations;
        opengv::translations_t translations;
        opengv::rotations_t* rotations_ptr;
        /// Logging variables
        int num_matches_refKF;
        int num_matches_refKF_lms;
        int num_matches_refKF_new;
        int num_matches_localMap;
        int num_inliers_refKF_lms;
        int num_inliers_localMap;
        int num_triangulated;
        vector<int> num_inliers_per_view, num_triangulated_per_view;


        /// todo: To be removed
        opengv::points_t points_3D;
        vector<support_pt> sparse_disparity;
        vector<Point3f> pts3D_;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};


#endif //LIGHT_FIELDS_ROS_LIGHTFIELDFRAME_H
