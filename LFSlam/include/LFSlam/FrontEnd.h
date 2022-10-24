//
// Created by auv on 6/6/20.
//

#ifndef LIGHT_FIELDS_ROS_FRONTEND_H
#define LIGHT_FIELDS_ROS_FRONTEND_H

#include "LFDataUtils/CamArrayConfig.h"
#include "LFSlam/ORBVocabulary.h"
#include "LFSlam/ORBextractor.h"
#include "LFSlam/FrontEndBase.h"
#include "LFSlam/LightFieldFrame.h"
#include "LFSlam/GlobalMap.h"
#include "common_utils/StringEnumerator.hpp"
#include "common_utils/matplotlibcpp.h"
#include "common_utils/utilities.h"
#include "LFSlam/LoopCloser.h"

#include "gtsam/geometry/triangulation.h"
#include "LFSlam/GtsamFactorHelpers.h"
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/slam/ProjectionFactorPPP.h>
#include <gtsam_unstable/slam/ProjectionFactorPPPC.h>

#include "LFSlam/utils.h"
#include <boost/assign.hpp>
#include <boost/assign/std/vector.hpp>

// OPENGV ////
#include <Eigen/Eigen>
#include <opengv/types.hpp>
#include <memory>
#include <opengv/relative_pose/NoncentralRelativeAdapter.hpp>
#include <opengv/sac_problems/relative_pose/NoncentralRelativePoseSacProblem.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/relative_pose/methods.hpp>

#include <opengv/absolute_pose/methods.hpp>
#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include <opengv/point_cloud/methods.hpp>
#include <opengv/point_cloud/PointCloudAdapter.hpp>

#include "LFSlam/time_measurement.hpp"
#include <opencv2/core/eigen.hpp>
#include <boost/filesystem.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace opengv;

typedef gtsam::CameraSet<gtsam::PinholeCamera<gtsam::Cal3_S2>> Cameras;

enum INIT_COND{
    MIN_FEATS =0,
    RANSAC_FILTER=1
};
enum POSEST_ALGO{
    PC_ALIGN=0,
    SEVENTEEN_PT=1,
    G_P3P=2
};
enum INTER_MATCH{
    BF_MATCH=0,
    BoW_MATCH=1
};

class FrontEnd: public FrontEndBase {
public:
    FrontEnd(string strSettingsFile, CamArrayConfig &camconfig, bool debug)
            : frontend_config_file(strSettingsFile),
              camconfig_(camconfig), DEBUG_MODE(debug){
        //Check and load settings file
        cv::FileStorage fSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if(!fSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

       //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
        string voc_file = fSettings["Vocabulary"];
        // read the vocabulary and create the voc object
        orb_vocabulary = new ORBVocabulary();
        bool bVocLoad = orb_vocabulary->loadFromTextFile(voc_file);
        if(!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Failed to open at: " << voc_file << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        //init_cond_ = (INIT_COND)fSettings["InitCondition"];
        //posestAlgo_ = (POSEST_ALGO)fSettings["PoseEstimation"];
        init_cond_=  static_cast<INIT_COND>((int)fSettings["InitCondition"]);
        posestAlgo_=  static_cast<POSEST_ALGO>((int)fSettings["PoseEstimation"]);
        interMatch_ = static_cast<INTER_MATCH>((int)fSettings["InterMatch"]);

        kf_translation_threshold = (double)fSettings["KFBaselineThresholdTranslation"];
        cout<<"kf threshold T: "<<kf_translation_threshold;

        kf_rotation_threshold = (double)fSettings["KFBaselineThresholdRotation"];
        cout<<"kf threshold R: "<<kf_rotation_threshold;

        string logDir = fSettings["LogDir"];
        boost::filesystem::path dir(logDir.c_str());
        if(boost::filesystem::create_directory(dir))
        {
            std::cerr<< "Directory Created: "<<logDir<<std::endl;
        }
        dir /= logFileName_;
        logFile_.open (dir.string());
        boost::filesystem::path dir2(logDir.c_str());
        dir2 /= logFileName2_;
        //logFile2_.open (dir2.string());
        //logFile_ << "Writing poses and landmarks to a file.\n";

        // Load ORB parameters
        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        // create the object to compute ORB features
        orBextractor = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
        for(int i =0; i <camconfig.num_cams_ ; i++){
            orBextractors.push_back(new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST));
        }

        //create the ORBDatabase object to parse and store images into Bag of words vector
        orb_database = new ORBDatabase(*orb_vocabulary, true , 2);

        //compute the rectification homography
        currentFramePose = cv::Mat::eye(3, 4, CV_64F);
        //currentFramePose1 = cv::Mat::eye(3, 4, CV_64F);
        //currentFramePose2 = cv::Mat::eye(3, 4, CV_64F);
        current_frameId = 0;
        currentKFID = 0;
        //initialized_ = false;
        initialized_ = NOT_INITIALIZED;
        initializationTrials = 0;
        //map object
        map = new GlobalMap();

        initialized_mono_ = false;
        currentFramePose_mono = cv::Mat::eye(3, 4, CV_64F);
        map_mono = new GlobalMap();

        //Create Loop closer objects
        lcLF_ = new LoopCloser(orb_vocabulary);
        lcMono_ = new LoopCloser(orb_vocabulary);

        convertCamConfig_CV2GTSAM(camconfig, RT_Mats);
        convertCamConfig_CV2GTSAM(camconfig, RT_Mats_init);

        clahe = cv::createCLAHE(3.0, cv::Size(8, 8));

    }


    ~FrontEnd(){

    }

    //MOno
    bool initialization_mono(vector<DMatch> &matches_mono);
    void estimatePose_Mono();
    void findMatchesMono(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur, int cam_ind, std::vector<DMatch>& matches);
    void insertKeyFrame_Mono();
    void normalizeKps(vector<cv::Point2f>& kps, vector<cv::Point2f>& kps_n, Mat& T);

    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
    void Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
    void getDataForGTSAMTriangulation(LightFieldFrame *lf, int iMatchInd, Cameras &cameras,
                                      gtsam::Point2Vector &measurements, vector<int> &compCamInds,
                                      vector<int> &octaves);
    void TriangulateGTSAM();
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::Point2f>& kps1,
                          const vector<cv::Point2f>& kps2, vector<bool> &inliers,
                          const cv::Mat &K, vector<cv::Point3f> &P3D, float th2,
                          vector<bool> &good, float &parallax, const cv::Mat &R1, const cv::Mat &t1);
    bool checkTriangulation(vector<Mat>& PJs, vector<Point2f>& kps, cv::Mat& P3D, float &parallax);
    bool ReconstructF(const vector<cv::Point2f>& kps1, const vector<cv::Point2f>& kps2,
                      vector<bool> &inliers, cv::Mat &F, cv::Mat &K,float sigma,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D,
                      vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
    cv::Mat solveF_8point(const vector<cv::Point2f> &kps1, const vector<cv::Point2f> &kps2);
    void generateRandomIndices(int max_ind, vector<vector<int>>& indices_set, int num_sets);
    float CheckFundamental(const cv::Mat &F, vector<cv::Point2f>& kps1,
                          vector<cv::Point2f>& kps2, vector<bool> &inliers, float sigma);
    void FindFundamentalMatrix(vector<Point2f> kps1, vector<Point2f> kps2, Mat& F, vector<bool> &inliers);

    cv::Mat getPose_Mono();
    vector<cv::Mat> getAllPoses_Mono();


    void createFrame(vector<Mat> img_set, vector<Mat> segmap_set, double timeStamp);
    void computeRefocusedImage();
    void filterIntraMatches( std::vector<IntraMatch>& matches_map, LightFieldFrame* currentFrame,
                             std::vector<IntraMatch>& filtered_intra_matches,
                             vector<DBoW2::NodeId >& words_, set<DBoW2::NodeId >& filtered_words);

    void bundleAdjustIntraMatches(std::vector<IntraMatch>& tmp_intra_matches);

    void obtainLfFeatures( std::vector<IntraMatch>& matches_map, LightFieldFrame* currentFrame,
                           std::vector<IntraMatch>& filtered_intra_matches,
                           vector<DBoW2::NodeId >& words_, set<DBoW2::NodeId >& filtered_words);

    void processFrame();
    void processFrameNon();
    bool trackLF();
    void initialization();
    bool initialization_non_overlapping(vector<DMatch> &mono_matches);
    void insertKeyFrame();
    void deleteCurrentFrame();

    void BruteForceMatching(Mat img1, Mat img2, vector<Mat> descs1, vector<Mat> descs2, vector<KeyPoint> kps1, vector<KeyPoint> kps2);
    void findInterMatches(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur, std::vector<DMatch>& matches_z_filtered,
                          std::vector<DMatch>& matches_mono_z_filtered, bool viz);
    void findInterMatches(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur, std::vector<DMatch>& matches_z_filtered,
                          bool viz);
    void findInterMatchesBow(LightFieldFrame *lf_prev, LightFieldFrame *lf_cur, std::vector<DMatch> &matches,
                             std::vector<DMatch> &mono_matches, bool viz);
    void findInterMatchesBow(LightFieldFrame *lf_prev, LightFieldFrame *lf_cur, std::vector<DMatch> &matches,
                             bool viz);
    void InterMatchingBow( DBoW2::FeatureVector lfFeatVec1, DBoW2::FeatureVector lfFeatVec2, std::vector<cv::Mat> img_desc1,
                           std::vector<cv::Mat> img_desc2, vector<unsigned int>& indices_1,
                           vector<unsigned int>& indices_2);
    void InterMatchingBow(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur, vector<unsigned int>& indices_1,
                          vector<unsigned int>& indices_2,set<DBoW2::NodeId>& words );
    void get3D_2DCorrs(LightFieldFrame *lf, int featInd, Mat pose, vector<Mat> &projmats, vector<Mat> &PJs, std::vector<Mat_<double>> &xs,
                       vector<int> &view_inds, vector<Point2f> &kps, vector<int> &octaves);
    int estimatePoseLF(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur,
                                 std::vector<DMatch>& matches, vector<bool>& inliers, Mat& T, bool SAC);
    int poseFromPCAlignment (LightFieldFrame* lf_prev, LightFieldFrame* lf_cur,
    std::vector<DMatch>& matches, vector<bool>& inliers, Mat& T,  bool SAC);
    int poseFromSeventeenPt(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur,
                            std::vector<DMatch>& matches, vector<bool>& inliers, Mat& T,  bool SAC);
    int absolutePoseFromGP3P(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur,
                            std::vector<DMatch>& matches, vector<bool>& inliers, Mat& T,  bool SAC);
    void OptimizePose(vector<DMatch> matches, vector<int> lids, vector<bool>& inliers,
                      transformation_t nonlinear_transformation, Mat& poseOut);

    cv::Mat getPose();
    cv::Mat getPose_seventeen();
    cv::Mat getPose_gp3p();
    vector<cv::Mat> getAllPoses_gp3p();
    vector<cv::Mat> getAllPoses();
    vector<cv::Mat> getAllPoses1();
    void getMapPoints(vector<Point3f>& mapPoints);

    void reset();
    void drawMatchesArrows(cv::Mat& img, vector<KeyPoint>& kps_prev, vector<KeyPoint>& kps_cur,  std::vector<DMatch> matches, Mat mask, cv::Scalar color);

    void check_tri_sp(int num_views, vector<int> view_inds, IntraMatch& matches,  LightFieldFrame* lf);

    void writeTrajectoryToFile(const string &filename,  bool mono);

    void featureStats();

    void drawInterMatch(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur, int prev_ind, int cur_ind);
    void findLoopCandidates(vector<Mat>& images, vector<Mat>& segMasks);
    void prepareOpengvData(vector<Point2f>& kps_1, vector<Point2f>& kps_2, int cam1, int cam2);
    void tracksForHist();
    void searchLocalMap(LightFieldFrame *prevKF, vector<DMatch> inter_matches_with_landmarks, vector<bool> inliers,
                        vector<DMatch> &inlierAllMapMatches, Mat &refinedPose, vector<int> &alllids);
    void writeLogs( string stats_file);



    //only for mono-comparison
    GlobalMap* map_mono;
    bool initialized_mono_;
    vector<LightFieldFrame*> lfFrames_mono;
    vector<cv::Mat> allPoses_mono;
    vector<double> poseTimeStamps_mono;
    cv::Mat currentFramePose_mono;
    int initializationTrials;
    Mat prev_rvec, prev_tvec;

    bool DEBUG_MODE;
    INIT_STATE initialized_ = NOT_INITIALIZED;
    INIT_COND init_cond_= RANSAC_FILTER;
    POSEST_ALGO posestAlgo_ = SEVENTEEN_PT;
    INTER_MATCH interMatch_ = BoW_MATCH;

    double kf_translation_threshold;
    double kf_rotation_threshold;

    int current_frameId, currentKFID;
    LightFieldFrame* currentFrame;
    vector<LightFieldFrame*> lfFrames;
    CamArrayConfig& camconfig_;
    string frontend_config_file;
    string logFileName_ = "poses_landmark.txt";
    string logFileName2_ = "poses_stats.txt";
    ofstream logFile_, logFile2_;
    ORBextractor* orBextractor;
    vector<ORBextractor*> orBextractors;
    ORBVocabulary* orb_vocabulary;
    ORBDatabase* orb_database;
    GlobalMap* map;
    //MulticamElas* reconstructor;
    LoopCloser* lcMono_;
    LoopCloser* lcLF_;
    LoopCloser* lcMasked_;

    //cv::Mat currentFramePose;
    //EXtra variables for pose visualization
    cv::Mat currentFramePose1, currentFramePose2, currentFramePose3;


    vector<double> poseTimeStamps;
    vector<cv::Mat> keyFramePoses;

    cv::Ptr<cv::CLAHE> clahe;

    // opengv pose estimation stuff
    vector<int> correspondences_1, correspondences_2;
    opengv::bearingVectors_t bearings_1, bearings_2;
    opengv::rotations_t rotations;
    opengv::translations_t translations;
    vector<gtsam::Pose3> RT_Mats, RT_Mats_init;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


};


#endif //LIGHT_FIELDS_ROS_FRONTEND_H
