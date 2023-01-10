//
// Created by Pushyami Kaveti on 1/24/22.
//

#ifndef SRC_MONOFRONTEND_H
#define SRC_MONOFRONTEND_H


#include "MCDataUtils/CamArrayConfig.h"
#include "MCSlam/ORBVocabulary.h"
#include "MCSlam/ORBextractor.h"
#include "common_utils/tools.h"
#include "MCSlam/FrontEndBase.h"
#include "MCSlam/GlobalMap.h"
#include "common_utils/StringEnumerator.hpp"
#include "common_utils/matplotlibcpp.h"
#include "common_utils/utilities.h"
#include "MCSlam/MonoFrame.h"
// OPENGV ////
#include <Eigen/Eigen>
#include <memory>

#include "MCSlam/time_measurement.hpp"
#include <opencv2/core/eigen.hpp>
#include <boost/filesystem.hpp>
#include <mutex>
#include <chrono>

using namespace std;
using namespace cv;


class MonoFrontEnd : public FrontEndBase{

public:
    MonoFrontEnd(string strSettingsFile, Mat Kmat, Mat distMat, bool debug): frontend_config_file(strSettingsFile),
    Kmat_(Kmat), distmat_(distMat), DEBUG_MODE(debug){
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

        //interMatch_ = static_cast<INTER_MATCH>((int)fSettings["InterMatch"]);

        kf_translation_threshold = (double)fSettings["KFBaselineThresholdTranslation"];
        cout<<"kf threshold T: "<<kf_translation_threshold;

        kf_rotation_threshold = (double)fSettings["KFBaselineThresholdRotation"];
        cout<<"kf threshold R: "<<kf_rotation_threshold;

        kf_triangulation_angle_threshold = (double)fSettings["KFTriangulationAngleThreshold"];

        string logDir = fSettings["LogDir"];
        boost::filesystem::path dir(logDir.c_str());
        if(boost::filesystem::create_directory(dir))
        {
            std::cerr<< "Directory Created: "<<logDir<<std::endl;
        }
        dir /= logFileName_;
        logFile_.open (dir.string());
        //logFile_ << "Writing poses and landmarks to a file.\n";

        // Load ORB parameters
        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        // create the object to compute ORB features
        orBextractor = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

        //create the ORBDatabase object to parse and store images into Bag of words vector
        orb_database = new ORBDatabase(*orb_vocabulary, true , 2);

        //compute the rectification homography
        currentFramePose = cv::Mat::eye(3, 4, CV_64F);

        current_frameId = 0;
        //initialized_ = false;
        initialized_ = NOT_INITIALIZED;
        //map object
        map = new GlobalMap();

        //lcMono_ = new LoopCloser(orb_vocabulary);

    }


    ~MonoFrontEnd(){

    }

    //MOno
    void initialization();
    void estimatePose_Mono();
    void findMatchesMono(MonoFrame* lf_prev, MonoFrame* lf_cur, std::vector<DMatch>& matches);
    void insertKeyFrame();
    void normalizeKps(vector<cv::Point2f>& kps, vector<cv::Point2f>& kps_n, Mat& T);

    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
    void Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::Point2f>& kps1,
                const vector<cv::Point2f>& kps2, vector<bool> &inliers,
                const cv::Mat &K, vector<cv::Point3f> &P3D, float th2,
                vector<bool> &good, float &parallax, const cv::Mat &R1, const cv::Mat &t1);
    bool ReconstructF(const vector<cv::Point2f>& kps1, const vector<cv::Point2f>& kps2,
                      vector<bool> &inliers, cv::Mat &F, cv::Mat &K,float sigma,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D,
                      vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
    cv::Mat solveF_8point(const vector<cv::Point2f> &kps1, const vector<cv::Point2f> &kps2);


    float CheckFundamental(const cv::Mat &F, vector<cv::Point2f>& kps1,
                           vector<cv::Point2f>& kps2, vector<bool> &inliers, float sigma);

    void generateRandomIndices(int max_ind, vector<vector<int>>& indices_set, int num_sets);

    void FindFundamentalMatrix(vector<Point2f> kps1, vector<Point2f> kps2, Mat& F, vector<bool> &inliers);

    void createFrame(Mat img_set, Mat segmap_set, double timeStamp);

    void processFrame();

    void CheckTriAngle(MonoFrame* prev_KF, MonoFrame* curFr, vector<DMatch> matches_with_lms,  vector<bool>& inliers);

    void tracksForHist();
    bool trackMono();

    cv::Mat getPose();
    vector<cv::Mat> getAllPoses();
    vector<cv::Mat> getAllPoses1();
    void getMapPoints(vector<Point3f>& mapPoints);
    void reset();
    void drawMatchesArrows(cv::Mat& img, vector<KeyPoint>& kps_prev, vector<KeyPoint>& kps_cur,  std::vector<DMatch> matches, Mat mask, cv::Scalar color);
    void featureStats();
    void writeTrajectoryToFile(const string &filename,  bool mono);

   // void findLoopCandidates(vector<Mat>& images, vector<Mat>& segMasks);
    void writeLogs();


    bool DEBUG_MODE;
    INIT_STATE initialized_ = NOT_INITIALIZED;

    double kf_translation_threshold;
    double kf_rotation_threshold;
    double kf_triangulation_angle_threshold;

    Mat prev_rvec, prev_tvec;

    int current_frameId;
    MonoFrame* currentFrame;
    vector<MonoFrame*> frames_;
    Mat Kmat_, distmat_;
    string frontend_config_file;
    string logFileName_ = "poses_landmark.txt";
    ofstream logFile_;
    ORBextractor* orBextractor;
    ORBVocabulary* orb_vocabulary;
    ORBDatabase* orb_database;
    GlobalMap* map;

   // LoopCloser* lcMasked_;

    //EXtra variables for pose visualization
    cv::Mat currentFramePose1, currentFramePose2, currentFramePose3;
    vector<cv::Mat> allPoses1, allPoses2;

    vector<double> poseTimeStamps;
    vector<cv::Mat> keyFramePoses;


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


#endif //SRC_MONOFRONTEND_H
