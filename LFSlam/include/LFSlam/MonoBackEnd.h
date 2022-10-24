//
// Created by Pushyami Kaveti on 1/24/22.
//

#ifndef SRC_MONOBACKEND_H
#define SRC_MONOBACKEND_H

#include "LFSlam/MonoFrontEnd.h"
#include "LFSlam/GlobalMap.h"
#include "common_utils/tools.h"

#include <gtsam/inference/Symbol.h>

// We want to use iSAM2 to solve the structure-from-motion problem incrementally, so
// include iSAM2 here
#include <gtsam/nonlinear/ISAM2.h>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>

// iSAM2 requires as input a set set of new factors to be added stored in a factor graph,
// and initial guesses for any new variables used in the added factors
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

// In GTSAM, measurement functions are represented as 'factors'. Several common factors
// have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
// Here we will use Projection factors to model the camera's landmark observations.
// Also, we will initialize the robot at some location using a Prior factor.
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>



class MonoBackEnd {
public:
    MonoBackEnd(string strSettingsFile, Mat Kmat, MonoFrontEnd *fe);

    ~MonoBackEnd();

    static gtsam::Point2 convertPoint2_CV2GTSAM(cv::KeyPoint &kp);

    static gtsam::Pose3 convertPose3_CV2GTSAM(cv::Mat &pose);

    static gtsam::Point3 convertPoint3_CV2GTSAM(cv::Mat &landmark);

    void displayMatches(MonoFrame *prev_frame, MonoFrame *current_frame);

    bool checkTriangulationAngle(gtsam::Point3 &pose1, gtsam::Point3 &pose2, gtsam::Point3 &landmark,
                                 double &angle_deg) const;
    double computeReprojectionError(gtsam::Pose3 &pose1, int camID1, gtsam::Point3 &landmark, gtsam::Point2 &obs);

    void addPosePrior(int lid, GlobalMap *map);

    void addLandmarkPrior(int lid, gtsam::Point3 &landmark, GlobalMap *map);

    bool obtainPrevMeasurement(Landmark *landmark, gtsam::Point2 &prev_measurement, int &prev_pose_id,
                               gtsam::Pose3 &prev_pose, bool init=false);

    void filterLandmarksStringent(MonoFrame* currentFrame, GlobalMap *map, vector<int> &lids_filtered,
                                  vector<gtsam::Point3> &landmarks, vector<gtsam::Point2> &current_measurements,
                                  vector<gtsam::Point2> &previous_measurements, vector<int> &previous_pose_ids);

    bool addKeyFrame();

    void optimizePosesLandmarks();

    void updateVariables();

    //variables

    unordered_map <int, vector <int>> landmark_log;
    unordered_map <int, int> pose_log;

    MonoFrontEnd* frontEnd;
    string backend_config_file;
    int windowSize;
    int windowCounter;
    int imageSkip;
    int camID;
    cv::Mat camTransformation;
    double angleThresh;
    int optimizationMethod;
    gtsam::Cal3_S2::shared_ptr K;
    gtsam::noiseModel::Isotropic::shared_ptr measurementNoise;
    gtsam::ISAM2Params parameters;
    gtsam::LevenbergMarquardtParams params;
    gtsam::ISAM2 isam;
    gtsam::LevenbergMarquardtOptimizer *optimizer;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initialEstimate;
    gtsam::Values currentEstimate;

};


#endif //SRC_MONOBACKEND_H
