//
// Created by Pushyami Kaveti on 8/21/21.
//

// Each variable in the system (poses and landmarks) must be identified with a unique key.
// We can either use simple integer keys (1, 2, 3, ...) or symbols (X1, X2, L1).
// Here we will use Symbols
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
#include <gtsam/slam/BetweenFactor.h>
#include "common_utils/tools.h"
#include "MCDataUtils/CamArrayConfig.h"
#include "MCSlam/GlobalMap.h"

// gtsam fixed lag smoother variables
#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>

#ifndef SRC_BACKEND_H
#define SRC_BACKEND_H



enum BACKEND_TYPE{
    MONO =0,
    MULTI=1,
    MULTI_RIGID=2
};

class Backend {
public:
    Backend(string strSettingsFile, CamArrayConfig &camconfig, FrontEnd *fe);

    ~Backend();

    /*static gtsam::Point2 convertPoint2_CV2GTSAM(cv::KeyPoint &kp);

    static gtsam::Pose3 convertPose3_CV2GTSAM(cv::Mat &pose);

    static gtsam::Point3 convertPoint3_CV2GTSAM(cv::Mat &landmark); */

    //bool addKeyFrame(cv::Mat &pose, vector<cv::KeyPoint> &keyPoints, vector<cv::Mat> &landmarks, int &frame_id, vector<int>* lids);
    void displayMatches(MultiCameraFrame *prev_frame, MultiCameraFrame *current_frame);

    bool checkTriangulationAngle(gtsam::Point3 &pose1, gtsam::Point3 &pose2, gtsam::Point3 &landmark,
                                 double &angle_deg) const;
    double computeReprojectionError(gtsam::Pose3 &pose1, int camID1, gtsam::Point3 &landmark, gtsam::Point2 &obs);

    gtsam::Pose3 calCompCamPose(gtsam::Pose3 &pose);
    gtsam::Pose3 calCompCamPose(MultiCameraFrame* lf, int cID);

    void addPosePrior(int lid, GlobalMap *map);

    void addLandmarkPrior(int lid, gtsam::Point3 &landmark);

    bool obtainPrevMeasurement(Landmark *landmark, gtsam::Point2 &prev_measurement, int &prev_pose_id,
                               gtsam::Pose3 &prev_pose, bool init=false);

    void filterLandmarks(MultiCameraFrame *currentFrame, GlobalMap *map, vector<int> &lids_filtered,
                         vector<gtsam::Point3> &landmarks, vector<gtsam::Point2> &current_measurements,
                         vector<gtsam::Point2> &previous_measurements, vector<int> &previous_pose_ids);

    void filterLandmarksStringent(MultiCameraFrame* currentFrame, GlobalMap *map, vector<int> &lids_filtered,
                                  vector<gtsam::Point3> &landmarks, vector<gtsam::Point2> &current_measurements,
                                  vector<gtsam::Point2> &previous_measurements, vector<int> &previous_pose_ids);

    void getCamIDs_Angles(Landmark* l, int KFID1, int KFID2 , Mat& angles , vector<int>& prevRaysCamIds,
                          vector<int>& curRaysCamIDs, int& maxAnglePrevCamID, int& maxAngleCurCamID);

    void insertPriors(int lid);

    void insertLandmarkInGraph(Landmark *l, int prevKFID, int curKFID, bool newlm, vector<int> previous_compcam_ids,
                               vector<gtsam::Point2> previous_measurements, vector<int> cur_compcam_ids,
                               vector<gtsam::Point2> current_measurements, vector<int> prevKPOctaves,
                               vector<int> curKPOctaves);
    bool addKeyFrameMulti();
    bool addKeyFrame();
    bool addKeyFrameMultiLatest();
    bool addKeyFrameMultiSensorOffset();

    void optimizePosesLandmarks();

    void removeVariables( gtsam::KeyVector tobeRemoved);
    void updateVariables();

    void globalOptimization();

    void optimizePose();

    //variables
    BACKEND_TYPE backendType = MULTI;
    vector<Mat> Rt_mats_;
    vector<Mat> Rt_mats_kalib_;
    CamArrayConfig* camArrayConfig;
    bool reinitialized_;

    FrontEnd* frontEnd;
    string backend_config_file;
    int windowSize;
    int windowCounter;
    int imageSkip;
    int camID;
    cv::Mat camTransformation;
    double angleThresh;
    int optimizationMethod;
    vector<gtsam::Cal3_S2::shared_ptr> K;
    gtsam::noiseModel::Isotropic::shared_ptr measurementNoise;
    noiseModel::Robust::shared_ptr huberModel;
    gtsam::ISAM2Params parameters;
    gtsam::LevenbergMarquardtParams params;
    gtsam::ISAM2 isam;
    gtsam::FactorIndices toBeRemovedFactorIndices;
    gtsam::LevenbergMarquardtOptimizer *optimizer;
    gtsam::BatchFixedLagSmoother fixedLagOptimizer;
    gtsam::FixedLagSmoother::KeyTimestampMap newTimeStamps;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initialEstimate;
    gtsam::Values currentEstimate;
    /// variable to keep track of landmark factors across multiple iterations
    std::map<int, vector<int>> lmFactorRecord;
    std::map<int, vector<int>> xFactorRecord;
    std::map<int, std::map<int, vector<int>>> lmFactorRecordMulti;

};

#endif //SRC_BACKEND_H

//if(!currentEstimate.exists(gtsam::Symbol('l', lids->at(i))) and
//!initialEstimate.exists(gtsam::Symbol('l', lids->at(i))))
//{
//if(numKFs <=2)
//continue;
///// Get the previous LF frame of the landmark and the observations in the component cameras
//vector<int> prevRaysCamIds, curRaysCamIDs;
//bool firstPair = true;
//for (int pairIdx=1 ; pairIdx< numKFs ; pairIdx++){
//int prevKFIdx =  pairIdx-1;     //(numKFs-2);
//int curKFIdx =   pairIdx;       //numKFs-1;
//MultiCameraFrame* lf1 = l->KFs[prevKFIdx]; // previous KF
//MultiCameraFrame* lf2 = l->KFs[curKFIdx];  // same as l->KFs[numKFs-1]; // last KF
//IntraMatch* im1 = &lf1->intraMatches.at(l->featInds[prevKFIdx]);
//IntraMatch* im2 = &lf2->intraMatches.at(l->featInds[curKFIdx]);
//
//Mat angles = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
//Mat tri_errors = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
//
/////check all combinations of rays and get the triangulation angles and reprojection errors
//
//int curKFCamID = -1, prevKFCamID = -1;
//double maxAngle = 0;
//curRaysCamIDs.clear();
//getCamIDs_Angles(l, prevKFIdx, curKFIdx, angles , prevRaysCamIds,curRaysCamIDs, prevKFCamID, curKFCamID);
//if(prevKFCamID != -1 and curKFCamID != -1)
//maxAngle = angles.at<double>(prevKFCamID,curKFCamID);
//// compare all the angles and choose the  largest one for insertion
////first check the diagonal angles i.e intersecting component cameras between two LF fram
//
//vector<int> acceptedPrevCamIDs, acceptedCurCamIDs;
//vector<gtsam::Point2> acceptedPrevMeas, acceptedCurMeas;
//if(maxAngle > 0){
//
//acceptedPrevCamIDs.push_back(prevKFCamID);
//acceptedPrevMeas.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[prevKFCamID][im1->matchIndex[prevKFCamID]]));
//acceptedCurCamIDs.push_back(curKFCamID);
//acceptedCurMeas.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[curKFCamID][im2->matchIndex[curKFCamID]]));
//
//vector<int>::iterator itr1 = prevRaysCamIds.begin();
//vector<int>::iterator itr2 = curRaysCamIDs.begin();
//
//for( ; itr1!=prevRaysCamIds.end() or itr2!= curRaysCamIDs.end() ; ){
//int camID1 = *itr1;
//int camID2 = *itr2;
///////////////// Check the latest in prevrays
//if( itr1!=prevRaysCamIds.end()){
///// if this is the first KF pair for the landmark
///// choose which rays to insert from the previous  component cams
//if(camID1 != prevKFCamID){
//if(firstPair){
//bool accept=true;
//vector<int>::iterator it_set = acceptedPrevCamIDs.begin();
//
//for( ; it_set != acceptedPrevCamIDs.end() ; ++it_set){
//int kp_ind1 = im1->matchIndex[camID1];
//int kp_ind2 = im1->matchIndex[*it_set];
//gtsam::Pose3 transformedPose1 = calCompCamPose(lf1, camID1); // take input of the LF frame pose and comp cam id to compute the comp cam pose
//gtsam::Pose3 transformedPose2 = calCompCamPose(lf1, *it_set);
//gtsam::Point2 obs1 = convertPoint2_CV2GTSAM(lf1->image_kps_undist[camID1][kp_ind1]);
//gtsam::Point2 obs2  = convertPoint2_CV2GTSAM(lf1->image_kps_undist[*it_set][kp_ind2]);
//
////check the reprojection error again to filter any outliers
//double error1 = computeReprojectionError(transformedPose1, camID1, landmark, obs1);
//double error2 = computeReprojectionError(transformedPose2, *it_set, landmark, obs2);
//double tri_angle_deg=0;
////compute angle
//bool angleCheck = checkTriangulationAngle(const_cast<gtsam::Point3 &>(transformedPose1.translation()),
//                                          const_cast<gtsam::Point3 &>(transformedPose2.translation()),
//                                          landmark , tri_angle_deg);
//
//if(!angleCheck || error1 >4 || error2 >4){
//accept = false;
//break;
//}
//}
//
//if(accept){
//it_set = acceptedCurCamIDs.begin();
//for( ; it_set != acceptedCurCamIDs.end() ; ++it_set){
//if(angles.at<double>(camID1, *it_set) == 0){
//accept = false;
//break;
//}
//}
//if(accept)
//{
//acceptedPrevCamIDs.push_back(camID1);
//acceptedPrevMeas.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[camID1][im1->matchIndex[camID1]]));
//
//}
//}
//
//}
///// if this is not the first KF pair for the landmark
///// accept all the rays in the previous frame, since they are already chosen
//else{
//acceptedPrevCamIDs.push_back(camID1);
//acceptedPrevMeas.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[camID1][im1->matchIndex[camID1]]));
//}
//}
//++itr1;
//}
//
//if(itr2!= curRaysCamIDs.end()){
//
//if(camID2 != curKFCamID){
//bool accept=true;
//
//vector<int>::iterator it_set = acceptedCurCamIDs.begin();
//
//for( ; it_set != acceptedCurCamIDs.end() ; ++it_set){
//int kp_ind1 = im2->matchIndex[camID2];
//int kp_ind2 = im2->matchIndex[*it_set];
//gtsam::Pose3 transformedPose1 = calCompCamPose(lf2, camID2); // take input of the LF frame pose and comp cam id to compute the comp cam pose
//gtsam::Pose3 transformedPose2 = calCompCamPose(lf2, *it_set);
//gtsam::Point2 obs1 = convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][kp_ind1]);
//gtsam::Point2 obs2  = convertPoint2_CV2GTSAM(lf2->image_kps_undist[*it_set][kp_ind2]);
//
////check the reprojection error again to filter any outliers
//double error1 = computeReprojectionError(transformedPose1, camID2, landmark, obs1);
//double error2 = computeReprojectionError(transformedPose2, *it_set, landmark, obs2);
//double tri_angle_deg=0;
////compute angle
//bool angleCheck = checkTriangulationAngle(const_cast<gtsam::Point3 &>(transformedPose1.translation()),
//                                          const_cast<gtsam::Point3 &>(transformedPose2.translation()),
//                                          landmark , tri_angle_deg);
//
//if(!angleCheck || error1 >4 || error2 >4){
//accept = false;
//break;
//}
//}
//
//if(accept){
//it_set = acceptedPrevCamIDs.begin();
//for( ; it_set != acceptedPrevCamIDs.end() ; ++it_set){
//if(angles.at<double>(*it_set, camID2) == 0){
//accept = false;
//break;
//}
//}
//if(accept)
//{
//acceptedCurCamIDs.push_back(camID2);
//acceptedCurMeas.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][im2->matchIndex[camID2]]));
//}
//}
//}
//++itr2;
//}
//}
//VLOG(2)<<"New landmark LID : "<<lids->at(i)<<"  KF1 :"<<lf1->frameId<<", KF2:"<<lf2->frameId<<", PT: "<<landmark.x()<<","<<landmark.y()<<","<<landmark.z()<<" , max angle : "<<maxAngle<<"Num meas: "<<acceptedCurCamIDs.size()+acceptedPrevCamIDs.size()<<endl;
//
//if(firstPair){
//firstPair  = false;
//num_lms_filtered++;
//if(num_lms_filtered == 1){
//insertPriors(lids->at(i));
//}
//insertLandmarkInGraph(l , prevKFIdx, curKFIdx, true, acceptedPrevCamIDs,
//acceptedPrevMeas, acceptedCurCamIDs,  acceptedCurMeas);
//
//prevRaysCamIds = acceptedCurCamIDs;
//
//}
//else{
//insertLandmarkInGraph(l , prevKFIdx, curKFIdx, false, acceptedPrevCamIDs,
//acceptedPrevMeas, acceptedCurCamIDs,  acceptedCurMeas);
//
//}
//
//}
//
///*  if ( maxAngle > 0 ) {
//
//      //lids_filtered.push_back(lids->at(i));
//      //landmarks.push_back(landmark);
//      //previous_KFs.push_back(lf1);
//      //prevKFInds.push_back(prevKFIdx);
//      //new_landmark_flags.push_back(true);
//      //current_measurements.push_back(acceptedCurMeas);
//      //cur_compcam_ids.push_back(acceptedCurCamIDs);
//      /// record the previous measurements and previous KF and comp cam ID
//     // previous_measurements.push_back(acceptedPrevMeas);
//      //previous_compcam_ids.push_back(acceptedPrevCamIDs);
//  } */
//}
//
//}
//else{
///// this is an existing landmark in the graph.
///// Grab the last observation
//std::map<int, vector<int>> facs = lmFactorRecordMulti[lids->at(i)];
//VLOG(2)<<"Existing Landmark"<<endl;
//assert(facs.size() != 0);
//
///// Get the prev factor's frameID and componenet camera IDs
//vector<int> prevRaysCamIds = facs.rbegin()->second;
//int prevKFIdx = facs.rbegin()->first;
//int curKFIdx = numKFs-1;
//MultiCameraFrame* lf1;
//IntraMatch* im1;
//
//lf1 = l->KFs[prevKFIdx];
//im1 = &lf1->intraMatches.at(l->featInds[prevKFIdx]);
//int prevFrameID = lf1->frameId;
//VLOG(3)<<"prevFrameID: "<<prevFrameID<<endl; //<<", CamIDs: "<<CamID1<<endl;
//
//MultiCameraFrame* lf2 = currentFrame;  // same as l->KFs[numKFs-1]; // last KF
//IntraMatch* im2 = &lf2->intraMatches.at(l->featInds.back());
//
//double maxAngle = 0;
//int curKFCamID = -1, prevKFCamID = -1;
//vector<int>  curRaysCamIDs;
//Mat angles = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
//Mat tri_errors = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
//
//getCamIDs_Angles(l, prevKFIdx, curKFIdx , angles , prevRaysCamIds,curRaysCamIDs, prevKFCamID, curKFCamID);
//if(prevKFCamID != -1 and curKFCamID != -1)
//maxAngle = angles.at<double>(prevKFCamID,curKFCamID);
//// compare all the angles and choose the  largest one for insertion
////first check the diagonal angles i.e intersecting component cameras between two LF frame
//vector<int> acceptedPrevCamIDs, acceptedCurCamIDs;
//vector<gtsam::Point2> acceptedPrevMeas, acceptedCurMeas;
//if(maxAngle > 0){
//acceptedPrevCamIDs.push_back(prevKFCamID);
//acceptedPrevMeas.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[prevKFCamID][im1->matchIndex[prevKFCamID]]));
//acceptedCurCamIDs.push_back(curKFCamID);
//acceptedCurMeas.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[curKFCamID][im2->matchIndex[curKFCamID]]));
//}
//
//vector<int>::iterator itr1 = prevRaysCamIds.begin();
//for( ; itr1!=prevRaysCamIds.end() ; ++itr1){
//int camID1 = *itr1;
//if(camID1 != prevKFCamID){
//acceptedPrevCamIDs.push_back(camID1);
//acceptedPrevMeas.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[camID1][im1->matchIndex[camID1]]));
//}
//}
//
//
//vector<int>::iterator itr2 = curRaysCamIDs.begin();
//for( ;  itr2!= curRaysCamIDs.end() ; ++itr2){
//int camID2 = *itr2;
//
//if(camID2 != curKFCamID){
//bool accept=true;
//
//vector<int>::iterator it_set = acceptedCurCamIDs.begin();
//
//for( ; it_set != acceptedCurCamIDs.end() ; ++it_set){
//int kp_ind1 = im2->matchIndex[camID2];
//int kp_ind2 = im2->matchIndex[*it_set];
//gtsam::Pose3 transformedPose1 = calCompCamPose(lf2, camID2); // take input of the LF frame pose and comp cam id to compute the comp cam pose
//gtsam::Pose3 transformedPose2 = calCompCamPose(lf2, *it_set);
//gtsam::Point2 obs1 = convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][kp_ind1]);
//gtsam::Point2 obs2  = convertPoint2_CV2GTSAM(lf2->image_kps_undist[*it_set][kp_ind2]);
//
////check the reprojection error again to filter any outliers
//double error1 = computeReprojectionError(transformedPose1, camID2, landmark, obs1);
//double error2 = computeReprojectionError(transformedPose2, *it_set, landmark, obs2);
//double tri_angle_deg=0;
////compute angle
//bool angleCheck = checkTriangulationAngle(const_cast<gtsam::Point3 &>(transformedPose1.translation()),
//                                          const_cast<gtsam::Point3 &>(transformedPose2.translation()),
//                                          landmark , tri_angle_deg);
//
//if(!angleCheck || error1 >4 || error2 >4){
//accept = false;
//break;
//}
//}
//
//if(accept){
//it_set = acceptedPrevCamIDs.begin();
//for( ; it_set != acceptedPrevCamIDs.end() ; ++it_set){
//if(angles.at<double>(*it_set, camID2) == 0){
//accept = false;
//break;
//}
//}
//if(accept)
//{
//acceptedCurCamIDs.push_back(camID2);
//acceptedCurMeas.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][im2->matchIndex[camID2]]));
//}
//}
//}
//}
//
//VLOG(2)<<"LID : "<<lids->at(i)<<", PT: "<<landmark.x()<<","<<landmark.y()<<","<<landmark.z()<<" , max angle : "<<maxAngle<<"Num meas: "<<acceptedCurCamIDs.size()+acceptedPrevCamIDs.size()<<endl;
//if ( maxAngle > 0 ) {
//num_lms_filtered++;
////lids_filtered.push_back(lids->at(i));
////landmarks.push_back(landmark);
////previous_KFs.push_back(lf1);
////prevKFInds.push_back(prevKFIdx);
//// new_landmark_flags.push_back(false);
////current_measurements.push_back(acceptedCurMeas);
////cur_compcam_ids.push_back(acceptedCurCamIDs);
///// record the previous measurements and previous KF and comp cam ID
////previous_measurements.push_back(acceptedPrevMeas);
////previous_compcam_ids.push_back(acceptedPrevCamIDs);
//
//insertLandmarkInGraph(l , prevKFIdx, curKFIdx, false, acceptedPrevCamIDs,
//acceptedPrevMeas, acceptedCurCamIDs,  acceptedCurMeas);
//
//}
//
//}

