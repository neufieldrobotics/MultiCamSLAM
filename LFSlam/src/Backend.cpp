//
// Created by pushyami Kaveti on 8/21/21.
//

#include <LFSlam/FrontEnd.h>
#include "LFSlam/Backend.h"
//// This is the class for writing optimization functions
//// todo: Create a ISAM instance, setup the factor graph and initialise the pose prior.
//// todo: at each time step update isam and get the optmized pose and land mark estimates
//// todo : backend methods should be called from lf_slam_app after trackLF(). trackLF() is a
//// todo : front-end method which generates the initial pose and landmark estimates.

Backend::Backend(string strSettingsFile, CamArrayConfig &camconfig, FrontEnd *fe)
        : backend_config_file(strSettingsFile) {

    cv::FileStorage fSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if (!fSettings.isOpened()) {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }

    camID = (int) fSettings["CamID"];

//    K[0] = gtsam::Cal3_S2::shared_ptr (new gtsam::Cal3_S2(50.0, 50.0, 0, 50.0, 50.0));

    camTransformation = cv::Mat(4, 4, CV_64F);
    for (int i = 0; i < 3; ++i) {
        camTransformation.at<double>(i, 3) = camconfig.t_mats_[camID].at<double>(i, 0);
        for (int j = 0; j < 3; ++j)
            camTransformation.at<double>(i, j) = camconfig.R_mats_[camID].at<double>(i, j);
    }

    camTransformation.at<double>(3,0) = 0;
    camTransformation.at<double>(3,1) = 0;
    camTransformation.at<double>(3,2) = 0;
    camTransformation.at<double>(3,3) = 1;

    camTransformation = camTransformation.inv();
    camArrayConfig = &camconfig;
    for(int i =0 ; i < camconfig.num_cams_ ; i++){

        K.push_back(gtsam::Cal3_S2::shared_ptr(new gtsam::Cal3_S2(camconfig.K_mats_[i].at<double>(0, 0),
                                                                 camconfig.K_mats_[i].at<double>(1, 1), 0,
                                                                 camconfig.K_mats_[i].at<double>(0, 2),
                                                                 camconfig.K_mats_[i].at<double>(1, 2))));

        Mat R = camconfig.Kalibr_R_mats_[i];/// store the pose of the cam chain
        Mat t = camconfig.Kalibr_t_mats_[i];
        Mat kalibrPose = Mat::eye(4,4, CV_64F);
        kalibrPose.rowRange(0,3).colRange(0,3) = R.t();
        kalibrPose.rowRange(0,3).colRange(3,4) = -1* R.t()*t;
        //gtsam::Matrix eigenRt_kalib;
       // cv2eigen(kalibrPose, eigenRt_kalib);
        Rt_mats_kalib_.push_back(kalibrPose.clone());

        Mat R2 = camconfig.R_mats_[i];/// store the pose of the cam chain
        Mat t2 = camconfig.t_mats_[i];
        Mat camPose = Mat::eye(4,4, CV_64F);
        camPose.rowRange(0,3).colRange(0,3) = R2.t();
        camPose.rowRange(0,3).colRange(3,4) = -1* R2.t()*t2;
        //gtsam::Matrix eigenRt;
        //cv2eigen(camPose, eigenRt);
        Rt_mats_.push_back(camPose.clone());
        VLOG(2)<<"RTMats cam: "<<i<<" : "<<camPose<<endl;

    }

    measurementNoise = gtsam::noiseModel::Isotropic::Sigma(2, (double)fSettings["MeasurementNoiseSigma"]);

    huberModel = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(
            sqrt(5.991)), measurementNoise);

    optimizationMethod = (int)fSettings["Optimization"];
    reinitialized_ = false;
    if(optimizationMethod == 0){
        //parameters.optimizationParams = ISAM2DoglegParams(1.0, 1e-5, DoglegOptimizerImpl::SEARCH_EACH_ITERATION);
        parameters.relinearizeThreshold = (double)fSettings["ISAMRelinearizeThreshold"];
        parameters.relinearizeSkip = (int)fSettings["ISAMRelinearizeSkip"];
        //parameters.factorization = gtsam::ISAM2Params::QR;
        isam = gtsam::ISAM2(parameters);
    }
    else if(optimizationMethod == 1){
        params.orderingType = gtsam::Ordering::METIS;
    }
    else if(optimizationMethod == 2){
        double lag = 3.0;
        params.lambdaInitial = 1e-2;
        params.maxIterations = 15;
        fixedLagOptimizer = gtsam::BatchFixedLagSmoother(lag, params );


    }
    else{
        VLOG(1)<<"WRONG OPTIMIZATION METHOD OPTION. Must be 0-ISAM2 1-LevenbergMarquardt 2-fixedlag levenberg";
        exit(0);
    }
//    optimizer = new gtsam::LevenbergMarquardtOptimizer(graph, initialEstimate, params);

//    imageSkip = (int)fSettings["ImageSkip"];
    windowSize = (int)fSettings["WindowBad"];
    windowCounter = 0;
    angleThresh = (double)fSettings["AngleThresh"];
    frontEnd = fe;
    backendType = static_cast<BACKEND_TYPE>((int)fSettings["BackEndType"]);

}
Backend::~Backend()= default;

void Backend::displayMatches(LightFieldFrame* prev_frame, LightFieldFrame* current_frame){

    vector<IntraMatch> *intramatches1 = &prev_frame->intraMatches;
    vector<IntraMatch> *intramatches2 = &current_frame->intraMatches;
    vector<int> *lids1 = &prev_frame->lIds;
    vector<int> *lids2 = &current_frame->lIds;
    vector<int> lids_filtered1, lids_filtered2;
//    vector<Landmark> landmarks;
    vector<cv::KeyPoint> keyPoints1, keyPoints2;

    for(int i=0; i<intramatches1->size(); ++i){
        if(lids1->at(i)!=-1 && intramatches1->at(i).matchIndex[camID]!=-1 /* this is to filter only 0th camera feature for now*/){
            lids_filtered1.push_back(lids1->at(i));
//            cout<<"lid at "<<i<<": "<<lids->at(i)<<endl;
            keyPoints1.push_back(prev_frame->image_kps[camID][intramatches1->at(i).matchIndex[camID]]);
           // keyPoints1.push_back(KeyPoint(Point2f(prev_frame->sparse_disparity[i].u, prev_frame->sparse_disparity[i].v), 3.0));
        }
    }

    for(int i=0; i<intramatches2->size(); ++i){
        if(lids2->at(i)!=-1 && intramatches2->at(i).matchIndex[camID]!=-1 /* this is to filter only 0th camera feature for now*/){
            lids_filtered2.push_back(lids2->at(i));
//            cout<<"lid at "<<i<<": "<<lids->at(i)<<endl;
            keyPoints2.push_back(current_frame->image_kps[camID][intramatches2->at(i).matchIndex[camID]]);
           // keyPoints2.push_back(KeyPoint(Point2f(current_frame->sparse_disparity[i].u, current_frame->sparse_disparity[i].v), 3.0));
        }
    }

    vector<cv::DMatch> matches;
    //assert(lids_filtered1.size() == lids_filtered2.size());
    std::cout<<"backend draw matches size: "<<lids_filtered1.size()<<std::endl;
    for(int i=0; i<lids_filtered1.size(); ++i){
        for(int j=0; j<lids_filtered2.size(); ++j){
            if(lids_filtered1.at(i) == lids_filtered2.at(j)){
                cv::DMatch match;
                match.trainIdx = j;
                match.queryIdx = i;
                matches.push_back(match);
                break;
            }
        }
    }

    cv::Mat outimg;
    cv::drawMatches(current_frame->imgs[camID], keyPoints1, prev_frame->imgs[camID], keyPoints2, matches, outimg);
    cv::imshow("backend_matches", outimg);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

bool Backend::checkTriangulationAngle(gtsam::Point3 &pose1, gtsam::Point3 &pose2, gtsam::Point3 &landmark,
                                      double &angle_deg) const{
    ///Get the Ray1 between landmark and the previous pose
    gtsam::Vector3 ray1 = landmark - pose1;
    double norm1 = ray1.norm();

    ///Get the Ray2 between landmark and the current pose
    gtsam::Vector3 ray2 = landmark - pose2;
    double norm2 = ray2.norm();

    /// Compute the angle between the two rays
    double cosTheta = ray1.dot(ray2)/(norm1*norm2);
    angle_deg = acos(cosTheta)*(180.0/3.141592653589793238463);

    if(angle_deg > angleThresh){
        return true;
    }

    return false;
}

double Backend::computeReprojectionError(gtsam::Pose3 &pose1, int camID1, gtsam::Point3 &landmark, gtsam::Point2 &obs) {
    //Get the extrisic parameters of the states
    gtsam::Matrix R1 = pose1.rotation().transpose();
    gtsam::Matrix t1 = -1 * R1 * pose1.translation();

    //COnvert the world point into the pose reference frames and apply K matrix
    gtsam::Vector pt_c1 = R1 * landmark + t1;
    double invZ1 = 1.0/pt_c1(2);
    double u_1 = camArrayConfig->K_mats_[camID1].at<double>(0,0) * pt_c1(0)*invZ1 + camArrayConfig->K_mats_[camID1].at<double>(0,2);
    double v_1 = camArrayConfig->K_mats_[camID1].at<double>(1,1) * pt_c1(1)*invZ1 + camArrayConfig->K_mats_[camID1].at<double>(1,2);
    double squareError = (u_1-obs.x())*(u_1-obs.x())+(v_1-obs.y())*(v_1-obs.y());
    return squareError;

}

gtsam::Pose3 Backend::calCompCamPose(gtsam::Pose3 &pose) {
    VLOG(3)<<"Cam Transformation: "<<camTransformation<<endl;
    return convertPose3_CV2GTSAM(camTransformation) * pose;
    return pose;
}

gtsam::Pose3 Backend::calCompCamPose(LightFieldFrame* lf, int cID){
    Mat finalPoseCV = lf->pose * Rt_mats_[cID];
    return  convertPose3_CV2GTSAM(finalPoseCV);
}

/// insert pose prior for first initialized frame to factor graph and add initial pose estimate
void Backend::addPosePrior(int lid, GlobalMap *map){
    Landmark* l = map->getLandmark(lid);
    LightFieldFrame* prevLF= l->KFs[0];
    gtsam::Pose3 previous_pose = convertPose3_CV2GTSAM(prevLF->pose);
    previous_pose = calCompCamPose(previous_pose);

    // Add a prior on pose x0
    gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas
            ((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.1),gtsam::Vector3::Constant(0.1)).finished());
    graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', prevLF->frameId), previous_pose, poseNoise));
    initialEstimate.insert(gtsam::Symbol('x', prevLF->frameId), previous_pose);
}

/// insert landmark prior for first landmark to factor graph
void Backend::addLandmarkPrior(int lid, gtsam::Point3 &landmark){

    gtsam::noiseModel::Isotropic::shared_ptr pointNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
    //cout<<lid<< landmark<<endl;
    graph.push_back(gtsam::PriorFactor<gtsam::Point3>(gtsam::Symbol('l', lid), landmark, pointNoise)); // add directly to graph
}

/// function for monocular backend
/// finds the previous keyframe in which the landmark was last observed in the same component camera. returns the
/// corresponding measurement, pose id and pose.
bool Backend::obtainPrevMeasurement(Landmark* landmark, gtsam::Point2 &prev_measurement, int &prev_pose_id, gtsam::Pose3 &prev_pose, bool init) {

    vector<LightFieldFrame *> KFs = landmark->KFs;
    vector<int> *feat_inds = &landmark->featInds;

    auto kf_it = KFs.rbegin();
    auto feat_it = feat_inds->rbegin();

    kf_it++;
    feat_it++;

    auto prev_KF_it = frontEnd->lfFrames.rbegin()++;
    int init_prev_pose_id = (*prev_KF_it)->frameId;

    for(; kf_it!=KFs.rend() && feat_it!=feat_inds->rend(); kf_it++, feat_it++){
        vector<IntraMatch> *intramatches = &((*kf_it)->intraMatches);
        int kp_ind = intramatches->at(*feat_it).matchIndex[camID];
        int pose_id = (*kf_it)->frameId;
        if(init){
            if(kp_ind!=-1 && pose_id == init_prev_pose_id){
                prev_measurement = convertPoint2_CV2GTSAM((*kf_it)->image_kps[camID][kp_ind]);
                prev_pose_id = (*kf_it)->frameId;
                prev_pose = convertPose3_CV2GTSAM((*kf_it)->pose);
                prev_pose = calCompCamPose(prev_pose);
                return true;
            }
        }
        else if(kp_ind!=-1 && (currentEstimate.exists(gtsam::Symbol('x', pose_id)) || initialEstimate.exists(gtsam::Symbol('x', pose_id)))){
            prev_measurement = convertPoint2_CV2GTSAM((*kf_it)->image_kps[camID][kp_ind]);
            prev_pose_id = (*kf_it)->frameId;
            prev_pose = convertPose3_CV2GTSAM((*kf_it)->pose);
            prev_pose = calCompCamPose(prev_pose);
            return true;
        }
    }
    return false;
}

/// function for monocular backend
/// filters landmarks in current frame based on observability in component camera (current and any of the previous frames) and triangulation angle
/// returns vectors of lids, landmarks, current measurements, previous measurements and previous pose ids
void Backend::filterLandmarks(LightFieldFrame* currentFrame, GlobalMap *map, vector<int>
        &lids_filtered, vector<gtsam::Point3> &landmarks, vector<gtsam::Point2> &current_measurements,
        vector<gtsam::Point2> &previous_measurements, vector<int> &previous_pose_ids){

    int landmarks_tot=0, landmarks_filtered=0;

    vector<IntraMatch> *intramatches = &currentFrame->intraMatches;
    vector<int> *lids = &currentFrame->lIds;

    gtsam::Pose3 current_pose = convertPose3_CV2GTSAM(currentFrame->pose);
    current_pose = calCompCamPose(current_pose);

    bool is_first_frame = currentEstimate.empty() && initialEstimate.empty();

    // browse through all instramatches
    for(int i=0; i<intramatches->size(); ++i){
        if(lids->at(i)!=-1 && intramatches->at(i).matchIndex[camID]!=-1){ // landmarks seen in current frame in the component camera

            landmarks_tot++;
            Landmark* l = map->getLandmark(lids->at(i));
            gtsam::Point3 landmark = convertPoint3_CV2GTSAM(l->pt3D);

            if(!currentEstimate.exists(gtsam::Symbol('l', lids->at(i))) and
               !initialEstimate.exists(gtsam::Symbol('l', lids->at(i)))){ // Newly triangulated landmark

                gtsam::Point2 previous_measurement;
                int previous_pose_id;
                gtsam::Pose3 previous_pose;
                bool add_landmark = obtainPrevMeasurement(l, previous_measurement, previous_pose_id, previous_pose, is_first_frame);

                //check if landmark has been observed in a previous kf on component camera
                if (!add_landmark)
                    continue;

                //check triangulation angle
                double tri_angle_deg=0;
                if(checkTriangulationAngle(const_cast<gtsam::Point3 &>(previous_pose.translation()),
                                           const_cast<gtsam::Point3 &>(current_pose.translation()), landmark, tri_angle_deg)) {
                    landmarks_filtered++;
                    lids_filtered.push_back(lids->at(i));
                    landmarks.push_back(landmark);
                    current_measurements.push_back(convertPoint2_CV2GTSAM(currentFrame->image_kps[camID][intramatches->at(i).matchIndex[camID]]));
                    previous_measurements.push_back(previous_measurement);
                    previous_pose_ids.push_back(previous_pose_id);
                }
            }

            else{ // Tracked landmark
                landmarks_filtered++;
                lids_filtered.push_back(lids->at(i));
                landmarks.push_back(landmark);
                current_measurements.push_back(convertPoint2_CV2GTSAM(currentFrame->image_kps[camID][intramatches->at(i).matchIndex[camID]]));
                previous_measurements.emplace_back();
                previous_pose_ids.push_back(-1);
            }
        }
    }
    VLOG(2)<<"Landmarks before and after filtering: "<<landmarks_tot<<" "<<landmarks_filtered<<endl;
}


/// function for monocular backend
/// filters landmarks in current frame based on observability in component camera (current and any of the previous frames) and triangulation angle
/// returns vectors of lids, landmarks, current measurements, previous measurements and previous pose ids
void Backend::filterLandmarksStringent(LightFieldFrame* currentFrame, GlobalMap *map, vector<int>
&lids_filtered, vector<gtsam::Point3> &landmarks, vector<gtsam::Point2> &current_measurements,
                              vector<gtsam::Point2> &previous_measurements, vector<int> &previous_pose_ids){

    int landmarks_tot=0, landmarks_filtered=0;

    vector<IntraMatch> *intramatches = &currentFrame->intraMatches;
    vector<int> *lids = &currentFrame->lIds;

    gtsam::Pose3 current_pose = convertPose3_CV2GTSAM(currentFrame->pose);
    current_pose = calCompCamPose(current_pose);

    bool is_first_frame = currentEstimate.empty() && initialEstimate.empty();
    double tri_ang;
    for(int i=0; i<intramatches->size(); ++i){
        if(lids->at(i)!=-1 && intramatches->at(i).matchIndex[camID]!=-1){ // landmarks seen in current frame in the component camera

            landmarks_tot++;
            Landmark* l = map->getLandmark(lids->at(i));
            gtsam::Point3 landmark = convertPoint3_CV2GTSAM(l->pt3D);

            gtsam::Point2 previous_measurement;
            int previous_pose_id;
            gtsam::Pose3 previous_pose;
            bool add_landmark = obtainPrevMeasurement(l, previous_measurement, previous_pose_id, previous_pose, is_first_frame);

            if(!currentEstimate.exists(gtsam::Symbol('l', lids->at(i))) and
               !initialEstimate.exists(gtsam::Symbol('l', lids->at(i)))){ // Newly triangulated landmark

                //check if landmark has been observed in a previous kf on component camera
                if (!add_landmark)
                    continue;

                //check triangulation angle
                if(checkTriangulationAngle(const_cast<gtsam::Point3 &>(previous_pose.translation()),
                                           const_cast<gtsam::Point3 &>(current_pose.translation()), landmark, tri_ang)) {
                    landmarks_filtered++;
                    lids_filtered.push_back(lids->at(i));
                    landmarks.push_back(landmark);
                    current_measurements.push_back(convertPoint2_CV2GTSAM(currentFrame->image_kps[camID][intramatches->at(i).matchIndex[camID]]));
                    previous_measurements.push_back(previous_measurement);
                    previous_pose_ids.push_back(previous_pose_id);
                }
            }

            else{ // Tracked landmark
                if(checkTriangulationAngle(const_cast<gtsam::Point3 &>(previous_pose.translation()),
                                           const_cast<gtsam::Point3 &>(current_pose.translation()), landmark, tri_ang)) {
                    landmarks_filtered++;
                    lids_filtered.push_back(lids->at(i));
                    landmarks.push_back(landmark);
                    current_measurements.push_back(convertPoint2_CV2GTSAM(
                            currentFrame->image_kps[camID][intramatches->at(i).matchIndex[camID]]));
                    previous_measurements.emplace_back();
                    previous_pose_ids.push_back(-1);
                }
            }
        }
    }
    VLOG(2)<<"Landmarks before and after filtering: "<<landmarks_tot<<" "<<landmarks_filtered<<endl;
}

bool Backend::addKeyFrameMultiSensorOffset() {

    LightFieldFrame *currentFrame = frontEnd->currentFrame;
    GlobalMap *map = frontEnd->map;
    bool ret = true;

    VLOG(2) << "Window counter: " << windowCounter << endl;

    /// Variable declaration
    std::map<int, set<int> > insertedRigid;
    int num_lms_filtered=0;
    vector<int> lids_filtered;
    vector<gtsam::Point3> landmarks;
    vector<gtsam::Point2> current_measurements;
    vector<gtsam::Point2> previous_measurements;
    vector<int>  previous_compcam_ids, cur_compcam_ids;
    vector<LightFieldFrame*> previous_KFs;
    vector<bool> new_landmark_flags;

    vector<IntraMatch> *intramatches = &currentFrame->intraMatches;
    vector<int> *lids = &currentFrame->lIds;
    int current_pose_id = currentFrame->frameId;

    // browse through all instramatches
    for (int i = 0; i < intramatches->size(); ++i) {
        // if this intra match is a landmark i.e it is tracked in the current frame and is a triangulated inlier
        if (lids->at(i) != -1 ) {

            //record the landmark
            Landmark* l = map->getLandmark(lids->at(i));
            //cout<<"LANDMARK :"<<l->pt3D<<endl;
            gtsam::Point3 landmark = convertPoint3_CV2GTSAM(l->pt3D);

            if(!currentEstimate.exists(gtsam::Symbol('l', lids->at(i))) and
               !initialEstimate.exists(gtsam::Symbol('l', lids->at(i))))
            {
                /// Get the previous LF frame of the landmark and the observtaions in the component cameras
                int numKFs = l->KFs.size();
                LightFieldFrame* lf1 = l->KFs[numKFs-2]; // previous KF
                LightFieldFrame* lf2 = currentFrame;  // same as l->KFs[numKFs-1]; // last KF
                IntraMatch* im1 = &lf1->intraMatches.at(l->featInds[numKFs-2]);
                IntraMatch* im2 = &lf2->intraMatches.at(l->featInds[numKFs-1]);

                Mat angles = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
                Mat tri_errors = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
                VLOG(3)<<"LID : "<<lids->at(i)<<", PT: "<<landmark.x()<<","<<landmark.y()<<","<<landmark.z()<<endl;
                VLOG(3)<<"Cam1 cam2 angle error1 error2"<<endl;
                ///check all combinations of rays and get the triangulation angles and reprojection errors
                for (int camID1 = 0; camID1 < lf1->num_cams_ ; camID1++){
                    int kp_ind1 = im1->matchIndex[camID1];
                    if(kp_ind1 == -1)
                        continue;
                    for (int camID2 = 0; camID2 < lf2->num_cams_ ; camID2++){
                        int kp_ind2 = im2->matchIndex[camID2];
                        if(kp_ind2 == -1)
                            continue;

                        gtsam::Pose3 transformedPose1 = calCompCamPose(lf1, camID1); // take input of the LF frame pose and comp cam id to compute the comp cam pose
                        gtsam::Pose3 transformedPose2 = calCompCamPose(lf2, camID2);
                        gtsam::Point2 obs1 = convertPoint2_CV2GTSAM(lf1->image_kps_undist[camID1][kp_ind1]);
                        gtsam::Point2 obs2  = convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][kp_ind2]);

                        //check the reprojection error again to filter any outliers
                        double error1 = computeReprojectionError(transformedPose1, camID1, landmark, obs1);
                        double error2 = computeReprojectionError(transformedPose2, camID2, landmark, obs2);
                        double tri_angle_deg=0;
                        //compute angle
                        bool angleCheck = checkTriangulationAngle(const_cast<gtsam::Point3 &>(transformedPose1.translation()),
                                                                  const_cast<gtsam::Point3 &>(transformedPose2.translation()),
                                                                  landmark , tri_angle_deg);
                        VLOG(3)<<camID1<<"  "<<camID2<<"  "<<tri_angle_deg<<"," <<error1<<", "<<error2<<endl;
                        if(angleCheck and error1 <=4 and error2 <=4 ){

                            angles.at<double>(camID1,camID2) = tri_angle_deg;
                            tri_errors.at<double>(camID1,camID2) = error1 + error2;

                        }
                    }
                }

                // compare all the angles and choose the  largest one for insertion
                //first check the diagonal angles i.e intersecting component cameras between two LF frame
                double maxAngleSameCam = 0;
                double maxAngle = 0;
                int curKFCamID = -1, prevKFCamID = -1, curKFSameCamID = -1, prevKFSameCamID = -1;
                for(int camID1 = 0; camID1 < camArrayConfig->num_cams_ ; camID1++){

                    for(int camID2 = 0; camID2 < camArrayConfig->num_cams_ ; camID2++){

                        if(maxAngle < angles.at<double>(camID1,camID2)){
                            maxAngle = angles.at<double>(camID1,camID2);
                            prevKFCamID = camID1;
                            curKFCamID = camID2;
                        }

                        if(camID1 == camID2){
                            if( maxAngleSameCam < angles.at<double>(camID1,camID2)){
                                maxAngleSameCam = angles.at<double>(camID1,camID2);
                                prevKFSameCamID = camID1;
                                curKFSameCamID = camID2;
                            }

                        }
                    }

                }
                VLOG(2)<<"LID : "<<lids->at(i)<<" , max angle : "<<maxAngle<<" , max angle same cam: "<<maxAngleSameCam<<endl;

                if (maxAngleSameCam > 0 || maxAngle > 0 ) {
                    num_lms_filtered++;
                    lids_filtered.push_back(lids->at(i));
                    landmarks.push_back(landmark);
                    previous_KFs.push_back(lf1);
                    new_landmark_flags.push_back(true);
                    // choose the landmark, pose id and comp cam id, observations
                    if(maxAngleSameCam > 0){ /// For now we are only choosing the same index cameras
                        current_measurements.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[curKFSameCamID][im2->matchIndex[curKFSameCamID]]));
                        cur_compcam_ids.push_back(curKFSameCamID);
                        /// record the previous measurements and previous KF and comp cam ID
                        previous_measurements.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[prevKFSameCamID][im1->matchIndex[prevKFSameCamID]]));
                        previous_compcam_ids.push_back(prevKFSameCamID);

                    }
                    else{

                        current_measurements.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[curKFCamID][im2->matchIndex[curKFCamID]]));
                        cur_compcam_ids.push_back(curKFCamID);
                        /// record the previous measurements and previous KF and comp cam ID
                        previous_measurements.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[prevKFCamID][im1->matchIndex[prevKFCamID]]));
                        previous_compcam_ids.push_back(prevKFCamID);
                    }
                }

            }
            else{

                /// this is an existing landmark in the graph.
                /// Grab the last observation
                vector<int> facs = lmFactorRecord[lids->at(i)];
                assert(facs.size() != 0);

                /// Get the prev factor's frameID and componenet camera ID
                int prevFacStateID = facs.back();
                int prevFrameID = 0;
                string s = to_string(prevFacStateID);
                int CamID1 = stoi(&s.back());
                s.pop_back();
                if(!s.empty()){
                    prevFrameID = stoi(s);
                }
                VLOG(3)<<"prevFrameID: "<<prevFrameID<<", CamID1: "<<CamID1<<endl;
                LightFieldFrame* lf1;
                IntraMatch* im1;
                vector<LightFieldFrame *> KFs = l->KFs;
                vector<int> *feat_inds = &l->featInds;
                /// iterate over the KF list of this landmark to get
                /// the last KF with the frme ID equal to prevFarmeID
                auto kf_it = KFs.rbegin();
                auto feat_it = feat_inds->rbegin();
                kf_it++;
                feat_it++;
                for(; kf_it!=KFs.rend() && feat_it!=feat_inds->rend(); kf_it++, feat_it++){
                    if((*kf_it)->frameId == prevFrameID){
                        lf1 = *kf_it;
                        im1 = &lf1->intraMatches.at(*feat_it);
                    }
                }



                LightFieldFrame* lf2 = currentFrame;  // same as l->KFs[numKFs-1]; // last KF
                IntraMatch* im2 = &lf2->intraMatches.at(l->featInds.back());

                /// check and make sure that the intramatch has an observation in CamID1 obtained form factor
                int kp_ind1 = im1->matchIndex[CamID1];
                assert(kp_ind1 != -1);
                gtsam::Pose3 transformedPose1 = calCompCamPose(lf1, CamID1);
                gtsam::Point2 obs1 = convertPoint2_CV2GTSAM(lf1->image_kps_undist[CamID1][kp_ind1]);
                double error1 = computeReprojectionError(transformedPose1, CamID1, landmark, obs1);

                Mat angles = Mat::zeros(1,lf2->num_cams_, CV_64FC1);
                Mat tri_errors = Mat::zeros(1,lf2->num_cams_, CV_64FC1);

                ///In the current frame see if we have the matching camera
                for (int camID2 = 0; camID2 < lf2->num_cams_ ; camID2++){
                    int kp_ind2 = im2->matchIndex[camID2];
                    if(kp_ind2 == -1)
                        continue;

                    // take input of the LF frame pose and comp cam id to compute the comp cam pose
                    gtsam::Pose3 transformedPose2 = calCompCamPose(lf2, camID2);
                    gtsam::Point2 obs2  = convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][kp_ind2]);

                    //check the reprojection error again to filter any outliers
                    double error2 = computeReprojectionError(transformedPose2, camID2, landmark, obs2);
                    double tri_angle_deg=0;
                    //compute angle
                    bool angleCheck = checkTriangulationAngle(const_cast<gtsam::Point3 &>(transformedPose1.translation()),
                                                              const_cast<gtsam::Point3 &>(transformedPose2.translation()),
                                                              landmark , tri_angle_deg);
                    VLOG(3)<<CamID1<<"  "<<camID2<<"  "<<tri_angle_deg<<"," <<error1<<", "<<error2<<endl;
                    if(angleCheck and error1 <=4 and error2 <=4 ){

                        angles.at<double>(0,camID2) = tri_angle_deg;
                        tri_errors.at<double>(0,camID2) = error1 + error2;

                    }
                }



                // compare all the angles and choose the  largest one for insertion
                //first check the diagonal angles i.e intersecting component cameras between two LF frame
                double maxAngleSameCam = angles.at<double>(0,CamID1);
                double maxAngle = 0;
                int curKFCamID = -1;
                for(int camID2 = 0; camID2 < camArrayConfig->num_cams_ ; camID2++){
                    if(maxAngle < angles.at<double>(0,camID2)){
                        maxAngle = angles.at<double>(0,camID2);
                        curKFCamID = camID2;
                    }

                }

                VLOG(2)<<"LID : "<<lids->at(i)<<" , max angle : "<<maxAngle<<" , max angle same cam: "<<maxAngleSameCam<<endl;
                if (maxAngleSameCam > 0 || maxAngle > 0 ) {
                    num_lms_filtered++;
                    lids_filtered.push_back(lids->at(i));
                    landmarks.push_back(landmark);
                    previous_KFs.push_back(lf1);
                    new_landmark_flags.push_back(false);
                    // choose the landmark, pose id and comp cam id, observations
                    if(maxAngleSameCam > 0){ /// For now we are only choosing the same index cameras
                        current_measurements.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[CamID1][im2->matchIndex[CamID1]]));
                        cur_compcam_ids.push_back(CamID1);
                        /// record the previous measurements and previous KF and comp cam ID
                        previous_measurements.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[CamID1][im1->matchIndex[CamID1]]));
                        previous_compcam_ids.push_back(CamID1);

                    }
                    else{

                        current_measurements.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[curKFCamID][im2->matchIndex[curKFCamID]]));
                        cur_compcam_ids.push_back(curKFCamID);
                        /// record the previous measurements and previous KF and comp cam ID
                        previous_measurements.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[CamID1][im1->matchIndex[CamID1]]));
                        previous_compcam_ids.push_back(CamID1);
                    }
                }


            }


        }
    }


    VLOG(2)<<"Backend Landmarks size: "<<landmarks.size()<<endl;
    if(landmarks.size() < 15){
        return false;
    }

    /// check if this is the first frame we are inserting
    /// if it is, we have to add prior to the pose and a landmark
    if(currentEstimate.empty() && initialEstimate.empty()) {

        VLOG(2)<<"Went inside GRAPH==0"<<endl;

        Landmark* l = map->getLandmark(lids_filtered.at(0));
        int numKFs = l->KFs.size();
        LightFieldFrame* prevLF=  l->KFs[numKFs-2];

        assert(previous_KFs.at(0)->frameId == prevLF->frameId);
        // Concatenate both pose id and the cam id
        // Convert the concatenated string
        // to integer

        /// Add a prior on pose x0
        gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas
                ((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.001),gtsam::Vector3::Constant(0.001)).finished());
                //((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.3),gtsam::Vector3::Constant(0.1)).finished());

        graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', prevLF->frameId), convertPose3_CV2GTSAM(prevLF->pose), poseNoise));
        VLOG(2)<<"Inserted prior for the state x"<<prevLF->frameId<<endl;
        /// LANDMARK PRIOR
        addLandmarkPrior(lids_filtered.at(0), landmarks[0]);
    }


    /// create graph edges
    /// insert initial values for the variables
    for (int i = 0; i < landmarks.size(); ++i) {
        ///Previous state ID
        Landmark* l = map->getLandmark(lids_filtered.at(i));
        LightFieldFrame* prev_LF = previous_KFs.at(i);

        string s1 = to_string(current_pose_id);
        string s2 = to_string(cur_compcam_ids[i]);
        // Concatenate both pose id and the cam id
        string s = s1 + s2;
        // Convert the concatenated string
        // to integer
        int curStateIDComp = stoi(s);


        s1 = to_string( prev_LF->frameId);
        s2 = to_string(previous_compcam_ids[i]);
        // Concatenate both pose id and the cam id
        s = s1 + s2;
        // Convert the concatenated string
        // to integer
        int prevStateIDComp = stoi(s);


        ///  Current State id
         int curStateID = current_pose_id;

        int prevStateID = prev_LF->frameId;

        if(!(currentEstimate.exists(gtsam::Symbol('x', curStateID)) || initialEstimate.exists(gtsam::Symbol('x', curStateID)))){
            /// Initial estimate for the current pose.
            initialEstimate.insert(gtsam::Symbol('x', curStateID), convertPose3_CV2GTSAM(currentFrame->pose));
            /// state factor record
            xFactorRecord[curStateID] = vector<int>();

            VLOG(2)<<"Inserting Initial Estimate for x"<<curStateID<<endl;
        }

        if(new_landmark_flags[i]){
            /// for each measurement add a projection factor
            if(!(currentEstimate.exists(gtsam::Symbol('x', prevStateID)) || initialEstimate.exists(gtsam::Symbol('x', prevStateID)))){
                // Initial estimate for the current pose.
                initialEstimate.insert(gtsam::Symbol('x', prevStateID), convertPose3_CV2GTSAM(prev_LF->pose));
                /// state factor record
                xFactorRecord[prevStateID] = vector<int>();
                VLOG(2)<<"Inserting Initial Estimate for x"<<prevStateID<<endl;
            }

            // Insert projection factor with previous kf to factor graph
            //VLOG(2)<<"measurement from prev frame: "<<previous_measurements[i].x()<<" "<<previous_measurements[i].y()<<endl;
            graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                    (previous_measurements[i], huberModel, gtsam::Symbol('x', prevStateID),
                                     gtsam::Symbol('l', lids_filtered.at(i)), K[previous_compcam_ids[i]], convertPose3_CV2GTSAM(Rt_mats_[previous_compcam_ids[i]])));
            /// state factor record
            xFactorRecord[prevStateID].push_back(lids_filtered.at(i));
            lmFactorRecord[lids_filtered.at(i)] = vector<int>({prevStateIDComp});

            initialEstimate.insert(gtsam::Symbol('l', lids_filtered.at(i)), landmarks[i]);
            VLOG(3)<<"Inserting Initial Estimate for l"<<lids_filtered.at(i)<<endl;
        }

        // insert projection factor with current frame to factor graph
        graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                (current_measurements[i], huberModel, gtsam::Symbol('x', curStateID),
                                 gtsam::Symbol('l', lids_filtered.at(i)), K[cur_compcam_ids[i]], convertPose3_CV2GTSAM(Rt_mats_[cur_compcam_ids[i]])));
        /// state factor record
        xFactorRecord[curStateID].push_back(lids_filtered.at(i));
        lmFactorRecord[lids_filtered.at(i)].push_back(curStateIDComp);


    }

    VLOG(2)<<"graph size: "<<graph.size()<<endl;
    if (windowCounter%windowSize != 0)
        ret = false;
    windowCounter++;
    VLOG(2)<<"add keyframe optimize flag: "<<ret<<endl;
    return ret;

}

bool Backend::addKeyFrameMultiLatest() {

    LightFieldFrame *currentFrame = frontEnd->currentFrame;
    GlobalMap *map = frontEnd->map;
    bool ret = true;

    VLOG(2) << "Window counter: " << windowCounter << endl;

    /// Variable declaration
    std::map<int, set<int> > insertedRigid;
    int num_lms_filtered=0;
    vector<int> lids_filtered;
    vector<gtsam::Point3> landmarks;
    vector<gtsam::Point2> current_measurements;
    vector<gtsam::Point2> previous_measurements;
    vector<int>  previous_compcam_ids, cur_compcam_ids;
    vector<LightFieldFrame*> previous_KFs;
    vector<bool> new_landmark_flags;

    vector<IntraMatch> *intramatches = &currentFrame->intraMatches;
    vector<int> *lids = &currentFrame->lIds;
    int current_pose_id = currentFrame->frameId;

    // browse through all instramatches
    for (int i = 0; i < intramatches->size(); ++i) {
        // if this intra match is a landmark i.e it is tracked in the current frame and is a triangulated inlier
        if (lids->at(i) != -1 ) {

            //record the landmark
            Landmark* l = map->getLandmark(lids->at(i));
            gtsam::Point3 landmark = convertPoint3_CV2GTSAM(l->pt3D);

            if(!currentEstimate.exists(gtsam::Symbol('l', lids->at(i))) and
               !initialEstimate.exists(gtsam::Symbol('l', lids->at(i))))
            {
                /// make sure it is seen only in two KFs
                //assert(l->KFs.size() == 2);

                /// Get the previous LF frame of the landmark and the observtaions in the component cameras
                int numKFs = l->KFs.size();
                LightFieldFrame* lf1 = l->KFs[numKFs-2]; // previous KF
                LightFieldFrame* lf2 = currentFrame;  // same as l->KFs[numKFs-1]; // last KF
                IntraMatch* im1 = &lf1->intraMatches.at(l->featInds[numKFs-2]);
                IntraMatch* im2 = &lf2->intraMatches.at(l->featInds[numKFs-1]);

                Mat angles = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
                Mat tri_errors = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
                VLOG(3)<<"LID : "<<lids->at(i)<<", PT: "<<landmark.x()<<","<<landmark.y()<<","<<landmark.z()<<endl;
                VLOG(3)<<"Cam1 cam2 angle error1 error2"<<endl;
                ///check all combinations of rays and get the triangulation angles and reprojection errors
                for (int camID1 = 0; camID1 < lf1->num_cams_ ; camID1++){
                    int kp_ind1 = im1->matchIndex[camID1];
                    if(kp_ind1 == -1)
                        continue;
                    for (int camID2 = 0; camID2 < lf2->num_cams_ ; camID2++){
                        int kp_ind2 = im2->matchIndex[camID2];
                        if(kp_ind2 == -1)
                            continue;

                        gtsam::Pose3 transformedPose1 = calCompCamPose(lf1, camID1); // take input of the LF frame pose and comp cam id to compute the comp cam pose
                        gtsam::Pose3 transformedPose2 = calCompCamPose(lf2, camID2);
                        gtsam::Point2 obs1 = convertPoint2_CV2GTSAM(lf1->image_kps_undist[camID1][kp_ind1]);
                        gtsam::Point2 obs2  = convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][kp_ind2]);

                        //check the reprojection error again to filter any outliers
                        double error1 = computeReprojectionError(transformedPose1, camID1, landmark, obs1);
                        double error2 = computeReprojectionError(transformedPose2, camID2, landmark, obs2);
                        double tri_angle_deg=0;
                        //compute angle
                        bool angleCheck = checkTriangulationAngle(const_cast<gtsam::Point3 &>(transformedPose1.translation()),
                                                                  const_cast<gtsam::Point3 &>(transformedPose2.translation()),
                                                                  landmark , tri_angle_deg);
                        VLOG(3)<<camID1<<"  "<<camID2<<"  "<<tri_angle_deg<<"," <<error1<<", "<<error2<<endl;
                        if(angleCheck and error1 <=4 and error2 <=4 ){

                            angles.at<double>(camID1,camID2) = tri_angle_deg;
                            tri_errors.at<double>(camID1,camID2) = error1 + error2;

                        }
                    }
                }

                // compare all the angles and choose the  largest one for insertion
                //first check the diagonal angles i.e intersecting component cameras between two LF frame
                double maxAngleSameCam = 0;
                double maxAngle = 0;
                int curKFCamID = -1, prevKFCamID = -1, curKFSameCamID = -1, prevKFSameCamID = -1;
                for(int camID1 = 0; camID1 < camArrayConfig->num_cams_ ; camID1++){

                    for(int camID2 = 0; camID2 < camArrayConfig->num_cams_ ; camID2++){

                        if(maxAngle < angles.at<double>(camID1,camID2)){
                            maxAngle = angles.at<double>(camID1,camID2);
                            prevKFCamID = camID1;
                            curKFCamID = camID2;
                        }

                        if(camID1 == camID2){
                            if( maxAngleSameCam < angles.at<double>(camID1,camID2)){
                                maxAngleSameCam = angles.at<double>(camID1,camID2);
                                prevKFSameCamID = camID1;
                                curKFSameCamID = camID2;
                            }

                        }
                    }

                }
                VLOG(2)<<"LID : "<<lids->at(i)<<" , max angle : "<<maxAngle<<" , max angle same cam: "<<maxAngleSameCam<<endl;

                if (maxAngleSameCam > 0 || maxAngle > 0 ) {
                    num_lms_filtered++;
                    lids_filtered.push_back(lids->at(i));
                    landmarks.push_back(landmark);
                    previous_KFs.push_back(lf1);
                    new_landmark_flags.push_back(true);
                    // choose the landmark, pose id and comp cam id, observations
                    if(maxAngleSameCam > 0){ /// For now we are only choosing the same index cameras
                        current_measurements.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[curKFSameCamID][im2->matchIndex[curKFSameCamID]]));
                        cur_compcam_ids.push_back(curKFSameCamID);
                        /// record the previous measurements and previous KF and comp cam ID
                        previous_measurements.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[prevKFSameCamID][im1->matchIndex[prevKFSameCamID]]));
                        previous_compcam_ids.push_back(prevKFSameCamID);

                    }
                    else{

                        current_measurements.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[curKFCamID][im2->matchIndex[curKFCamID]]));
                        cur_compcam_ids.push_back(curKFCamID);
                        /// record the previous measurements and previous KF and comp cam ID
                        previous_measurements.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[prevKFCamID][im1->matchIndex[prevKFCamID]]));
                        previous_compcam_ids.push_back(prevKFCamID);
                    }
                }

            }
            else{

                /// this is an existing landmark in the graph.
                /// Grab the last observation
                vector<int> facs = lmFactorRecord[lids->at(i)];
                assert(facs.size() != 0);

                /// Get the prev factor's frameID and componenet camera ID
                int prevFacStateID = facs.back();
                int prevFrameID = 0;
                string s = to_string(prevFacStateID);
                VLOG(2)<<s<<endl;
                int CamID1 = stoi(&s.back());
                s.pop_back();
                if(!s.empty()){
                    prevFrameID = stoi(s);
                }
                VLOG(2)<<"prevFrameID: "<<prevFrameID<<", CamID1: "<<CamID1<<endl;
                LightFieldFrame* lf1;
                IntraMatch* im1;
                vector<LightFieldFrame *> KFs = l->KFs;
                vector<int> *feat_inds = &l->featInds;
                /// iterate over the KF list of this landmark to get
                /// the last KF with the frme ID equal to prevFarmeID
                auto kf_it = KFs.rbegin();
                auto feat_it = feat_inds->rbegin();
                kf_it++;
                feat_it++;
                for(; kf_it!=KFs.rend() && feat_it!=feat_inds->rend(); kf_it++, feat_it++){
                    if((*kf_it)->frameId == prevFrameID){
                        lf1 = *kf_it;
                        im1 = &lf1->intraMatches.at(*feat_it);
                    }
                }



                LightFieldFrame* lf2 = currentFrame;  // same as l->KFs[numKFs-1]; // last KF
                IntraMatch* im2 = &lf2->intraMatches.at(l->featInds.back());

                /// check and make sure that the intramatch has an observation in CamID1 obtained form factor
                int kp_ind1 = im1->matchIndex[CamID1];
                assert(kp_ind1 != -1);
                gtsam::Pose3 transformedPose1 = calCompCamPose(lf1, CamID1);
                gtsam::Point2 obs1 = convertPoint2_CV2GTSAM(lf1->image_kps_undist[CamID1][kp_ind1]);
                double error1 = computeReprojectionError(transformedPose1, CamID1, landmark, obs1);

                Mat angles = Mat::zeros(1,lf2->num_cams_, CV_64FC1);
                Mat tri_errors = Mat::zeros(1,lf2->num_cams_, CV_64FC1);

                ///In the current frame see if we have the matching camera
                for (int camID2 = 0; camID2 < lf2->num_cams_ ; camID2++){
                    int kp_ind2 = im2->matchIndex[camID2];
                    if(kp_ind2 == -1)
                        continue;

                    // take input of the LF frame pose and comp cam id to compute the comp cam pose
                    gtsam::Pose3 transformedPose2 = calCompCamPose(lf2, camID2);
                    gtsam::Point2 obs2  = convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][kp_ind2]);

                    //check the reprojection error again to filter any outliers
                    double error2 = computeReprojectionError(transformedPose2, camID2, landmark, obs2);
                    double tri_angle_deg=0;
                    //compute angle
                    bool angleCheck = checkTriangulationAngle(const_cast<gtsam::Point3 &>(transformedPose1.translation()),
                                                              const_cast<gtsam::Point3 &>(transformedPose2.translation()),
                                                              landmark , tri_angle_deg);
                    VLOG(3)<<CamID1<<"  "<<camID2<<"  "<<tri_angle_deg<<"," <<error1<<", "<<error2<<endl;
                    if(angleCheck and error1 <=4 and error2 <=4 ){

                        angles.at<double>(0,camID2) = tri_angle_deg;
                        tri_errors.at<double>(0,camID2) = error1 + error2;

                    }
                }



                // compare all the angles and choose the  largest one for insertion
                //first check the diagonal angles i.e intersecting component cameras between two LF frame
                double maxAngleSameCam = angles.at<double>(0,CamID1);
                double maxAngle = 0;
                int curKFCamID = -1;
                for(int camID2 = 0; camID2 < camArrayConfig->num_cams_ ; camID2++){
                    if(maxAngle < angles.at<double>(0,camID2)){
                        maxAngle = angles.at<double>(0,camID2);
                        curKFCamID = camID2;
                    }

                }

                VLOG(2)<<"LID : "<<lids->at(i)<<" , max angle : "<<maxAngle<<" , max angle same cam: "<<maxAngleSameCam<<endl;
                if (maxAngleSameCam > 0 || maxAngle > 0 ) {
                    num_lms_filtered++;
                    lids_filtered.push_back(lids->at(i));
                    landmarks.push_back(landmark);
                    previous_KFs.push_back(lf1);
                    new_landmark_flags.push_back(false);
                    // choose the landmark, pose id and comp cam id, observations
                    if(maxAngleSameCam > 0){ /// For now we are only choosing the same index cameras
                        current_measurements.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[CamID1][im2->matchIndex[CamID1]]));
                        cur_compcam_ids.push_back(CamID1);
                        /// record the previous measurements and previous KF and comp cam ID
                        previous_measurements.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[CamID1][im1->matchIndex[CamID1]]));
                        previous_compcam_ids.push_back(CamID1);

                    }
                    else{

                        current_measurements.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[curKFCamID][im2->matchIndex[curKFCamID]]));
                        cur_compcam_ids.push_back(curKFCamID);
                        /// record the previous measurements and previous KF and comp cam ID
                        previous_measurements.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[CamID1][im1->matchIndex[CamID1]]));
                        previous_compcam_ids.push_back(CamID1);
                    }
                }


            }


        }
    }


    VLOG(2)<<"Backend Landmarks size: "<<landmarks.size()<<endl;
    if(landmarks.size() < 15){

        return false;
    }

    /// check if this is the first frame we are inserting
    /// if it is, we have to add prior to the pose and a landmark
    if(currentEstimate.empty() && initialEstimate.empty()) {

        VLOG(2)<<"Went inside GRAPH==0"<<endl;

        Landmark* l = map->getLandmark(lids_filtered.at(0));
        int numKFs = l->KFs.size();
        LightFieldFrame* prevLF=  l->KFs[numKFs-2];
        gtsam::Pose3 previous_pose = calCompCamPose(prevLF , previous_compcam_ids[0]);

        assert(previous_KFs.at(0)->frameId == prevLF->frameId);
        // Concatenate both pose id and the cam id
        // Convert the concatenated string
        // to integer
        int prevStateID = stoi(to_string( prevLF->frameId) + to_string(previous_compcam_ids[0]));

        /// Add a prior on pose x0
        gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas
                ((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.3),gtsam::Vector3::Constant(0.1)).finished());
        graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', prevStateID), previous_pose, poseNoise));
        VLOG(2)<<"Inserted prior for the state x"<<prevStateID<<endl;
        /// LANDMARK PRIOR
        addLandmarkPrior(lids_filtered.at(0), landmarks[0]);
    }


    /// create graph edges
    /// insert initial values for the variables
    for (int i = 0; i < landmarks.size(); ++i) {
        ///Previous state ID
        Landmark* l = map->getLandmark(lids_filtered.at(i));
        LightFieldFrame* prev_LF = previous_KFs.at(i);

        ///  Current State id
        string s1 = to_string(current_pose_id);
        string s2 = to_string(cur_compcam_ids[i]);
        // Concatenate both pose id and the cam id
        string s = s1 + s2;
        // Convert the concatenated string
        // to integer
        int curStateID = stoi(s);


        s1 = to_string( prev_LF->frameId);
        s2 = to_string(previous_compcam_ids[i]);
        // Concatenate both pose id and the cam id
        s = s1 + s2;
        // Convert the concatenated string
        // to integer
        int prevStateID = stoi(s);


        if(!(currentEstimate.exists(gtsam::Symbol('x', curStateID)) || initialEstimate.exists(gtsam::Symbol('x', curStateID)))){
            /// Initial estimate for the current pose.
            gtsam::Pose3 current_pose = calCompCamPose(currentFrame , cur_compcam_ids[i]);
            initialEstimate.insert(gtsam::Symbol('x', curStateID), current_pose);
            /// state factor record
            xFactorRecord[curStateID] = vector<int>();

            VLOG(2)<<"Inserting Initial Estimate for x"<<curStateID<<endl;
            if(backendType == MULTI_RIGID){
                /// if the pose is being inserted for the firt time
                if(insertedRigid.find(current_pose_id) == insertedRigid.end())
                    insertedRigid[current_pose_id] = set<int>({ cur_compcam_ids[i]});
                else
                    insertedRigid[current_pose_id].insert( cur_compcam_ids[i]);
            }


        }

        if(new_landmark_flags[i]){
            /// for each measurement add a projection factor
            if(!(currentEstimate.exists(gtsam::Symbol('x', prevStateID)) || initialEstimate.exists(gtsam::Symbol('x', prevStateID)))){
                // Initial estimate for the current pose.
                gtsam::Pose3 previous_pose = calCompCamPose(prev_LF , previous_compcam_ids[i]);
                initialEstimate.insert(gtsam::Symbol('x', prevStateID), previous_pose);
                /// state factor record
                xFactorRecord[prevStateID] = vector<int>();
                VLOG(2)<<"Inserting Initial Estimate for x"<<prevStateID<<endl;
                if(backendType == MULTI_RIGID){
                    /// if the pose is being inserted for the firt time
                    if(insertedRigid.find( prev_LF->frameId) == insertedRigid.end())
                        insertedRigid[ prev_LF->frameId] = set<int>({ previous_compcam_ids[i]});
                    else
                        insertedRigid[ prev_LF->frameId].insert( previous_compcam_ids[i]);
                }

            }

            // Insert projection factor with previous kf to factor graph
            //VLOG(2)<<"measurement from prev frame: "<<previous_measurements[i].x()<<" "<<previous_measurements[i].y()<<endl;
            graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                    (previous_measurements[i], measurementNoise, gtsam::Symbol('x', prevStateID),
                                     gtsam::Symbol('l', lids_filtered.at(i)), K[previous_compcam_ids[i]]));
            /// state factor record
            xFactorRecord[prevStateID].push_back(lids_filtered.at(i));
            lmFactorRecord[lids_filtered.at(i)] = vector<int>({prevStateID});
            //record the lm factors
            /*if(lmFactorRecord.find(lids_filtered.at(i)) != lmFactorRecord.end()){
                lmFactorRecord[lids_filtered.at(i)].push_back(prevStateID);
            }
            else{
                lmFactorRecord[lids_filtered.at(i)] = vector<int>({prevStateID});
            }*/

            initialEstimate.insert(gtsam::Symbol('l', lids_filtered.at(i)), landmarks[i]);
            VLOG(2)<<"Inserting Initial Estimate for l"<<lids_filtered.at(i)<<endl;
        }

        // insert projection factor with current frame to factor graph
        graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                (current_measurements[i], measurementNoise, gtsam::Symbol('x', curStateID),
                                 gtsam::Symbol('l', lids_filtered.at(i)), K[cur_compcam_ids[i]]));
        /// state factor record
        xFactorRecord[curStateID].push_back(lids_filtered.at(i));
        lmFactorRecord[lids_filtered.at(i)].push_back(curStateID);
        //record the lm factors
        /*if(lmFactorRecord.find(lids_filtered.at(i)) != lmFactorRecord.end()){
            lmFactorRecord[lids_filtered.at(i)].push_back(curStateID);
        }
        else{
            lmFactorRecord[lids_filtered.at(i)] = vector<int>({curStateID});
        }*/

    }

    if(backendType == MULTI_RIGID){
        gtsam::noiseModel::Diagonal::shared_ptr  betweenNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6)<<gtsam::Vector3::Constant(0.001), gtsam::Vector3::Constant(0.001)).finished());
        std::map<int, std::set<int>>::iterator it;
        for(it = insertedRigid.begin() ; it != insertedRigid.end() ; ++it){
            int poseid = it->first;
            set<int> compcamids = it->second;
            set<int>::iterator it_set = compcamids.begin();
            int comp_cam_id_prev = *it_set;
            ++it_set;
            for( ; it_set != compcamids.end() ; ++it_set){

                Mat poseDiff = Mat::eye(4,4, CV_64FC1);
                for(int ii=comp_cam_id_prev+1; ii <= *it_set ; ii++ ){
                    poseDiff =  poseDiff * Rt_mats_kalib_[ii];
                }
                gtsam::Pose3 betweenPose = convertPose3_CV2GTSAM(poseDiff);
                int prev_stateid = stoi(to_string(poseid) + to_string(comp_cam_id_prev));
                int state_id = stoi(to_string(poseid) + to_string(*it_set));
                graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(gtsam::Symbol('x', prev_stateid), gtsam::Symbol('x', state_id),  betweenPose, betweenNoise);
                VLOG(2) <<"Inserted Between factor for PoseId: "<<poseid<<" Between: "<<comp_cam_id_prev<<" and "<<*it_set<<endl;
                comp_cam_id_prev = *it_set;
            }

        }

    }


    VLOG(2)<<"graph size: "<<graph.size()<<endl;
    if (windowCounter%windowSize != 0)
        ret = false;
    VLOG(2)<<"add keyframe optimize flag: "<<ret<<endl;
    windowCounter++;
    return ret;

}

void Backend::getCamIDs_Angles(Landmark* l, int KFID1, int KFID2 , Mat& angles , vector<int>& prevRaysCamIds,
                               vector<int>& curRaysCamIDs, int& maxAnglePrevCamID, int& maxAngleCurCamID){

    LightFieldFrame* lf1 = l->KFs[KFID1]; // previous KF
    LightFieldFrame* lf2 = l->KFs[KFID2];  // same as l->KFs[numKFs-1]; // last KF
    IntraMatch* im1 = &lf1->intraMatches.at(l->featInds[KFID1]);
    IntraMatch* im2 = &lf2->intraMatches.at(l->featInds[KFID2]);
    gtsam::Point3 landmark = convertPoint3_CV2GTSAM(l->pt3D);

    angles = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
    Mat tri_errors = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
    VLOG(3)<<"LID : "<<l->lId<<", PT: "<<landmark.x()<<","<<landmark.y()<<","<<landmark.z()<<endl;
    VLOG(3)<<"Cam1 cam2 angle error1 error2"<<endl;

    ///check all combinations of rays and get the triangulation angles and reprojection errors
    double maxAngle = 0;
    maxAngleCurCamID = -1;
    maxAnglePrevCamID = -1;
    if(prevRaysCamIds.size() == 0){
        for (int camID1 = 0; camID1 < lf1->num_cams_ ; camID1++) {
            int kp_ind1 = im1->matchIndex[camID1];
            if (kp_ind1 == -1)
                continue;
            prevRaysCamIds.push_back(camID1);
        }
    }

    int firstcID = prevRaysCamIds[0];
    for (auto& camID1 : prevRaysCamIds){
        int kp_ind1 = im1->matchIndex[camID1];

        for (int camID2 = 0; camID2 < lf2->num_cams_ ; camID2++){
            int kp_ind2 = im2->matchIndex[camID2];
            if(kp_ind2 == -1)
                continue;
            if(camID1 == firstcID)
                curRaysCamIDs.push_back(camID2);
            gtsam::Pose3 transformedPose1 = calCompCamPose(lf1, camID1); // take input of the LF frame pose and comp cam id to compute the comp cam pose
            gtsam::Pose3 transformedPose2 = calCompCamPose(lf2, camID2);
            gtsam::Point2 obs1 = convertPoint2_CV2GTSAM(lf1->image_kps_undist[camID1][kp_ind1]);
            gtsam::Point2 obs2  = convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][kp_ind2]);

            //check the reprojection error again to filter any outliers
            double error1 = computeReprojectionError(transformedPose1, camID1, landmark, obs1);
            double error2 = computeReprojectionError(transformedPose2, camID2, landmark, obs2);
            double tri_angle_deg=0;
            //compute angle
            bool angleCheck = checkTriangulationAngle(const_cast<gtsam::Point3 &>(transformedPose1.translation()),
                                                      const_cast<gtsam::Point3 &>(transformedPose2.translation()),
                                                      landmark , tri_angle_deg);
            VLOG(3)<<camID1<<"  "<<camID2<<"  "<<tri_angle_deg<<"," <<error1<<", "<<error2<<endl;
            if(angleCheck and error1 <=4 and error2 <=4 ){

                angles.at<double>(camID1,camID2) = tri_angle_deg;
                tri_errors.at<double>(camID1,camID2) = error1 + error2;

                if(maxAngle < angles.at<double>(camID1,camID2)){
                    maxAngle = angles.at<double>(camID1,camID2);
                    maxAnglePrevCamID = camID1;
                    maxAngleCurCamID = camID2;
                }

            }
        }
    }

}

void
Backend::insertLandmarkInGraph(Landmark *l, int prevKFID, int curKFID, bool newlm, vector<int> previous_compcam_ids,
                               vector<gtsam::Point2> previous_measurements, vector<int> cur_compcam_ids,
                               vector<gtsam::Point2> current_measurements, vector<int> prevKPOctaves,
                               vector<int> curKPOctaves) {
    ///Previous state ID
    LightFieldFrame* prev_LF = l->KFs[prevKFID];
    LightFieldFrame* currentFrame = l->KFs[curKFID];
    int curStateID = currentFrame->frameId;
    int prevStateID = prev_LF->frameId;
    int lid = l->lId;

    newTimeStamps[gtsam::Symbol('l', lid)] = (windowCounter-1);

    if(!(currentEstimate.exists(gtsam::Symbol('x', curStateID)) || initialEstimate.exists(gtsam::Symbol('x', curStateID)))){
        gtsam::Pose3 current_pose = convertPose3_CV2GTSAM(currentFrame->pose);
        /// Initial estimate for the current pose.
        initialEstimate.insert(gtsam::Symbol('x', curStateID), current_pose);
        /// state factor record
        xFactorRecord[curStateID] = vector<int>();
        //This will be used only in case of fixedlag smoothing
        newTimeStamps[gtsam::Symbol('x', curStateID)] = (windowCounter-1);
        VLOG(3)<<"Inserting Initial Estimate for x"<<curStateID<<endl;

    }

    if(newlm){

        /// for each measurement add a projection factor
        if(!(currentEstimate.exists(gtsam::Symbol('x', prevStateID)) || initialEstimate.exists(gtsam::Symbol('x', prevStateID)))){
            // Initial estimate for the current pose.
            gtsam::Pose3 previous_pose = convertPose3_CV2GTSAM(prev_LF->pose);
            initialEstimate.insert(gtsam::Symbol('x', prevStateID), previous_pose);
            /// state factor record
            xFactorRecord[prevStateID] = vector<int>();
            newTimeStamps[gtsam::Symbol('x', prevStateID)] = (windowCounter-1);
            VLOG(3)<<"Inserting Initial Estimate for x"<<prevStateID<<endl;
        }


        int c=0;
        for(auto& prevCompId : previous_compcam_ids){

            // Insert projection factor with previous kf to factor graph
            //VLOG(2)<<"measurement from prev frame: "<<previous_measurements[i].x()<<" "<<previous_measurements[i].y()<<endl;
            noiseModel::Robust::shared_ptr huberModel1;
            if(!prevKPOctaves.empty()){
                auto measurementNoise1 =
                        noiseModel::Isotropic::Sigma(2, frontEnd->orBextractor->GetInverseScaleSigmaSquares()[ prevKPOctaves[c]]);
                huberModel1 = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(
                        sqrt(5.991)), measurementNoise1);
                graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                        (previous_measurements[c], huberModel, gtsam::Symbol('x', prevStateID),
                                         gtsam::Symbol('l', lid), K[prevCompId], convertPose3_CV2GTSAM(Rt_mats_[prevCompId]) ));
            }
            else{
                graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                        (previous_measurements[c], huberModel, gtsam::Symbol('x', prevStateID),
                                         gtsam::Symbol('l', lid), K[prevCompId], convertPose3_CV2GTSAM(Rt_mats_[prevCompId]) ));
            }


            if(lmFactorRecordMulti.find(lid) != lmFactorRecordMulti.end() ){
                std::map<int, vector<int>> rec;
                rec[prevKFID] = vector<int>({prevCompId});
                lmFactorRecordMulti[lid] = rec;
            }
            else{
                std::map<int, vector<int>> rec = lmFactorRecordMulti[lid];
                if(rec.find(prevStateID) != rec.end()){
                    rec[prevKFID].push_back(prevCompId);
                    lmFactorRecordMulti[lid] = rec;
                }
                else{
                    rec[prevKFID] = vector<int>({prevCompId});
                    lmFactorRecordMulti[lid] = rec;
                }

            }

            c++;
        }
        /// state factor record
        xFactorRecord[prevStateID].push_back(lid);
        initialEstimate.insert(gtsam::Symbol('l', lid), convertPoint3_CV2GTSAM(l->pt3D));
        VLOG(3)<<"Inserting Initial Estimate for l"<<lid<<endl;
    }

    int c=0;


    for(auto& curCompId : cur_compcam_ids){
        // insert projection factor with current frame to factor graph
        noiseModel::Robust::shared_ptr huberModel1;
        if(!curKPOctaves.empty()){
            auto measurementNoise1 =
                    noiseModel::Isotropic::Sigma(2, frontEnd->orBextractor->GetInverseScaleSigmaSquares()[ curKPOctaves[c]]);
            huberModel1 = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(
                    sqrt(5.991)), measurementNoise1);
            graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                    (current_measurements[c], huberModel, gtsam::Symbol('x', curStateID),
                                     gtsam::Symbol('l',lid), K[curCompId], convertPose3_CV2GTSAM(Rt_mats_[curCompId]) ));
        }
        else{
            graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                    (current_measurements[c], huberModel, gtsam::Symbol('x', curStateID),
                                     gtsam::Symbol('l',lid), K[curCompId], convertPose3_CV2GTSAM(Rt_mats_[curCompId]) ));
        }


        std::map<int, vector<int>> rec = lmFactorRecordMulti[lid];
        if(rec.find(curKFID) != rec.end()){
            rec[curKFID].push_back(curCompId);
            lmFactorRecordMulti[lid] = rec;
        }
        else{
            rec[curKFID] = vector<int>({curCompId});
            lmFactorRecordMulti[lid] = rec;
        }

        c++;
    }

    /// state factor record
    xFactorRecord[curStateID].push_back(lid);
}

void Backend::insertPriors(int lid){
    /// check if this is the first frame we are inserting
    /// if it is, we have to add prior to the pose and a landmark
    if(currentEstimate.empty() && initialEstimate.empty()) {

        VLOG(2)<<"Went inside GRAPH==0"<<endl;

        Landmark* l = frontEnd->map->getLandmark(lid);
        int numKFs = l->KFs.size();
        LightFieldFrame* prevLF=  l->KFs[numKFs-2];
        //gtsam::Pose3 previous_pose = calCompCamPose(prevLF , previous_compcam_ids[0]);

        int prevStateID = prevLF->frameId;
        gtsam::Pose3 previous_pose = convertPose3_CV2GTSAM(prevLF->pose);
        /// Add a prior on pose x0
        gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas
                ((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.01),gtsam::Vector3::Constant(0.01)).finished());
        graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', prevStateID), previous_pose, poseNoise));
        VLOG(2)<<"Inserted prior for the state x"<<prevStateID<<endl;
        /// LANDMARK PRIOR
        gtsam::Point3 lmgtsam = convertPoint3_CV2GTSAM(l->pt3D);
        addLandmarkPrior(lid, lmgtsam);


        if(false){
            reinitialized_ = false;
            vector<IntraMatch> *intramatches = &frontEnd->currentFrame->intraMatches;
            vector<int> *lids = &frontEnd->currentFrame->lIds;

            std::set<LightFieldFrame*> kfSet;
            for (int i = 0; i < intramatches->size(); ++i) {
                if (lids->at(i) != -1 ) {
                    int l_id = lids->at(i);
                    vector<LightFieldFrame*> observedKFs = frontEnd->map->getLandmark(l_id)->KFs;
                    std::copy(observedKFs.begin(), observedKFs.end(), std::inserter(kfSet, kfSet.end()));
                }
            }
            for(auto kf : kfSet){
                int prevStateID = kf->frameId;
                gtsam::Pose3 previous_pose = convertPose3_CV2GTSAM(kf->pose);
                /// Add a prior on pose x0
                gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas
                        ((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.01),gtsam::Vector3::Constant(0.01)).finished());
                graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', prevStateID), previous_pose, poseNoise));
                VLOG(2)<<"Inserted prior for the state x"<<prevStateID<<endl;
            }

        }

    }

}

bool Backend::addKeyFrameMulti() {

    ///check if camconfig has been update, if so update here as well
    if(true){
        camArrayConfig = &frontEnd->camconfig_;
        camTransformation = cv::Mat(4, 4, CV_64F);
        for (int i = 0; i < 3; ++i) {
            camTransformation.at<double>(i, 3) = camArrayConfig->t_mats_[camID].at<double>(i, 0);
            for (int j = 0; j < 3; ++j)
                camTransformation.at<double>(i, j) = camArrayConfig->R_mats_[camID].at<double>(i, j);
        }

        camTransformation.at<double>(3,0) = 0;
        camTransformation.at<double>(3,1) = 0;
        camTransformation.at<double>(3,2) = 0;
        camTransformation.at<double>(3,3) = 1;

        camTransformation = camTransformation.inv();

        for(int i =0 ; i < camArrayConfig->num_cams_ ; i++){

            K.push_back(gtsam::Cal3_S2::shared_ptr(new gtsam::Cal3_S2(camArrayConfig->K_mats_[i].at<double>(0, 0),
                                                                      camArrayConfig->K_mats_[i].at<double>(1, 1), 0,
                                                                      camArrayConfig->K_mats_[i].at<double>(0, 2),
                                                                      camArrayConfig->K_mats_[i].at<double>(1, 2))));

            Mat R = camArrayConfig->Kalibr_R_mats_[i];/// store the pose of the cam chain
            //VLOG(2)<<R;
            Mat t = camArrayConfig->Kalibr_t_mats_[i];
            Mat kalibrPose = Mat::eye(4,4, CV_64F);
            kalibrPose.rowRange(0,3).colRange(0,3) = R.t();
            kalibrPose.rowRange(0,3).colRange(3,4) = -1* R.t()*t;
            //gtsam::Matrix eigenRt_kalib;
            // cv2eigen(kalibrPose, eigenRt_kalib);
            Rt_mats_kalib_.push_back(kalibrPose.clone());

            Mat R2 = camArrayConfig->R_mats_[i];/// store the pose of the cam chain
            Mat t2 = camArrayConfig->t_mats_[i];
            Mat camPose = Mat::eye(4,4, CV_64F);
            camPose.rowRange(0,3).colRange(0,3) = R2.t();
            camPose.rowRange(0,3).colRange(3,4) = -1* R2.t()*t2;
            //gtsam::Matrix eigenRt;
            //cv2eigen(camPose, eigenRt);
            Rt_mats_.push_back(camPose.clone());
            //VLOG(2)<<"RTMats cam: "<<i<<" : "<<camPose<<endl;

        }
    }

    LightFieldFrame *currentFrame = frontEnd->currentFrame;
    GlobalMap *map = frontEnd->map;
    bool ret = true;

    VLOG(2) << "Window counter: " << windowCounter << endl;

    /// Variable declaration
    std::map<int, set<int> > insertedRigid;
    int num_lms_filtered=0;
    vector<int> lids_filtered;
    vector<gtsam::Point3> landmarks;
    vector<vector<gtsam::Point2>> current_measurements;
    vector<vector<gtsam::Point2>> previous_measurements;
    vector<vector<int>>  previous_compcam_ids, cur_compcam_ids;
    vector<LightFieldFrame*> previous_KFs;
    vector<int> prevKFInds;
    vector<bool> new_landmark_flags;

    vector<IntraMatch> *intramatches = &currentFrame->intraMatches;
    vector<int> *lids = &currentFrame->lIds;
    int current_pose_id = currentFrame->frameId;

    std::set<LightFieldFrame*> kfSetOlder;

    // browse through all instramatches
    for (int i = 0; i < intramatches->size(); ++i) {
        // if this intra match is a landmark i.e it is tracked in the current frame and is a triangulated inlier
        if (lids->at(i) != -1 ) {

            //record the landmark
            Landmark* l = map->getLandmark(lids->at(i));
            gtsam::Point3 landmark = convertPoint3_CV2GTSAM(l->pt3D);
            int numKFs = l->KFs.size();
            if(!currentEstimate.exists(gtsam::Symbol('l', lids->at(i))) and
               !initialEstimate.exists(gtsam::Symbol('l', lids->at(i))))
            {
               // if(numKFs <=2)
               //     continue;
                /// Get the previous LF frame of the landmark and the observations in the component cameras
                vector<int> prevRaysCamIds, curRaysCamIDs;
                bool firstPair = true;
                for (int pairIdx = (numKFs-1) ; pairIdx< numKFs ; pairIdx++){
                    int prevKFIdx =  pairIdx-1;     //(numKFs-2);
                    int curKFIdx =   pairIdx;       //numKFs-1;
                    LightFieldFrame* lf1 = l->KFs[prevKFIdx]; // previous KF
                    LightFieldFrame* lf2 = l->KFs[curKFIdx];  // same as l->KFs[numKFs-1]; // last KF
                    IntraMatch* im1 = &lf1->intraMatches.at(l->featInds[prevKFIdx]);
                    IntraMatch* im2 = &lf2->intraMatches.at(l->featInds[curKFIdx]);

                    Mat angles = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
                    Mat tri_errors = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);

                    ///check all combinations of rays and get the triangulation angles and reprojection errors

                    int curKFCamID = -1, prevKFCamID = -1;
                    double maxAngle = 0;
                    curRaysCamIDs.clear();
                    getCamIDs_Angles(l, prevKFIdx, curKFIdx, angles , prevRaysCamIds,curRaysCamIDs, prevKFCamID, curKFCamID);
                    if(prevKFCamID != -1 and curKFCamID != -1)
                        maxAngle = angles.at<double>(prevKFCamID,curKFCamID);
                    // compare all the angles and choose the  largest one for insertion
                    //first check the diagonal angles i.e intersecting component cameras between two LF fram

                    vector<int> acceptedPrevCamIDs, acceptedCurCamIDs, acceptedPrevKPOctaves, acceptedCurKPOctaves;
                    vector<gtsam::Point2> acceptedPrevMeas, acceptedCurMeas;
                    VLOG(3)<<"MAx Angle : "<<maxAngle;
                    if(maxAngle > 0){

                        acceptedPrevCamIDs.push_back(prevKFCamID);
                        acceptedPrevKPOctaves.push_back(lf1->image_kps_undist[prevKFCamID][im1->matchIndex[prevKFCamID]].octave);
                        acceptedPrevMeas.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[prevKFCamID][im1->matchIndex[prevKFCamID]]));

                        acceptedCurCamIDs.push_back(curKFCamID);
                        acceptedCurMeas.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[curKFCamID][im2->matchIndex[curKFCamID]]));
                        acceptedCurKPOctaves.push_back(lf2->image_kps_undist[curKFCamID][im2->matchIndex[curKFCamID]].octave);

                        vector<int>::iterator itr1 = prevRaysCamIds.begin();
                        vector<int>::iterator itr2 = curRaysCamIDs.begin();

                        for( ; itr1!=prevRaysCamIds.end() or itr2!= curRaysCamIDs.end() ; ){
                            int camID1 = *itr1;
                            int camID2 = *itr2;
                            /////////////// Check the latest in prevrays
                            if( itr1!=prevRaysCamIds.end()){
                                /// if this is the first KF pair for the landmark
                                /// choose which rays to insert from the previous  component cams
                                if(camID1 != prevKFCamID){
                                    if(firstPair){
                                        bool accept=true;
                                        vector<int>::iterator it_set = acceptedPrevCamIDs.begin();

                                        for( ; it_set != acceptedPrevCamIDs.end() ; ++it_set){
                                            int kp_ind1 = im1->matchIndex[camID1];
                                            int kp_ind2 = im1->matchIndex[*it_set];
                                            gtsam::Pose3 transformedPose1 = calCompCamPose(lf1, camID1); // take input of the LF frame pose and comp cam id to compute the comp cam pose
                                            gtsam::Pose3 transformedPose2 = calCompCamPose(lf1, *it_set);
                                            gtsam::Point2 obs1 = convertPoint2_CV2GTSAM(lf1->image_kps_undist[camID1][kp_ind1]);
                                            gtsam::Point2 obs2  = convertPoint2_CV2GTSAM(lf1->image_kps_undist[*it_set][kp_ind2]);

                                            //check the reprojection error again to filter any outliers
                                            double error1 = computeReprojectionError(transformedPose1, camID1, landmark, obs1);
                                            double error2 = computeReprojectionError(transformedPose2, *it_set, landmark, obs2);
                                            double tri_angle_deg=0;
                                            //compute angle
                                            bool angleCheck = checkTriangulationAngle(const_cast<gtsam::Point3 &>(transformedPose1.translation()),
                                                                                      const_cast<gtsam::Point3 &>(transformedPose2.translation()),
                                                                                      landmark , tri_angle_deg);
                                            //VLOG(2)<<"Prev Cams Tri ANgle : "<<tri_angle_deg;
                                            if(!angleCheck || error1 >5.991*frontEnd->orBextractor->GetScaleSigmaSquares()[lf1->image_kps_undist[camID1][kp_ind1].octave] ||
                                            error2 >5.991*frontEnd->orBextractor->GetScaleSigmaSquares()[lf1->image_kps_undist[*it_set][kp_ind2].octave]){
                                                accept = false;
                                                break;
                                            }
                                        }

                                        if(accept){
                                            it_set = acceptedCurCamIDs.begin();
                                            for( ; it_set != acceptedCurCamIDs.end() ; ++it_set){
                                                if(angles.at<double>(camID1, *it_set) == 0){
                                                    accept = false;
                                                    break;
                                                }
                                            }
                                            if(accept)
                                            {
                                                acceptedPrevCamIDs.push_back(camID1);
                                                acceptedPrevMeas.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[camID1][im1->matchIndex[camID1]]));
                                                acceptedPrevKPOctaves.push_back(lf1->image_kps_undist[camID1][im1->matchIndex[camID1]].octave);

                                            }
                                        }

                                    }
                                        /// if this is not the first KF pair for the landmark
                                        /// accept all the rays in the previous frame, since they are already chosen
                                    else{
                                        acceptedPrevCamIDs.push_back(camID1);
                                        acceptedPrevMeas.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[camID1][im1->matchIndex[camID1]]));
                                        acceptedPrevKPOctaves.push_back(lf1->image_kps_undist[camID1][im1->matchIndex[camID1]].octave);
                                    }
                                }
                                ++itr1;
                            }

                            if(itr2!= curRaysCamIDs.end()){

                                if(camID2 != curKFCamID){
                                    bool accept=true;

                                    vector<int>::iterator it_set = acceptedCurCamIDs.begin();

                                    for( ; it_set != acceptedCurCamIDs.end() ; ++it_set){
                                        int kp_ind1 = im2->matchIndex[camID2];
                                        int kp_ind2 = im2->matchIndex[*it_set];
                                        gtsam::Pose3 transformedPose1 = calCompCamPose(lf2, camID2); // take input of the LF frame pose and comp cam id to compute the comp cam pose
                                        gtsam::Pose3 transformedPose2 = calCompCamPose(lf2, *it_set);
                                        gtsam::Point2 obs1 = convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][kp_ind1]);
                                        gtsam::Point2 obs2  = convertPoint2_CV2GTSAM(lf2->image_kps_undist[*it_set][kp_ind2]);

                                        //check the reprojection error again to filter any outliers
                                        double error1 = computeReprojectionError(transformedPose1, camID2, landmark, obs1);
                                        double error2 = computeReprojectionError(transformedPose2, *it_set, landmark, obs2);
                                        double tri_angle_deg=0;
                                        //compute angle
                                        bool angleCheck = checkTriangulationAngle(const_cast<gtsam::Point3 &>(transformedPose1.translation()),
                                                                                  const_cast<gtsam::Point3 &>(transformedPose2.translation()),
                                                                                  landmark , tri_angle_deg);

                                        if(!angleCheck || error1 >5.991*frontEnd->orBextractor->GetScaleSigmaSquares()[lf2->image_kps_undist[camID2][kp_ind1].octave] ||
                                        error2 >5.991*frontEnd->orBextractor->GetScaleSigmaSquares()[lf2->image_kps_undist[*it_set][kp_ind2].octave]){
                                            accept = false;
                                            break;
                                        }
                                    }

                                    if(accept){
                                        it_set = acceptedPrevCamIDs.begin();
                                        for( ; it_set != acceptedPrevCamIDs.end() ; ++it_set){
                                            if(angles.at<double>(*it_set, camID2) == 0){
                                                accept = false;
                                                break;
                                            }
                                        }
                                        if(accept)
                                        {
                                            acceptedCurCamIDs.push_back(camID2);
                                            acceptedCurMeas.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][im2->matchIndex[camID2]]));
                                            acceptedCurKPOctaves.push_back(lf2->image_kps_undist[camID2][im2->matchIndex[camID2]].octave);
                                        }
                                    }
                                }
                                ++itr2;
                            }
                        }
                        VLOG(3)<<"New landmark LID : "<<lids->at(i)<<"  KF1 :"<<lf1->frameId<<", KF2:"<<lf2->frameId<<", PT: "<<landmark.x()<<","<<landmark.y()<<","<<landmark.z()<<" , max angle : "<<maxAngle<<"Num meas: "<<acceptedCurCamIDs.size()+acceptedPrevCamIDs.size()<<endl;

                        if(firstPair){
                            firstPair  = false;
                            num_lms_filtered++;
                            if(num_lms_filtered == 1){
                                insertPriors(lids->at(i));
                            }
                            else{
                                kfSetOlder.insert(lf1);
                            }

                            insertLandmarkInGraph(l, prevKFIdx, curKFIdx, true, acceptedPrevCamIDs,
                                                  acceptedPrevMeas, acceptedCurCamIDs, acceptedCurMeas, acceptedPrevKPOctaves,
                                                  acceptedCurKPOctaves);

                            prevRaysCamIds = acceptedCurCamIDs;

                        }
                        else{
                            insertLandmarkInGraph(l, prevKFIdx, curKFIdx, false, acceptedPrevCamIDs,
                                                  acceptedPrevMeas, acceptedCurCamIDs, acceptedCurMeas, acceptedPrevKPOctaves,
                                                  acceptedCurKPOctaves);
                            VLOG(2)<<"NEVER COMES HERE";

                        }

                    }

                }

            }
            else{
                /// this is an existing landmark in the graph.
                /// Grab the last observation
                std::map<int, vector<int>> facs = lmFactorRecordMulti[lids->at(i)];
                VLOG(3)<<"Existing Landmark"<<endl;
                assert(facs.size() != 0);

                /// Get the prev factor's frameID and componenet camera IDs
                vector<int> prevRaysCamIds = facs.rbegin()->second;
                int prevKFIdx = facs.rbegin()->first;
                int curKFIdx = numKFs-1;
                LightFieldFrame* lf1;
                IntraMatch* im1;

                lf1 = l->KFs[prevKFIdx];
                im1 = &lf1->intraMatches.at(l->featInds[prevKFIdx]);
                int prevFrameID = lf1->frameId;
                VLOG(3)<<"prevFrameID: "<<prevFrameID<<endl; //<<", CamIDs: "<<CamID1<<endl;

                LightFieldFrame* lf2 = currentFrame;  // same as l->KFs[numKFs-1]; // last KF
                IntraMatch* im2 = &lf2->intraMatches.at(l->featInds.back());

                double maxAngle = 0;
                int curKFCamID = -1, prevKFCamID = -1;
                vector<int>  curRaysCamIDs;
                Mat angles = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);
                Mat tri_errors = Mat::zeros(lf1->num_cams_,lf2->num_cams_, CV_64FC1);

                getCamIDs_Angles(l, prevKFIdx, curKFIdx , angles , prevRaysCamIds,curRaysCamIDs, prevKFCamID, curKFCamID);
                if(prevKFCamID != -1 and curKFCamID != -1)
                    maxAngle = angles.at<double>(prevKFCamID,curKFCamID);
                // compare all the angles and choose the  largest one for insertion
                //first check the diagonal angles i.e intersecting component cameras between two LF frame
                vector<int> acceptedPrevCamIDs, acceptedCurCamIDs, acceptedPrevKPOctaves, acceptedCurKPOctaves;
                vector<gtsam::Point2> acceptedPrevMeas, acceptedCurMeas;
                if(maxAngle > 0){
                    acceptedPrevCamIDs.push_back(prevKFCamID);
                    acceptedPrevMeas.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[prevKFCamID][im1->matchIndex[prevKFCamID]]));
                    acceptedPrevKPOctaves.push_back(lf1->image_kps_undist[prevKFCamID][im1->matchIndex[prevKFCamID]].octave);

                    acceptedCurCamIDs.push_back(curKFCamID);
                    acceptedCurMeas.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[curKFCamID][im2->matchIndex[curKFCamID]]));
                    acceptedCurKPOctaves.push_back(lf2->image_kps_undist[curKFCamID][im2->matchIndex[curKFCamID]].octave);
                }

                vector<int>::iterator itr1 = prevRaysCamIds.begin();
                for( ; itr1!=prevRaysCamIds.end() ; ++itr1){
                    int camID1 = *itr1;
                    if(camID1 != prevKFCamID){
                        acceptedPrevCamIDs.push_back(camID1);
                        acceptedPrevMeas.push_back(convertPoint2_CV2GTSAM(lf1->image_kps_undist[camID1][im1->matchIndex[camID1]]));
                        acceptedPrevKPOctaves.push_back(lf1->image_kps_undist[camID1][im1->matchIndex[camID1]].octave);
                    }
                }


                vector<int>::iterator itr2 = curRaysCamIDs.begin();
                for( ;  itr2!= curRaysCamIDs.end() ; ++itr2){
                    int camID2 = *itr2;

                    if(camID2 != curKFCamID){
                        bool accept=true;

                        vector<int>::iterator it_set = acceptedCurCamIDs.begin();

                        for( ; it_set != acceptedCurCamIDs.end() ; ++it_set){
                            int kp_ind1 = im2->matchIndex[camID2];
                            int kp_ind2 = im2->matchIndex[*it_set];
                            gtsam::Pose3 transformedPose1 = calCompCamPose(lf2, camID2); // take input of the LF frame pose and comp cam id to compute the comp cam pose
                            gtsam::Pose3 transformedPose2 = calCompCamPose(lf2, *it_set);
                            gtsam::Point2 obs1 = convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][kp_ind1]);
                            gtsam::Point2 obs2  = convertPoint2_CV2GTSAM(lf2->image_kps_undist[*it_set][kp_ind2]);

                            //check the reprojection error again to filter any outliers
                            double error1 = computeReprojectionError(transformedPose1, camID2, landmark, obs1);
                            double error2 = computeReprojectionError(transformedPose2, *it_set, landmark, obs2);
                            double tri_angle_deg=0;
                            //compute angle
                            bool angleCheck = checkTriangulationAngle(const_cast<gtsam::Point3 &>(transformedPose1.translation()),
                                                                      const_cast<gtsam::Point3 &>(transformedPose2.translation()),
                                                                      landmark , tri_angle_deg);

                            if(!angleCheck || error1 >5.991*frontEnd->orBextractor->GetScaleSigmaSquares()[lf2->image_kps_undist[camID2][kp_ind1].octave] ||
                            error2 > 5.991*frontEnd->orBextractor->GetScaleSigmaSquares()[lf2->image_kps_undist[*it_set][kp_ind2].octave]){
                                accept = false;
                                break;
                            }
                        }

                        if(accept){
                            it_set = acceptedPrevCamIDs.begin();
                            for( ; it_set != acceptedPrevCamIDs.end() ; ++it_set){
                                if(angles.at<double>(*it_set, camID2) == 0){
                                    accept = false;
                                    break;
                                }
                            }
                            if(accept)
                            {
                                acceptedCurCamIDs.push_back(camID2);
                                acceptedCurMeas.push_back(convertPoint2_CV2GTSAM(lf2->image_kps_undist[camID2][im2->matchIndex[camID2]]));
                                acceptedCurKPOctaves.push_back(lf2->image_kps_undist[camID2][im2->matchIndex[camID2]].octave);
                            }
                        }
                    }
                }

                VLOG(3)<<"LID : "<<lids->at(i)<<", PT: "<<landmark.x()<<","<<landmark.y()<<","<<landmark.z()<<" , max angle : "<<maxAngle<<"Num meas: "<<acceptedCurCamIDs.size()+acceptedPrevCamIDs.size()<<endl;
                if ( maxAngle > 0 ) {
                    num_lms_filtered++;
                    //lids_filtered.push_back(lids->at(i));
                    //landmarks.push_back(landmark);
                    //previous_KFs.push_back(lf1);
                    //prevKFInds.push_back(prevKFIdx);
                    // new_landmark_flags.push_back(false);
                    //current_measurements.push_back(acceptedCurMeas);
                    //cur_compcam_ids.push_back(acceptedCurCamIDs);
                    /// record the previous measurements and previous KF and comp cam ID
                    //previous_measurements.push_back(acceptedPrevMeas);
                    //previous_compcam_ids.push_back(acceptedPrevCamIDs);

                    insertLandmarkInGraph(l, prevKFIdx, curKFIdx, false, acceptedPrevCamIDs,
                                          acceptedPrevMeas, acceptedCurCamIDs, acceptedCurMeas, acceptedPrevKPOctaves,
                                          acceptedCurKPOctaves);

                }

            }

        }
    }


    if(reinitialized_){
        reinitialized_ = false;
        for(auto kf : kfSetOlder){
            int prevStateID = kf->frameId;
            gtsam::Pose3 previous_pose = convertPose3_CV2GTSAM(kf->pose);
            /// Add a prior on pose x0
            gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas
                    ((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.01),gtsam::Vector3::Constant(0.01)).finished());
            graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', prevStateID), previous_pose, poseNoise));
            VLOG(3)<<"Inserted prior for the state x"<<prevStateID<<endl;
        }
    }
    VLOG(2)<<"Backend Landmarks size: "<<num_lms_filtered<<endl;
    if(num_lms_filtered < 15){
        graph.resize(0);
        initialEstimate.clear();
        newTimeStamps.clear();
        return false;
    }

    VLOG(2)<<"graph size: "<<graph.size()<<endl;
    if(optimizationMethod != 2){
        if (windowCounter%windowSize != 0)
            ret = false;
    }
    windowCounter++;
    VLOG(2)<<"add keyframe optimize flag: "<<ret<<endl;

    return ret;

}

/// function for monocular backend
/// Add the factors for the new keyframe and the landmarks
bool Backend::addKeyFrame() {
//Backend::addKeyFrame(cv::Mat &pose, vector<cv::KeyPoint> &keyPoints, vector<cv::Mat> &landmarks, int &frame_id, vector<int>* lids){
    LightFieldFrame* currentFrame = frontEnd->currentFrame;
    GlobalMap* map = frontEnd->map;
    bool ret = true;

    VLOG(2)<<"Window counter: "<<windowCounter<<endl;

    vector<int> lids_filtered;
    vector<int> previous_pose_ids;
    int current_pose_id = currentFrame->frameId;

    vector<gtsam::Point3> landmarks;
    vector<gtsam::Point2> current_measurements;
    vector<gtsam::Point2> previous_measurements;

    gtsam::Pose3 current_pose = convertPose3_CV2GTSAM(currentFrame->pose);
    current_pose = calCompCamPose(current_pose);

//    filterLandmarks(currentFrame, map, lids_filtered, landmarks, current_measurements, previous_measurements, previous_pose_ids);
    filterLandmarksStringent(currentFrame, map, lids_filtered, landmarks, current_measurements, previous_measurements, previous_pose_ids);

    VLOG(2)<<"Backend Landmarks size: "<<landmarks.size()<<endl;
    if(landmarks.size() < 12){
        VLOG(2)<<"skipping key frame: "<<current_pose_id<<endl<<endl;
        windowCounter--;
        return false;
    }

    /// check if this is the first frame we are inserting
    /// if it is, we have to add prior to the pose and a landmark
    if(currentEstimate.empty() && initialEstimate.empty()) {

        VLOG(2)<<"Went inside GRAPH==0"<<endl;
        addPosePrior(lids_filtered.at(0), map);
        addLandmarkPrior(lids_filtered.at(0), landmarks[0]);
    }

    initialEstimate.insert(gtsam::Symbol('x', current_pose_id), current_pose);

    /// create graph edges
    /// insert initial values for the variables
    for (int i = 0; i < landmarks.size(); ++i) {
        /// for each measurement add a projection factor

        if(previous_pose_ids[i] != -1){ // Newly tracked landmark

//            if(!(currentEstimate.exists(gtsam::Symbol('x', previous_pose_ids[i])) || initialEstimate.exists(gtsam::Symbol('x', previous_pose_ids[i])))){
//
//                //Landmark's previous frame is not inserted in backend, skip landmark
//                VLOG(2)<<"Skipping landmark lid, frameid: "<<lids_filtered.at(i)<<" "<<previous_pose_ids[i]<<" "<<current_pose_id<<endl;
//                continue;
//            }

            // Insert projection factor with previous kf to factor graph
            VLOG(2)<<"measurement from prev frame: "<<previous_measurements[i].x()<<" "<<previous_measurements[i].y()<<endl;
            graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                    (previous_measurements[i], measurementNoise, gtsam::Symbol('x', previous_pose_ids[i]), gtsam::Symbol('l', lids_filtered.at(i)), K[camID]));
            initialEstimate.insert(gtsam::Symbol('l', lids_filtered.at(i)), landmarks[i]);
        }

        // insert projection factor with current frame to factor graph
        graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                    (current_measurements[i], measurementNoise, gtsam::Symbol('x', current_pose_id), gtsam::Symbol('l', lids_filtered.at(i)), K[camID]));
    }
    VLOG(2)<<"graph size: "<<graph.size()<<endl;
    if (windowCounter%windowSize != 0)
        ret = false;
    VLOG(2)<<"add keyframe optimize flag: "<<ret<<endl;
    windowCounter++;
    return ret;
}

void Backend::optimizePosesLandmarks() {
    int frame_id = frontEnd->current_frameId;
    if(optimizationMethod == 0){
        ISAM2 isam2 = ISAM2(isam);
        try {
            isam.update(graph, initialEstimate, toBeRemovedFactorIndices);
            graph.resize(0);
            initialEstimate.clear();
            toBeRemovedFactorIndices.clear();
            newTimeStamps.clear();
            // Each call to iSAM2 update(*) performs one iteration of the iterative nonlinear solver.
            // If accuracy is desired at the expense of time, update(*) can be called additional times
            // to perform multiple optimizer iterations every step.
            currentEstimate = isam.calculateBestEstimate();
            VLOG(2) << "Frame " << frame_id << ": " << endl;
            //    currentEstimate.print("Current estimate: ");
            // graph.print("current graph");
            VLOG(2) << "BackEnd Optimization done!!!!" << endl;
        }
        catch (IndeterminantLinearSystemException& e)
        {
            VLOG(2)<<"Near by variable: "<<e.nearbyVariable();
            Symbol sym  = gtsam::Symbol(e.nearbyVariable());
            if(sym.chr() == 'l'){
                // it is  landmark.
                //extract the edges corresponding to that landmark

                map<int, vector<int>> lmFactorRecord = lmFactorRecordMulti[sym.index()];
                Landmark* l = frontEnd->map->getLandmark(sym.index());
                VLOG(2)<<"lid : "<<sym.index()<<" Pt: "<<l->pt3D;
//                for(map<int, vector<int>>::iterator itr_l = lmFactorRecord.begin() ; itr_l != lmFactorRecord.end() ; ++itr_l){
//                    int iD = (*itr_l).first;
//                    VLOG(2)<<"xid : "<<l->KFs[iD]->frameId;
//                    vector<int> xids = (*itr_l).second;
//                    for(auto& x : xids)
//                    {
//                        VLOG(2)<<"comp cam id :"<< x;
//                        Mat img ;

//                        if(!l->KFs[iD]->imgs[x].empty()){
//                            cvtColor( l->KFs[iD]->imgs[x],img , COLOR_GRAY2BGR);
//                            IntraMatch im = l->KFs[iD]->intraMatches[l->featInds[iD]];
//                            circle(img, l->KFs[iD]->image_kps_undist[x][im.matchIndex[x]].pt, 3,Scalar(0,200,0), 3);
//                            imshow("error lm", img);
//                            waitKey(0);
//                        }
//                    }

               // }

            }

            /////////////////////////////////////////////////////////////////////
            //// This is purely damage control as I am not able to figure out
            //// why isam2 is giving indeterminant linea system error
            /////////////////////////////////////////////////////////////////////
            int fr_ind=0;
            for (auto& fr : frontEnd->lfFrames){
                int poseid = fr->frameId;
                if(currentEstimate.exists(gtsam::Symbol('x', poseid))){
                    try{
                       // gtsam::Matrix  covgtsam = isam2.marginalCovariance(gtsam::Symbol('x', poseid));
                       // cv::eigen2cv(covgtsam, fr->cov);
                    }
                    catch (IndeterminantLinearSystemException& e) {
                        VLOG(2)<<"Exception occured in marginal covariance computation:"<<endl;
                        fr->cov = Mat::eye(6,6, CV_64F);
                    }

                    //VLOG(2)<<"Covariance"<<isam.marginalCovariance(gtsam::Symbol('x', poseid));

                }
                fr_ind++;
            }
            isam.clear();
            lmFactorRecordMulti.clear();
            xFactorRecord.clear();
            currentEstimate.clear();
            toBeRemovedFactorIndices.clear();
            newTimeStamps.clear();
            graph.resize(0);
            initialEstimate.clear();
            currentEstimate.clear();
            isam = gtsam::ISAM2(parameters);
            reinitialized_ = true;
            windowCounter=0;

            VLOG(2)<<"Resetting ISAM";
            std::cout << "Exception caught : " << e.what() << std::endl;
        }
    }

    else if(optimizationMethod == 1){
        VLOG(2)<<"LM Optimization"<<endl;
        graph.print("graph:");
        initialEstimate.print("initial estimate");
        optimizer = new gtsam::LevenbergMarquardtOptimizer(graph, initialEstimate, params);
        currentEstimate = optimizer->optimize();
        delete optimizer;

    }
    else if(optimizationMethod == 2){
        if(windowCounter >= 2){
            try{
                fixedLagOptimizer.update(graph, initialEstimate, newTimeStamps);
                graph.resize(0);
                initialEstimate.clear();
                newTimeStamps.clear();
                VLOG(2)<<"Number of factors in graph : "<<fixedLagOptimizer.getFactors().size();
                // Each call to iSAM2 update(*) performs one iteration of the iterative nonlinear solver.
                // If accuracy is desired at the expense of time, update(*) can be called additional times
                // to perform multiple optimizer iterations every step.
//            isam.update();
                currentEstimate = fixedLagOptimizer.calculateEstimate();

            }
            catch (IndeterminantLinearSystemException& e)
            {
                Symbol sym  = gtsam::Symbol(e.nearbyVariable());
                if(sym.chr() == 'l'){
                    // it is  landmark.
                    //extract the edges corresponding to that landmark

                    map<int, vector<int>> lmFactorRecord = lmFactorRecordMulti[sym.index()];
                    Landmark* l = frontEnd->map->getLandmark(sym.index());
                    VLOG(2)<<"lid : "<<sym.index()<<" Pt: "<<l->pt3D;
                    for(map<int, vector<int>>::iterator itr_l = lmFactorRecord.begin() ; itr_l != lmFactorRecord.end() ; ++itr_l){
                        int iD = (*itr_l).first;
                        VLOG(2)<<"xid : "<<l->KFs[iD]->frameId;
                        vector<int> xids = (*itr_l).second;
                        for(auto& x : xids)
                        {
                            VLOG(2)<<"comp cam id :"<< x;
                            Mat img ;

                            if(!l->KFs[iD]->imgs[x].empty()){
                                cvtColor( l->KFs[iD]->imgs[x],img , COLOR_GRAY2BGR);
                                IntraMatch im = l->KFs[iD]->intraMatches[l->featInds[iD]];
                                circle(img, l->KFs[iD]->image_kps_undist[x][im.matchIndex[x]].pt, 3,Scalar(0,200,0), 3);
                                imshow("error lm", img);
                                waitKey(0);
                            }
                        }

                    }

                }
                graph.resize(0);
                initialEstimate.clear();
                currentEstimate = isam.calculateBestEstimate();

                std::cout << "Exception caught : " << e.what() << std::endl;
            }

        }

    }

}

void Backend::removeVariables( gtsam::KeyVector tobeRemoved){
    if(optimizationMethod == 0){
        const NonlinearFactorGraph& nfg = isam.getFactorsUnsafe();
        set<size_t> removedFactorSlots;
        const VariableIndex variableIndex(nfg);
        for(Key key: tobeRemoved) {
            int lid = symbolIndex(key);
            const auto& slots = variableIndex[key];
            toBeRemovedFactorIndices.insert(toBeRemovedFactorIndices.end(), slots.begin(), slots.end());
//            lmFactorRecordMulti.erase(lid);
//            for(auto fr: frontEnd->map->getLandmark(lid)->KFs){
//                if(xFactorRecord.find(fr->frameId) != xFactorRecord.end()){
//                    vector<int> lms = xFactorRecord[fr->frameId];
//                    for(int i=0; i <lms.size(); i++) {
//                        if (lms[i] == lid)
//                        {
//                            xFactorRecord[fr->frameId].erase(xFactorRecord[fr->frameId].begin() + i);
//                            break;
//                        }
//                    }
//                }
//            }
//            frontEnd->map->deleteLandmark(lid);
        }

      }
    else if(optimizationMethod == 1){
        //Not implemented
    }
    else if(optimizationMethod == 2){
        //Not implemented
    }
}
void Backend::updateVariables() {
    GlobalMap* map = frontEnd->map;
    double mean_correction=0.0, max_correction=0.0;
    int num_lms = 0;
    gtsam::KeyVector tobeRemoved;
    for (gtsam::Values::iterator it = currentEstimate.begin(); it != currentEstimate.end(); ++it) {
        gtsam::Symbol lid = it->key;

        if (lid.chr() == 'l') {
            num_lms++;
            gtsam::Point3 b = currentEstimate.at<gtsam::Point3>(lid);
            double aa = b.x();
            double bb = b.y();
            double cc = b.z();
            double a[3][1] = {aa, bb, cc};
            cv::Mat point3(3, 1, CV_64F, a);
            double diff_norm;
            bool success = map->updateLandmark(lid.index(), point3, diff_norm);
            // if the update of the landmark is not successful
            //remove the landmark from the graph
            if(!success)
                tobeRemoved.push_back(lid);
            mean_correction += diff_norm;
            if(max_correction < diff_norm)
                max_correction = diff_norm;


        }

    }
    //Remove the following variables from the back-end
    VLOG(2)<<"To Be Removed landmarks : "<<tobeRemoved.size()<<endl;
    removeVariables( tobeRemoved);
    mean_correction /= num_lms;
    VLOG(2)<< "Mean correction for landmarks : "<<mean_correction<<", Max Correction : "<<max_correction<<endl;
    //cout<<"Number of LF Frames in front-end:"<<frontEnd->lfFrames.size()<<endl;
    unique_lock<mutex> lock(frontEnd->mMutexPose);
    //frontEnd->allPoses.clear();
    if(backendType == MULTI_RIGID ){
        for (auto& fr : frontEnd->lfFrames){
            int poseid = fr->frameId;
            for (int cid =0; cid< 1; cid++){
                int stateid = stoi(to_string(poseid) + to_string(cid));
                if(currentEstimate.exists(gtsam::Symbol('x', stateid))){
                    gtsam::Pose3 pose = currentEstimate.at<gtsam::Pose3>(  gtsam::Symbol('x', stateid));
                    gtsam::Matrix mat = pose.matrix();
                    Mat Pose_wc;
                    cv::eigen2cv(mat, Pose_wc);
                    Mat R = camArrayConfig->R_mats_[cid];/// store the pose of the cam chain
                    Mat t = camArrayConfig->t_mats_[cid];
                    Mat Pose_cb = Mat::eye(4,4, CV_64F);
                    Pose_cb.rowRange(0,3).colRange(0,3) = R;
                    Pose_cb.rowRange(0,3).colRange(3,4) = t;
                    Mat Pose_wb = Pose_wc* Pose_cb;
                    frontEnd->allPoses.push_back(Pose_wb.rowRange(0,3).clone());

                    Mat diffPose = fr->pose.inv() * Pose_wb;
                    VLOG(3)<<"diff in pose of x"<<stateid<<" : "<<diffPose<<endl;
                    fr->pose = Pose_wb.clone();
                    break; // break from this loop and go to next frameID
                }
            }
        }
    }
    else if (backendType == MONO or backendType == MULTI){
        int fr_ind=0;
        for (auto& fr : frontEnd->lfFrames){
            int poseid = fr->frameId;
            if(currentEstimate.exists(gtsam::Symbol('x', poseid))){

                gtsam::Pose3 pose = currentEstimate.at<gtsam::Pose3>(  gtsam::Symbol('x', poseid));
                gtsam::Matrix mat = pose.matrix();
                Mat curPos;
                cv::eigen2cv(mat, curPos);
                Mat diffPose = fr->pose.inv() * curPos;
                VLOG(3)<<"diff in pose of x"<<poseid<<" : "<<diffPose<<endl;
                //frontEnd->allPoses.push_back(curPos.rowRange(0,3).clone());
                frontEnd->allPoses[fr_ind] = curPos.rowRange(0,3).clone();
                fr->pose = curPos.clone();
            }
            fr_ind++;
        }
    }

}

////// THESE methods are for Later.
void Backend::globalOptimization(){

}
void Backend::optimizePose(){

}
