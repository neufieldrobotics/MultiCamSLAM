//
// Created by Pushyami Kaveti on 1/24/22.
//

#include "MCSlam/MonoBackEnd.h"

MonoBackEnd::MonoBackEnd(string strSettingsFile, Mat Kmat, MonoFrontEnd *fe) : backend_config_file(strSettingsFile) {

        cv::FileStorage fSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if (!fSettings.isOpened()) {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

        camID = (int) fSettings["CamID"];


        camTransformation = cv::Mat::eye(4, 4, CV_64F);

        K = gtsam::Cal3_S2::shared_ptr(new gtsam::Cal3_S2(Kmat.at<double>(0, 0),Kmat.at<double>(1, 1), 0,
                                                          Kmat.at<double>(0, 2),Kmat.at<double>(1, 2)));

        measurementNoise = gtsam::noiseModel::Isotropic::Sigma(2, (double)fSettings["MeasurementNoiseSigma"]);

        optimizationMethod = (int)fSettings["Optimization"];

        if(optimizationMethod == 0){
            parameters.relinearizeThreshold = (double)fSettings["ISAMRelinearizeThreshold"];
            parameters.relinearizeSkip = (int)fSettings["ISAMRelinearizeSkip"];
            isam = gtsam::ISAM2(parameters);
        }
        else if(optimizationMethod == 1){
            params.orderingType = gtsam::Ordering::METIS;
        }

        windowSize = (int)fSettings["WindowBad"];
        windowCounter = 0;
        angleThresh = (double)fSettings["AngleThresh"];
        frontEnd = fe;
}

MonoBackEnd::~MonoBackEnd() = default;

gtsam::Pose3 MonoBackEnd::convertPose3_CV2GTSAM(cv::Mat &pose){

    gtsam::Rot3 R(pose.at<double>(0,0), pose.at<double>(0,1), pose.at<double>(0,2),
                  pose.at<double>(1,0), pose.at<double>(1,1), pose.at<double>(1,2),
                  pose.at<double>(2,0), pose.at<double>(2,1), pose.at<double>(2,2));

    gtsam::Point3 t(pose.at<double>(0,3), pose.at<double>(1,3), pose.at<double>(2,3));

    return gtsam::Pose3(R, t);
}

gtsam::Point2 MonoBackEnd::convertPoint2_CV2GTSAM(cv::KeyPoint &kp){
    gtsam::Point2 pt(kp.pt.x, kp.pt.y);
    return pt;
}

gtsam::Point3 MonoBackEnd::convertPoint3_CV2GTSAM(cv::Mat &landmark){
    gtsam::Point3 pt(landmark.at<double>(0,0), landmark.at<double>(1,0), landmark.at<double>(2,0));
    return pt;
}

bool MonoBackEnd::checkTriangulationAngle(gtsam::Point3 &pose1, gtsam::Point3 &pose2, gtsam::Point3 &landmark,
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

void MonoBackEnd::addPosePrior(int lid, GlobalMap *map){
    Landmark* l = map->getLandmark(lid);
    MonoFrame* prevFrame= l->KFs_mono[0];
    gtsam::Pose3 previous_pose = convertPose3_CV2GTSAM(prevFrame->pose);

    // Add a prior on pose x0
    gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas
            ((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.1),gtsam::Vector3::Constant(0.1)).finished());
    graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', prevFrame->frameId), previous_pose, poseNoise));
    initialEstimate.insert(gtsam::Symbol('x', prevFrame->frameId), previous_pose);
    pose_log.insert({prevFrame->frameId, 0});
}

/// insert landmark prior for first landmark to factor graph
void MonoBackEnd::addLandmarkPrior(int lid, gtsam::Point3 &landmark, GlobalMap *map){

    gtsam::noiseModel::Isotropic::shared_ptr pointNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
    //cout<<lid<< landmark<<endl;
    graph.push_back(gtsam::PriorFactor<gtsam::Point3>(gtsam::Symbol('l', lid), landmark, pointNoise)); // add directly to graph
}

/// finds the previous keyframe in which the landmark was last observed in the same component camera. returns the
/// corresponding measurement, pose id and pose.
bool MonoBackEnd::obtainPrevMeasurement(Landmark* landmark, gtsam::Point2 &prev_measurement, int &prev_pose_id, gtsam::Pose3 &prev_pose, bool init) {

    vector<MonoFrame *> KFs = landmark->KFs_mono;
    vector<int> *feat_inds = &landmark->featInds;
    auto kf_it = KFs.rbegin();
    auto feat_it = feat_inds->rbegin();

    kf_it++;
    feat_it++;

    auto prev_KF_it = ++frontEnd->frames_.rbegin();
    int init_prev_pose_id = (*prev_KF_it)->frameId;

    for(; kf_it!=KFs.rend() && feat_it!=feat_inds->rend(); kf_it++, feat_it++){
        int kp_ind = *feat_it;
        int pose_id = (*kf_it)->frameId;
        if(init){
            if(pose_id == init_prev_pose_id){
                prev_measurement = convertPoint2_CV2GTSAM((*kf_it)->image_kps_undist[kp_ind]);
                prev_pose_id = (*kf_it)->frameId;
                prev_pose = convertPose3_CV2GTSAM((*kf_it)->pose);
                return true;
            }
        }
        else if(currentEstimate.exists(gtsam::Symbol('x', pose_id)) || initialEstimate.exists(gtsam::Symbol('x', pose_id))){
            prev_measurement = convertPoint2_CV2GTSAM((*kf_it)->image_kps_undist[kp_ind]);
            prev_pose_id = (*kf_it)->frameId;
            prev_pose = convertPose3_CV2GTSAM((*kf_it)->pose);
            return true;
        }
    }
    return false;
}

/// filters landmarks in current frame based on observability in component camera (current and any of the previous frames) and triangulation angle
/// returns vectors of lids, landmarks, current measurements, previous measurements and previous pose ids
void MonoBackEnd::filterLandmarksStringent(MonoFrame* currentFrame, GlobalMap *map, vector<int>
&lids_filtered, vector<gtsam::Point3> &landmarks, vector<gtsam::Point2> &current_measurements,
                                       vector<gtsam::Point2> &previous_measurements, vector<int> &previous_pose_ids){

    int landmarks_tot=0, landmarks_filtered=0;

    vector<KeyPoint>* kps = &currentFrame->image_kps_undist;
    vector<int> *lids = &currentFrame->lIds;
    cout<<"Number of tracked landmarks " <<currentFrame->numTrackedLMs<<endl;

    gtsam::Pose3 current_pose = convertPose3_CV2GTSAM(currentFrame->pose);

    bool is_first_frame = currentEstimate.empty() && initialEstimate.empty();
    double tri_ang;
    for(int i=0; i<kps->size(); ++i){
        if(lids->at(i)!=-1){ // landmarks seen in current frame
            landmarks_tot++;
            Landmark* l = map->getLandmark(lids->at(i));
            gtsam::Point3 landmark = convertPoint3_CV2GTSAM(l->pt3D);

            gtsam::Point2 previous_measurement;
            int previous_pose_id;
            gtsam::Pose3 previous_pose;
            bool add_landmark = obtainPrevMeasurement(l, previous_measurement, previous_pose_id, previous_pose, is_first_frame);

            if(!currentEstimate.exists(gtsam::Symbol('l', lids->at(i))) and
               !initialEstimate.exists(gtsam::Symbol('l', lids->at(i)))){ // Newly triangulated landmark

                //check if landmark has been observed in a previous kf
                if (!add_landmark)
                    continue;
                //check triangulation angle

                landmarks_filtered++;
                lids_filtered.push_back(lids->at(i));
                landmarks.push_back(landmark);
                current_measurements.push_back(convertPoint2_CV2GTSAM(kps->at(i)));
                previous_measurements.push_back(previous_measurement);
                previous_pose_ids.push_back(previous_pose_id);

            }

            else{ // Tracked landmark

                landmarks_filtered++;
                lids_filtered.push_back(lids->at(i));
                landmarks.push_back(landmark);
                current_measurements.push_back(convertPoint2_CV2GTSAM(kps->at(i)));
                previous_measurements.emplace_back();
                previous_pose_ids.push_back(-1);
            }
        }
    }
    VLOG(2)<<"Landmarks before and after filtering: "<<landmarks_tot<<" "<<landmarks_filtered<<endl;
}


/// Add the factors for the new keyframe and the landmarks
bool MonoBackEnd::addKeyFrame() {
    MonoFrame* currentFrame = frontEnd->currentFrame;
    GlobalMap* map = frontEnd->map;
    bool ret = true;
    windowCounter++;

    VLOG(2)<<"Window counter: "<<windowCounter<<endl;

    vector<int> lids_filtered;
    vector<int> previous_pose_ids;
    int current_pose_id = currentFrame->frameId;

    vector<gtsam::Point3> landmarks;
    vector<gtsam::Point2> current_measurements;
    vector<gtsam::Point2> previous_measurements;

    gtsam::Pose3 current_pose = convertPose3_CV2GTSAM(currentFrame->pose);
    VLOG(2)<<"current pose id: "<<current_pose_id<<endl;

    filterLandmarksStringent(currentFrame, map, lids_filtered, landmarks, current_measurements, previous_measurements, previous_pose_ids);

    VLOG(2)<<"Backend Landmarks size: "<<landmarks.size()<<endl;
//    if(landmarks.size() < 12){
//        VLOG(2)<<"skipping key frame: "<<current_pose_id<<endl<<endl;
//        windowCounter--;
//        return false;
//    }

    /// check if this is the first frame we are inserting
    /// if it is, we have to add prior to the pose and a landmark
    if(currentEstimate.empty() && initialEstimate.empty()) {

        VLOG(2)<<"Went inside GRAPH==0"<<endl;
        addPosePrior(lids_filtered.at(0), map);
        addLandmarkPrior(lids_filtered.at(0), landmarks[0], map);
    }

    initialEstimate.insert(gtsam::Symbol('x', current_pose_id), current_pose);
    pose_log.insert({current_pose_id, 0});
    /// create graph edges
    /// insert initial values for the variables
    for (int i = 0; i < landmarks.size(); ++i) {
        /// for each measurement add a projection factor

        if(previous_pose_ids[i] != -1){ // Newly tracked landmark

            // Insert projection factor with previous kf to factor graph
            VLOG(3)<<"measurement from prev frame: "<<previous_measurements[i].x()<<" "<<previous_measurements[i].y()<<endl;
            graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                    (previous_measurements[i], measurementNoise,
                                     gtsam::Symbol('x', previous_pose_ids[i]),
                                     gtsam::Symbol('l', lids_filtered.at(i)), K));
            landmark_log.insert(make_pair(lids_filtered.at(i), vector<int>(1, previous_pose_ids[i])));
            pose_log.at(previous_pose_ids[i])++;
            initialEstimate.insert(gtsam::Symbol('l', lids_filtered.at(i)), landmarks[i]);
        }

        // insert projection factor with current frame to factor graph
        graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                (current_measurements[i], measurementNoise,
                                 gtsam::Symbol('x', current_pose_id),
                                 gtsam::Symbol('l', lids_filtered.at(i)), K));
        landmark_log.at(lids_filtered.at(i)).push_back(current_pose_id);
        pose_log.at(current_pose_id)++;
    }

    VLOG(2)<<"graph size: "<<graph.size()<<endl;
    if (windowCounter%windowSize != 0)
        ret = false;
    VLOG(2)<<"add keyframe optimize flag: "<<ret<<endl;

    VLOG(2)<<"Pose Log"<<endl;
    VLOG(2)<<"----------------------------------------------------"<<endl;
    for(const auto & it : pose_log){
        VLOG(2)<<it.first<<" "<<it.second<<endl;
    }

    VLOG(2)<<"Landmarks log"<<endl;
    VLOG(2)<<"----------------------------------------------------"<<endl;
    for(const auto & it : landmark_log){
        assert(it.second.size() > 1);
//        VLOG(2)<< it.first << " " << it.second.size()<<endl;
        for(auto &i : it.second){
//            VLOG(2)<<i;
            assert(pose_log.count(i) == 1);
        }
//        VLOG(2)<<endl;
    }

    return ret;
}

void MonoBackEnd::optimizePosesLandmarks() {
    int frame_id = frontEnd->current_frameId;
//    filebuf fb;
//    fb.open ("/home/auv/NUFRL/lf_ws/src/light-fields-pack/log/graph.dot",std::ios::out);
//    std::ostream os(&fb);
//    graph.saveGraph(os, currentEstimate);
    if(optimizationMethod == 0){
        //initialEstimate.print("Initial estimate: ");
        cout<<"---------------------------------------------"<<endl;
        //graph.print("current graph");

        // Update iSAM with the new factors
//    graph.print("graph:");
//    initialEstimate.print("initial estimate");
        isam.update(graph, initialEstimate);
        // Each call to iSAM2 update(*) performs one iteration of the iterative nonlinear solver.
        // If accuracy is desired at the expense of time, update(*) can be called additional times
        // to perform multiple optimizer iterations every step.
        isam.update();
        currentEstimate = isam.calculateBestEstimate();
        VLOG(2)<<"Frame " << frame_id << ": " << endl;
//    currentEstimate.print("Current estimate: ");
        // graph.print("current graph");
        graph.resize(0);
        initialEstimate.clear();
        VLOG(2)<<"BackEnd Optimization done!!!!"<<endl;
    }

    else if(optimizationMethod == 1){
        VLOG(2)<<"LM Optimization"<<endl;
//        graph.print("graph:");
//        initialEstimate.print("initial estimate");
        optimizer = new gtsam::LevenbergMarquardtOptimizer(graph, initialEstimate, params);
        currentEstimate = optimizer->optimize();

    }

}

void MonoBackEnd::updateVariables() {
    GlobalMap* map = frontEnd->map;
    double mean_correction=0.0, max_correction=0.0;
    int num_lms = 0;
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
            bool success = map->updateLandmark(lid.index(), point3,diff_norm);
            mean_correction += diff_norm;
            if(max_correction < diff_norm)
                max_correction = diff_norm;

        }

    }
    mean_correction /= num_lms;
    VLOG(2)<< "Mean correction for landmarks : "<<mean_correction<<", Max Correction : "<<max_correction<<endl;
    //cout<<"Number of LF Frames in front-end:"<<frontEnd->lfFrames.size()<<endl;
    unique_lock<mutex> lock(frontEnd->mMutexPose);
    frontEnd->allPoses.clear();

    for (auto& fr : frontEnd->frames_) {
        int poseid = fr->frameId;
        if (currentEstimate.exists(gtsam::Symbol('x', poseid))) {

            gtsam::Pose3 pose = currentEstimate.at<gtsam::Pose3>(gtsam::Symbol('x', poseid));
            gtsam::Matrix mat = pose.matrix();
            Mat curPos;
            cv::eigen2cv(mat, curPos);
            Mat diffPose = fr->pose.inv() * curPos;
            VLOG(1)<<"diff in pose of x"<<poseid<<" : "<<diffPose<<endl;
            frontEnd->allPoses.push_back(curPos.rowRange(0, 3).clone());
            fr->pose = curPos.clone();
        }
    }
}
