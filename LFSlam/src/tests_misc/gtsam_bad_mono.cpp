//
// Created by Pushyami Kaveti on 11/1/21.
//
#include <gflags/gflags.h>
#include <fstream>
#include <glog/logging.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/slam/BetweenFactor.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <set>
#include <chrono>

using namespace std;
using namespace gtsam;
using namespace cv;
using namespace std::chrono;

double FOCAL_LENGTH_X = 0.0;
double FOCAL_LENGTH_Y = 0.0;
double CENTER_X = 0.0;
double CENTER_Y = 0.0;

DEFINE_string(calib_file_mono, "", "A text file with calibration params for a single camera");
DEFINE_string(calib_file, "", "A text file with calibration params");
DEFINE_string(factor_file, "", "A text file with poses and landmark initial values");

vector<Matrix> K_mats_;
vector<Matrix> Rt_mats_;
vector<Matrix> Rt_mats_kalib_;
vector<Mat> t_vecs_ ;
vector<Mat> dist_coeffs_;
vector<Mat> R_mats_;

Mat build_Rt(Mat R, Mat t) {

    Mat_<double> Rt = Mat_<double>::zeros(3,4);
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            Rt(i,j) = R.at<double>(i,j);
        }
        Rt(i,3) = t.at<double>(0,i);
    }

    return Rt;

}


void writeOptimized(Values& result,  std::map<int, double>& tStamps, bool multi){
    ofstream fil("optimized_poses.txt");
    Values pose_values = result.filter<Pose3>();
    for(Values::iterator it = pose_values.begin(); it != pose_values.end() ; ++it){
        Symbol sym = it->key;
        int x_id = sym.index();
        Pose3 pose = pose_values.at<Pose3>(sym);
        Matrix mat = pose.matrix();
        int stamp_key=0;
        if (multi){
            string s = to_string(x_id);
            s.pop_back();
            if(!s.empty())
                stamp_key=stoi(s);
        }
        else{
            stamp_key = x_id;
        }


        double stamp = tStamps[stamp_key];
        fil <<"x"<<" "<< std::setprecision(5)<< std::fixed<< stamp <<" "<<x_id<<" "<<mat(0,0)<<" "<<mat(0,1)<<" "<<mat(0,2)<<" "<<mat(0,3)<<" "
                                 <<mat(1,0)<<" "<<mat(1,1)<<" "<<mat(1,2)<<" "<<mat(1,3)<<" "
                                 <<mat(2,0)<<" "<<mat(2,1)<<" "<<mat(2,2)<<" "<<mat(2,3)<<" "
                                 <<mat(3,0)<<" "<<mat(3,1)<<" "<<mat(3,2)<<" "<<mat(3,3)<<"\n";


    }

    //fil<<"l"<<" "<<l->lId<<" "<<l->pt3D.at<double>(0,0)<<" "<<l->pt3D.at<double>(1,0)<<" "<<l->pt3D.at<double>(2,0)<<"\n";
    fil.close();
}


void read_kalibr_chain(string path){

    LOG(INFO) << "Reading calibration data"<<path<<endl;
    bool radtan = true;

    FileStorage fs(path, FileStorage::READ);
    FileNode fn = fs.root();

    FileNodeIterator fi = fn.begin(), fi_end = fn.end();
    int i=0;
    Size calib_img_size_;
    for (; fi != fi_end; ++fi, i++) {

        FileNode f = *fi;
        if (f.name().find("cam",0) == string::npos)
            break;

        // READING CAMERA PARAMETERS from here coz its only one time now due to static camera array
        // in future will have to subscribe from camera_info topic
        // Reading distortion coefficients
        vector<double> dc;
        Mat_<double> dist_coeff = Mat_<double>::zeros(1,5);
        f["distortion_coeffs"] >> dc;
        if(radtan){
            for (int j=0; j < dc.size(); j++)
                dist_coeff(0,j) = (double)dc[j];
        }else{

            for (int j=0; j < 3; j++){
                if( j < 2)
                    dist_coeff(0,j) = (double)dc[j];
                else
                    dist_coeff(0,j+2) = (double)dc[j];
            }
        }

        vector<int> ims;
        f["resolution"] >> ims;
        if (i>0) {
            if (((int)ims[0] != calib_img_size_.width) || ((int)ims[1] != calib_img_size_.height))
                LOG(FATAL)<<"Resolution of all images is not the same!";
        } else {
            calib_img_size_ = Size((int)ims[0], (int)ims[1]);
        }

        // Reading K (camera matrix)
        vector<double> intr;
        f["intrinsics"] >> intr;
        Mat_<double> K_mat = Mat_<double>::zeros(3,3);
        K_mat(0,0) = (double)intr[0]; K_mat(1,1) = (double)intr[1];
        K_mat(0,2) = (double)intr[2]; K_mat(1,2) = (double)intr[3];
        K_mat(2,2) = 1.0;

        // Reading R and t matrices
        Mat_<double> R = Mat_<double>::zeros(3,3);
        Mat_<double> t = Mat_<double>::zeros(3,1);
        FileNode tn = f["T_cn_cnm1"];
        if (tn.empty()) {
            R(0,0) = 1.0; R(1,1) = 1.0; R(2,2) = 1.0;
            t(0,0) = 0.0; t(1,0) = 0.0; t(2,0) = 0.0;
        } else {
            FileNodeIterator fi2 = tn.begin(), fi2_end = tn.end();
            int r = 0;
            for (; fi2 != fi2_end; ++fi2, r++) {
                if (r==3)
                    continue;
                FileNode f2 = *fi2;
                R(r,0) = (double)f2[0]; R(r,1) = (double)f2[1]; R(r,2) = (double)f2[2];
                t(r,0) = (double)f2[3];  // * 1000; //. 1000 for real rig we donot have a scale yet.. not sure what metrics this will be
            }
        }
        /// store the pose of the cam chain
        Mat kalibrPose = Mat::eye(4,4, CV_64F);
        kalibrPose.rowRange(0,3).colRange(0,3) = R.t();
        kalibrPose.rowRange(0,3).colRange(3,4) = -1* R.t()*t;
        Matrix eigenRt_kalib;
        cv2eigen(kalibrPose, eigenRt_kalib);
        Rt_mats_kalib_.push_back(eigenRt_kalib);
        cout<<"-----------"<<endl;
        cout<<eigenRt_kalib<<endl;

       // Converting R and t matrices to be relative to world coordinates
        if (i>0) {
            Mat R3 = R.clone()*R_mats_[i-1].clone();
            Mat t3 = R.clone()*t_vecs_[i-1].clone() + t.clone();
            R = R3.clone(); t = t3.clone();
        }



        Mat Rt = build_Rt(R, t);
        Mat P = K_mat*Rt;

        VLOG(2)<<K_mat;
        VLOG(2)<<Rt;
        VLOG(1)<<P;

        Mat camPose = Mat::eye(4,4, CV_64F);
        camPose.rowRange(0,3).colRange(0,3) = R.t();
        camPose.rowRange(0,3).colRange(3,4) = -1* R.t()*t;
        Matrix eigenRt;
        cv2eigen(camPose, eigenRt);
        Rt_mats_.push_back(eigenRt);
        dist_coeffs_.push_back(dist_coeff);
        Matrix eigenK;
        cv2eigen(K_mat, eigenK);
        K_mats_.push_back(eigenK);
        R_mats_.push_back(R);
        t_vecs_.push_back(t);

    }
}

void gtsam_bad_allcams(bool rigidEdges){
    //Read the calibration params
    VLOG(1)<<"Reading calibration info"<<endl;
    read_kalibr_chain(FLAGS_calib_file);
    for (auto item: Rt_mats_){
        cout<<item<<endl;
    }
    vector<Cal3_S2::shared_ptr> K;
    for (auto item: K_mats_){
        Cal3_S2::shared_ptr tmpK(new Cal3_S2( item(0,0), item(1,1), 0 , item(0,2), item(1,2)));
        K.push_back(tmpK);
        cout<<item<<endl;
    }

    ifstream factor_file(FLAGS_factor_file.c_str());
    VLOG(1)<<"Reading factor graph info"<<endl;

    //Declare the graph and initial values
    Values initialValues;
    NonlinearFactorGraph graph;
    noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0); // pixel error in (x,y)

    //Read poses and ther initial values
    int num_poses;
    factor_file>>num_poses;
    int x_id;
    double stamp;
    char typ;
    MatrixRowMajor m(4,4);
    bool init_prior = true;
    bool init_prior_pose = true;

    int l_id;
    double X,Y,Z,u,v;
    int cam_id;
    vector<double> u_coord, v_coord;
    vector<int> pose_ids;
    vector<int> comp_cam_ids;
    /// this s to choose only poses with enough baseline
    double baseline = 0.3;
    double angleThresh = 0.5;
    std::map<int, MatrixRowMajor> chosen_poses;
    std::map<int, int> insertedPoses;
    std::map<int, double> tStamps, tStamps_allcams;
    int x_id_prev= -1; /// this to maintain a god enoough baseline.
    MatrixRowMajor m_prev(4,4);
    // Read the projection factors between the poses and the landmarks
    while(factor_file>>typ){
        switch (typ){
            case 'x':
            {   factor_file>>stamp;
                factor_file>>x_id;
                //get the pose of the state
                for (int i = 0; i < 16; i++) {
                    factor_file >> m.data()[i];
                }
                Pose3 pose(m);
                Vector diffVec = m_prev.col(3) - m.col(3);
                double diff = diffVec.norm();
                if(diff < baseline)
                    break;

                //cout<<"X id :"<<x_id<<endl;
                chosen_poses[x_id]=m;
                tStamps[x_id] = stamp;
                x_id_prev = x_id;
                m_prev = m;
                break;
            }
            case 'l':
            {
                //if the poseid are > 2 dont insert
                set<int> unique_poses;
                for (auto pid: pose_ids){
                    unique_poses.insert(pid);
                }

                if(u_coord.size() >= 2 and unique_poses.size()>1){  /// if the landmark is atleast seen from two LF frames

                    /// add the factors
                    Point3 lm(X,Y,Z);

                    if(init_prior){
                        init_prior=false;
                        noiseModel::Isotropic::shared_ptr point_noise = noiseModel::Isotropic::Sigma(3, 0.1);
                        graph.addPrior(Symbol('l',l_id),lm, point_noise );
                    }
                    initialValues.insert(Symbol('l',l_id), lm);
                    VLOG(2)<<"Inserted Initial Value for Landmark ID: "<<l_id<<": ";
                    VLOG(2)<<lm<<endl;
                    for(int idx = 0 ; idx < u_coord.size() ; idx++ ){

                        //Insert the state if it is not already inserted
                        //if this is the first pose add a prior on the pose
                        int poseid = pose_ids[idx];
                        int comp_cam_id =  comp_cam_ids[idx];
                        string s1 = to_string(poseid);
                        string s2 = to_string(comp_cam_id);
                        // Concatenate both pose id and the cam id
                        string s = s1 + s2;
                        // Convert the concatenated string
                        // to integer
                        int state_id = stoi(s);
                        //If it is the first time we are inserting this state
                        if(!initialValues.exists(Symbol('x',state_id))){
                            Matrix tranformedPose = chosen_poses[poseid]*Rt_mats_[comp_cam_id];
                            Pose3 pose(tranformedPose);
                            tStamps_allcams[state_id] = tStamps[poseid];

                            if(init_prior_pose){
                                init_prior_pose= false;
                                //noiseModel::Diagonal::shared_ptr  poseNoise = noiseModel::Diagonal::Sigmas((Vector(6)<<Vector3::Constant(0.1), Vector3::Constant(0.1)).finished());
                                //graph.addPrior(Symbol('x', poseid),pose, poseNoise);
                                graph.emplace_shared<NonlinearEquality<Pose3>>(Symbol('x',state_id), pose);
                                VLOG(2)<<"Inserted Prior for X ID: "<<state_id<<endl;
                            }

                            initialValues.insert(Symbol('x',state_id), pose);
                            if(rigidEdges){
                                /// if the pose is being inserted for the firt time
                                if(insertedPoses.find(poseid) == insertedPoses.end())
                                    insertedPoses[poseid] = comp_cam_id;
                                else{ /// if the pose has already been inserted, it is time to insert th between factors
                                    noiseModel::Diagonal::shared_ptr  betweenNoise = noiseModel::Diagonal::Sigmas((Vector(6)<<Vector3::Constant(0.001), Vector3::Constant(0.001)).finished());
                                    int comp_cam_id_prev = insertedPoses[poseid];
                                    Matrix poseDiff =Matrix::Identity(4,4);
                                    for(int ii=comp_cam_id_prev+1; ii <= comp_cam_id ; ii++ ){
                                        poseDiff =  poseDiff * Rt_mats_kalib_[ii];
                                    }
                                    Pose3 betweenPose(poseDiff);
                                    int prev_stateid = stoi(to_string(poseid) + to_string(comp_cam_id_prev));
                                    graph.emplace_shared<BetweenFactor<Pose3>>(Symbol('x', prev_stateid), Symbol('x', state_id),  betweenPose, betweenNoise);
                                    VLOG(2) <<"Inserted Between factor for PoseId: "<<poseid<<" Between: "<<comp_cam_id_prev<<" and "<<comp_cam_id<<endl;
                                }
                            }

                            VLOG(2)<<"Inserted Initial Value for X ID: "<<state_id<<endl;
                        }

                        Point2 measurement(u_coord[idx],v_coord[idx]);
                        graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(measurement, measurement_noise, Symbol('x',state_id), Symbol('l',l_id), K[comp_cam_id]);
                        VLOG(2)<<"Inserted projection factor L ID: "<<l_id<<"X ID: "<<state_id<<endl;
                    }
                    u_coord.clear();
                    v_coord.clear();
                    pose_ids.clear();
                    comp_cam_ids.clear();
                }
                else{
                    u_coord.clear();
                    v_coord.clear();
                    pose_ids.clear();
                    comp_cam_ids.clear();

                }
                factor_file>>l_id>>X>>Y>>Z;
                break;
            }
            case 'e':{
                factor_file>>x_id>>cam_id>>u>>v;
                if (chosen_poses.find(x_id) != chosen_poses.end()){
                    /// make sure that the angle between the rays is not acute.
                    if(u_coord.size() >= 1){
                        /// Check parallax
                        ///Get the landmarks coordinates, prev pose id and u, v coordinates
                        Vector3 pt(X,Y,Z);
                        int x_id_p = pose_ids[u_coord.size()-1];
                        int cam_id_p = comp_cam_ids[u_coord.size()-1];
                        double u_p = u_coord[u_coord.size()-1];
                        double v_p = v_coord[u_coord.size()-1];

                        ///Get the Ray1 between landmark and the previous pose
                        Matrix tranformedPose1 = chosen_poses[x_id_p]*Rt_mats_[cam_id_p];
                        Vector3 ray1 = pt - tranformedPose1.block(0,3,3,1);
                        double norm1 = ray1.norm();

                        ///Get the Ray2 between landmark and the current pose
                        Matrix tranformedPose2 = chosen_poses[x_id]*Rt_mats_[cam_id];
                        Vector3 ray2 = pt - tranformedPose2.block(0,3,3,1);
                        double norm2 = ray2.norm();

                        /// Compute the angle between the two rays
                        double cosTheta = ray1.dot(ray2)/(norm1*norm2);
                        double angle_deg = acos(cosTheta)*(180.0/3.141592653589793238463);
                        //if(x_id == 54){
                        ///cout<<pt<<endl;
                        ///cout<<chosen_poses[x_id].block(0,0,3,1)<<endl;
                        ///cout<<chosen_poses[x_id_p].block(0,0,3,1)<<endl;

                        //cout<<tranformedPose1<<endl;
                        //cout<<tranformedPose2<<endl;
                        //Get the extrisic parameters of the states
                        Matrix R1 = tranformedPose1.block(0,0,3,3).transpose();
                        Vector t1 = -1*R1*tranformedPose1.block(0,3,3,1);
                        Matrix R2 = tranformedPose2.block(0,0,3,3).transpose();
                        Vector t2 = -1*R2*tranformedPose2.block(0,3,3,1);

                        //COnvert the world point into the pose reference frames and apply K matrix
                        Vector pt_c1 = R1*pt+t1;
                        double invZ1 = 1.0/pt_c1(2);
                        double u_1 = K_mats_[cam_id_p](0,0)*pt_c1(0)*invZ1+K_mats_[cam_id_p](0,2);
                        double v_1 = K_mats_[cam_id_p](1,1)*pt_c1(1)*invZ1+K_mats_[cam_id_p](1,2);
                        float squareError1 = (u_1-u_p)*(u_1-u_p)+(v_1-v_p)*(v_1-v_p);

                        Vector pt_c2 = R2*pt+t2;
                        double invZ2 = 1.0/pt_c2(2);
                        double u_2 = K_mats_[cam_id](0,0)*pt_c2(0)*invZ2+K_mats_[cam_id](0,2);
                        double v_2 = K_mats_[cam_id](1,1)*pt_c2(1)*invZ2+K_mats_[cam_id](1,2);
                        float squareError2 = (u_2-u)*(u_2-u)+(v_2-v)*(v_2-v);


                        VLOG(1)<<"Squared Error 1 :"<<squareError1<<endl;
                        VLOG(1)<<"Square Error 2 :"<<squareError2<<endl;

                        //cout<<"u,v for x ID "<<x_id_p<<": "<<u_p<<","<<v_p<<"Predicted :"<<u_1<<","<<v_1<<endl;
                        //cout<<"u,v for x ID "<<x_id<<": "<<u<<","<<v<<"Predicted :"<<u_2<<","<<v_2<<endl;

                        //cout<<"--------------------------"<<endl;
                        // }


                        if(angle_deg > angleThresh){
                            VLOG(2)<<"Lid: "<<l_id<<"X ID: "<<x_id<<cam_id<<" Angle: "<<angle_deg<<endl;
                            u_coord.push_back(u);
                            v_coord.push_back(v);
                            pose_ids.push_back(x_id);
                            comp_cam_ids.push_back(cam_id);
                        }
                    }
                    else{
                        u_coord.push_back(u);
                        v_coord.push_back(v);

                        pose_ids.push_back(x_id);
                        comp_cam_ids.push_back(cam_id);

                    }

                }
                break;
            }

            default:
                cout<<"wrong type of line"<<endl;
        }


    }

    auto start_timer = high_resolution_clock::now();
    graph.print("current graph");
    initialValues.print("Initial Values");
    cout<<"Total Number of variables"<<initialValues.size()<<endl;
    cout<<"Size of the graph"<<graph.size()<<endl;
    ofstream os("graph1.dot");
    graph.saveGraph(os, initialValues);
    //cout << endl;
    LevenbergMarquardtParams params;
    params.orderingType = Ordering::METIS;
    LevenbergMarquardtOptimizer optimizer(graph, initialValues, params);
    Values result = optimizer.optimize();
    //Values result = DoglegOptimizer(graph, initialValues).optimize();
    result.print("Final results:\n");
    cout << "initial graph error = " << graph.error(initialValues) << endl;
    cout << "final graph error = " << graph.error(result) << endl;
    auto stop_timer = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_timer - start_timer);
    cout<<"time taken for optimization: "<<duration.count()<<endl;

    writeOptimized(result, tStamps, true);
}

void gtsam_bad_decoupled_mono(bool rigidEdges){
    ///Read the calibration params
    VLOG(1)<<"Reading calibration info"<<endl;
    read_kalibr_chain(FLAGS_calib_file);
    for (auto item: Rt_mats_){
        cout<<item<<endl;
    }
    vector<Cal3_S2::shared_ptr> K;
    for (auto item: K_mats_){
        Cal3_S2::shared_ptr tmpK(new Cal3_S2( item(0,0), item(1,1), 0 , item(0,2), item(1,2)));
        K.push_back(tmpK);
        cout<<item<<endl;
    }

    ifstream factor_file(FLAGS_factor_file.c_str());
    VLOG(1)<<"Reading factor graph info"<<endl;

    //Declare the graph and initial values
    Values initialValues;
    NonlinearFactorGraph graph;
    noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0); // pixel error in (x,y)

    //Read poses and ther initial values
    int num_poses;
    factor_file>>num_poses;
    int x_id;
    double stamp;
    char typ;
    MatrixRowMajor m(4,4);
    bool init_prior = true;
    bool init_prior_pose = true;

    int l_id;
    double X,Y,Z,u,v;
    int cam_id;
    //vector<int> pose_ids;
    //vector<int> comp_cam_ids;
    std::map<int, vector<int>> lm_cam_record;
    std::map<int, vector<Point2f>> uv_coords;
    std::map<int, Point3> lm_initial;
    /// this is to choose only poses with enough baseline
    double baseline = 0.3;
    double angleThresh = 0.5;
    std::map<int, MatrixRowMajor> chosen_poses;
    //std::map<int, int> insertedPoses;
    std::map<int, set<int> > insertedRigid;
    std::map<int, double> tStamps, tStamps_allcams;
    int cnt_lms =0, cnt_lms_removed=0;
    int x_id_prev= -1; /// this to maintain a god enoough baseline.
    MatrixRowMajor m_prev(4,4);
    // Read the projection factors between the poses and the landmarks
    while(factor_file>>typ){
        switch (typ){
            case 'x':
            {
                /// get the time stamp of the state
                factor_file>>stamp;
                /// get the state id
                factor_file>>x_id;
                /// get the pose of the state
                for (int i = 0; i < 16; i++) {
                    factor_file >> m.data()[i];
                }
                Pose3 pose(m);
                Vector diffVec = m_prev.col(3) - m.col(3);
                double diff = diffVec.norm();
                if(diff < baseline)
                    break;

                //cout<<"X id :"<<x_id<<endl;
                /// select only those poses which are separated
                chosen_poses[x_id]=m;
                tStamps[x_id] = stamp;
                x_id_prev = x_id;
                m_prev = m;
                break;
            }
            case 'l':
            {
                //if the poseid are > 2 dont insert
                if(lm_cam_record.size() >1){  /// if the landmark is atleast seen from two LF frames

                    /// find the intersection of the component cameras to add landmark
                    /// factors.
                    std::map<int, vector<int>>::iterator itr;
                    /*itr = lm_cam_record.begin();
                    vector<int> cam_vec_prev = itr->second;
                    ++itr;
                    for(; itr != lm_cam_record.end() ; ++itr){
                        vector<int> inter;
                        vector<int> cam_vec = itr->second;
                        std::set_intersection(cam_vec_prev.begin(),cam_vec_prev.end(), cam_vec.begin(), cam_vec.end(), back_inserter(inter));
                        cam_vec_prev = inter;
                    }
                    VLOG(1)<<"intersecting cameras for landmark "<<l_id<<":  ";
                    for(auto c : cam_vec_prev)
                        VLOG(1)<<c<<",";
                    VLOG(1)<< "\n";
                    if(cam_vec_prev.size() == 0 )
                        cnt_lms_removed++; */

                    vector<int> intersecting_poses;
                    vector<Point2f> uv_coords_chosen;
                    int max_freq_poses=1;
                    int comp_cam_chosen= -1;
                    for(itr = lm_cam_record.begin() ; itr != lm_cam_record.end(); ++itr){
                        int freq = itr->second.size();
                        if(max_freq_poses < freq){
                            max_freq_poses = freq;
                            intersecting_poses = itr->second;
                            comp_cam_chosen = itr->first;
                            uv_coords_chosen = uv_coords[comp_cam_chosen];
                        }
                    }

                    //VLOG(1)<<" component camera chosen for landmark : "<<l_id<<" is "<<comp_cam_chosen<<". poses : ";
                   // for(auto p: intersecting_poses)
                   //     VLOG(1)<<p<<", ";
                    if(comp_cam_chosen == -1){
                        /// we did not find a common camera. But, we will handle it later.
                        cnt_lms_removed++;
                    }
                    else{
                        /// chck the angular threshold////////////////////////////////////////////////////
                        /// Check parallax
                        ///Get the landmarks coordinates, prev pose id and u, v coordinates
                        Vector3 pt(X,Y,Z);
                        vector<double> u_coord, v_coord;
                        vector<int> poseids;
                        int prev_ind = 0;
                        for(int i = 1; i < max_freq_poses ; i++){

                            int x_id_p = intersecting_poses[prev_ind];
                            int x_id_n = intersecting_poses[i];

                            double u_p = uv_coords_chosen[prev_ind].x;
                            double v_p =  uv_coords_chosen[prev_ind].y;
                            double u_n = uv_coords_chosen[i].x;
                            double v_n =  uv_coords_chosen[i].y;

                            ///Get the Ray1 between landmark and the previous pose
                            Matrix tranformedPose1 = chosen_poses[x_id_p]*Rt_mats_[comp_cam_chosen];
                            Vector3 ray1 = pt - tranformedPose1.block(0,3,3,1);
                            double norm1 = ray1.norm();

                            ///Get the Ray2 between landmark and the current pose
                            Matrix tranformedPose2 = chosen_poses[x_id_n]*Rt_mats_[comp_cam_chosen];
                            Vector3 ray2 = pt - tranformedPose2.block(0,3,3,1);
                            double norm2 = ray2.norm();

                            /// Compute the angle between the two rays
                            double cosTheta = ray1.dot(ray2)/(norm1*norm2);
                            double angle_deg = acos(cosTheta)*(180.0/3.141592653589793238463);

                            //Get the extrisic parameters of the states
                            Matrix R1 = tranformedPose1.block(0,0,3,3).transpose();
                            Vector t1 = -1*R1*tranformedPose1.block(0,3,3,1);
                            Matrix R2 = tranformedPose2.block(0,0,3,3).transpose();
                            Vector t2 = -1*R2*tranformedPose2.block(0,3,3,1);

                            //COnvert the world point into the pose reference frames and apply K matrix
                            Vector pt_c1 = R1*pt+t1;
                            double invZ1 = 1.0/pt_c1(2);
                            double u_1 = K_mats_[comp_cam_chosen](0,0)*pt_c1(0)*invZ1+K_mats_[comp_cam_chosen](0,2);
                            double v_1 = K_mats_[comp_cam_chosen](1,1)*pt_c1(1)*invZ1+K_mats_[comp_cam_chosen](1,2);
                            float squareError1 = (u_1-u_p)*(u_1-u_p)+(v_1-v_p)*(v_1-v_p);

                            Vector pt_c2 = R2*pt+t2;
                            double invZ2 = 1.0/pt_c2(2);
                            double u_2 = K_mats_[comp_cam_chosen](0,0)*pt_c2(0)*invZ2+K_mats_[comp_cam_chosen](0,2);
                            double v_2 = K_mats_[comp_cam_chosen](1,1)*pt_c2(1)*invZ2+K_mats_[comp_cam_chosen](1,2);
                            float squareError2 = (u_2-u_n)*(u_2-u_n)+(v_2-v_n)*(v_2-v_n);

                            VLOG(1)<<"Squared Error 1 :"<<squareError1<<endl;
                            VLOG(1)<<"Square Error 2 :"<<squareError2<<endl;
                            VLOG(2)<<"Lid: "<<l_id<<" X ID pre: "<<x_id_p<<" X ID cur: "<<x_id_n<<" cam ID: "<<comp_cam_chosen<<" Angle: "<<angle_deg<<endl;
                            if(angle_deg > angleThresh and squareError1 <= 4 and squareError2 <= 4){
                                VLOG(2)<<"inserted"<<endl;
                                if(u_coord.size() == 0 and v_coord.size() ==0){
                                    u_coord.push_back(u_p);
                                    v_coord.push_back(v_p);
                                    poseids.push_back(x_id_p);
                                }
                                u_coord.push_back(u_n);
                                v_coord.push_back(v_n);
                                poseids.push_back(x_id_n);
                                prev_ind = i;
                            }

                        }

                        //////////////////////////////////////////////////////////////////////////////////
                        /// add the factors
                        if(u_coord.size() >1 and v_coord.size() >1){
                            Point3 lm(X,Y,Z);
                            ///Save the initial value of the mandmark
                            lm_initial[l_id] = lm;
                            /// if the prior is not added on the landmark, add the prior
                            if(init_prior){
                                init_prior=false;
                                noiseModel::Isotropic::shared_ptr point_noise = noiseModel::Isotropic::Sigma(3, 0.1);
                                graph.addPrior(Symbol('l',l_id),lm, point_noise );
                            }
                            /// add the initial value for the landmark
                            initialValues.insert(Symbol('l',l_id), lm);
                            VLOG(2)<<"Inserted Initial Value for Landmark ID: "<<l_id<<": ";
                            VLOG(2)<<lm<<endl;

                            for(int idx = 0 ; idx < u_coord.size() ; idx++ ){

                                ///Insert the state if it is not already inserted
                                ///if this is the first pose add a prior on the pose
                                int poseid = poseids[idx];
                                string s1 = to_string(poseid);
                                string s2 = to_string(comp_cam_chosen);
                                /// Concatenate both pose id and the cam id
                                string s = s1 + s2;
                                /// Convert the concatenated string
                                /// to integer
                                int state_id = stoi(s);
                                ///If it is the first time we are inserting this state
                                if(!initialValues.exists(Symbol('x',state_id))){
                                    Matrix tranformedPose = chosen_poses[poseid]*Rt_mats_[comp_cam_chosen];
                                    Pose3 pose(tranformedPose);
                                    tStamps_allcams[state_id] = tStamps[poseid];

                                    if(init_prior_pose){
                                        init_prior_pose= false;
                                        //noiseModel::Diagonal::shared_ptr  poseNoise = noiseModel::Diagonal::Sigmas((Vector(6)<<Vector3::Constant(0.1), Vector3::Constant(0.1)).finished());
                                        //graph.addPrior(Symbol('x', poseid),pose, poseNoise);
                                        graph.emplace_shared<NonlinearEquality<Pose3>>(Symbol('x',state_id), pose);
                                        VLOG(2)<<"Inserted Prior for X ID: "<<state_id<<endl;
                                    }

                                    initialValues.insert(Symbol('x',state_id), pose);
                                    if(rigidEdges){
                                        /// if the pose is being inserted for the firt time
                                        if(insertedRigid.find(poseid) == insertedRigid.end())
                                            insertedRigid[poseid] = set<int>({comp_cam_chosen});
                                        else
                                            insertedRigid[poseid].insert(comp_cam_chosen);
                                    }

                                    VLOG(2)<<"Inserted Initial Value for X ID: "<<state_id<<endl;
                                }

                                Point2 measurement(u_coord[idx],v_coord[idx]);
                                graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(measurement, measurement_noise, Symbol('x',state_id), Symbol('l',l_id), K[comp_cam_chosen]);
                                VLOG(2)<<"Inserted projection factor L ID: "<<l_id<<"X ID: "<<state_id<<endl;
                            }

                        }
                    }

                    lm_cam_record.clear();
                    uv_coords.clear();
                }
                else{

                    lm_cam_record.clear();
                    uv_coords.clear();

                }
                factor_file>>l_id>>X>>Y>>Z;
                cnt_lms++;
                break;
            }
            case 'e':{
                factor_file>>x_id>>cam_id>>u>>v;
                if (chosen_poses.find(x_id) != chosen_poses.end()){
                    /// make sure that the angle between the rays is not acute.

                    if(lm_cam_record.find(cam_id) != lm_cam_record.end()){
                        lm_cam_record[cam_id].push_back(x_id);
                        uv_coords[cam_id].push_back(Point2f(u, v));
                    }
                    else{
                        lm_cam_record[cam_id] = vector<int>({x_id});
                        uv_coords[cam_id] = vector<Point2f>({Point2f(u, v)});
                    }

                }
                break;
            }

            default:
                cout<<"wrong type of line"<<endl;
        }


    }

    // Now insert the rigid edges based on the information
    // of which states have which component cameras
    if(rigidEdges){
        noiseModel::Diagonal::shared_ptr  betweenNoise = noiseModel::Diagonal::Sigmas((Vector(6)<<Vector3::Constant(0.001), Vector3::Constant(0.001)).finished());
        std::map<int, std::set<int>>::iterator it;
        for(it = insertedRigid.begin() ; it != insertedRigid.end() ; ++it){
            int poseid = it->first;
            set<int> compcamids = it->second;
            set<int>::iterator it_set = compcamids.begin();
            int comp_cam_id_prev = *it_set;
            ++it_set;
            for( ; it_set != compcamids.end() ; ++it_set){

                Matrix poseDiff = Matrix::Identity(4,4);
                for(int ii=comp_cam_id_prev+1; ii <= *it_set ; ii++ ){
                    poseDiff =  poseDiff * Rt_mats_kalib_[ii];
                }
                Pose3 betweenPose(poseDiff);
                int prev_stateid = stoi(to_string(poseid) + to_string(comp_cam_id_prev));
                int state_id = stoi(to_string(poseid) + to_string(*it_set));
                graph.emplace_shared<BetweenFactor<Pose3>>(Symbol('x', prev_stateid), Symbol('x', state_id),  betweenPose, betweenNoise);
                VLOG(2) <<"Inserted Between factor for PoseId: "<<poseid<<" Between: "<<comp_cam_id_prev<<" and "<<*it_set<<endl;
                comp_cam_id_prev = *it_set;
            }

        }

    }

    cout<<"Total number of landmarks: "<<cnt_lms<<endl;
    cout<<"Number of Landmarks removed due to lack of intersecting cameras: "<<cnt_lms_removed<<endl;
    auto start_timer = high_resolution_clock::now();
    //graph.print("current graph");
    //initialValues.print("Initial Values");
    cout<<"Total Number of variables"<<initialValues.size()<<endl;
    cout<<"Size of the graph"<<graph.size()<<endl;
    ofstream os("graph1.dot");
    graph.saveGraph(os, initialValues);
    //cout << endl;
    LevenbergMarquardtParams params;
    params.orderingType = Ordering::METIS;
    LevenbergMarquardtOptimizer optimizer(graph, initialValues, params);
    Values result = optimizer.optimize();
    //Values result = DoglegOptimizer(graph, initialValues).optimize();
    result.print("Final results:\n");
    cout << "initial graph error = " << graph.error(initialValues) << endl;
    cout << "final graph error = " << graph.error(result) << endl;
    auto stop_timer = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_timer - start_timer);
    cout<<"time taken for optimization: "<<duration.count()<<endl;

    writeOptimized(result, tStamps, true);


    Values lm_optimized = result.filter<Point3>();
    double mean_diff=0, max_diff=0;
    for(Values::iterator it = lm_optimized.begin(); it != lm_optimized.end() ; ++it) {
        Symbol sym = it->key;
        int lid = sym.index();
        Point3 lm = lm_optimized.at<Point3>(sym);
        Point3 lm_prev = lm_initial[lid];
        double diff = (lm-lm_prev).norm();
        mean_diff+=diff;
        if(max_diff < diff)
            max_diff = diff;
        cout<<"lid diff "<<diff<<endl;
    }
    mean_diff/=  lm_optimized.size();
    cout<<"Mean correction for landmarks : "<<mean_diff<<", Max correction: "<<max_diff<<endl;

}

void gtsam_bad(int select_cam){
    //Read the calibration params
    ifstream calib_file(FLAGS_calib_file.c_str());
    VLOG(1)<<"Reading calibration info"<<endl;
    calib_file>>FOCAL_LENGTH_X>>FOCAL_LENGTH_Y>>CENTER_X>>CENTER_Y;
    VLOG(1)<<"fx :"<<FOCAL_LENGTH_X<<endl;
    VLOG(1)<<"fy :"<<FOCAL_LENGTH_Y<<endl;
    VLOG(1)<<"cx :"<<CENTER_X<<endl;
    VLOG(1)<<"cy :"<<CENTER_Y<<endl;
    ifstream factor_file(FLAGS_factor_file.c_str());
    VLOG(1)<<"Reading factor graph info"<<endl;

    Cal3_S2::shared_ptr K(new Cal3_S2(FOCAL_LENGTH_X, FOCAL_LENGTH_Y, 0 , CENTER_X, CENTER_Y));

    //Declare the graph and initial values
    Values initialValues;
    NonlinearFactorGraph graph;
    noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0); // pixel error in (x,y)

    //Read poses and ther initial values
    int num_poses;
    factor_file>>num_poses;
    int x_id;
    double stamp;
    char typ;
    MatrixRowMajor m(4,4);
    bool init_prior = true;
    bool init_prior_pose = true;

    int l_id;
    double X,Y,Z,u,v;
    int cam_id;
    vector<double> u_coord, v_coord;
    vector<int> pose_ids;
    /// this s to choose only poses with enough baseline
    double baseline = 0.3;
    double angleThresh = 0.5;
    std::map<int, MatrixRowMajor> chosen_poses;
    std::map<int, double> tStamps;
    int x_id_prev= -1; /// this to maintain a god enoough baseline.
    MatrixRowMajor m_prev(4,4);
    // Read the projection factors between the poses and the landmarks
    while(factor_file>>typ){
        switch (typ){
            case 'x':
            {   factor_file>>stamp;
                factor_file>>x_id;
                //get the pose of the state
                for (int i = 0; i < 16; i++) {
                    factor_file >> m.data()[i];
                }
                Pose3 pose(m);
                Vector diffVec = m_prev.col(3)-m.col(3);
                double diff = diffVec.norm();
                if(diff < baseline)
                    break;
                //if this is the first pose add a prior on the pose
                /*if(init_prior_pose){
                    init_prior_pose= false;
                    noiseModel::Diagonal::shared_ptr  poseNoise = noiseModel::Diagonal::Sigmas((Vector(6)<<Vector3::Constant(0.1), Vector3::Constant(0.1)).finished());
                    graph.addPrior(Symbol('x', x_id),pose, poseNoise);
                }
                //insert the initial value
                initialValues.insert(Symbol('x',x_id), pose); */
                //cout<<"X id :"<<x_id<<endl;
                chosen_poses[x_id]=m;
                tStamps[x_id] = stamp;
                x_id_prev = x_id;
                m_prev = m;
                break;
            }
            case 'l':
            {
                if(u_coord.size() >= 2){  /// if the landmark is atleast seen from two LF frames
                    /// add the factors
                    Point3 lm(X,Y,Z);

                    if(init_prior){
                        init_prior=false;
                        noiseModel::Isotropic::shared_ptr point_noise = noiseModel::Isotropic::Sigma(3, 0.1);
                        graph.addPrior(Symbol('l',l_id),lm, point_noise );
                    }
                    initialValues.insert(Symbol('l',l_id), lm);
                    VLOG(2)<<"Inserted Initial Value for Landmark ID: "<<l_id<<": ";
                    VLOG(2)<<lm<<endl;
                    for(int idx = 0 ; idx < u_coord.size() ; idx++ ){

                        //Insert the state if it is not already inserted
                        //if this is the first pose add a prior on the pose
                        int poseid = pose_ids[idx];
                        if(!initialValues.exists(Symbol('x',poseid))){
                            Pose3 pose(chosen_poses[poseid]);
                            if(init_prior_pose){
                                init_prior_pose= false;
                                //noiseModel::Diagonal::shared_ptr  poseNoise = noiseModel::Diagonal::Sigmas((Vector(6)<<Vector3::Constant(0.1), Vector3::Constant(0.1)).finished());
                                //graph.addPrior(Symbol('x', poseid),pose, poseNoise);
                                graph.emplace_shared<NonlinearEquality<Pose3>>(Symbol('x',poseid), pose);
                                VLOG(2)<<"Inserted Prior for X ID: "<<poseid<<endl;
                            }

                            initialValues.insert(Symbol('x',poseid), pose);
                            VLOG(2)<<"Inserted Initial Value for X ID: "<<poseid<<endl;
                        }

                        Point2 measurement(u_coord[idx],v_coord[idx]);
                        graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(measurement, measurement_noise, Symbol('x',pose_ids[idx]), Symbol('l',l_id), K);
                        VLOG(2)<<"Inserted projection factor L ID: "<<l_id<<"X ID: "<<poseid<<endl;
                    }
                    u_coord.clear();
                    v_coord.clear();
                    pose_ids.clear();
                }
                else{
                    u_coord.clear();
                    v_coord.clear();
                    pose_ids.clear();
                }
                factor_file>>l_id>>X>>Y>>Z;
                break;
            }
            case 'e':{
                factor_file>>x_id>>cam_id>>u>>v;
                if (cam_id == select_cam and chosen_poses.find(x_id) != chosen_poses.end()){
                    /// make sure that the angle between the rays is not acute.
                    if(u_coord.size() >= 1){
                        /// Check parallax
                        ///Get the landmarks coordinates, prev pose id and u, v coordinates
                        Vector3 pt(X,Y,Z);
                        int x_id_p = pose_ids[u_coord.size()-1];
                        double u_p = u_coord[u_coord.size()-1];
                        double v_p = v_coord[u_coord.size()-1];

                        ///Get the Ray1 between landmark and the previous pose
                        Vector3 ray1 = pt - chosen_poses[x_id_p].block(0,3,3,1);
                        double norm1 = ray1.norm();

                        ///Get the Ray2 between landmark and the current pose
                        Vector3 ray2 = pt - chosen_poses[x_id].block(0,3,3,1);
                        double norm2 = ray2.norm();

                        /// Compute the angle between the two rays
                        double cosTheta = ray1.dot(ray2)/(norm1*norm2);
                        double angle_deg = acos(cosTheta)*(180.0/3.141592653589793238463);
                        //if(x_id == 54){
                            ///cout<<pt<<endl;
                            ///cout<<chosen_poses[x_id].block(0,0,3,1)<<endl;
                            ///cout<<chosen_poses[x_id_p].block(0,0,3,1)<<endl;


                            //Get the extrisic parameters of the states
                            Matrix R1 = chosen_poses[x_id_p].block(0,0,3,3).transpose();
                            Vector t1 = -1*R1*chosen_poses[x_id_p].block(0,3,3,1);
                            Matrix R2 = chosen_poses[x_id].block(0,0,3,3).transpose();
                            Vector t2 = -1*R2*chosen_poses[x_id].block(0,3,3,1);

                            //COnvert the world point into the pose reference frames and apply K matrix
                            Vector pt_c1 = R1*pt+t1;
                            double invZ1 = 1.0/pt_c1(2);
                            double u_1 = FOCAL_LENGTH_X*pt_c1(0)*invZ1+CENTER_X;
                            double v_1 = FOCAL_LENGTH_Y*pt_c1(1)*invZ1+CENTER_Y;
                            float squareError1 = (u_1-u_p)*(u_1-u_p)+(v_1-v_p)*(v_1-v_p);

                            Vector pt_c2 = R2*pt+t2;
                            double invZ2 = 1.0/pt_c2(2);
                            double u_2 = FOCAL_LENGTH_X*pt_c2(0)*invZ2+CENTER_X;
                            double v_2 = FOCAL_LENGTH_Y*pt_c2(1)*invZ2+CENTER_Y;
                            float squareError2 = (u_2-u)*(u_2-u)+(v_2-v)*(v_2-v);


                            VLOG(1)<<"Squared Error 1 :"<<squareError1<<endl;
                            VLOG(1)<<"Square Error 2 :"<<squareError2<<endl;

                            //cout<<"u,v for x ID "<<x_id_p<<": "<<u_p<<","<<v_p<<"Predicted :"<<u_1<<","<<v_1<<endl;
                            //cout<<"u,v for x ID "<<x_id<<": "<<u<<","<<v<<"Predicted :"<<u_2<<","<<v_2<<endl;

                            //cout<<"--------------------------"<<endl;
                       // }


                        if(angle_deg > angleThresh){
                            VLOG(2)<<"Lid: "<<l_id<<"X ID: "<<x_id<<" Angle: "<<angle_deg<<endl;
                            u_coord.push_back(u);
                            v_coord.push_back(v);
                            pose_ids.push_back(x_id);
                        }
                    }
                    else{
                        u_coord.push_back(u);
                        v_coord.push_back(v);
                        pose_ids.push_back(x_id);
                    }

                }
                break;
            }

            default:
                cout<<"wrong type of line"<<endl;
        }


    }
    graph.print("current graph");
    initialValues.print("Initial Values");
    cout<<"Total Number of variables"<<initialValues.size()<<endl;
    cout<<"Size of the graph"<<graph.size()<<endl;
    ofstream os("graph1.dot");
    graph.saveGraph(os, initialValues);
    //cout << endl;
    LevenbergMarquardtParams params;
    params.orderingType = Ordering::METIS;
    LevenbergMarquardtOptimizer optimizer(graph, initialValues, params);
    Values result = optimizer.optimize();
    //Values result = DoglegOptimizer(graph, initialValues).optimize();
    result.print("Final results:\n");
    cout << "initial graph error = " << graph.error(initialValues) << endl;
    cout << "final graph error = " << graph.error(result) << endl;
    writeOptimized(result, tStamps, false);
}

int main(int argc, char** argv){

    //google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    gtsam_bad_decoupled_mono(true);
    //gtsam_bad_allcams(true);
    //gtsam_bad(0);

}