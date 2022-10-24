//
// Created by Pushyami Kaveti on 6/6/20.
//

// This is a starter application to run LF -SLAM on the static portions of the scene using multi-view data


#include "LFSlam/FrontEnd.h"
#include "LFSlam/Backend.h"
#include "LFSlam/OpenGlViewer.h"
#include "LFDataUtils/RosDataReader.h"
#include "ParseSettings.h"
#include "LFReconstruct/SegmentNet.h"
#include "LFDataUtils/CamArrayConfig.h"
#include "LFDataUtils/LFDataUtilParams.h"
#include "LFReconstruct/LFReconstructParams.h"

#include "LiveViewer.h"
#include "LFDataUtils/DatasetReader.h"
#include "LFDataUtils/RosbagParser.h"
#include <chrono>
// OPENGV ////
#include <Eigen/Eigen>
#include <opengv/types.hpp>
#include <memory>
#include <opengv/relative_pose/NoncentralRelativeAdapter.hpp>
#include <opengv/sac_problems/relative_pose/NoncentralRelativePoseSacProblem.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/relative_pose/methods.hpp>
#include "LFSlam/time_measurement.hpp"
///////
#include <ros/ros.h>
#include <opencv2/core/eigen.hpp>

void handleKeyboardInput(FrontEnd& frontend, Backend& backend, SegmentNet &segnet , DatasetReaderBase &dataReader, OpenGlViewer &glViewer);

void process_frames(FrontEnd& frontend, Backend& backend);

bool updateData(FrontEnd& frontend, SegmentNet &segnet, DatasetReaderBase &dataReader);

DEFINE_bool(fhelp, false, "show config file options");

DEFINE_string(config_file, "", "config file path");

DEFINE_string(log_file, "pose_stats.txt", "log_file file path");
DEFINE_string(traj_file, "Tum_trajectory.txt", "trajectory file file path");

using namespace cv;
using namespace std;
using namespace opengv;
using namespace std::chrono;
float tracking_time=0.0;
float optim_time = 0.0;
float feat_xtract_time = 0.0;
int frame_counter=0;
int optim_cnt=0;


int main(int argc , char** argv){
    // parse arguments
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr=1;

    //parse the settings in config file
    //parse the settings in config file
    LFDataUtilSettings settings;
    LFReconstructSettings reconstructSettings;
    parse_settings(FLAGS_config_file, settings, reconstructSettings, FLAGS_fhelp);

    //INitialize get the data from ros bag using ros reader
    //read the dataset and calibration given settings for the refocus object
    DatasetReaderBase* datareader;
    // if ROS is enabled Initialize the ROS node and
    // Also set all the callbacks
    ros::init(argc, argv, "slam_node");
    ros::NodeHandle nh;
    if (settings.is_ros){
        // Initializing the ros node
        datareader = new RosDataReader(nh);
    }
    else{

        datareader = new DatasetReader();
    }
    datareader->initialize(settings);

    //create refocus object
    //Refocus refocus(settings);

    // initialize segmentation object to detect the static parts
    //Segmentation object
    SegmentNet *segnet ;
    if(settings.use_segment){
        if (settings.read_segmask)
            segnet = new SegmentNet(datareader->num_cams_, settings.mask_type,3);
        else
            segnet = new SegmentNet(settings.seg_settings_path, datareader->num_cams_,3);
    }

    // create a camera configuration object with came intrinisc and extrinsic params
    CamArrayConfig* cam_cfg = new CamArrayConfig(datareader->getKMats(),datareader->getDistMats(),datareader->getRMats(),\
    datareader->getTMats(), datareader->getKalibrRMats(), datareader->getKalibrTMats(), datareader->img_size_, datareader->num_cams_);

    // create the SLAM front end object

    FrontEnd *slam_frontend;
    slam_frontend = new FrontEnd(settings.frontend_params_file, *cam_cfg, settings.debug_mode);

    Backend *slam_backend;
    slam_backend = new Backend(settings.backend_params_file, *cam_cfg, slam_frontend);

    // create a openGlViewer object
    OpenGlViewer* glViewer;
    glViewer = new OpenGlViewer(settings.frontend_params_file, slam_frontend);
    std::thread* viewerThread = new thread(&OpenGlViewer::goLive, glViewer);

    //IN a loop datareader->getNext() or datareader->getNextGPU(), refocus.updateImgs() or refocus.updateImgsGPU()
    // refocus.updateZMap() , optional processing , perform refocus and show in GUI
    ros::AsyncSpinner spinner(0); // Use max cores possible for mt
    if(settings.is_ros){
        spinner.start();
    }

    // This method runs the SLAM program in a loop for each new image.
    // user can give keyboard inputs to control the loop for obtaining next image,
    // visualizing and exiting
    handleKeyboardInput(*slam_frontend, *slam_backend, *segnet,  *datareader, *glViewer);
    int fr_ind=0;
    /*for (auto& fr : slam_frontend->lfFrames){
        int poseid = fr->frameId;
        if(slam_backend->currentEstimate.exists(gtsam::Symbol('x', poseid))){
            try{
                gtsam::Matrix  covgtsam = slam_backend->isam.marginalCovariance(gtsam::Symbol('x', poseid));
                cv::eigen2cv(covgtsam, fr->cov);
                //VLOG(2)<<"Covariance"<<isam.marginalCovariance(gtsam::Symbol('x', poseid));
            }
            catch (IndeterminantLinearSystemException& e) {
                VLOG(2)<<"Exception occured in marginal covariance computation:"<<endl;
                fr->cov = Mat::eye(6,6, CV_64F);
            }
        }
        fr_ind++;
    }*/
    slam_frontend->writeLogs(FLAGS_log_file);
    slam_frontend->logFile_.close();
    slam_frontend->logFile2_.close();
    slam_frontend->writeTrajectoryToFile(FLAGS_traj_file, false);
    return 0;
}

bool updateData(FrontEnd& frontend, SegmentNet &segnet, DatasetReaderBase &dataReader) {
    vector<Mat> imgs, segMasks;
    vector<string> segmaskNames;
    double timeStamp;
    if(dataReader.settings.read_segmask){
        dataReader.getNext(imgs,segmaskNames, timeStamp);
    }else{
        dataReader.getNext(imgs, timeStamp);
    }
    if (imgs.empty()){
        VLOG(2)<<"Reached the end of images";
        return false;
    }


    if(dataReader.settings.use_segment){
        if(dataReader.settings.read_segmask) {
            segnet.updateSegmaskImgs(segmaskNames);
            segnet.updateImgs(imgs);
            segMasks = segnet.readSegmentationMasks();
            if (segMasks.size() < dataReader.num_cams_)
                return false;
        }
        else{
            //All image segmnetation
            segnet.updateImgs(imgs);
            segMasks = segnet.computeSegmentation_allimgs();
        }
    }
    else{

        for(int i =0; i <dataReader.num_cams_; i++)
            segMasks.push_back(Mat::zeros(dataReader.img_size_, CV_32FC1));
    }

    frontend.createFrame(imgs, segMasks, timeStamp);

    return true;
}
void process_frames(FrontEnd& frontend, Backend& backend){

    frame_counter++;
    // process the LF snapshot. extract features, obtain intra matches
    // and triangulate to get the 3D points
    auto startT = high_resolution_clock::now();
    frontend.processFrameNon();
    //frontend.processFrame();
    auto stopT = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stopT - startT);
    feat_xtract_time += duration.count();
    VLOG(0)<<"Total time taken for Computing intramatch features: "<<duration.count()<<"Average: " <<(feat_xtract_time/frame_counter)<<endl;
    //track thecurrent frame WRT the last keyframe
    // Visualize the estimated Pose
    startT = high_resolution_clock::now();
    bool new_kf = frontend.trackLF();
    stopT = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stopT - startT);
    tracking_time += duration.count();
    VLOG(0)<<"Total time taken for tracking LF Frame: "<<duration.count()<<"Average: " <<(tracking_time/frame_counter)<<endl;
    ////////////////////// BACK_END//////////////////
    /////////////////////////////////////////////////
    startT = high_resolution_clock::now();
    if(frontend.initialized_ == NOT_INITIALIZED){
        frontend.reset();
        return;
    }
    else if(frontend.initialized_ == INITIALIZED and new_kf){
        bool optimtize_ = false;
        if(backend.backendType == MULTI_RIGID or backend.backendType == MULTI){
             optimtize_ = backend.addKeyFrameMulti();
        }
        else if(backend.backendType == MONO ){
            optimtize_ = backend.addKeyFrame();
        }
        else{
            return;
        }
        if(optimtize_){
            optim_cnt++;
            backend.optimizePosesLandmarks();
            backend.updateVariables();
        }
    }
    stopT = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stopT - startT);
    optim_time+=duration.count();
    VLOG(0)<<"Total time taken for Back-end optimization: "<<duration.count()<<" Average: "<<(optim_time/optim_cnt)<<endl;

    //BACKEND - add the keyframe, intra matches and observations to a pose graph
    // initially do a sliding window optimization - window fo size 10

    // plot the trajectories
    //1. mono FE
    // 2. multi FE
    //3. mono BE
    // 4. multi-backend

    frontend.reset();

}

void handleKeyboardInput(FrontEnd& frontend, Backend& backend, SegmentNet &segnet, DatasetReaderBase &dataReader, OpenGlViewer &glViewer) {
    int seq = 1;
    //create live-viewer object
    LiveViewer gui;

    // initialize
    bool initialized = false;
    float time_per_frame = 0.0;
    while (true) {

        if ( dataReader.settings.is_ros && ! ros::ok())
            return;
        int key = waitKey(10);
        VLOG(4) << "Key press: " << (key & 255) << endl;
        if ((key & 255) != 255)
        {
            int condition = ((key & 255));
            switch (condition) {
                case 46:{

                    bool success = updateData(frontend, segnet, dataReader);
                    if(success){
                        process_frames(frontend, backend);

                    } else
                        continue;

                    break;
                }
                case 27: // finishing
                {
                    destroyAllWindows();
                    glViewer.requestFinish();
                    while(!glViewer.isFinished()){
                        usleep(5000);
                    }
                    //frontend.writeTrajectoryToFile("/home/auv/ros_ws/trajectory.txt", true);
                    //frontend.featureStats();
                    return ;
                }

                default: // code to be executed if n doesn't match any cases
                    cout << "INVALID INPUT" << endl;
            }
        }
        bool live = true; //Hard coded.
        if(live){
            auto startT = high_resolution_clock::now();
            //update the new Camera data to corresponding objects
            bool success = updateData(frontend, segnet, dataReader);
            if(success){
                process_frames(frontend, backend);
            } else{
                //continue;
                destroyAllWindows();
                glViewer.requestFinish();
                while(!glViewer.isFinished()){
                    usleep(5000);
                }
                return ;
            }
            auto stopT = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stopT - startT);
            time_per_frame += duration.count();
            VLOG(0)<<"Time taken per frame : "<<duration.count()<<" Average: "<<  (time_per_frame/seq)<<endl;


        }
        // Visualize
        seq++;

    }
}

// get the key points from the images and then use bag of words to group features together across images

// find cross image matches only between the grouped static features

// triangulate and get the 3D points

// now we have key points associated with camera view and the depth at those key points. This is our initialization

// For each new LF frame , we perform the same processing - kps, camera views, 3D points.

// find the Kp matches between two frames - avoid matching between same cameras to apply
// 1. 17-point algo- but this is time taking for refinement via ransac
// 2. or  5 -point between same camera points to essential but this is not absolute
// 3. or pnp to get absolute pose.

// optimization???

