//
// Created by Pushyami Kaveti on 1/24/22.
//

// This is a starter application to run Monocular -SLAM

#include "LFSlam/FrontEndBase.h"
#include "LFSlam/MonoFrontEnd.h"
#include "LFSlam/MonoBackEnd.h"
#include "LFSlam/OpenGlViewer.h"
#include "LFDataUtils/RosDataReader.h"
#include "ParseSettings.h"
// #include "LFReconstruct/SegmentNet.h"
#include "LFDataUtils/CamArrayConfig.h"
#include "LFDataUtils/LFDataUtilParams.h"


#include "LiveViewer.h"
#include "LFDataUtils/DatasetReader.h"
#include "LFDataUtils/RosbagParser.h"

// OPENGV ////
#include <memory>
#include "LFSlam/time_measurement.hpp"
///////
#include <ros/ros.h>
#include <opencv2/core/eigen.hpp>

void handleKeyboardInput(MonoFrontEnd& frontend, MonoBackEnd& backend, SegmentNet &segnet , DatasetReaderBase &dataReader, OpenGlViewer &glViewer);

void process_frames(MonoFrontEnd& frontend, MonoBackEnd& backend);

bool updateData(MonoFrontEnd& frontend, SegmentNet &segnet, DatasetReaderBase &dataReader);

DEFINE_bool(fhelp, false, "show config file options");

DEFINE_string(config_file, "", "config file path");

using namespace cv;
using namespace std;
using namespace opengv;

int main(int argc , char** argv){
    // parse arguments
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr=1;

    //parse the settings in config file
    LFDataUtilSettings settings;
    parse_settings(FLAGS_config_file, settings, FLAGS_fhelp);

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

    // initialize segmentation object to detect the static parts
    //Segmentation object
    SegmentNet *segnet ;
    if(settings.use_segment){
        if (settings.read_segmask)
            segnet = new SegmentNet(datareader->num_cams_, settings.mask_type,3);
        else
            segnet = new SegmentNet(settings.seg_settings_path, datareader->num_cams_,3);
    }

    // create a camera configuration object with came intrinisc and extrinsic params...
    CamArrayConfig* cam_cfg = new CamArrayConfig(datareader->getKMats(),datareader->getDistMats(),datareader->getRMats(),\
    datareader->getTMats(), datareader->getKalibrRMats(), datareader->getKalibrTMats(), datareader->img_size_, datareader->num_cams_);

    // create the SLAM front end object

    FrontEndBase* slam_frontend;
    slam_frontend = new MonoFrontEnd (settings.frontend_params_file, cam_cfg->K_mats_[0], cam_cfg->dist_coeffs_[0], settings.debug_mode );
    // create a openGlViewer object
    OpenGlViewer* glViewer;
    glViewer = new OpenGlViewer(settings.frontend_params_file, slam_frontend);
    std::thread* viewerThread = new thread(&OpenGlViewer::goLive, glViewer);

    MonoBackEnd *slam_backend;
    slam_backend = new MonoBackEnd(settings.backend_params_file,cam_cfg->K_mats_[0], static_cast<MonoFrontEnd*>(slam_frontend));



    //IN a loop datareader->getNext() or datareader->getNextGPU(), refocus.updateImgs() or refocus.updateImgsGPU()
    // refocus.updateZMap() , optional processing , perform refocus and show in GUI
    ros::AsyncSpinner spinner(0); // Use max cores possible for mt
    if(settings.is_ros){
        spinner.start();
    }

    // This method runs the SLAM program in a loop for each new image.
    // user can give keyboard inputs to control the loop for obtaining next image,
    // visualizing and exiting
    handleKeyboardInput(*static_cast<MonoFrontEnd*>(slam_frontend), *slam_backend, *segnet,  *datareader, *glViewer);
    //slam_frontend->writeLogs();
    //slam_frontend->logFile_.close();
    return 0;
}

bool updateData(MonoFrontEnd& frontend, SegmentNet &segnet, DatasetReaderBase &dataReader) {
    vector<Mat> imgs, segMasks;
    vector<string> segmaskNames;
    double timeStamp;
    if(dataReader.settings.read_segmask){
        dataReader.getNext(imgs,segmaskNames, timeStamp);
    }else{
        dataReader.getNext(imgs, timeStamp);
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

    frontend.createFrame(imgs[0], segMasks[0], timeStamp);

    return true;
}
void process_frames(MonoFrontEnd& frontend, MonoBackEnd& backend){

    // process the LF snapshot. extract features, obtain intra matches
    //extract features
    frontend.processFrame();
    // track frame
    bool new_kf = frontend.trackMono();

    if(frontend.initialized_ == NOT_INITIALIZED){
        frontend.reset();
        return;
    }
    else if(frontend.initialized_ == INITIALIZED and new_kf){
        bool optimtize_ = backend.addKeyFrame();
        if(optimtize_){
            backend.optimizePosesLandmarks();
            backend.updateVariables();
        }
    }

    //reset frontend
    frontend.reset();
}

void handleKeyboardInput(MonoFrontEnd& frontend, MonoBackEnd& backend, SegmentNet &segnet, DatasetReaderBase &dataReader, OpenGlViewer &glViewer) {
    int seq = 0;
    //create live-viewer object
    LiveViewer gui;

    // initialize
    bool initialized = false;

    while (true) {

        if ( dataReader.settings.is_ros && ! ros::ok())
            return;
        int key = waitKey(10);
        // VLOG(4) << "Key press: " << (key & 255) << endl;
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
            //update the new Camera data to corresponding objects
            bool success = updateData(frontend, segnet, dataReader);
            if(success){
                process_frames(frontend, backend);
            } else
                continue;

        }
        // Visualize
        seq++;

    }
}

