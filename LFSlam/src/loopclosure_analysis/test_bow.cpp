//
// Created by Pushyami Kaveti on 4/22/21.
//

//read the rosbag or read images from folder and select the required images

// create the Bow Vector for LF based and normal images.
// get the bag of words vector and compare
#include "ParseSettings.h"
#include "LFReconstruct/SegmentNet.h"
#include "LFSlam/FrontEnd.h"
#include "LFDataUtils/CamArrayConfig.h"
#include "LFDataUtils/RosbagParser.h"
#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <DBoW2/DBoW2.h>
#include "DLoopDetector.h"
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DVision/DVision.h>
#include <opencv2/sfm/projection.hpp>
#include <opencv2/sfm/triangulation.hpp>
#include "ParseSettings.h"


using namespace std;
using namespace cv;
using namespace DLoopDetector;
using namespace DBoW2;


// entry point to the program
// we need bag file,
DEFINE_string(bagfile, "", "rosbag path");
DEFINE_string(config_file, "", "config_file path");
DEFINE_bool(debug, false, "Debug mode");

std::vector<std::string> topics;
Size img_size;
vector<Mat> R_mats_, t_vecs_, dist_coeffs_, K_mats_, P_mats_;
int num_cams_;

void read_settings(LFDataUtilSettings settings){
    bool CAMCHAIN=true;
    string path = settings.calib_file_path;
    LOG(INFO) << "Reading calibration data from ROS ..."<<path<<endl;
    VLOG(1)<<"segmask path"<<settings.segmasks_path<<endl;

    FileStorage fs(path, FileStorage::READ);
    FileNode fn = fs.root();

    FileNodeIterator fi = fn.begin(), fi_end = fn.end();
    int i=0;
    for (; fi != fi_end; ++fi, i++) {

        FileNode f = *fi;
        if (f.name().find("cam",0) == string::npos)
            break;
        string cam_name; f["rostopic"]>>cam_name;

        topics.push_back(cam_name);
        VLOG(2)<<"Camera "<<i<<" topic: "<<topics[i]<<endl;

        // READING CAMERA PARAMETERS from here coz its only one time now due to static camera array
        // in future will have to subscribe from camera_info topic
        // Reading distortion coefficients
        vector<double> dc;
        Mat_<double> dist_coeff = Mat_<double>::zeros(1,5);
        f["distortion_coeffs"] >> dc;
        if(settings.radtan){
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
            if (((int)ims[0] != img_size.width) || ((int)ims[1] != img_size.height))
                LOG(FATAL)<<"Resolution of all images is not the same!";
        } else {
            img_size = Size((int)ims[0], (int)ims[1]);
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

        R_mats_.push_back(R);
        t_vecs_.push_back(t);
        dist_coeffs_.push_back(dist_coeff);
        K_mats_.push_back(K_mat);
        P_mats_.push_back(P);

    }

    num_cams_ = i;
}

void filterMatches( std::vector<array<int, 5>>& matches_map, LightFieldFrame* lf_frame,
                    vector<cv::KeyPoint>& keys, vector<Mat>& descriptors){
    Mat P_1 = build_Rt(lf_frame->camconfig_.R_mats_[1], lf_frame->camconfig_.t_mats_[1]);
    std::vector<array<int, 5> >::iterator matches_iter;
    int ind=0;
    // iterate through all the intramatches
    for(matches_iter = matches_map.begin() ;  matches_iter!= matches_map.end(); ++matches_iter, ++ind ){
        array<int, 5> temp = *matches_iter;

        int first=-1, last=-1; // initialize this with ref camera, since that is the index which we will be avoiding
        vector<Mat> descs;
        vector<int> view_inds;
        int num_views=0;
        int c=0;
        for(auto& featInd :temp){
            //print the intra matches so that we know.
            if(featInd != -1 and c==0) // this is to avoid matches containing reference frame.
                break;
            if(featInd != -1){
                // if it is a valid index
                Point2f p = lf_frame->image_kps[c][featInd].pt;
                if (lf_frame->segMasks[c].at<float>(p.y, p.x) < 0.7){
                    descs.push_back(lf_frame->image_descriptors[c][featInd]);
                    view_inds.push_back(c);
                    num_views++;
                }
            }
            c = c+1;
        }

        //get the disparity of those points WRT to the biggest baseline it is visible in
        if(num_views>1){

            ///// Traingulation using  opencv sfm and normalized coordinates///////////
            vector<Mat> PJs;
            std::vector<Mat_<double> >  xx;
            for(int ii=0; ii<num_views ; ii++){
                int cur_view_ind= view_inds[ii];
                Mat Rt = build_Rt(lf_frame->camconfig_.R_mats_[cur_view_ind], lf_frame->camconfig_.t_mats_[cur_view_ind]);
                Mat P1 = cv::Mat_<double>(Rt);
                PJs.push_back(P1);
                Mat_<double> x1(2, 1);
                x1(0, 0) = (lf_frame->image_kps[cur_view_ind][temp[cur_view_ind]].pt.x - lf_frame->camconfig_.K_mats_[cur_view_ind].at<double>(0,2))/lf_frame->camconfig_.K_mats_[cur_view_ind].at<double>(0,0);
                x1(1, 0) = (lf_frame->image_kps[cur_view_ind][temp[cur_view_ind]].pt.y - lf_frame->camconfig_.K_mats_[cur_view_ind].at<double>(1,2)) / lf_frame->camconfig_.K_mats_[cur_view_ind].at<double>(1,1);
                xx.push_back(x1.clone());
            }

            cv::Mat pt3d_sfm;
            cv::sfm::triangulatePoints(xx, PJs, pt3d_sfm);
            //////////////////////////////////////////////////////////////////////////
            //update the support point
            Mat projected = lf_frame->camconfig_.K_mats_[0] * pt3d_sfm;
            double expected_x = projected.at<double>(0,0) / projected.at<double>(2,0);
            double expected_y = projected.at<double>(1,0) / projected.at<double>(2,0);
            double inv_depth = 1.0/ projected.at<double>(2,0);
            //from multi-cam elas. just for comparison
            double base_0 = -1*P_1.at<double>(0, 3);
            double f_0 =  lf_frame->camconfig_.K_mats_[0].at<double>(0, 0);
            inv_depth = f_0 * base_0 / (double) pt3d_sfm.at<double>(2,0);
            // if this point projects on to a dynamic object in refrence frame
            // then get this intra match
            if(lf_frame->segMasks[0].at<float>((int)expected_y, (int)expected_x) > 0.7){
                if(inv_depth > 6 and inv_depth < 100){
                    //collect all the valid intra matches
                    keys.push_back(KeyPoint(expected_x,expected_y, 1.0 ));
                    descriptors.insert(descriptors.end(), descs.begin(), descs.end());
                }
            }

        }

    }

}

void extractFeaturesMono(FrontEnd* frontend,vector<Mat>& images, vector<Mat>& segMasks,
                         vector<cv::KeyPoint>& keys, vector<FORB::TDescriptor>& descriptors){

    cv::Mat descs;
    Mat img = images[0].clone();
    multiply(img, 255, img);
    img.convertTo(img,CV_8U);
    if (img.channels() == 3){
        Mat imgGray;
        cvtColor(img,imgGray , CV_BGR2GRAY);
        img = imgGray;
    }
    Mat undistImg;
    cv::undistort(img, undistImg, frontend->camconfig_.K_mats_[0], frontend->camconfig_.dist_coeffs_[0] );
    vector<cv::KeyPoint> keys_;
    (*frontend->orBextractor)(undistImg,cv::Mat(),  keys_, descs);
    descriptors.clear();
    keys.clear();
    descriptors.reserve(descs.rows);
    for (int j=0;j<descs.rows;j++){
        Point2f p = keys_[j].pt;
        if(segMasks[0].at<float>(p.y, p.x) < 0.7){
            keys.push_back(keys_[j]);
            descriptors.push_back(descs.row(j));
        }
    }


}

void extractFeatures(FrontEnd* frontend,vector<Mat>& images, vector<Mat>& segMasks,
                     vector<cv::KeyPoint>& keys, vector<FORB::TDescriptor>& descriptors){

    LightFieldFrame* fr = new LightFieldFrame(images, segMasks, frontend->orb_vocabulary,
                                              frontend->orBextractor, frontend->orBextractors, frontend->camconfig_,
                                              frontend->current_frameId++,0.0,false);
    fr->extractFeaturesParallel();
    fr->parseandadd_BoW();
    /////////////// IF INTRA MATCHES nEED TO BE CALCULATED//////////////////
    std::vector<array<int, 5>> matches_map;
    vector<DBoW2::NodeId > words_;
    fr->computeIntraMatches(matches_map, words_);
    Mat img_draw, img_tmp;

    descriptors.clear();
    keys.clear();
    // filter out the intra matches and extract only those that are
    // 1. not seen in reference camera as we extract them below
    // 2. seen in other camera and reproject on to the dynamic portions of the reference camera
    filterMatches( matches_map,fr, keys, descriptors);
    int dynamic_features = keys.size() ;
    if(FLAGS_debug){
        img_tmp = images[0] * 255.0;
        img_tmp.convertTo(img_tmp, CV_8U);
        cv::drawKeypoints(img_tmp, keys, img_draw, Scalar(0,0,200));

    }
    // push back all the key points and descriptors in refrence camera
    // that belong to static portions
    vector<cv::KeyPoint> keys_static;
    for(int j=0; j <fr->image_descriptors[0].size(); j++){
        Point2f p = fr->image_kps[0][j].pt;
        if(fr->segMasks[0].at<float>(p.y, p.x) < 0.7){
            keys.push_back(fr->image_kps[0][j]);
            keys_static.push_back(fr->image_kps[0][j]);
            descriptors.push_back(fr->image_descriptors[0][j]);
        }
    }

    if(FLAGS_debug){
        cv::drawKeypoints(img_draw, keys_static, img_draw, Scalar(255,0,0));
        cv::imshow("image", img_draw);
        cv::waitKey(0);

        cv::drawKeypoints(img_tmp, fr->image_kps[0], img_tmp);
        cv::imshow("image original", img_tmp);
        cv::waitKey(0);

    }

    int static_features = keys.size() - dynamic_features;


    cout<<"total number of keypoints in ref image : "<<fr->image_descriptors[0].size()<<endl;
    cout<<"total number of static features in ref image : "<<static_features<<endl;
    cout<<"total number of recovered features in dynamic region in ref image : "<<dynamic_features<<endl;

    //all_descriptors.insert(all_descriptors.end(), currentFrame->image_descriptors[i].begin(), currentFrame->image_descriptors[i].end());

}


void extractFeaturesAll(FrontEnd* frontend,vector<Mat>& images, vector<Mat>& segMasks,
                     vector<vector<cv::KeyPoint>>& keys, vector<FORB::TDescriptor>& descriptors){

    LightFieldFrame* fr = new LightFieldFrame(images, segMasks, frontend->orb_vocabulary,
                                              frontend->orBextractor, frontend->orBextractors, frontend->camconfig_,
                                              frontend->current_frameId++,0.0,false);
    fr->extractFeaturesParallel();
    fr->parseandadd_BoW();
    descriptors.clear();
    keys.clear();

    for(int i =0 ; i < fr->num_cams_; i++){
        vector<KeyPoint> keys_single;
        for(int j=0; j <fr->image_descriptors[i].size(); j++){
            Point2f p = fr->image_kps[i][j].pt;
            if(fr->segMasks[i].at<float>(p.y, p.x) < 0.7){
                keys_single.push_back(fr->image_kps[i][j]);
                descriptors.push_back(fr->image_descriptors[i][j]);
            }
        }
        keys.push_back(keys_single);
        //all_descriptors.insert(all_descriptors.end(), currentFrame->image_descriptors[i].begin(), currentFrame->image_descriptors[i].end());
    }

}


void run(RosbagParser& ros_parser, LFDataUtilSettings& settings, SegmentNet* segnet, FrontEnd* slam_frontend ){

    // Set loop detector parameters
    typename OrbLoopDetector::Parameters params(img_size.height, img_size.width, 1);

    // We are going to change these values individually:
    params.use_nss = true; // use normalized similarity score instead of raw score
    params.alpha = 0.3; // nss threshold
    params.k = 3; // a loop must be consistent with 1 previous matches
    params.geom_check = GEOM_NONE; // use direct index for geometrical checking
    params.di_levels = 4; // use two direct index levels

    // Initiate loop detector with the vocabulary
    cout << "Processing sequence..." << endl;
    OrbLoopDetector detector(*(slam_frontend->orb_vocabulary), params);
    // we can allocate memory for the expected number of images
    //detector.allocate(1000);
    //// VISUALIZATION VARIABLES ////////
    vector<DetectionResult> results_;

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////  Get the images one by one in a loop and process//////////////
    ///////////////////////////////////////////////////////////////////////////////
    vector<Mat> images;
    // loop to get the images
    int sample_ind =0;
    int cnt_processed =0;
    int loops_detected = 0;
    do {

        //get the images
        double tStamp = 0;
        ros_parser.getImagesAt(images, tStamp);
        if (tStamp == -1 )
            //end of bag reached so, exit
            break;
        if(images.size() < num_cams_)
            continue;
        sample_ind++;
        if(sample_ind%10 != 0)
            continue;
        cnt_processed++;

        //get segmentation ma sks and update images to their respective classes
        vector<Mat> imgs_float;
        for(auto& im_tmp: images){
            Mat imgg;
            //cvtColor(ros_imgs[cam_ind], imgg, CV_BGR2GRAY);
            im_tmp.convertTo(imgg, CV_32F);
            imgg /= 255.0;
            imgs_float.push_back(imgg.clone());
        }
        vector<Mat>  segMasks;
        vector<string> segmaskNames;
        if(settings.use_segment){
            if(settings.read_segmask) {
                for(int i =0; i <num_cams_ ; i++){
                    string name = settings.segmasks_path+"/cam"+to_string(i) +"/"+ to_string(tStamp)+"."+settings.mask_type;
                    segmaskNames.push_back(name);
                }
                segnet->updateSegmaskImgs(segmaskNames);
                segnet->updateImgs(imgs_float);
                segMasks = segnet->readSegmentationMasks();
                assert (segMasks.size() == imgs_float.size());
            }
            else{
                //All image segmnetation
                segnet->updateImgs(imgs_float);
                segMasks = segnet->computeSegmentation_allimgs();
            }
        }
        else{
            for(int i =0; i <num_cams_; i++)
                segMasks.push_back(Mat::zeros(img_size, CV_32FC1));
        }

        // Find the key points and descriptors
        vector<vector<cv::KeyPoint>> keys;
        vector<cv::KeyPoint> keysMono;
        vector<FORB::TDescriptor> descriptors;
        extractFeatures(slam_frontend, imgs_float, segMasks, keysMono,descriptors);
        //extractFeaturesMono(slam_frontend, imgs_float, segMasks, keysMono,descriptors);

        //visualization of images if needed
        if(FLAGS_debug) {
            // show image
            cv::Mat outimg;
            cv::drawKeypoints(images[0], keysMono, outimg);
            cv::imshow("image", outimg);
            cv::waitKey(50);
            //cv::imshow("segment image", segMasks[0]);
            //cv::waitKey(0);

        }

        cout<<"Current image:"<<cnt_processed<<" timestamp : "<<to_string(tStamp)<<endl;
        // cout<<"Number of Keypoints : "<<keys.size()<<endl;
        cout<<"Number of descriptors : "<<descriptors.size()<<endl;

        // add image to the collection and check if there is some loop
        DetectionResult result;
        detector.detectLoop(keysMono, descriptors, to_string(tStamp), result);
        //detector.findTopN(keysMono, descriptors, to_string(tStamp), result);
        results_.push_back(result);
        if(result.detection())
        {
            cout << "- Loop found with image " << result.match << "!"
            << endl;
            vector<string> imgNames = detector.getFileNames();
            cout<<"image name : "<<imgNames.at(result.match)<<endl;
            loops_detected++;
        }
        /*else{
            cout << "- No loop: ";
            switch(result.status)
            {
                case CLOSE_MATCHES_ONLY:
                    cout << "All the images in the database are very recent" << endl;
                    break;
                case NO_DB_RESULTS:
                    cout << "There are no matches against the database (few features in"
                                " the image?)" << endl;
                    break;
                case LOW_NSS_FACTOR:
                    cout << "Little overlap between this image and the previous one"
                             << endl;
                    break;

                case LOW_SCORES:
                    cout << "No match reaches the score threshold (alpha: " <<
                             params.alpha << ")" << endl;
                    break;

                case NO_GROUPS:
                    cout << "Not enough close matches to create groups. "
                             << "Best candidate: " << result.match << endl;
                    break;

                case NO_TEMPORAL_CONSISTENCY:
                    cout << "No temporal consistency (k: " << params.k << "). "
                             << "Best candidate: " << result.match << endl;
                    break;

                case NO_GEOMETRICAL_CONSISTENCY:
                    cout << "No geometrical consistency. Best candidate: "
                             << result.match << endl;
                    break;

                default:
                    break;
            }
        }

        cout << endl; */

    }while (true);

    //Run the visualization here
    // create heatmap of the  similarity scores
    Mat res_map = Mat::zeros(results_.size(), results_.size(), CV_64F);
    for(auto& res: results_){
        int query = res.query;
        int match = res.match;
        if(query != match){
            res_map.at<double>(query , match) = res.score;
        }

    }
    Mat norm_img, norm_img_color;
    cv::normalize(res_map, norm_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cvtColor(vdisp, vdisp_color, CV_GRAY2BGR);
    cv::applyColorMap(norm_img, norm_img_color, cv::COLORMAP_JET);
    namedWindow("Disparity_planes", 0);
    resizeWindow("Disparity_planes", res_map.size().width*8, res_map.size().height*8);
    cv::imshow("Disparity_planes", norm_img_color);
    cv::waitKey(0);

}


int main(int argc , char** argv) {

    ///////////////////////////////////////////////////////////////////////
    ///////////////  INITIALIZAING ALL THE REQUIRED OBJECTS///////////////
    //////////////////////////////////////////////////////////////////////
    // parse arguments
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // Initializing the ros node
    ros::init(argc, argv, "slam_node");
    ros::NodeHandle nh;

    //parse the settings in config file
    LFDataUtilSettings settings;
    parse_settings(FLAGS_config_file, settings, false);
    read_settings(settings);

    RosbagParser ros_parser(FLAGS_bagfile , topics );
    std::thread thread(&RosbagParser::parseBag, &ros_parser);

    SegmentNet *segnet ;
    if(settings.use_segment){
        if (settings.read_segmask)
            segnet = new SegmentNet(ros_parser.topics.size(), settings.mask_type,3);
        else
            segnet = new SegmentNet(settings.seg_settings_path, ros_parser.topics.size(),3);
    }

    // create a camera configuration object with came intrinisc and extrinsic params
    CamArrayConfig* cam_cfg = new CamArrayConfig(K_mats_,dist_coeffs_,R_mats_,\
    t_vecs_, img_size, num_cams_);
    FrontEnd *slam_frontend;
    slam_frontend = new FrontEnd(settings.frontend_params_file, NULL, *cam_cfg, settings.debug_mode);


    run(ros_parser,settings, segnet,slam_frontend);

    thread.join();


}