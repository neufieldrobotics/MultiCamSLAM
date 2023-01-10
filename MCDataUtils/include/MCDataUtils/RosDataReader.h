//
// Created by Pushyami Kaveti on 9/16/19.
//
#pragma once
#include "DatasetReaderBase.h"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include <image_transport/image_transport.h>

using namespace cv;
using namespace std;

class RosDataReader : public DatasetReaderBase{

public:
    //constructor and destructor
    RosDataReader(ros::NodeHandle nh): nh_(nh) , it_(nh_){

    }
    ~RosDataReader(){

    }

    bool CAMCHAIN;
    //Inner class
    class callBackFunctor
    {

    private:
        int counter;
        int cam_ind;
        RosDataReader *re;
    public:
        bool got_frame;
        callBackFunctor(RosDataReader &obj, int in){
            this->re = &obj;
            this->counter=0;
            this->cam_ind= in;
            this->got_frame = false;
            cout<<"camera index in callbackfunctor :"<<this->cam_ind<<"\n";
        }

        int getcam_ind(){ return this->cam_ind ;}
        int getcounter(){ return this->counter ;}
        void CB(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& info_msg);
    };

    vector<cv::Mat> ros_imgs;
    //todo: only required temporarily to save and read segmentation results. We should do this in a better way
    vector<string> ros_img_seq_nums;
    bool grab_frame;
    ros::Time tStamp;

    void initialize(MCDataUtilSettings refocus_set);
    void loadNext(vector<cv::Mat>& imgs);
    void getNext(vector<cv::Mat>& imgs, double& timeStamp);
    void getNext(vector<cv::Mat>& imgs , vector<string>& segmaskImgs,double& timeStamp);

    void read_ros_data(MCDataUtilSettings settings);
    bool isDataLoaded();


private:
    //ROS variables
    ros::NodeHandle nh_;
    vector<string> cam_topics_;
    image_transport::ImageTransport it_;
    vector<image_transport::CameraSubscriber> cam_subs;
    vector<sensor_msgs::ImagePtr> img_msgs;
    vector<sensor_msgs::CameraInfo> cam_info_msgs;
    vector<callBackFunctor*> cbs;
};



