//
// Created by auv on 4/19/21.
//

#ifndef SRC_ROSBAGPARSER_H
#define SRC_ROSBAGPARSER_H

#include <boost/foreach.hpp>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/Image.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/message_instance.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/simple_filter.h>

using namespace std;
using namespace cv;

template <class M>
class BagSubscriber : public message_filters::SimpleFilter<M>
{
public:
    void newMessage(const boost::shared_ptr<M const>& msg)
    //void newMessage(const sensor_msgs::Image::ConstPtr& msg)
    {
        this->signalMessage(msg);
    }
};



class RosbagParser {
public:
    RosbagParser(string bag_path, std::vector<std::string> topicnames );
    void parseBag();
    void getImagesAt(vector<Mat>& imgs, double& timeStamp);
    void callback(const sensor_msgs::Image::ConstPtr &img1,
                  const sensor_msgs::Image::ConstPtr &img2,
                  const sensor_msgs::Image::ConstPtr &img3,
                  const sensor_msgs::Image::ConstPtr &img4,
                  const sensor_msgs::Image::ConstPtr &img5);


    string bagPath;
    bool done_parse;
    rosbag::Bag bag;
    vector<string> topics;
    vector<Mat> latestImages;
    double latestTimeStamp;
    std::mutex mtx;
    std::mutex mtx_bag;
    std::condition_variable con;


};


#endif //SRC_ROSBAGPARSER_H
