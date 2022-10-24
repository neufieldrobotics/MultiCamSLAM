//
// Created by auv on 4/19/21.
//

#include "LFDataUtils/RosbagParser.h"


bool ready = true;

RosbagParser::RosbagParser(string bag_path, std::vector<std::string> topicnames ){
    bagPath = bag_path;
    topics = topicnames;
    latestTimeStamp =0 ;
    done_parse = false;
}

// Callback for synchronized messages
void RosbagParser::callback(const sensor_msgs::Image::ConstPtr &img1,
              const sensor_msgs::Image::ConstPtr &img2,
              const sensor_msgs::Image::ConstPtr &img3,
              const sensor_msgs::Image::ConstPtr &img4,
              const sensor_msgs::Image::ConstPtr &img5)
{
    ready=false;
    cv_bridge::CvImagePtr cv_ptr;
    //std::lock_guard<std::mutex> lk(mtx_bag);
    mtx.lock();
    try
    {
        latestTimeStamp = img1->header.stamp.toSec();
        latestImages.clear();
        cv_ptr = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::BGR8);
        Mat cvImg = cv_ptr->image;
        latestImages.push_back(cvImg.clone());

        cv_ptr = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::BGR8);
        cvImg = cv_ptr->image;
        latestImages.push_back(cvImg.clone());

        cv_ptr = cv_bridge::toCvCopy(img3, sensor_msgs::image_encodings::BGR8);
        cvImg = cv_ptr->image;
        latestImages.push_back(cvImg.clone());

        cv_ptr = cv_bridge::toCvCopy(img4, sensor_msgs::image_encodings::BGR8);
        cvImg = cv_ptr->image;
        latestImages.push_back(cvImg.clone());

        cv_ptr = cv_bridge::toCvCopy(img5, sensor_msgs::image_encodings::BGR8);
        cvImg = cv_ptr->image;
        latestImages.push_back(cvImg.clone());

        //copying done
        //cout<<"camera :"<<img1->header.seq<<"\n";
       // cout<<"camera :"<<img1->header.stamp<<"\n";
       // cout<<"camera :"<<img1->header.frame_id<<"\n";

    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    mtx.unlock();

}


void RosbagParser::parseBag(){

    rosbag::Bag bag;
    bag.open(bagPath, rosbag::bagmode::Read);

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    //setup the time synchronization
    // Set up fake subscribers to capture images
    BagSubscriber<sensor_msgs::Image> img1_sub, img2_sub,img3_sub,img4_sub,img5_sub;

    message_filters::TimeSynchronizer<sensor_msgs::Image,sensor_msgs::Image,sensor_msgs::Image, \
    sensor_msgs::Image , sensor_msgs::Image> sync(img1_sub, img2_sub,img3_sub,img4_sub,img5_sub, 1);
    sync.registerCallback(boost::bind(&RosbagParser::callback, this, _1, _2, _3, _4, _5));
    // Load all messages into our stereo dataset

    BOOST_FOREACH(rosbag::MessageInstance const m, view)
    {

        std::unique_lock<std::mutex> lck(mtx_bag);
        con.wait(lck,[]{return ready;});

        if (m.getTopic() == topics.at(0) || ("/" + m.getTopic() == topics.at(0)))
        {
            sensor_msgs::Image::ConstPtr img = m.instantiate<sensor_msgs::Image>();
            if (img != NULL)
                img1_sub.newMessage(img);
        }

        if (m.getTopic() == topics.at(1) || ("/" + m.getTopic() == topics.at(1)))
        {
            sensor_msgs::Image::ConstPtr img = m.instantiate<sensor_msgs::Image>();
            if (img != NULL)
                img2_sub.newMessage(img);
        }

        if (m.getTopic() == topics.at(2) || ("/" + m.getTopic() == topics.at(2)))
        {
            sensor_msgs::Image::ConstPtr img = m.instantiate<sensor_msgs::Image>();
            if (img != NULL)
                img3_sub.newMessage(img);
        }

        if (m.getTopic() == topics.at(3) || ("/" + m.getTopic() == topics.at(3)))
        {
            sensor_msgs::Image::ConstPtr img = m.instantiate<sensor_msgs::Image>();
            if (img != NULL)
                img4_sub.newMessage(img);
        }

        if (m.getTopic() == topics.at(4) || ("/" + m.getTopic() == topics.at(4)))
        {
            sensor_msgs::Image::ConstPtr img = m.instantiate<sensor_msgs::Image>();
            if (img != NULL)
                img5_sub.newMessage(img);
        }


    }
    done_parse= true;
    bag.close();

}

void RosbagParser::getImagesAt(vector<Mat>& imgs, double& timeStamp){

    imgs.clear();
    std::lock_guard<std::mutex> lk(mtx_bag);
    mtx.lock();
    if(latestImages.size() == topics.size() and latestTimeStamp >= timeStamp){
        //cout<<"timeStamp: "<<(latestTimeStamp-timeStamp)<<endl;
        for (auto& i : latestImages){
            imgs.push_back(i.clone());
        }
        latestImages.clear();
        timeStamp = latestTimeStamp;
    }
    if(done_parse)
        timeStamp = -1;

    mtx.unlock();
    ready=true;
    con.notify_one();
}