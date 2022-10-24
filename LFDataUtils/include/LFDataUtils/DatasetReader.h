//
// Created by Pushyami Kaveti on 8/29/19.
//

#ifndef PROJECT_DATASETREADER_H
#define PROJECT_DATASETREADER_H

#include "LFDataUtilParams.h"
#include "DatasetReaderBase.h"
#include "common_utils/serialization.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/filesystem.hpp>

using namespace std;

class DatasetReader : public DatasetReaderBase{

public:

    //constructor and destructor
    DatasetReader(){

    }
    ~DatasetReader(){

    }

    void initialize(LFDataUtilSettings refocus_set);
    void loadNext(vector<cv::Mat>& imgs);
    void getNext(vector<cv::Mat>& imgs, double& timeStamp);
    void getNext(vector<cv::Mat>& imgs, vector<string>& segmaskImgs, double& timeStamp);


    int MTIFF_FLAG;
    int MP4_FLAG;
    int BINARY_FLAG;
    int ASYNC_FLAG;
    int ALL_FRAME_FLAG;
    int RESIZE_IMAGES;
    int NUM_IMAGES = 0;
    int start_frame_;
    int end_frame_;
    int skip_frame_;
    vector<int> shifts_;
    double rf_;
    vector<string> cam_names_;
    vector<int> frames_;
    vector<vector<string>> seg_img_names_ , all_img_names_;

    vector< vector<Mat> > imgs;
    vector<double> tStamps_;
    bool imgs_read_;
    int current_index;

    // COnfiguration and datareading functions

    void read_vo_data( string path);
    void read_kalibr_data( string path);
    // DocString: read_imgs
    //! Read images when they are as individual files in separate folders
    void read_imgs(string path);
    // DocString: read_binary_imgs
    //! Read images when they are as individual files but in binary format
    void read_binary_imgs(string path);
    // DocString: read_imgs_mtiff
    //! Read images when they are in multipage TIFF files
   // void read_imgs_mtiff(string path);
    //! Read images when they are in mp4 files
    //! Read images when they are in mp4 files
    void read_imgs_mp4( string path);
};


#endif //PROJECT_DATASETREADER_H
