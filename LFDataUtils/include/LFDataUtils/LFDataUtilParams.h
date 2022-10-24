//
// Created by Pushyami Kaveti on 7/1/21.
//

#pragma once
#include <string>
#include <vector>

using namespace std;

struct LFDataUtilSettings {


    //! Flag to use a debug mode or not
    int debug_mode; // 1 for debug mode
    // the root path of the dataset
    string data_path;
    string calib_file_path;
    string calib_images_path;
    string frontend_params_file;
    string backend_params_file;

    //! Flag indicating if calibration images should be resized or not
    int resize_images;
    //! Factor by which to resize calibration images
    double rf;

    //! Whether calibration file is out of Kalibr
    int kalibr;

    //! Whether to undistort images or not
    int undistort;

    //! Whether to use rad tan distortion model or not
    int radtan;
    //! if the data is streamed in ROS
    bool is_ros;
    bool imgs_read;
    int ref_cam;

    /**
   * Segmentation settings
   *
   */
    //! path to the segmentation masks if they are computed already
    string segmasks_path;
    //! Path to the segmentatrion model settings. we have bodypix or maskrcnn settings
    string seg_settings_path;
    //! 1 to use segmentaion masks on the images to refocus past them. otherwise all pixels are refocused
    bool use_segment;
    bool read_segmask;
    string mask_type;

    // settings for the dataset reader which i stored on disk
    //! Flag indicating if data is in mtiff format files
    int mtiff; // 1 for using multipage tiffs
    //! Flag indicating if data is in mp4 videos
    int mp4; // 1 for using mp4 data
    //! Flag indicating if data is in binary format
    int binary; // 1 for using binary data
    //! Path to directory where images to be refocused are
    string images_path;

    string frames;
    //! Flag indicating if all frames in the input data should be processed
    int all_frames;
    //! Start frame number (used if all_frames is set to 0)
    int start_frame;
    //! End frame number (used if all_frames is set to 0)
    int end_frame;
    //! Successive frames to skip (used if all_frames is set to 0)
    int skip;

    //! Time shift values for each camera
    vector<int> shifts;
    //! Whether images are async or not
    int async;

};

