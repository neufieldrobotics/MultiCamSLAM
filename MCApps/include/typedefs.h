/**
 * THis is the deifnition file for all the parameters used for Light Field SLAM & Refocusing
 */

#ifndef DATA_TYPES
#define DATA_TYPES

#include "std_include.h"

using namespace std;
using namespace cv;

const double pi = 3.14159;

struct Settings {

    //General settings for the program
     //! Flag to use a GPU or not
    int use_gpu; // 1 for GPU
    //! Flag to use a debug mode or not
    int debug_mode; // 1 for debug mode
    //! Flag to evaluate or not
    int eval_mode; // 1 for evaluation
    int datagen_mode; // 1 to generate refocused image and depth image
    // the root path of the dataset
    string data_path;
    //! Path to calibration data file
    string calib_file_path;
    string calib_images_path;

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

    //! live streaming refocus or not. currently works only with ROS
    bool live_refocus;
    //! if the data is streamed in ROS
    bool is_ros;

    bool color_imgs;
    int ref_cam;
    //! SLAM settings file
    string frontend_params_file;
    //! LF Reconstructor params
    string reconstructor_params_path;

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


    /**
     * Refocus settings
     */
    //! Use Homography Fit (HF) method or not
    int hf_method; // 1 to use corner method
    //! 1 to calculate refocus map vs refocus homography
    bool calc_map;
    //! 1 to perform depth reconstruction
    bool calc_depth;
    int gpu_median_rec;
    int gpu_average_rec;


    /**
     * Depth estimation via stereo to checkl the quality of reconstruction
     *
     */
    bool depth_est;
    bool use_q;
    int algo ;
    int lcam;
    int rcam;

    // settings for the dataset reader which i stored on disk

    //! Flag indicating if data is in mtiff format files
    int mtiff; // 1 for using multipage tiffs
    //! Flag indicating if data is in mp4 videos
    int mp4; // 1 for using mp4 data
    //! Flag indicating if data is in binary format
    int binary; // 1 for using binary data
    //! Path to directory where images to be refocused are
    string images_path;

    // string frames;
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

/*! Settings container passed to DepthReconstructor constructor. */
struct depth_settings {

    bool use_q;
    int algo ;
    int lcam;
    int rcam;
    int debug_mode;

};

struct calibration_settings {

    //! Path to directory where input images / videos lie
    string images_path;
    //! Path to file where detected corners should be written
    string corners_file_path;
    //! Number of corners in grid (horizontal x vertical)
    Size grid_size;
    //! Physical size of grid in [mm]
    double grid_size_phys;

    //! Flag indicating if calibration refractive or not
    int refractive;
    //! Flag indicating if calibration data is in mtiff files or not
    int mtiff;
    //! Flag indicating if calibration data is in mp4 files or not
    int mp4;
    //! Flag indicating if radial distortion should be accounted for or not
    int distortion;
    //! Flag indicating if corners should be written to file or not
    int write_corners;

    //! Frames to skip between successive reads
    int skip;
    //! Number of the frame to start reading at
    int start_frame;
    //! Number of the frame to end reading at
    int end_frame;

    //! Time shift values for each camera
    vector<int> shifts;

    //! Flag indicating if calibration images should be resized or not
    int resize_images;
    //! Factor by which to resize calibration images
    double rf;

};



#endif
