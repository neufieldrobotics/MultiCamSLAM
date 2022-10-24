#include "ParseSettings.h"

using namespace cv;
using namespace std;

void programOptions(po::variables_map& vm, string filename, bool h){
    namespace po = boost::program_options;

    po::options_description desc("Allowed config file options");
    desc.add_options()
            ("use_gpu", po::value<int>()->default_value(0), "ON to use GPU")
            ("debug_mode", po::value<int>()->default_value(0), "ON to use debug mode")
            ("eval_mode", po::value<int>()->default_value(0), "ON to use evaluation mode")
            ("datagen_mode",  po::value<int>()->default_value(0), "ON to write refocused and depth images to disk")
            ("data_path", po::value<string>()->default_value(""), "data path contaning imgs, segmasks, calibration file to use")
            ("calib_file_path", po::value<string>()->default_value("camchain.yaml"), "calibration file to use")
            ("calib_images_path", po::value<string>()->default_value(""), "calibration images path with respective cam folders")
            ("resize_images", po::value<int>()->default_value(0), "ON to resize all input images")
            ("rf", po::value<double>()->default_value(1.0), "Factor to resize input images by")
            ("kalibr", po::value<int>()->default_value(0), "ON to use Kalibr calibration output")
            ("undistort", po::value<int>()->default_value(0), "ON to undistort images")
            ("radtan", po::value<int>()->default_value(0), "ON to use rad tan distortion model")
            ("ros", po::value<int>()->default_value(0), "ON if running using ROS. Otherwise, the images are read from folders.")
            ("imgs_read", po::value<int>()->default_value(0), "ON if we ould lilke to read the images from disk ahead of time")
            ("live_refocus", po::value<int>()->default_value(0), "live streaming refocus or not. currently works only with ROS")
            ("color_imgs", po::value<int>()->default_value(0), "1 to specify that image are RGB")
            ("ref_cam", po::value<int>()->default_value(0), "index of the reference camera for refocusing")
            ("frontend_params_file", po::value<string>()->default_value("lf_frontend.yaml"), "params for frontend file to use")
            ("backend_params_file", po::value<string>()->default_value("lf_backend.yaml"), "params for backend file to use")
            ("reconstructor_params_path", po::value<string>()->default_value("recon_params.yaml"), "params for Light Field Reconstruction file to use")
            ("segmasks_path", po::value<string>()->default_value("segmasks"), "segmentation masks file to use")
            ("seg_settings_file", po::value<string>()->default_value("SegmentSettings.yaml"), "Segmentation CNN settings file to use")
            ("use_segment", po::value<int>()->default_value(0), "1 to run segmentation on the images")
            ("read_segmask", po::value<int>()->default_value(0), "1 to read the precomputed segmentation on the images")
            ("mask_type", po::value<string>()->default_value("bmp"), "Segmentation mask either binary (saved as an image) or continous (saved as a csv) or npy")

            ("hf_method", po::value<int>()->default_value(0), "ON to use HF method")
            ("calc_map", po::value<int>()->default_value(0), "1 to calculate refocus map vs refocus homography")
            ("calc_depth", po::value<int>()->default_value(0), "1 to perform depth reconstruction")
            ("gpu_median_rec", po::value<int>()->default_value(0), "1 ro specify that use median refocusing with gpu")
            ("gpu_average_rec", po::value<int>()->default_value(0), "1 ro specify that use average refocusing with gpu")

            ("depth_est", po::value<int>()->default_value(0), "1 to specify that depth reconstruction needs to be performed")
            ("use_q", po::value<int>()->default_value(0), "ON to use Q matrix for depth reconstruction")
            ("depth_algo", po::value<int>()->default_value(1), "1 to use LIBELAS , 2 for BLOCK MATCHING method")
            ("lcam_index", po::value<int>()->default_value(0), "index of left camera for depth reconstruction")
            ("rcam_index", po::value<int>()->default_value(1), "index of right camera for depth reconstruction")

            ("mtiff", po::value<int>()->default_value(0), "ON if data is in tiff files")
            ("mp4", po::value<int>()->default_value(0), "ON if data is in mp4 files")
            ("binary", po::value<int>()->default_value(0), "ON if data is in binary format")



            ("images_path", po::value<string>()->default_value(""), "path where data is located")
            ("frames", po::value<string>()->default_value(""), "Array of values in format start, end, skip")
            ("all_frames", po::value<int>()->default_value(1), "ON to process all frames in a multipage tiff file")
            ("start_frame", po::value<int>()->default_value(0), "first frame in range of frames to process")
            ("end_frame", po::value<int>(), "last frame in range of frames to process")
            ("skip", po::value<int>(), "Successive frames to skip (used if all_frames is set to 0)")
            ("shifts", po::value<string>()->default_value(""), "path where data is located")
            ("async", po::value<int>()->default_value(0), "ON if rosbag images with different timestamps")
            ;

    if (h) {
        cout<<desc;
        exit(1);
    }

    po::store(po::parse_config_file<char>(filename.c_str(), desc), vm);
    po::notify(vm);
}

void parse_settings(string filename, LFDataUtilSettings &settings, LFReconstructSettings &reconSettings, bool h) {

    po::variables_map vm;
    programOptions(vm, filename,h);

    //parse data util params
    parseLFDataUtilParams(settings, vm);
    parseLFReconstructParams(reconSettings, vm);

}

void parse_settings(string filename, LFDataUtilSettings &settings, bool h) {

    po::variables_map vm;
    programOptions(vm, filename,h);

    //parse data util params
    parseLFDataUtilParams(settings, vm);

}

void parseLFDataUtilParams(LFDataUtilSettings &settings,  po::variables_map vm ){


    boost::filesystem::path dataP(vm["data_path"].as<string>());
    if(dataP.string().empty()) {
        LOG(FATAL)<<"data_path is a REQUIRED variable";
    }
    settings.data_path = dataP.string();

    boost::filesystem::path calibP(vm["calib_file_path"].as<string>());
    if(calibP.string().empty()) {
        LOG(FATAL) << "calib_file_path is a REQUIRED variable";
    }
    if (calibP.is_absolute()) {
        settings.calib_file_path = calibP.string();
    } else {
        boost::filesystem::path config_file_path(dataP);
        config_file_path /= calibP.string();
        settings.calib_file_path = config_file_path.string();
    }

    boost::filesystem::path imgsP(vm["images_path"].as<string>());
    if(!imgsP.string().empty()) {

        if (imgsP.is_absolute()) {
            settings.images_path = imgsP.string();
        } else {
            boost::filesystem::path config_file_path(dataP);
            config_file_path.remove_leaf() /= imgsP.string();
            settings.images_path = config_file_path.string();
        }
    }

    boost::filesystem::path frontendfile(vm["frontend_params_file"].as<string>());
    if(frontendfile.string().empty()) {
        LOG(FATAL) << "frontend_params_file is a REQUIRED variable";
    }
    if (frontendfile.is_absolute()) {
        settings.frontend_params_file = frontendfile.string();
    } else {
        boost::filesystem::path config_file_path(dataP);
        config_file_path /= frontendfile.string();
        settings.frontend_params_file = config_file_path.string();
    }

    boost::filesystem::path backendfile(vm["backend_params_file"].as<string>());
    if(backendfile.string().empty()) {
        LOG(FATAL) << "backend_params_file is a REQUIRED variable";
    }
    if (backendfile.is_absolute()) {
        settings.backend_params_file = backendfile.string();
    } else {
        boost::filesystem::path config_file_path(dataP);
        config_file_path /= backendfile.string();
        settings.backend_params_file = config_file_path.string();
    }

    settings.is_ros = vm["ros"].as<int>();
    settings.imgs_read = vm["imgs_read"].as<int>();
    settings.debug_mode = vm["debug_mode"].as<int>();
    settings.resize_images = vm["resize_images"].as<int>();
    settings.rf = vm["rf"].as<double>();
    settings.kalibr = vm["kalibr"].as<int>();
    settings.undistort = vm["undistort"].as<int>();
    settings.radtan = vm["radtan"].as<int>();
    settings.ref_cam = vm["ref_cam"].as<int>();
    settings.mask_type = vm["mask_type"].as<string>();
    boost::filesystem::path segmaskP(vm["segmasks_path"].as<string>());
    if(segmaskP.string().empty()) {
        LOG(FATAL) << "calib_file_path is a REQUIRED variable";
    }
    if (segmaskP.is_absolute()) {
        settings.segmasks_path = segmaskP.string();
    } else {
        boost::filesystem::path config_file_path(dataP);
        config_file_path /= segmaskP.string();
        settings.segmasks_path = config_file_path.string();

    }

    boost::filesystem::path seg_settings(vm["seg_settings_file"].as<string>());
    if(seg_settings.string().empty()) {
        LOG(FATAL) << "seg_settings_file is a REQUIRED variable";
    }
    if (seg_settings.is_absolute()) {
        settings.seg_settings_path = seg_settings.string();
    } else {
        boost::filesystem::path config_file_path(dataP);
        config_file_path /= seg_settings.string();
        settings.seg_settings_path = config_file_path.string();

    }
    settings.use_segment = vm["use_segment"].as<int>();
    settings.read_segmask = vm["read_segmask"].as<int>();

    settings.mp4 = vm["mp4"].as<int>();
    settings.binary = vm["binary"].as<int>();
    settings.async = vm["async"].as<int>();


    vector<int> frames;
    stringstream frames_stream(vm["frames"].as<string>());
    int i;
    while (frames_stream >> i) {
        frames.push_back(i);

        if(frames_stream.peek() == ',' || frames_stream.peek() == ' ') {
            frames_stream.ignore();
        }
    }
    if (frames.size() == 0) {
        settings.all_frames = 1;
        settings.skip = 0;
    } else if (frames.size() == 1) {
        settings.start_frame = frames.at(0);
        settings.end_frame = frames.at(0);
        settings.skip = 0;
        settings.all_frames = 0;
    } else if (frames.size() == 2) {
        settings.start_frame = frames.at(0);
        settings.end_frame = frames.at(1);
        settings.skip = 0;
        settings.all_frames = 0;
    } else if (frames.size() >= 3) {
        settings.start_frame = frames.at(0);
        settings.end_frame = frames.at(1);
        settings.skip = frames.at(2);
        settings.all_frames = 0;
    }
    if (settings.start_frame<0) {

        LOG(FATAL)<<"Can't have starting frame less than 0. Terminating..."<<endl;
    }

    // Reading time shift values
    vector<int> shifts;
    stringstream shifts_stream(vm["shifts"].as<string>());
    while (shifts_stream >> i) {
        shifts.push_back(i);
        if(shifts_stream.peek() == ',' || shifts_stream.peek() == ' ') {
            shifts_stream.ignore();
        }
    }
    settings.shifts = shifts;


}

void parseLFReconstructParams(LFReconstructSettings &settings,  po::variables_map vm){

    boost::filesystem::path dataP(vm["data_path"].as<string>());
    if(dataP.string().empty()) {
        LOG(FATAL)<<"data_path is a REQUIRED variable";
    }
    settings.data_path = dataP.string();

    boost::filesystem::path reconParamFile(vm["reconstructor_params_path"].as<string>());
    if(reconParamFile.string().empty()) {
        LOG(FATAL) << "reconstructor_params_path is a REQUIRED variable";
    }
    if (reconParamFile.is_absolute()) {
        settings.reconstructor_params_path = reconParamFile.string();
    } else {
        boost::filesystem::path config_file_path(dataP);
        config_file_path /= reconParamFile.string();
        settings.reconstructor_params_path = config_file_path.string();
    }


    settings.use_gpu = vm["use_gpu"].as<int>();
    settings.debug_mode = vm["debug_mode"].as<int>();
    settings.eval_mode = vm["eval_mode"].as<int>();
    settings.hf_method = vm["hf_method"].as<int>();

    settings.live_refocus = vm["live_refocus"].as<int>();
    settings.calc_map = vm["calc_map"].as<int>();
    settings.calc_depth = vm["calc_depth"].as<int>();

    settings.color_imgs =  vm["color_imgs"].as<int>();
    //settings.depth_est =  vm["depth_est"].as<int>();
    //settings.use_q = vm["use_q"].as<int>();
    //settings.algo = vm["depth_algo"].as<int>();
    //settings.lcam = vm["lcam_index"].as<int>();
    //settings.rcam = vm["rcam_index"].as<int>();

    settings.gpu_median_rec = vm["gpu_median_rec"].as<int>();
    settings.gpu_average_rec = vm["gpu_average_rec"].as<int>();
    settings.datagen_mode = vm["datagen_mode"].as<int>();

}

/*void parse_slam_settings(string filename, Settings &settings, bool help) {

    namespace po = boost::program_options;

    po::options_description desc("Allowed config file options");
    desc.add_options()
            ("debug_mode", po::value<int>()->default_value(0), "ON to use debug mode")
            ("ros", po::value<int>()->default_value(0), "ON if running using ROS. Otherwise, the images are read from folders.")
            ("live_refocus", po::value<int>()->default_value(0), "live streaming refocus or not. currently works only with ROS")
            ("data_path", po::value<string>()->default_value(""), "data path contaning imgs, segmasks, calibration file to use")
            ("calib_file_path", po::value<string>()->default_value("camchain.yaml"), "calibration file to use")
            ("segmasks_path", po::value<string>()->default_value("segmasks"), "segmentation masks file to use")
            ("radtan", po::value<int>()->default_value(0), "ON to use rad tan distortion model")
            ("use_segment", po::value<int>()->default_value(0), "1 to run segmentation on the images")
            ("read_segmask", po::value<int>()->default_value(0), "1 to read the precomputed segmentation on the images")
            ("seg_settings_file", po::value<string>()->default_value("SegmentSettings.yaml"), "Segmentation CNN settings file to use")
            ("mask_type", po::value<string>()->default_value("bmp"), "Segmentation mask either binary (saved as an image) or continous (saved as a csv) or npy")
            ("frontend_params_file", po::value<string>()->default_value("lf_frontend.yaml"), "params for frontend file to use")
            ;

    if (help) {
        cout<<desc;
        exit(1);
    }


    po::variables_map vm;
    po::store(po::parse_config_file<char>(filename.c_str(), desc), vm);
    po::notify(vm);
    settings.debug_mode = vm["debug_mode"].as<int>();
    settings.live_refocus = vm["live_refocus"].as<int>();
    settings.radtan = vm["radtan"].as<int>();
    settings.is_ros = vm["ros"].as<int>();
    settings.read_segmask = vm["read_segmask"].as<int>();
    settings.use_segment = vm["use_segment"].as<int>();
    settings.mask_type = vm["mask_type"].as<string>();

    boost::filesystem::path dataP(vm["data_path"].as<string>());
    if(dataP.string().empty()) {
        LOG(FATAL)<<"data_path is a REQUIRED variable";
    }
    settings.data_path = dataP.string();
    boost::filesystem::path calibP(vm["calib_file_path"].as<string>());
    if(calibP.string().empty()) {
        LOG(FATAL) << "calib_file_path is a REQUIRED variable";
    }
    if (calibP.is_absolute()) {
        settings.calib_file_path = calibP.string();
    } else {
        boost::filesystem::path config_file_path(dataP);
        config_file_path /= calibP.string();
        settings.calib_file_path = config_file_path.string();
    }

    boost::filesystem::path frontendfile(vm["frontend_params_file"].as<string>());
    if(frontendfile.string().empty()) {
        LOG(FATAL) << "frontend_params_file is a REQUIRED variable";
    }
    if (frontendfile.is_absolute()) {
        settings.frontend_params_file = frontendfile.string();
    } else {
        boost::filesystem::path config_file_path(dataP);
        config_file_path /= frontendfile.string();
        settings.frontend_params_file = config_file_path.string();
    }

    boost::filesystem::path segmaskP(vm["segmasks_path"].as<string>());
    if(segmaskP.string().empty()) {
        LOG(FATAL) << "calib_file_path is a REQUIRED variable";
    }
    if (segmaskP.is_absolute()) {
        settings.segmasks_path = segmaskP.string();
    } else {
        boost::filesystem::path config_file_path(dataP);
        config_file_path /= segmaskP.string();
        settings.segmasks_path = config_file_path.string();

    }

    boost::filesystem::path seg_settings(vm["seg_settings_file"].as<string>());
    if(seg_settings.string().empty()) {
        LOG(FATAL) << "seg_settings_file is a REQUIRED variable";
    }
    if (seg_settings.is_absolute()) {
        settings.seg_settings_path = seg_settings.string();
    } else {
        boost::filesystem::path config_file_path(dataP);
        config_file_path /= seg_settings.string();
        settings.seg_settings_path = config_file_path.string();

    }

}



void parse_calibration_settings(string filename, calibration_settings &settings, bool h) {

    namespace po = boost::program_options;

    po::options_description desc("Allowed config file options");
    desc.add_options()
        ("images_path", po::value<string>()->default_value(""), "path where data is located")
        ("corners_file", po::value<string>()->default_value(""), "file where to write corners")
        ("refractive", po::value<int>()->default_value(0), "ON if calibration data is refractive")
        ("mp4", po::value<int>()->default_value(0), "ON if data is in mp4 files")
        ("distortion", po::value<int>()->default_value(0), "ON if radial distortion should be accounted for")
        ("hgrid", po::value<int>()->default_value(5), "Horizontal number of corners in the grid")
        ("vgrid", po::value<int>()->default_value(5), "Vertical number of corners in the grid")
        ("grid_size_phys", po::value<double>()->default_value(5), "Physical size of grid in [mm]")
        ("skip", po::value<int>()->default_value(1), "Number of frames to skip (used mostly when mtiff on)")
        ("frames", po::value<string>()->default_value(""), "Array of values in format start, end, skip")
        ("shifts", po::value<string>()->default_value(""), "Array of time shift values separated by commas")
        ("resize_images", po::value<int>()->default_value(0), "ON if calibration images should be resized")
        ("rf", po::value<double>()->default_value(1), "Factor by which to resize calibration images")
        ;

    if (h) {
        cout<<desc;
        exit(1);
    }

    po::variables_map vm;
    po::store(po::parse_config_file<char>(filename.c_str(), desc), vm);
    po::notify(vm);

    settings.grid_size = Size(vm["hgrid"].as<int>(), vm["vgrid"].as<int>());
    settings.grid_size_phys = vm["grid_size_phys"].as<double>();
    settings.refractive = vm["refractive"].as<int>();
    settings.distortion = vm["distortion"].as<int>();
    settings.skip = vm["skip"].as<int>();
    settings.resize_images = vm["resize_images"].as<int>();
    settings.rf = vm["rf"].as<double>();
    settings.mp4 = vm["mp4"].as<int>();

    vector<int> frames;
    stringstream frames_stream(vm["frames"].as<string>());
    int i;
    while (frames_stream >> i) {
        frames.push_back(i);

        if(frames_stream.peek() == ',' || frames_stream.peek() == ' ') {
            frames_stream.ignore();
        }
    }
    if (frames.size() == 3) {
        settings.start_frame = frames.at(0);
        settings.end_frame = frames.at(1);
        settings.skip = frames.at(2);
    } else {
        LOG(FATAL)<<"frames expects 3 comma or space separated values";
    }

    if (settings.start_frame<0) {
        LOG(FATAL)<<"Can't have starting frame less than 0. Terminating..."<<endl;
    }

    // Reading time shift values
    vector<int> shifts;
    stringstream shifts_stream(vm["shifts"].as<string>());
    while (shifts_stream >> i) {
        shifts.push_back(i);
        if(shifts_stream.peek() == ',' || shifts_stream.peek() == ' ') {
            shifts_stream.ignore();
        }
    }
    settings.shifts = shifts;

    boost::filesystem::path imgsP(vm["images_path"].as<string>());
    if(imgsP.string().empty()) {
        LOG(FATAL)<<"images_path is a REQUIRED variable";
    }
    if (imgsP.is_absolute()) {
        settings.images_path = imgsP.string();
    } else {
        boost::filesystem::path config_file_path(filename);
        config_file_path.remove_leaf() /= imgsP.string();
        settings.images_path = config_file_path.string();
    }

    settings.corners_file_path = vm["corners_file"].as<string>();

}

 */