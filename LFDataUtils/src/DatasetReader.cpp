//
// Created by Pushyami Kaveti on 8/29/19.
//

#include "LFDataUtils/DatasetReader.h"
using namespace cv;
using namespace std;
// COnfiguration and datareading functions

void DatasetReader::initialize(LFDataUtilSettings refocus_set) {

    //Ideally this should be
    //Stage 1 : read the calibration data
    //Stage 2 : read the dataset
    settings = refocus_set;
    MP4_FLAG = settings.mp4;
    //MTIFF_FLAG = settings.mtiff;
    BINARY_FLAG = settings.binary;
    ASYNC_FLAG = settings.async;
    imgs_read_ = settings.imgs_read;
    start_frame_ = settings.start_frame;
    end_frame_ = settings.end_frame;
    skip_frame_ = settings.skip;
    shifts_ = settings.shifts;
    RESIZE_IMAGES = settings.resize_images;
    rf_ = settings.rf;
    UNDISTORT_IMAGES = settings.undistort;
    ALL_FRAME_FLAG = settings.all_frames;
    RADTAN_FLAG = settings.radtan;
    string path = settings.calib_file_path;
    imgs.clear();
    tStamps_.clear();

    current_index = 0;

    //parse the calibration file

        if (settings.kalibr) {
            read_kalibr_data(settings.calib_file_path);
        } else {
            read_vo_data(settings.calib_file_path);
        }

        if (MP4_FLAG) {

            if (!ALL_FRAME_FLAG) {

                for (int i = settings.start_frame; i <= settings.end_frame; i += settings.skip + 1)
                    frames_.push_back(i);
            }
            read_imgs_mp4(settings.images_path);

        }
        /*else if (MTIFF_FLAG) {

            if (!ALL_FRAME_FLAG) {

                for (int i = settings.start_frame; i <= settings.end_frame; i += settings.skip + 1)
                    frames_.push_back(i);
            }
            read_imgs_mtiff( settings.images_path);

        }*/
         else if (BINARY_FLAG) {

            //read_binary_imgs( settings.images_path);

        } else {

            read_imgs(settings.images_path);

        }

}


void DatasetReader::read_vo_data(string path){

    LOG(INFO) << "Reading calibration data from " << path << "...";

    FileStorage fs(path, FileStorage::READ);
    FileNode fn = fs.root();

    FileNodeIterator fi = fn.begin(), fi_end = fn.end();

    int i=0;
    for (; fi != fi_end; ++fi, i++) {

        FileNode f = *fi;
        string cam_name; f["rostopic"]>>cam_name;
        if (MP4_FLAG)
            cam_names_.push_back(cam_name.substr(1,8) + ".MP4");
        else
            cam_names_.push_back(cam_name.substr(14,4));

        cout<<"Camera: "<<cam_names_[i]<<endl;

        string cam_model; f["camera_model"]>>cam_model;
        string dist_model; f["distortion_model"]>>dist_model;

        if (cam_model.compare("pinhole"))
            LOG(FATAL)<<"Only pinhole camera model is supported as of now!";
        // if (dist_model.compare("equidistant"))
        //     LOG(FATAL)<<"Only equidistant distortion model is supported as of now!";

        // Reading distortion coefficients
        vector<double> dc;
        Mat_<double> dist_coeff = Mat_<double>::zeros(1,4);
        f["distortion_coeffs"] >> dc;
        for (int j=0; j < dc.size(); j++)
            dist_coeff(0,j) = (double)dc[j];

        vector<int> ims;
        f["resolution"] >> ims;
        if (i>0) {
            if (((int)ims[0] != calib_img_size_.width) || ((int)ims[1] != calib_img_size_.height))
                LOG(FATAL)<<"Resolution of all images is not the same!";
        } else {
            calib_img_size_ = Size((int)ims[0], (int)ims[1]);
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
                t(r,0) = (double)f2[3]; //we donot have a scale yet.. not sure what metrics this will be
            }
        }


        Mat Rt = build_Rt(R, t);
        Mat P = K_mat*Rt;

        VLOG(2)<<K_mat;
        VLOG(2)<<Rt;
        VLOG(3)<<P;

        R_mats_.push_back(R);
        t_vecs_.push_back(t);
        dist_coeffs_.push_back(dist_coeff);
        K_mats_.push_back(K_mat);
        P_mats_.push_back(P);

    }
    num_cams_ = i;


}

void DatasetReader::read_kalibr_data(string path) {

    LOG(INFO) << "Reading calibration (kalibr) data from " << path << "...";

    FileStorage fs(path, FileStorage::READ);
    FileNode fn = fs.root();

    FileNodeIterator fi = fn.begin(), fi_end = fn.end();

    int i=0;
    for (; fi != fi_end; ++fi, i++) {

        FileNode f = *fi;
        string cam_name; f["rostopic"]>>cam_name;
        if (MP4_FLAG)
            cam_names_.push_back(cam_name.substr(1,8) + ".MP4");
        else
            cam_names_.push_back(cam_name.substr(14,4));

        cout<<"Camera: "<<cam_names_[i]<<endl;

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
            if (((int)ims[0] != calib_img_size_.width) || ((int)ims[1] != calib_img_size_.height))
                LOG(FATAL)<<"Resolution of all images is not the same!";
        } else {
            calib_img_size_ = Size((int)ims[0], (int)ims[1]);
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
                t(r,0) = (double)f2[3]; // multiply by 1000 to convert from [m] to [mm]
            }
        }
        R_mats_kalibr.push_back(R.clone());
        t_vecs_kalibr.push_back(t.clone());

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
        VLOG(3)<<P;
        Mat T =Mat::eye(4,4,CV_64F);
        Rt.copyTo(T.rowRange(0,3));
        VLOG(2)<<T;
        VLOG(2)<<T.inv();
        R_mats_.push_back(R);
        t_vecs_.push_back(t);
        dist_coeffs_.push_back(dist_coeff);
        K_mats_.push_back(K_mat);
        P_mats_.push_back(P);

    }

    img_size_ = calib_img_size_;
    num_cams_ = i;

}

void DatasetReader::read_imgs(string path) {

    DIR *dir;
    struct dirent *ent;

    string dir1(".");
    string dir2("..");
    string temp_name;
    string img_prefix = "";

    Mat image, fimage;



    LOG(INFO)<<"READING IMAGES TO REFOCUS...";

    VLOG(1)<<"UNDISTORT_IMAGES flag is "<<UNDISTORT_IMAGES;
    VLOG(1)<<"RADTAN_FLAG flag is "<<RADTAN_FLAG;
    VLOG(1)<<"ASYNC_FLAG flag is "<<ASYNC_FLAG;

    vector<string> img_names;
    vector<vector<string>> all_img_names;

    for (int i=0; i<num_cams_; i++) {

        VLOG(1)<<"Camera "<<i+1<<" of "<<num_cams_<<"..."<<endl;

        string path_tmp;
        path_tmp = path+cam_names_[i]+"/"+img_prefix;
        dir = opendir(path_tmp.c_str());
        while(ent = readdir(dir)) {
            temp_name = ent->d_name;
            if (temp_name.compare(dir1)) {
                if (temp_name.compare(dir2)) {
                    string path_img = path_tmp+temp_name;
                    img_names.push_back(path_img);
                }
            }
        }


        sort(img_names.begin(), img_names.end());
        all_img_names.push_back(img_names);

        img_names.clear();
        path_tmp = "";

    }


    vector<string> timestamps;
    if (ASYNC_FLAG) {

        LOG(WARNING) << "Filtering async timestamps. This will only work for 18 digit timestamps!";

        for (int i=0; i<all_img_names[0].size(); i++) {

            string key = all_img_names[0][i];
            string filename = key.substr(key.length()-23);
            long int tstamp_ref  = atol(filename.substr(0,19).c_str());

            int find_count = 1;
            for (int j=1; j<num_cams_; j++) {
                for (int k=0; k<all_img_names[j].size(); k++) {
                    string key1 = all_img_names[j][k];
                    string filename1 = key1.substr(key1.length()-23);
                    long int tstamp_comp = atol(filename1.substr(0,19).c_str());
                    if (tstamp_comp == tstamp_ref)
                        find_count++;
                }
            }

            if (find_count == num_cams_)
                timestamps.push_back(filename);

        }

    }

    int begin;
    int end;
    int skip;

    if(ALL_FRAME_FLAG) {
        begin = 0;
        skip = 0;
        end = all_img_names[0].size();
        if (ASYNC_FLAG)
            end = timestamps.size();
    } else {
        begin = start_frame_;
        skip = skip_frame_;
        end = end_frame_+1;
        if (ASYNC_FLAG) {
            if (end>timestamps.size()) {
                LOG(WARNING)<<"End frame is greater than number of frames!" <<endl;
                end = timestamps.size();
            }
        } else {
            if (end>all_img_names[0].size()) {
                LOG(WARNING)<<"End frame is greater than number of frames!" <<endl;
                end = all_img_names[0].size();
            }
        }
    }


    for (int i = 0; i < num_cams_; i++) {

        vector<Mat> refocusing_imgs_sub;
        vector<string> seg_imgs_sub, image_names_sub;
        vector<string> img_names = all_img_names[i];

        string segPath = settings.segmasks_path + "/cam" + to_string(i) +
                         "/"; //msg->header.seq should be seq. but seeing some lag in recorded rosbag

        string path_tmp;
        path_tmp = path + cam_names_[i] + "/" + img_prefix;


        int count = 0;
        for (int j = begin; j < end; j += skip + 1) {
            string key = img_names.at(j);
            string filename = key.substr(key.length() - 23, 19);
            /// This below line is only needed to read segmasks writte in decimal format
            string segFileName = filename.substr(0, 10) + "." + to_string(atol(filename.substr(10, 9).c_str()));
            seg_imgs_sub.push_back(segPath + segFileName + "." + settings.mask_type);
            image_names_sub.push_back(img_names.at(j));
            if(imgs_read_){
                if (ASYNC_FLAG) {
                    VLOG(1) << j << ": " << path_tmp + timestamps.at(j) << " (" << count << ")";
                    //pushyami :changed reading image in grayscale // chnaged to bgr
                    image = imread(path_tmp + timestamps.at(j), IMREAD_UNCHANGED);
                } else {
                    VLOG(3) << j << ": " << img_names.at(j) << " (" << count << ")";
                    image = imread(img_names.at(j), IMREAD_UNCHANGED);
                }

                Mat image2;
                if (UNDISTORT_IMAGES) {
                    if (RADTAN_FLAG)
                        undistort(image, image2, K_mats_[i], dist_coeffs_[i]);
                    else
                        fisheye::undistortImage(image, image2, K_mats_[i], dist_coeffs_[i], K_mats_[i]);
                } else {
                    image2 = image.clone();
                }

                if (VLOG_IS_ON(4))
                    qimshow(image2);
                refocusing_imgs_sub.push_back(image2);


                if (j == begin)
                    img_size_ = Size(image.cols, image.rows);
            }

            count++;

            if (i == 0) {
                long int tstamp_ref = atol(filename.substr(0, 19).c_str());
                double stmp = tstamp_ref * 1e-9;
                VLOG(2) << "Timestamp: " << setprecision(9) << std::fixed << stmp << endl;
                tStamps_.push_back(stmp);
                frames_.push_back(j);
            }

        }


        imgs.push_back(refocusing_imgs_sub);
        seg_img_names_.push_back(seg_imgs_sub);
        all_img_names_.push_back(image_names_sub);

        img_names.clear();
        seg_imgs_sub.clear();
        image_names_sub.clear();

        VLOG(1) << "done!\n";
    }

        VLOG(1) << "DONE READING IMAGES" << endl;
    NUM_IMAGES =   all_img_names_[0].size();
}

void DatasetReader::read_binary_imgs(string path) {

    DIR *dir;
    struct dirent *ent;

    string dir1(".");
    string dir2("..");
    string temp_name;
    string img_prefix = "";

    Mat image, fimage;

    vector<string> img_names;

    if(!imgs_read_) {

        LOG(INFO)<<"READING IMAGES TO REFOCUS...";

        VLOG(1)<<"UNDISTORT_IMAGES flag is "<<UNDISTORT_IMAGES;

        for (int i=0; i<num_cams_; i++) {

            VLOG(1)<<"Camera "<<i+1<<" of "<<num_cams_<<"..."<<endl;

            string path_tmp;
            vector<Mat> refocusing_imgs_sub;

            path_tmp = path+cam_names_[i]+"/"+img_prefix;

            dir = opendir(path_tmp.c_str());
            while(ent = readdir(dir)) {
                temp_name = ent->d_name;
                if (temp_name.compare(dir1)) {
                    if (temp_name.compare(dir2)) {
                        string path_img = path_tmp+temp_name;
                        img_names.push_back(path_img);
                    }
                }
            }

            sort(img_names.begin(), img_names.end());

            int begin;
            int end;
            int skip;

            if(ALL_FRAME_FLAG) {
                begin = 0;
                end = img_names.size();
                skip = 0;
            } else {
                begin = start_frame_;
                end = end_frame_+1;
                skip = skip_frame_;
                if (end>img_names.size()) {
                    LOG(WARNING)<<"End frame is greater than number of frames!" <<endl;
                    end = img_names.size();
                }
            }

            for (int j=begin; j<end; j+=skip+1) {
                VLOG(1)<<j<<": "<<img_names.at(j)<<endl;

                ifstream ifs(img_names.at(j).c_str());
                boost::archive::binary_iarchive ia(ifs);
                ia >> image;

                if (j==begin)
                    img_size_ = Size(image.cols, image.rows);

                Mat image2;
                if (UNDISTORT_IMAGES) {
                    fisheye::undistortImage(image, image2, K_mats_[i], dist_coeffs_[i], K_mats_[i]);
                } else {
                    image2 = image.clone();
                }

                refocusing_imgs_sub.push_back(image2);
                if (i==0) {
                   frames_.push_back(j);
                }

            }
            img_names.clear();

            imgs.push_back(refocusing_imgs_sub);
            path_tmp = "";

            VLOG(1)<<"done!\n";
            imgs_read_ = 1;

        }

        VLOG(1)<<"DONE READING IMAGES"<<endl;
    } else {
        LOG(INFO)<<"Images already read!"<<endl;
    }

}

/*
void DatasetReader::read_imgs_mtiff(string path) {

    LOG(INFO)<<"READING IMAGES TO REFOCUS...";

    DIR *dir;
    struct dirent *ent;

    string dir1(".");
    string dir2("..");
    string temp_name;

    vector<string> img_names;

    dir = opendir(path.c_str());
    while(ent = readdir(dir)) {
        temp_name = ent->d_name;
        if (temp_name.compare(dir1)) {
            if (temp_name.compare(dir2)) {
                if (temp_name.compare(temp_name.size()-3,3,"tif") == 0) {
                    string img_name = path+temp_name;
                    img_names.push_back(img_name);
                }
            }
        }
    }

    sort(img_names.begin(), img_names.end());
    vector<mtiffReader> tiffs;

    VLOG(1)<<"Images in path:"<<endl;
    for (int i=0; i<img_names.size(); i++) {
        VLOG(1)<<img_names[i]<<endl;
        mtiffReader tiff(img_names[i]);
        VLOG(2)<<tiff.num_frames()<<" frames in file.";
        tiffs.push_back(tiff);
    }

    // TODO: add check for whether all tiffs are equal in size or not

    if (ALL_FRAME_FLAG) {
        VLOG(1)<<"READING ALL FRAMES..."<<endl;
        for (int i=0; i<tiffs[0].num_frames(); i++)
            frames_.push_back(i);
    }

    VLOG(1)<<"Reading images..."<<endl;
    for (int n=0; n<img_names.size(); n++) {

        VLOG(1)<<"Camera "<<n+1<<"...";

        vector<Mat> refocusing_imgs_sub;
        int count=0;
        for (int f=0; f<frames_.size(); f++) {
            Mat img = tiffs[n].get_frame(frames_.at(f));
            refocusing_imgs_sub.push_back(img.clone());
            count++;
        }

        imgs.push_back(refocusing_imgs_sub);
        VLOG(1)<<"done! "<<count<<" frames read."<<endl;

    }


    VLOG(1)<<"DONE READING IMAGES"<<endl;

}

 */
void DatasetReader::read_imgs_mp4(string path) {

    for (int i=0; i<cam_names_.size(); i++) {

        string file_path = path+cam_names_[i];
        mp4Reader mf(file_path);

        int total_frames = mf.num_frames();
        VLOG(1)<<"Total frames: "<<total_frames;

        Mat frame, frame2, frame3;

        vector<Mat> refocusing_imgs_sub;
        for (int j=0; j<frames_.size(); j++) {

            frame = mf.get_frame(frames_[j] + shifts_[i]);

            if (RESIZE_IMAGES) {
                resize(frame, frame2, Size(int(frame.cols*rf_), int(frame.rows*rf_)));
            } else {
                frame2 = frame.clone();
            }

            if (UNDISTORT_IMAGES) {
                fisheye::undistortImage(frame2, frame3, K_mats_[i], dist_coeffs_[i], K_mats_[i]);
            } else {
                frame3 = frame2.clone();
            }

            refocusing_imgs_sub.push_back(frame3.clone()); // store frame

            if (i==0 && j==0) {
                img_size_ = Size(refocusing_imgs_sub[0].cols, refocusing_imgs_sub[0].rows);
            } else {
                if (refocusing_imgs_sub[j].cols != img_size_.width || refocusing_imgs_sub[j].rows != img_size_.height)
                    LOG(FATAL)<<"Size of images to refocus is not the same!";
            }

        }

        imgs.push_back(refocusing_imgs_sub);

    }

    // img_size_ = Size(imgs[0][0].cols, imgs[0][0].rows);
    if (img_size_.width != calib_img_size_.width || img_size_.height != calib_img_size_.height)
        LOG(FATAL)<<"Resolution of images used for calibration and size of images to refocus is not the same!";

    LOG(INFO)<<"DONE READING IMAGES!";

}

void DatasetReader::loadNext(vector<cv::Mat>& imgset) {

    imgset.clear();
    if (current_index >= NUM_IMAGES)
        return ;

    for (int cam_ind=0; cam_ind< num_cams_; cam_ind++){

        ///read the images if they are not read earlier
        Mat imgg, imgc;
        if(!imgs_read_){
            imgc = imread(all_img_names_[cam_ind][current_index], IMREAD_UNCHANGED);
            VLOG(3)<<"Img path being read: "<<all_img_names_[cam_ind][current_index]<<endl;
           }
        else{
            imgc = imgs[cam_ind][current_index];
        }
        if(imgc.channels() == 3)
            cvtColor(imgc, imgg, COLOR_BGR2GRAY);
        else
            imgg = imgc;
        imgg.convertTo(imgg, CV_32F);
        imgg /= 255.0;
        imgset.push_back(imgg.clone());

        if (cam_ind == 0){
            img_size_ = Size(imgg.cols, imgg.rows);
        }
    }

}

void DatasetReader::getNext(vector<cv::Mat>& imgs, double& timeStamp){
     loadNext(imgs);
    timeStamp = tStamps_[current_index];
    // increase the index to go next image
    current_index++;
}
void DatasetReader::getNext(vector<cv::Mat>& imgs, vector<string>& segmaskImgs, double& timeStamp){
    loadNext(imgs);
    timeStamp = tStamps_[current_index];
    //todo: code for updating the segmentation mask names.
    // increase the index to go next image'
    segmaskImgs.clear();
    for(int i =0; i <num_cams_ ; i++){
        segmaskImgs.push_back(seg_img_names_[i][current_index]);
    }
   // copy(seg_img_names_[current_index].begin(), seg_img_names_[current_index].end(), back_inserter(segmaskImgs));
    current_index++;
}

