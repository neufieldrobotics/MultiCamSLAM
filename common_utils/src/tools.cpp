//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                For Open Source Flow Visualization Library
//
// Copyright 2013-2017 Abhishek Bajpayee
//
// This file is part of OpenFV.
//
// OpenFV is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License version 2 as published by the Free Software Foundation.
//
// OpenFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License version 2 for more details.
//
// You should have received a copy of the GNU General Public License version 2 along with
// OpenFV. If not, see https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html.

#include "common_utils/tools.h"

using namespace std;
using namespace cv;
//using namespace libtiff;

const double pi = 3.14159;

void init_logging(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::InstallFailureFunction(&failureFunction);
    FLAGS_logtostderr=1;

}



void T_from_P(Mat P, Mat &H, double z, double scale, Size img_size) {

    Mat_<double> A = Mat_<double>::zeros(3,3);

    for (int i=0; i<3; i++) {
        for (int j=0; j<2; j++) {
            A(i,j) = P.at<double>(i,j);
        }
    }

    for (int i=0; i<3; i++) {
        A(i,2) = P.at<double>(i,2)*z+P.at<double>(i,3);
    }

    Mat A_inv = A.inv();

    Mat_<double> D = Mat_<double>::zeros(3,3);
    D(0,0) = scale;
    D(1,1) = scale;
    D(2,2) = 1;
    D(0,2) = img_size.width*0.5;
    D(1,2) = img_size.height*0.5;

    Mat T = D*A_inv;

    H = T.clone();

}

bool dirExists(string dirPath) {

    if ( dirPath.c_str() == NULL) return false;

    DIR *pDir;
    bool bExists = false;

    pDir = opendir (dirPath.c_str());

    if (pDir != NULL)
    {
        bExists = true;
        (void) closedir (pDir);
    }

    return bExists;

}

// Function to calculate mean of any matrix
// Returns 1 if success
int matrixMean(vector<Mat> mats_in, Mat &mat_out) {

    if (mats_in.size()==0) {
        cout<<"\nInput matrix vector empty!\n";
        return 0;
    }

    for (int i=0; i<mats_in.size(); i++) {
        for (int j=0; j<mats_in[0].rows; j++) {
            for (int k=0; k<mats_in[0].cols; k++) {
                mat_out.at<double>(j,k) += mats_in[i].at<double>(j,k);
            }
        }
    }

    mat_out = mat_out/double(mats_in.size());

    return 1;

}

// Construct aligned and unaligned P matrix from K, R and T matrices
Mat P_from_KRT(Mat K, Mat rvec, Mat tvec, Mat rmean, Mat &P_u, Mat &P) {

    Mat rmean_t;
    cv::transpose(rmean, rmean_t);

    Mat R = rvec*rmean_t;

    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            P_u.at<double>(i,j) = rvec.at<double>(i,j);
            P.at<double>(i,j) = R.at<double>(i,j);
        }
        P_u.at<double>(i,3) = tvec.at<double>(0,i);
        P.at<double>(i,3) = tvec.at<double>(0,i);
    }

    P_u = K*P_u;
    P = K*P;

}

Mat build_Rt(Mat R, Mat t) {

    Mat_<double> Rt = Mat_<double>::zeros(3,4);
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            Rt(i,j) = R.at<double>(i,j);
        }
        Rt(i,3) = t.at<double>(0,i);
    }

    return Rt;

}

Mat build_camera_matrix(Mat K, Mat rvec, Mat tvec) {

    Mat R; Rodrigues(rvec, R);
    Mat Rt = build_Rt(R, tvec);
    Mat P = K*Rt;

    return P.clone();

}

double dist(Point3f p1, Point3f p2) {

    double distance = sqrt(pow(p2.x-p1.x,2) + pow(p2.y-p1.y,2) + pow(p2.z-p1.z,2));

    return(distance);

}

void qimshow(Mat image) {

    stringstream info;
    info<<"Image Type: ";
    int type = image.type();
    switch(type) {
    case CV_8U:
        info<<"CV_8U";
        break;
    case CV_16U:
        info<<"CV_16U";
        break;
    case CV_32F:
        info<<"CV_32F";
        break;
    case CV_64F:
        info<<"CV_64F";
        break;
    default:
        info<<type;
        break;
    }

    info<<", Size: ";
    info<<image.cols<<" x "<<image.rows;

    namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image", image);
    displayOverlay("Image", info.str().c_str());

    int key;
    while(1) {
        key = waitKey(10);
        if ((key & 255) == 27)
            break;
    }
    destroyWindow("Image");

}

// Quick image show for n number of images
void qimshow2(vector<Mat> imgs) {

    for (int i=0; i<imgs.size(); i++) {
        stringstream wname;
        wname<<"Image"<<i;
        namedWindow(wname.str(), WINDOW_AUTOSIZE);
        imshow(wname.str(), imgs[i]);
    }

    int key;
    while(1) {
        key = waitKey(10);
        if ((key & 255) == 27)
            break;
    }

    for (int i=0; i<imgs.size(); i++) {
        stringstream wname;
        wname<<"Image"<<i;
        destroyWindow(wname.str());
    }

}

void pimshow(Mat image, double z, int n) {

    namedWindow("Image", WINDOW_AUTOSIZE);

    char title[50];
    sprintf(title, "z = %f, n = %d", z, n);
    putText(image, title, Point(10,20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0));
    imshow("Image", image);

    waitKey(0);
    destroyWindow("Image");

}

// Yaw Pitch Roll Rotation Matrix
Mat getRotMat(double x, double y, double z) {

    x = x*pi/180.0;
    y = y*pi/180.0;
    z = z*pi/180.0;

    Mat_<double> Rx = Mat_<double>::zeros(3,3);
    Mat_<double> Ry = Mat_<double>::zeros(3,3);
    Mat_<double> Rz = Mat_<double>::zeros(3,3);

    Rx(0,0) = 1;
    Rx(1,1) = cos(x);
    Rx(1,2) = -sin(x);
    Rx(2,1) = sin(x);
    Rx(2,2) = cos(x);

    Ry(0,0) = cos(y);
    Ry(1,1) = 1;
    Ry(2,0) = -sin(y);
    Ry(2,2) = cos(y);
    Ry(0,2) = sin(y);

    Rz(0,0) = cos(z);
    Rz(0,1) = -sin(z);
    Rz(1,0) = sin(z);
    Rz(1,1) = cos(z);
    Rz(2,2) = 1;

    Mat R = Rz*Ry*Rx;

    return(R);

}

void failureFunction() {

    LOG(INFO)<<"Good luck debugging that X-|";
    exit(1);

}

void writeMat(Mat M, string path) {

    ofstream file;
    file.open(path.c_str());

    Mat_<double> A = M;

    for (int i=0; i<M.rows; i++) {
        for (int j=0; j<M.cols; j++) {
            file<<A.at<double>(i,j)<<"\t";
        }
        file<<"\n";
    }

    VLOG(3)<<"Written matrix to file "<<path<<endl;

    file.close();

}

Mat getTransform(vector<Point2f> src, vector<Point2f> dst) {

    Mat_<double> A1 = Mat_<double>::zeros(8,8);
    Mat_<double> B1 = Mat_<double>::zeros(8,1);
    for (int i=0; i<4; i++) {
        A1(i*2,0) = src[i].x; A1(i*2,1) = src[i].y; A1(i*2,2) = 1;
        A1(i*2+1,3) = src[i].x; A1(i*2+1,4) = src[i].y; A1(i*2+1,5) = 1;
        A1(i*2,6) = -dst[i].x*src[i].x; A1(i*2,7) = -dst[i].x*src[i].y;
        A1(i*2+1,6) = -dst[i].y*src[i].x; A1(i*2+1,7) = -dst[i].y*src[i].y;
        B1(i*2,0) = dst[i].x;
        B1(i*2+1,0) = dst[i].y;
    }

    Mat A1t;
    transpose(A1, A1t);

    Mat C1;
    invert(A1t*A1, C1, DECOMP_SVD);
    Mat C2 = A1t*B1;

    Mat_<double> R = C1*C2;

    Mat_<double> H = Mat_<double>::zeros(3,3);
    H(0,0) = R(0,0); H(0,1) = R(1,0); H(0,2) = R(2,0);
    H(1,0) = R(3,0); H(1,1) = R(4,0); H(1,2) = R(5,0);
    H(2,0) = R(6,0); H(2,1) = R(7,0); H(2,2) = 1.0;


    // old


    Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.data);
    double a[8][8], b[8];
    Mat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);

    for( int i = 0; i < 4; ++i )
    {
        a[i][0] = a[i+4][3] = src[i].x;
        a[i][1] = a[i+4][4] = src[i].y;
        a[i][2] = a[i+4][5] = 1;
        a[i][3] = a[i][4] = a[i][5] =
        a[i+4][0] = a[i+4][1] = a[i+4][2] = 0;
        a[i][6] = -src[i].x*dst[i].x;
        a[i][7] = -src[i].y*dst[i].x;
        a[i+4][6] = -src[i].x*dst[i].y;
        a[i+4][7] = -src[i].y*dst[i].y;
        b[i] = dst[i].x;
        b[i+4] = dst[i].y;
    }

    solve( A, B, X, DECOMP_SVD );
    ((double*)M.data)[8] = 1.;

    return M;

}

void listDir(string path, vector<string> &files) {

    DIR *dir;
    struct dirent *ent;

    string temp_name;

    dir = opendir(path.c_str());
    while(ent = readdir(dir)) {
        temp_name = ent->d_name;
        if (temp_name.compare(".")) {
            if (temp_name.compare("..")) {
                string path_file = path+temp_name;
                files.push_back(path_file);
            }
        }
    }

}

void readImgStack(vector<string> img_names, vector<Mat> &imgs) {

    for (int i=0; i<img_names.size(); i++) {
        Mat img = imread(img_names[i], 0);
        imgs.push_back(img);
    }

}

vector<double> linspace(double a, double b, int n) {

    vector<double> array;
    double step = (b-a) / (n-1);

    while(a <= b) {
        array.push_back(a);
        a += step;           // could recode to better handle rounding errors
    }
    return array;

}

// returns normalized(A x B) where A and B are 2 vectors
Mat cross(Mat_<double> A, Mat_<double> B) {

    Mat_<double> result = Mat_<double>::zeros(3,1);

    result(0,0) = A(1,0)*B(2,0) - A(2,0)*B(1,0);
    result(1,0) = A(2,0)*B(0,0) - A(0,0)*B(2,0);
    result(2,0) = A(0,0)*B(1,0) - A(1,0)*B(0,0);

    return(normalize(result));

}

// Normalizes a vector and then returns it
Mat normalize(Mat_<double> A) {

    double d = sqrt(pow(A(0,0), 2) + pow(A(1,0), 2) + pow(A(2,0), 2));
    A(0,0) /= d; A(1,0) /= d; A(2,0) /= d;
    return(A);

}

// ----------------------------------------------------
// Temporary location of particle propagation functions
// ----------------------------------------------------

vector<double> hill_vortex(double x, double y, double z, double t) {

    double R = sqrt(x*x + y*y + z*z);
    double r = sqrt(x*x + z*z);
    double theta = atan2(z, x);

    double a = 32; double us = 0.8;
    double A = 7.5*us/(a*a); // needs tweaking maybe

    double V = (-A/10)*(4*r*r + 2*y*y - 2*a*a);
    double U = A*r*y/5;

    double mag = sqrt(V*V + U*U);

    double Vo = us*(pow((a*a/(r*r + y*y)), 2.5)*(2*y*y - r*r)/(2*a*a)-1);
    double Uo = (1.5*us/(a*a))*r*y*pow((a*a/(r*r + y*y)), 2.5);

    if (R<=a) {
        Vo = 0; Uo = 0;
    }

    double Mo = sqrt(Vo*Vo + Uo*Uo);


    if (R>a) {
        V = Vo; U=Uo;
    }

    double u = U*cos(theta); double v = V; double w = U*sin(theta);

    vector<double> np;
    np.push_back(x+u*t); np.push_back(y+v*t); np.push_back(z+w*t);

    return(np);

}

vector<double> vortex(double x, double y, double z, double t) {

    double omega = 2.0;

    double r = sqrt(x*x + z*z);
    double theta = atan2(z, x);
    double dTheta = omega*t;
    double theta2 = theta+dTheta;

    double x2 = r*cos(theta2); double z2 = r*sin(theta2); double y2 = y;

    vector<double> np;
    np.push_back(x2); np.push_back(y2); np.push_back(z2);

    return(np);

}

vector<double> burgers_vortex(double x, double y, double z, double t) {

    //
    Mat_<double> R = Mat_<double>::zeros(3,3);
    R(0,0) = 1/sqrt(2); R(1,0) = 1/sqrt(2);
    R(0,1) = -1/sqrt(2); R(1,1) = 1/sqrt(2);
    R(2,2) = 1;

    Mat_<double> p = Mat_<double>::zeros(3,1);
    p(0,0) = x; p(1,0) = y, p(2,0) = z;

    Mat_<double> rp = R*p;

    x  = rp(0,0); y = rp(1,0); z = rp(2,0);
    //

    double tau = 200;
    double sigma = 0.01;
    double nu = 1;

    double r = sqrt(x*x + z*z);
    double theta = atan2(z, x);

    double omega = (tau/(2*pi*r*r))*(1 - exp(-sigma*r*r/4/nu));
    double dTheta = omega*t;
    double theta2 = theta+dTheta;

    double x2 = r*cos(theta2); double z2 = r*sin(theta2); double y2 = y;

    //
    Mat_<double> p2 = Mat_<double>::zeros(3,1);
    p2(0,0) = x2; p2(1,0) = y2, p2(2,0) = z2;

    Mat_<double> p3 = R.inv()*p2;
    x2 = p3(0,0); y2 = p3(1,0); z2 = p3(2,0);
    //

    vector<double> np;
    np.push_back(x2); np.push_back(y2); np.push_back(z2);

    return(np);

}

vector<double> test_field(double x, double y, double z, double t) {

    double v = 1.15; //z*1.01;
    double d = v*t;
    //double z2 = z + d;

    vector<double> np;
    np.push_back(x+d); np.push_back(y+d); np.push_back(z+d);

    return(np);

}

// Directional field
vector<double> dir_field(double x, double y, double z, double t) {

    double v = 0.3125;
    double c = 10.0;
    double dy = (v*x + c)*t;
    double dz = 0.5*(v*z + c)*t;

    vector<double> np;
    np.push_back(x); np.push_back(y+dy); np.push_back(z+dz);

    return(np);

}

// ----------------------------------------------------
// Movie class functions
// ----------------------------------------------------

Movie::Movie(vector<Mat> frames) {

    frames_ = frames;
    active_frame_ = 0;
    play();

}

void Movie::play() {

    namedWindow("Movie", WINDOW_AUTOSIZE);
    updateFrame();

    while (1) {

        int key = waitKey(10);

        if ( (key & 255) == 83) {
            if (active_frame_ < frames_.size()-1) {
                active_frame_++;
                updateFrame();
            }
        } else if ( (key & 255) == 81) {
            if (active_frame_ > 0) {
                active_frame_--;
                updateFrame();
            }
        } else if ( (key & 255) == 27) {
            destroyAllWindows();
            break;
        }

    }

}

void Movie::updateFrame() {

    char title[50];
    int size = frames_.size();
    sprintf(title, "Frame %d/%d", active_frame_+1, size);
    imshow("Movie", frames_[active_frame_]);
    displayOverlay("Movie", title);

}

// ----------------------------------------------------
// fileIO class functions
// ----------------------------------------------------

fileIO::fileIO(string filename) {

    file.open(filename.c_str());
    if(file.is_open()) {
        VLOG(1)<<"Successfully opened file "<<filename;
    } else {
        LOG(INFO)<<"Could not open file "<<filename<<"!";
        string reroute_path = "../temp/" + getFilename(filename);
        file.open(reroute_path.c_str());
        if (file.is_open())
            LOG(INFO)<<"Routing output to "<<reroute_path;
    }

}

fileIO& fileIO::operator<< (int val) {
    file<<val;
    return(*this);
}

fileIO& fileIO::operator<< (float val) {
    file<<val;
    return(*this);
}

fileIO& fileIO::operator<< (double val) {
    file<<val;
    return(*this);
}

fileIO& fileIO::operator<< (string val) {
    file<<val;
    return(*this);
}

fileIO& fileIO::operator<< (const char* val) {
    file<<val;
    return(*this);
}

fileIO& fileIO::operator<< (vector<int> val) {

    for (int i=0; i<val.size(); i++)
        file<<val[i]<<endl;

    return(*this);

}

fileIO& fileIO::operator<< (vector< vector<int> > val) {

    for (int i=0; i<val.size(); i++) {
        for (int j=0; j<val[i].size(); j++)
            file<<val[i][j]<<"\t";
        file<<endl;
    }

    return(*this);

}

fileIO& fileIO::operator<< (vector<float> val) {

    for (int i=0; i<val.size(); i++)
        file<<val[i]<<endl;

    return(*this);

}

fileIO& fileIO::operator<< (vector< vector<float> > val) {

    for (int i=0; i<val.size(); i++) {
        for (int j=0; j<val[i].size(); j++)
            file<<val[i][j]<<"\t";
        file<<endl;
    }

    return(*this);

}

fileIO& fileIO::operator<< (vector<double> val) {

    for (int i=0; i<val.size(); i++)
        file<<val[i]<<endl;

    return(*this);

}

fileIO& fileIO::operator<< (vector< vector<double> > val) {

    for (int i=0; i<val.size(); i++) {
        for (int j=0; j<val[i].size(); j++)
            file<<val[i][j]<<"\t";
        file<<endl;
    }

    return(*this);

}

fileIO& fileIO::operator<< (Mat mat) {

    int type = mat.type();

    file<<type<<"\t"<<mat.rows<<"\t"<<mat.cols<<endl;

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            switch(type) {
            case CV_8U:
                file<<mat.at<char>(i,j)<<"\t";
                break;
            case CV_32F:
                file<<mat.at<float>(i,j)<<"\t";
                break;
            case CV_64F:
                file<<mat.at<double>(i,j)<<"\t";
                break;
            }
        }
        file<<endl;
    }

    return(*this);

}

string fileIO::getFilename(string filename) {

    string slash = "/";
    int i;
    for (i=filename.length()-1; i>=0; i--) {
        if (string(1, filename[i]) == slash)
            break;
    }

    return(filename.substr(i+1));

}

// ----------------------------------------------------
// imageIO class functions
// ----------------------------------------------------

// imageIO class allows writing of a single image or a sequence of
// images to a certain path
imageIO::imageIO(string path) {

    DIR *dir;
    struct dirent *ent;
    string slash = "/";

    dir = opendir(path.c_str());
    if (!dir) {

        LOG(INFO)<<"Could not open directory "<<path;

        int i=1;
        dir_path_ = "../temp/folder001/";
        while(opendir(dir_path_.c_str())) {
            i++;
            dir_path_ = "../temp/folder";
            char buf[10];
            sprintf(buf, "%03d", i);
            dir_path_ += string(buf) + "/";
        }
        DIR_CREATED = 0;
        LOG(INFO)<<"Redirecting output to directory "<<dir_path_;

    } else {

        if (string(1, path[path.length()-1]) == slash) {
            dir_path_ = path;
        } else {
            dir_path_ = path + "/";
        }

        VLOG(1)<<"Successfully initialized image writer in "<<dir_path_;

        int files=0;
        while(ent = readdir(dir)) {
            files++;
            if(files>2)
                break;
        }

        if(files>2)
            LOG(INFO)<<"Warning: "<<dir_path_<<" is not empty";

    }

    counter_ = 1;
    prefix_ = string("");
    ext_ = ".tif";

}

void imageIO::operator<< (Mat img) {

    if (!DIR_CREATED)
        mkdir(dir_path_.c_str(), S_IRWXU);

    stringstream filename;
    filename<<dir_path_<<prefix_;

    char num[10];
    sprintf(num, "%03d", counter_);

    filename<<string(num);
    filename<<ext_;

    // TODO: specify quality depending on extension
    imwrite(filename.str(), img);
    counter_++;

}

void imageIO::operator<< (vector<Mat> imgs) {

    if (!DIR_CREATED)
        mkdir(dir_path_.c_str(), S_IRWXU);

    for (int i = 0; i < imgs.size(); i++) {

        stringstream filename;
        filename<<dir_path_<<prefix_;

        char num[10];
        sprintf(num, "%03d", counter_);

        filename<<string(num);
        filename<<ext_;

        // TODO: specify quality depending on extension
        imwrite(filename.str(), imgs[i]*255.0);
        counter_++;

    }

}

void imageIO::setPrefix(string prefix) {

    prefix_ = prefix;

}

/*
mtiffReader::mtiffReader(string path) {

    path_ = path;

    VLOG(1)<<"Opening "<<path<<endl;
    tiff_ = TIFFOpen(path_.c_str(), "r");

    num_frames_ = 0;
    if (tiff_) {
        VLOG(1)<<"Counting number of frames...";
	do {
	    num_frames_++;
	} while (TIFFReadDirectory(tiff_));
    }
    VLOG(1)<<"done! ("<<num_frames_<<" frames found.)"<<endl;

}

int mtiffReader::num_frames() { return num_frames_; }

Mat mtiffReader::get_frame(int n) {

    Mat img;
    uint32 c, r;
    size_t npixels;
    uint32* raster;

    if (n>=num_frames_) {
        LOG(WARNING)<<"Multipage tiff file only contains "<<num_frames_<<" frames and frame "<<n<<" requested! Blank image will be returned.";
        return(img);
    }

    TIFFSetDirectory(tiff_, n);

    TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &c);
    TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &r);
    npixels = r * c;
    raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
    if (raster != NULL) {
        if (TIFFReadRGBAImageOriented(tiff_, c, r, raster, ORIENTATION_TOPLEFT, 0)) {
            img.create(r, c, CV_32F);
            for (int i=0; i<r; i++) {
                for (int j=0; j<c; j++) {
                    img.at<float>(i,j) = TIFFGetR(raster[i*c+j]);
                }
            }
        }
        _TIFFfree(raster);
    }

    img /= 255;

    return(img);

}

 */
mp4Reader::mp4Reader(string path) {

    path_ = path;
    color_ = 0;
    cap_.open(path_);
    if (!cap_.isOpened())
        LOG(FATAL)<<"Could not open "<<path;

    num_frames_ = cap_.get(CAP_PROP_FRAME_COUNT);
    VLOG(1)<<"Total number of frames in "<<path<<": "<<num_frames_;

}

mp4Reader::mp4Reader(string path, int color) {

    path_ = path;
    color_ = color;
    cap_.open(path_);
    if (!cap_.isOpened())
        LOG(FATAL)<<"Could not open "<<path;

    num_frames_ = cap_.get(CAP_PROP_FRAME_COUNT);
    VLOG(1)<<"Total number of frames in "<<path<<": "<<num_frames_;

}

int mp4Reader::num_frames() { return num_frames_; }

double mp4Reader::time_stamp(int n) { return cap_.get(CAP_PROP_POS_MSEC); }

Mat mp4Reader::get_frame(int n) {

    Mat frame, img;

    if (n>=num_frames_) {
        LOG(WARNING)<<"mp4 file only contains "<<num_frames_<<" frames and frame "<<n<<" requested! Blank image will be returned.";
        return(img);
    }

    cap_.set(CAP_PROP_POS_FRAMES, n);
    cap_ >> frame;

    if (color_) {
        img = frame.clone();
    } else {
        cvtColor(frame, img, COLOR_BGR2GRAY);
        img.convertTo(img, CV_8U);
    }

    VLOG(3)<<n<<"\'th frame read.";
    return img;

}


