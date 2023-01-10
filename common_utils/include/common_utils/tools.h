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

#ifndef TOOLS_LIBRARY
#define TOOLS_LIBRARY

#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
// Tiff library (included in namespace because of typedef conflict with some OpenCV versions)
//namespace libtiff {
//#include <tiffio.h>
//}
#include "glog/logging.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*! Initialize logging via glog. Should be called at the start of
  the main() function to enable logging since most output in openFV
  is routed to the glog library.
 */
void init_logging(int argc, char** argv);

void T_from_P(Mat P, Mat &H, double z, double scale, Size img_size);



bool dirExists(string dirPath);

int matrixMean(vector<Mat> mats_in, Mat &mat_out);

Mat P_from_KRT(Mat K, Mat rvec, Mat tvec, Mat rmean, Mat &P_u, Mat &P);

Mat build_Rt(Mat R, Mat t);

Mat build_camera_matrix(Mat K, Mat rvec, Mat tvec);

double dist(Point3f p1, Point3f p2);

/*! Display an image and wait for Esc press
  \param image Image to display
 */
void qimshow(Mat image);

void qimshow2(vector<Mat> imgs);

void pimshow(Mat image, double z, int n);

Mat getRotMat(double x, double y, double z);

void failureFunction();

void writeMat(Mat M, string path);

Mat getTransform(vector<Point2f> src, vector<Point2f> dst);

void listDir(string, vector<string> &);

void readImgStack(vector<string>, vector<Mat> &);

/*! Function to generate a given number of linearly spaced points
  between specified lower and upper limits
*/
vector<double> linspace(double, double, int);

Mat cross(Mat_<double>, Mat_<double>);

Mat normalize(Mat_<double>);


vector<double> hill_vortex(double, double, double, double);

vector<double> vortex(double, double, double, double);

vector<double> burgers_vortex(double, double, double, double);

vector<double> test_field(double, double, double, double);

vector<double> dir_field(double, double, double, double);

// Movie class

class Movie {

 public:
    ~Movie() {}

    Movie(vector<Mat>);

 private:

    void play();
    void updateFrame();
    vector<Mat> frames_;
    int active_frame_;

};

//! Class to enable easy writing of data to files
class fileIO {

 public:
    ~fileIO() {
        LOG(INFO)<<"Closing file...";
        file.close();
    }

    /*! Constructor to create a file to write data to. This
      also checks whether or not the path to the file being
      created exists or not which determines whether the file
      is successfully created or not. In case the path is not
      valid or for some other reason the constructor is unable
      to create the specified file, all output is routed to a
      temporary file.
      \param filename Name of file to write data to
     */
    fileIO(string filename);

    //! Write int type to file
    fileIO& operator<< (int);
    //! Write float type to file
    fileIO& operator<< (float);
    //! Write double type to file
    fileIO& operator<< (double);
    //! Write a string to the file
    fileIO& operator<< (string);
    //! Write a const char* to file
    fileIO& operator<< (const char*);
    //! Write a vector of int variables to file
    fileIO& operator<< (vector<int>);
    fileIO& operator<< (vector< vector<int> >);
    fileIO& operator<< (vector<float>);
    fileIO& operator<< (vector< vector<float> >);
    fileIO& operator<< (vector<double>);
    fileIO& operator<< (vector< vector<double> >);
    fileIO& operator<< (Mat);

    // TODO: add templated Mat data output to file

    /*
    void write(Mat);
    */

 protected:

 private:

    string getFilename(string filename);
    ofstream file;

};

class imageIO {

 public:
    ~imageIO() {

    }

    imageIO(string path);

    void setPrefix(string prefix);

    void operator<< (Mat);
    void operator<< (vector<Mat>);

 protected:

 private:

    string dir_path_;
    string prefix_;
    string ext_;

    int counter_;

    int DIR_CREATED;

};

/*class mtiffReader {

public:

    ~mtiffReader() {}
    mtiffReader(string path);

    Mat get_frame(int);
    int num_frames();

protected:

private:

    libtiff::TIFF* tiff_;
    int num_frames_;
    string path_;

};
*/

class mp4Reader {

public:

    ~mp4Reader() {}
    mp4Reader(string path);
    mp4Reader(string path, int color);

    Mat get_frame(int);
    int num_frames();
    double time_stamp(int);

protected:

private:

    VideoCapture cap_;
    int num_frames_;
    string path_;
    int color_;

};

#endif
