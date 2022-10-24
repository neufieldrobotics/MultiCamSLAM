//
// Created by jagat on 9/16/19.
//

#ifndef SRC_LIVE_VIEWER_H
#define SRC_LIVE_VIEWER_H


#endif //SRC_LIVE_VIEWER_H

#include "std_include.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

class LiveViewer
{
    // Declaring public and private methods
public:
    /* initialises the live viewer window with its buttons and status bar
     * input: GPU or CPU ??
     *
     */
    LiveViewer();
    ~LiveViewer(){

    }

    /* shows the output of the processed images from refocus functions
     * displays a status bar: that shows what kind of pre-processing is being performed on the images
     * such as segmentation, depth processing and so on.
     * to do: decide what inputs are given
     */

    void UpdateRefocusedImage();

private:
    void initializeCallbacks(){

    }


    // Declaring public and private variables
public:
    int window_size;
    Mat imageToDisplay;
private:
    double dthresh;
    double tlimit ;
    double mult_exp_limit;
    double mult_thresh ;
    bool sig;
    int interval;


};