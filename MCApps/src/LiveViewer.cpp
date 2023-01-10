//
// Created by jagat on 9/16/19.
//
#include "LiveViewer.h"

LiveViewer::LiveViewer()
{
    namedWindow("Live View", WINDOW_AUTOSIZE | WINDOW_GUI_EXPANDED);

   /* if (array_all.size()-1)
        createTrackbar("Frame", "Live View", &active_frame_, array_all.size()-1, cb_frames, this);

    createButton("Multiplicative", cb_mult, this, CV_CHECKBOX);
    createButton("MedianRec", cb_v_rec, this, CV_CHECKBOX);
    createButton("GPUMedianRec", cb_gv_rec, this, CV_CHECKBOX);
    createButton("StDevRec", cb_stdev, this, CV_CHECKBOX);
    createButton("dz = 0.1", cb_dz_p1, this, CV_RADIOBOX, 0);
    createButton("dz = 1", cb_dz_1, this, CV_RADIOBOX, 0);
    createButton("dz = 10", cb_dz_10, this, CV_RADIOBOX, 0);
    createButton("dz = 100", cb_dz_100, this, CV_RADIOBOX, 0);
    createButton("dz = 1000", cb_dz_1000, this, CV_RADIOBOX, 0);
    createButton("dz = 10000", cb_dz_10000, this, CV_RADIOBOX, 0); */
}
void LiveViewer::UpdateRefocusedImage(){
    char title[200];
    imshow("Live View", imageToDisplay);
}
