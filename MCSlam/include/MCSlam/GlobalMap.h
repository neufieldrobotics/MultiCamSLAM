//
// Created by Pushyami Kaveti on 3/24/21.
//

#ifndef SRC_GLOBALMAP_H
#define SRC_GLOBALMAP_H

#include <set>
#include <map>
#include <Eigen/Eigen>
#include <opengv/types.hpp>
#include "MultiCameraFrame.h"
#include "MonoFrame.h"

using namespace std;

class Landmark{
public:
    Landmark();
    Landmark(Mat p, MultiCameraFrame* lf_frame, int featInd, Point2f uv, int id);
    Landmark(Mat p, MonoFrame* frame, int featInd, Point2f uv, int id);

    ~Landmark();

    void addLfFrame(MultiCameraFrame* lf_frame, int featInd, Point2f uv);
    void addMonoFrame(MonoFrame* lf_frame, int featInd, Point2f uv);

    int lId;
    Mat pt3D;
    vector<MultiCameraFrame*> KFs;
    vector<MonoFrame*> KFs_mono;
    vector<int> featInds;
    vector<cv::Point2f> uv_ref;
    //for future
    // std::map<int,IntraMatch*> observations;

    bool operator<(Landmark lm1){
        return lId < lm1.lId;
    }
};


class GlobalMap {
public:
    GlobalMap();
    ~GlobalMap();

    void insertLandmark(Landmark* p);
    bool updateLandmark(int lid, cv::Mat &point_new, double& diff_norm);
    int insertLandmark(Mat p, MultiCameraFrame* lf_frame, int featInd, Point2f uv);
    int insertLandmark(Mat p, MonoFrame* frame, int featInd, Point2f uv);
    void deleteLandmark(int lid);
    Landmark* getLandmark(int lid);
    void printMap();
    map<int, Landmark*> mapPoints;
    int num_lms;
    int lmID;

};



#endif //SRC_GLOBALMAP_H
