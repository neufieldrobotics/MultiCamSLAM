//
// Created by Pushyami Kaveti on 3/24/21.
//

#include "LFSlam/GlobalMap.h"
Landmark::Landmark(Mat p, LightFieldFrame* lf_frame, int featInd, Point2f uv, int id) {
    lId = id;
    pt3D = p;
    KFs.push_back(lf_frame);
    featInds.push_back(featInd);
    uv_ref.push_back(uv);
}

Landmark::Landmark(Mat p, MonoFrame* frame, int featInd, Point2f uv, int id) {
    lId = id;
    pt3D = p.clone();
    KFs_mono.push_back(frame);
    featInds.push_back(featInd);
    uv_ref.push_back(uv);
}

void Landmark::addLfFrame(LightFieldFrame* lf_frame, int featInd, Point2f uv){
    KFs.push_back(lf_frame);
    featInds.push_back(featInd);
    uv_ref.push_back(uv);
}

void Landmark::addMonoFrame(MonoFrame* frame, int featInd, Point2f uv){
    KFs_mono.push_back(frame);
    featInds.push_back(featInd);
    uv_ref.push_back(uv);
}



GlobalMap::GlobalMap() {

    num_lms = 0;
    lmID = 0;
}

int GlobalMap::insertLandmark(Mat p, LightFieldFrame* lf_frame,  int featInd, Point2f uv) {
    Landmark* l = new Landmark(p,lf_frame, featInd, uv, lmID);
    auto res = mapPoints.insert(pair<int,Landmark*>(lmID , l));
    if(!res.second){
        // the landmark ID is already present
        cout<<"Something wrong. The landmark is already inserted"<<endl;
        return -1;
    }
    num_lms++;
    lmID++;
    return (lmID-1);
}

int GlobalMap::insertLandmark(Mat p, MonoFrame* frame,  int featInd, Point2f uv) {
    Landmark* l = new Landmark(p,frame, featInd, uv, lmID);
    auto res = mapPoints.insert(pair<int,Landmark*>(lmID , l));
    if(!res.second){
        // the landmark ID is already present
        cout<<"Something wrong. The landmark is already inserted"<<endl;
        return -1;
    }
    num_lms++;
    lmID++;
    return (lmID-1);
}

void GlobalMap::insertLandmark(Landmark* p) {

    auto res = mapPoints.insert(pair<int,Landmark*>(lmID , p));
    if(!res.second){
        // the landmark ID is already present
        return;
    }
    lmID++;
    num_lms++;
}

void GlobalMap::deleteLandmark(int lid){
    Landmark* l = getLandmark(lid);
    int i =0;
    for(auto lf : l->KFs){
        lf->lIds[l->featInds[i]] = -1;
        i++;
    }
    mapPoints.erase(lid);
    num_lms--;
}

bool GlobalMap::updateLandmark(int lid, cv::Mat &point_new, double& diff_norm){

    Landmark* landmark = getLandmark(lid);
    Mat diff_lm = landmark->pt3D - point_new;
    VLOG(3)<<"diff in lm: "<<diff_lm.at<double>(0,0)<<","<<diff_lm.at<double>(1,0)<<","<<diff_lm.at<double>(2,0)<<endl;
    diff_norm = cv::norm(diff_lm);
    if(diff_norm > 0.5)
        VLOG(2)<<"LiD: "<<lid<<", ("<<landmark->pt3D.at<double>(0,0)<<","<<landmark->pt3D.at<double>(1,0)<<","<<landmark->pt3D.at<double>(2,0)
               <<")--->("<<point_new.at<double>(0,0)<<","<<point_new.at<double>(1,0)<<","<<point_new.at<double>(2,0)<<")"<<endl;
    if(diff_norm < 2.0){
        landmark->pt3D = point_new.clone();
        return true;
    }
    else{
        //delete landmrak form the map
        VLOG(2)<<"Deleting landmark with ID : "<<lid<<endl;
        deleteLandmark(lid);
        return false;
    }
}

Landmark* GlobalMap::getLandmark(int lid){
    auto res = mapPoints.find(lid);
    if (res != mapPoints.end()) {
        return res->second;
    } else {
        //std::cout << "Not found\n";
        return NULL;
    }

}

void GlobalMap::printMap(){
    std::map<int, Landmark*>::iterator it;
    for(it = mapPoints.begin() ; it != mapPoints.end() ; ++it){
        Landmark* l = it->second;
        cout<<"landmark "<<it->first<<","<<l->lId << " pt: "<<l->pt3D<<" frame seen : ";
        for (auto i = l->KFs.begin(); i !=l->KFs.end(); ++i)
            cout<<(*i)->frameId<<"," ;
        cout<<endl;
    }
}