//
// Created by Pushyami Kaveti on 6/15/20.
//

#include "MCSlam/OpenGlViewer.h"
#include <pangolin/plot/plotter.h>
using namespace std;

OpenGlViewer::OpenGlViewer( const string& strSettingPath, FrontEndBase* front_end):frontEnd(front_end){

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    mImageWidth = fSettings["Camera.width"];
    mImageHeight = fSettings["Camera.height"];
    if(mImageWidth<1 || mImageHeight<1)
    {
        mImageWidth = 640;
        mImageHeight = 480;
    }

    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];
    finishRequested = false;
    finished = false;
}

void OpenGlViewer::goLive(){
    pangolin::CreateWindowAndBind("GenCam SLAM: Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    //pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    //pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
    //pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
            pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,0.0, 1.0)
    );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc, Twc_mono;
    Twc.SetIdentity();
    //mono
    Twc_mono.SetIdentity();

    ////EXTRA pose visualizationvaiables
    pangolin::OpenGlMatrix Twc_seventeen, Twc_gp3p;
    Twc_seventeen.SetIdentity();
    Twc_gp3p.SetIdentity();

    ///////////////////////////////////////////////////////////////////////////////
    // Data logger object
   /*pangolin::DataLog log;

    // Optionally add named labels
    std::vector<std::string> labels;
    labels.push_back(std::string("LF Features"));
    labels.push_back(std::string("Matches"));
    labels.push_back(std::string("Tracked Landmarks"));
    labels.push_back(std::string("Triangulated"));
    labels.push_back(std::string("Keyframes"));
    log.SetLabels(labels);

    // OpenGL 'view' of data. We might have many views of the same data.
    pangolin::Plotter plotter(&log,0.0f, 200,0.0f,500.0f,50,0.5f);
    plotter.SetBounds(0.0, pangolin::Attach::Pix(200), 0.0, 1.0);
    plotter.Track("$i");

    // Add some sample annotations to the plot
    plotter.AddMarker(pangolin::Marker::Vertical,   0, pangolin::Marker::LessThan, pangolin::Colour::Blue().WithAlpha(0.2f) );
    plotter.AddMarker(pangolin::Marker::Horizontal,   0, pangolin::Marker::GreaterThan, pangolin::Colour::Red().WithAlpha(0.2f) );
    plotter.AddMarker(pangolin::Marker::Horizontal,    0, pangolin::Marker::Equal, pangolin::Colour::Green().WithAlpha(0.2f) );
    plotter.AddMarker(pangolin::Marker::Horizontal,    0, pangolin::Marker::Equal, pangolin::Colour::White().WithAlpha(0.2f) );
    plotter.AddMarker(pangolin::Marker::Horizontal,    0, pangolin::Marker::Equal, pangolin::Colour::White().WithAlpha(1.0f) );

    pangolin::DisplayBase().AddDisplay(plotter);*/

    ///////////////////////////////////////////////////////////////////////////////

    bool bFollow = true;

    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        ////////////////////////////////////////////////////////

        //log.Log(frontEnd->log_num_intramatches_, frontEnd->log_num_matches_, frontEnd->log_num_tracked_lms_, frontEnd->log_num_triangulated_, frontEnd->log_KF_*300);

        frontEnd->log_KF_=0;
        //////////////////////////////////////////////////////

        cv::Mat poseCV = frontEnd->getPose();
        convertToOpenGlCameraMatrix( poseCV, Twc);
        //extra variables for viz poses
        //cv::Mat poseCV_seventeen = frontEnd->getPose_seventeen();
        //convertToOpenGlCameraMatrix( poseCV_seventeen, Twc_seventeen);
        //cv::Mat poseCV_gp3p = frontEnd->getPose_gp3p();
        //convertToOpenGlCameraMatrix( poseCV_gp3p, Twc_gp3p);


        //Mono
        //cv::Mat poseCV_mono = frontEnd->getPose_Mono();
        //convertToOpenGlCameraMatrix( poseCV_mono, Twc_mono);

        if(menuFollowCamera && bFollow)
        {
            s_cam.Follow(Twc);
        }
        else if(menuFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }


        d_cam.Activate(s_cam);
        //glClearColor(1.0f,1.0f,1.0f,1.0f);
        glClearColor(0.0f,0.0f,0.0f,0.0f);
        DrawCurrentCamera(Twc, 1.0, 0.0, 0.0);
        DrawAllCameras(1, 0.0,1.0, 0.0);
        //
        //glClearColor(1.0f,1.0f,1.0f,1.0f);
        glClearColor(0.0f,0.0f,0.0f,0.0f);
        //DrawCurrentCamera(Twc_seventeen, 0.0, 1.0, 0.0);
        //DrawAllCameras(2, 1.0,0.0,0.0);

        //glClearColor(1.0f,1.0f,1.0f,1.0f);
        //DrawCurrentCamera(Twc_gp3p, 0.0, 0.0, 1.0);
        //DrawAllCameras(3, 0.0,0.0, 1.0);

       // DrawCurrentCamera(Twc_mono, 1.0, 0.0, 1.0);
       // DrawAllCameras(true);
        //show the 3D reconstructed points
        //if(menuShowPoints)
        //    DrawMapPoints();

        pangolin::FinishFrame();

        //cv::Mat im = DrawFrame();
        //cv::imshow("GenCam SLAM: Current Frame",im);
        //cv::waitKey(mT);
        usleep(mT*1000);

        if(menuReset)
        {
            menuShowPoints = true;

            bFollow = true;
            menuFollowCamera = true;

            menuReset = false;
        }

        if(CheckFinish())
            break;
    }
    setFinish();


}
bool OpenGlViewer::CheckFinish(){
    unique_lock<mutex> lock(mMutexFinish);
    return finishRequested;
}
void OpenGlViewer::requestFinish(){
    unique_lock<mutex> lock(mMutexFinish);
    finishRequested = true;
}
bool OpenGlViewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return finished;
}
void OpenGlViewer::setFinish(){

    unique_lock<mutex> lock(mMutexFinish);
    finished = true;
}

void OpenGlViewer::DrawAllCameras(int whichPoses, float r, float g, float b) {
    vector<cv::Mat> posesCV;
    switch(whichPoses){
        case 1:
            posesCV = frontEnd->getAllPoses();
            break;
        case 2:
            posesCV = frontEnd->getAllPoses1();
            break;
        //case 3:
        //    posesCV = frontEnd->getAllPoses1();
        //    break;
        default:
            break;
    }
    if (posesCV.size() == 0 )
        return;
    for(int i =0; i < posesCV.size() ; i++){
        pangolin::OpenGlMatrix Twc;
        convertToOpenGlCameraMatrix( posesCV[i], Twc);
        DrawCurrentCamera(Twc, r, g, b);
    }
    glLineWidth(2);
    glColor4f(0.0f,0.0f,1.0f,0.6f);
    glBegin(GL_LINES);
    for(int i =0; i < (posesCV.size()-1) ; i++){
        cv::Mat Ow = posesCV[i].colRange(3,4);
        cv::Mat Ow2 = posesCV[i+1].colRange(3,4);
        glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
        glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));

    }
    glEnd();
    ////////// Plot the points in each frame /////////////////
   // for(int i =0; i < (posesCV.size()-1) ; i++){
   //     DrawMapPoints(frontEnd->sparse_recon[i], frontEnd->sparse_recon_color[i]);
  //  }
}


/*void OpenGlViewer::DrawAllCameras(bool mono){
    vector<cv::Mat> posesCV;
    vector<cv::Mat> posesCV_seventeen ,posesCV_gp3p;
    if(mono){
        posesCV = frontEnd->getAllPoses_Mono();
    }
    else{
        posesCV = frontEnd->getAllPoses();
    }
    if (posesCV.size() == 0 )
        return;
    for(int i =0; i < posesCV.size() ; i++){
        pangolin::OpenGlMatrix Twc;
        convertToOpenGlCameraMatrix( posesCV[i], Twc);
        if(mono)
            DrawCurrentCamera(Twc, 0.0, 0.0, 1.0);
        else
            DrawCurrentCamera(Twc, 0.0, 1.0, 0.0);
    }
    glLineWidth(2);
    glColor4f(0.0f,0.0f,1.0f,0.6f);
    glBegin(GL_LINES);
    for(int i =0; i < (posesCV.size()-1) ; i++){
        cv::Mat Ow = posesCV[i].colRange(3,4);
        cv::Mat Ow2 = posesCV[i+1].colRange(3,4);
        glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
        glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
    }

    glEnd();
} */
void OpenGlViewer::DrawCurrentCamera(pangolin::OpenGlMatrix& Twc, float r, float g, float b ){

    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
    glMultMatrixf(Twc.m);
#else
    glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(r,g,b);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}
void  OpenGlViewer::DrawMapPoints(vector<Point3f>& mapPoints,vector<Point3f>& pointColors){

    //vector<Point3f> mapPoints;
    //frontEnd->getMapPoints(mapPoints); //for now only to compile
    if(mapPoints.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    //glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=mapPoints.size(); i<iend;i++)
    {
        //if(mapPoints[i]->isBad())
        //    continue;
        glColor3f(pointColors[i].x, pointColors[i].y, pointColors[i].z);
        cv::Point3f pos = mapPoints[i];
        glVertex3f(pos.x,pos.y,pos.z);
    }
    glEnd();

}
void  OpenGlViewer::DrawMapPoints(){

    vector<Point3f> mapPoints;
    frontEnd->getMapPoints(mapPoints); //for now only to compile
    if(mapPoints.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,1.0,0.8);

    for(size_t i=0, iend=mapPoints.size(); i<iend;i++)
    {
        //if(mapPoints[i]->isBad())
        //    continue;
        //glColor3f(pointColors[i].x, pointColors[i].y, pointColors[i].z);
        cv::Point3f pos = mapPoints[i];
        glVertex3f(pos.x,pos.y,pos.z);
    }
    glEnd();

}

void OpenGlViewer::DrawFrame(){

}


void OpenGlViewer::convertToOpenGlCameraMatrix(cv::Mat& pose, pangolin::OpenGlMatrix &M)
{

    if(!pose.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            //Rwc = pose.rowRange(0,3).colRange(0,3).t();
            //twc = -Rwc*pose.rowRange(0,3).col(3);
            pose.convertTo(pose, CV_32FC1);
            Rwc = pose.rowRange(0,3).colRange(0,3);
            twc = pose.rowRange(0,3).col(3);
        }

        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();

}
