//
// Created by Pushyami Kaveti on 7/23/19.
//

#include "MCSlam/DepthReconstructor.h"

void DepthReconstructor::init(Mat K1, Mat D1, Mat K2, Mat D2, Mat Rot, Mat trans){

    cv::stereoRectify(K1, D1, K2, D2, img_size ,Rot, trans /*/1000.0*/ ,\
                      rectLeft, rectRight , projLeft, projRight , Q, CALIB_ZERO_DISPARITY, 1.0, img_size, &validRoi[0], &validRoi[1]); //dividing by 1000 to convert form mm to meters
    // 0.654848 alpha 1.0 = blackareas 0.0 = zoomedin
    // cout<<"\nP Mat-left:";
    // cout<<projLeft;
    // cout<<"\nP MAT-right:";
    // cout<<projRight;
    // cout<<"\nRECTIFY Mat-left:";
    // cout<<rectLeft;
    // cout<<"\nRECTIFY Mat-right:";
    // cout<<rectRight;
    initUndistortRectifyMap(K1 , D1, rectLeft ,projLeft, img_size, CV_32FC1 , rectMapLeft_x, rectMapLeft_y);
    initUndistortRectifyMap(K2 ,D2, rectRight ,projRight, img_size, CV_32FC1 , rectMapRight_x, rectMapRight_y);

    // make settings based on which algorithm to use for depth reconstruction
    if(depthAlgo == BLOCK_MATCH){
        //minDisparity , numDisparities = 16, blockSize = 3, P1 = 0, P2 = 0,disp12MaxDiff = 0, preFilterCap = 0, uniquenessRatio = 0, speckleWindowSize = 0,
        // speckleRange = 0,mode
        stereo_proc =  cv::StereoSGBM::create(0, 48, 15, 100, 1000, 32, 0, 15, 1000,\
                       16, cv::StereoSGBM::MODE_HH); //StereoSGBM::create( -64, 128, 100, 100, 1000, 32, 0, 15, 1000, 16, cv::StereoSGBM::MODE_HH);

    }
    else if (depthAlgo == ELAS){
        Elas::parameters param(Elas::ROBOTICS);
        param.postprocess_only_left = true;
        elas_proc = new Elas(param);
        //elas_proc = &elas_obj;
    }
   // else{
       // VLOG(2)<<" SPECIFY EITHER BLOCK_MATCH OR ELAS FOR ALGORITHM\n";
   // }

   return;
}


void DepthReconstructor::updateImgs(Mat &imgl , Mat &imgr){
    lcam_img = imgl.clone();
    rcam_img = imgr.clone();
    return;
}

void DepthReconstructor::calcDisparity(Mat &disp){
    Mat depth;
    calcDisparity(lcam_img , rcam_img, disp, depth);
    return;
}

void DepthReconstructor::calcDisparity(Mat &img1 , Mat &img2, Mat &disp, Mat &depthMap){

    cv::Mat img1r, img2r , vdisp;
    if (img1.empty() || img2.empty()){
        cout<<"ERROR: IMAGE IS EMPTY";
        return;
    }

    cv::remap(img1, img1r, rectMapLeft_x, rectMapLeft_y, cv::INTER_LINEAR);
    cv::remap(img2, img2r, rectMapRight_x, rectMapRight_y, cv::INTER_LINEAR);

    // Calculate disparity
    if(depthAlgo == BLOCK_MATCH){
        Mat dispSGBM;
        stereo_proc->compute(img1r , img2r , dispSGBM);
        dispSGBM.convertTo(disp, CV_16S, 1.f/16.f);
    }
    else if (depthAlgo == ELAS) {
        const int32_t dims[3] = {img_size.width, img_size.height, img_size.width};
        Mat leftdpf = Mat::zeros(img_size, CV_32F);
        Mat rightdpf = Mat::zeros(img_size, CV_32F);

        elas_proc->process(img1r.data, img2r.data, leftdpf.ptr<float>(0), rightdpf.ptr<float>(0), dims);
        //Mat dmap = Mat(img_size, CV_8UC1, Scalar(0));
        //leftdpf.convertTo(disp, CV_32F, 1.);
        //leftdpf.convertTo(disp, CV_16S, 1.);
        disp = leftdpf.clone();
        //cv::normalize(disp, vdisp, 0, 256, cv::NORM_MINMAX, CV_8U);
        //cv::imshow("depth", vdisp);
        //cv::waitKey(0);

    }

    //cvWaitKey(0);
    img1_rect = img1r.clone();
    img2_rect = img2r.clone();


    ////////////////////compute depth /////////////////////

    Mat V = Mat(4, 1, CV_64FC1);
    Mat pos = Mat(4, 1, CV_64FC1);
    //depthMap = Mat(disp.size(), CV_16U, Scalar(0));
    depthMap = Mat(disp.size(), CV_32FC3, Scalar(0.0, 0.0, 0.0));
    for (int i = 0; i < disp.cols; i++) {
        for (int j = 0; j < disp.rows; j++) {
            int d = disp.at<float>(j,i);
            // if low disparity, then ignore
            if (d < 2) {
                continue;
            }
            // V is the vector to be multiplied to Q to get
            // the 3D homogenous coordinates of the image point
            V.at<double>(0,0) = (double)(i);
            V.at<double>(1,0) = (double)(j);
            V.at<double>(2,0) = (double)d;
            V.at<double>(3,0) = 1.;
            double X,Y,Z;
            if(with_q){
                pos = Q * V; // 3D homogeneous coordinate
                X = pos.at<double>(0,0) / pos.at<double>(3,0);
                Y = pos.at<double>(1,0) / pos.at<double>(3,0);
                Z = pos.at<double>(2,0) / pos.at<double>(3,0);
            }
            else{
                double base = -1.0 *projRight.at<double>(0,3)/projRight.at<double>(0,0);
                X= ((double)i-projRight.at<double>(0,2)) * base/(double)d;
                Y= ((double)j-projRight.at<double>(1,2)) * base/(double)d;
                Z= projRight.at<double>(0,0) *base / (double)d;

            }

            //depthMap.at<short>(j,i) = Z*5000;
            depthMap.at<Vec3f>(j,i) = Vec3f(X,Y,Z);
            // transform 3D point from camera frame to robot frame

        }
    }

    /*cv::Mat dense_points_(disp.size(), CV_32FC3);
    cv::reprojectImageTo3D(disp, dense_points_, Q, true);
    depthMap = Mat(disp.size(), CV_16U);
    if(dense_points_.rows != disp.rows && dense_points_.cols != disp.cols){
        VLOG(2)<<"ERROR THE DENSE POINT CLOUD SIZE IS DIFERENT FROM IMAGE\n";
        return;
    }

    vector<Mat> channels;
    split(dense_points_, channels);
    channels[2].convertTo(depthMap, CV_16U, 5000);

    //Mat d = cv::imread("/home/auv/software/DynaSLAM/Examples/RGB-D/rgbd_dataset_freiburg3_walking_xyz/depth/1341846314.022058.png", CV_LOAD_IMAGE_UNCHANGED);*/
   // cv::normalize(depthMap, vdisp, 0, 256, cv::NORM_MINMAX, CV_8U);
   // cv::imshow("depth", depthMap);
   // cv::waitKey(0);

    return;

}

void DepthReconstructor::convertToDepthMap(cv::Mat &disp, cv::Mat &depthMap) {
    //cv::Mat_<cv::Vec3f> dense_points_;
   /* cv::Mat dense_points_(disp.size(), CV_32FC3);

    cv::reprojectImageTo3D(disp, dense_points_, Q, true, CV_16S);
    depthMap = Mat(dense_points_.rows, dense_points_.cols, CV_16S);
    if(dense_points_.rows != disp.rows && dense_points_.cols != disp.cols){
        VLOG(2)<<"ERROR THE DENSE POINT CLOUD SIZE IS DIFERENT FROM IMAGE\n";
        return;
    }

    vector<Mat> channels;
    split(dense_points_, channels);
    depthMap = channels[2].clone();

    //float bad_point = std::numeric_limits<float>::quiet_NaN (); */

   return;
}




//////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// STEREO CALIBRATION SHOULD NOT BE HERE //////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

void DepthReconstructor::stereo_calibration(const char *imageList, int nx, int ny, vector<vector<cv::Point2f> >* points_final\
                             , string cam1 , string cam2, cv::Mat &M1, cv::Mat &M2, cv::Mat &D1, cv::Mat &D2,\
                             cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F , cv::Size &imageSize , int &nframes , vector<string> *imageNames){

    bool displayCorners = true;
    bool showUndistorted = true;
    // horiz or vert cams
    const int maxScale = 1;
    const float squareSize = 0.08f;

    // actual square size

    int i, j, lr;
    int N = nx * ny;
    cv::Size board_sz = cv::Size(nx, ny);
    //vector<string> imageNames[2];
    vector<cv::Point3f> boardModel;
    vector<vector<cv::Point3f> > objectPoints;
    vector<vector<cv::Point2f> > points[2];
    vector<cv::Point2f> corners[2];
    bool found[2] = {false, false};
    //cv::Size imageSize;

    // READ IN THE LIST OF CIRCLE GRIDS:
    //

    for (i = 0; i < ny; i++)
        for (j = 0; j < nx; j++)
            boardModel.push_back(
                    cv::Point3f((float)(i * squareSize), (float)(j * squareSize), 0.f));

    FILE *f = fopen(imageList, "rt");
    if (!f) {
        cout << "Cannot open file " << imageList << endl;
        return;
    }
    i = 0;
    for (;;) {
        char buf[1024];
        string pathe;
        lr = i % 2;
        if (lr == 0){
            found[0] = found[1] = false;
            if (!fgets(buf, sizeof(buf) - 3, f))
                break;
            size_t len = strlen(buf);
            while (len > 0 && isspace(buf[len - 1]))
                buf[--len] = '\0';
            if (buf[0] == '#')
                continue;
            pathe = cam1 + string(buf);
        }
        else{
            pathe = cam2 + string(buf);
        }
        cv::Mat img = cv::imread(pathe, 0);
        if (img.empty())
            break;
        imageSize = img.size();
        imageNames[lr].push_back(pathe);
        i++;

        // If we did not find board on the left image,
        // it does not make sense to find it on the right.
        //
        if (lr == 1 && !found[0])
            continue;

        // Find circle grids and centers therein:
        for (int s = 1; s <= maxScale; s++) {
            cv::Mat timg = img;
            if (s > 1)
                resize(img, timg, cv::Size(), s, s, cv::INTER_CUBIC);
            // Just as example, this would be the call if you had circle calibration
            // boards ...
            //      found[lr] = cv::findCirclesGrid(timg, cv::Size(nx, ny),
            //      corners[lr],
            //                                      cv::CALIB_CB_ASYMMETRIC_GRID |
            //                                          cv::CALIB_CB_CLUSTERING);
            //...but we have chessboards in our images
            found[lr] = cv::findChessboardCorners(timg, board_sz, corners[lr]);

            if (found[lr] || s == maxScale) {
                cv::Mat mcorners(corners[lr]);
                mcorners *= (1. / s);
            }
            if (found[lr])
                break;
        }
        if (displayCorners) {
            cout << buf << endl;
            cv::Mat cimg;
            cv::cvtColor(img, cimg, cv::COLOR_GRAY2BGR);

            // draw chessboard corners works for circle grids too
            cv::drawChessboardCorners(cimg, cv::Size(nx, ny), corners[lr], found[lr]);
            cv::imshow("Corners", cimg);
            if ((cv::waitKey(0) & 255) == 27) // Allow ESC to quit
                exit(-1);
        } else
            cout << '.';
        if (lr == 1 && found[0] && found[1]) {
            objectPoints.push_back(boardModel);
            points[0].push_back(corners[0]);
            points[1].push_back(corners[1]);
        }
    }
    fclose(f);

    // CALIBRATE THE STEREO CAMERAS
    /*cv::Mat M1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat M2 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D1, D2, R, T, E, F; */
    cout << "\nRunning stereo calibration ...\n";
    cv::stereoCalibrate(
            objectPoints, points[0], points[1], M1, D1, M2, D2, imageSize, R, T, E, F,
            /*cv::CALIB_FIX_PRINCIPAL_POINT | */cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_ZERO_TANGENT_DIST|cv::CALIB_SAME_FOCAL_LENGTH,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-5));
    cout << "Done! Press any key to step through images, ESC to exit\n\n";

    // CALIBRATION QUALITY CHECK
    // because the output fundamental matrix implicitly
    // includes all the output information,
    // we can check the quality of calibration using the
    // epipolar geometry constraint: m2^t*F*m1=0
    vector<cv::Point3f> lines[2];
    double avgErr = 0;
    nframes = (int)objectPoints.size();
    for (i = 0; i < nframes; i++) {
        vector<cv::Point2f> &pt0 = points[0][i];
        vector<cv::Point2f> &pt1 = points[1][i];
        cv::undistortPoints(pt0, pt0, M1, D1, cv::Mat(), M1);
        cv::undistortPoints(pt1, pt1, M2, D2, cv::Mat(), M2);
        cv::computeCorrespondEpilines(pt0, 1, F, lines[0]);
        cv::computeCorrespondEpilines(pt1, 2, F, lines[1]);

        for (j = 0; j < N; j++) {
            double err = fabs(pt0[j].x * lines[1][j].x + pt0[j].y * lines[1][j].y +
                              lines[1][j].z) +
                         fabs(pt1[j].x * lines[0][j].x + pt1[j].y * lines[0][j].y +
                              lines[0][j].z);
            avgErr += err;
        }
    }
    cout << "avg err = " << avgErr / (nframes * N) << endl;
    cout<<"Intrinsic Matrices :\n";
    cout<<"M1:\n";
    cout<<M1;
    cout<<"D1:\n";
    cout<<D1;
    cout<<"\nM2:\n";
    cout<<M2;
    cout<<"\nD2:\n";
    cout<<D2;
    cout<<"\nExtrinsics :\n";
    cout<<"Rot:\n";
    cout<<R;
    cout<<"\nTrans:\n";
    cout<<T;
    points_final = points;
}
