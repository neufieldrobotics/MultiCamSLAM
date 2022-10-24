//
// Created by Pushyami Kaveti on 1/24/22.
//

#include "LFSlam/MonoFrontEnd.h"

void MonoFrontEnd::createFrame(Mat img_set, Mat segmap_set, double timeStamp){
    currentFrame = new MonoFrame(img_set, segmap_set, orb_vocabulary, orBextractor, Kmat_, distmat_, current_frameId++,
                                       timeStamp,false);
}


/// Method which returns the latest estimated pose
/// \return
cv::Mat MonoFrontEnd::getPose(){
    unique_lock<mutex> lock(mMutexPose);
    return currentFramePose;
}

vector<cv::Mat> MonoFrontEnd::getAllPoses(){
    unique_lock<mutex> lock(mMutexPose);
    return allPoses;
}

vector<cv::Mat> MonoFrontEnd::getAllPoses1(){
    unique_lock<mutex> lock(mMutexPose);
    return allPoses1;
}

void MonoFrontEnd::insertKeyFrame(){
    VLOG(2)<<"INSERTING KEYFRAME..."<<endl;
    unique_lock<mutex> lock(mMutexPose);
    VLOG(2)<<"Pose of KF"<<currentFrame->pose<<endl;
    frames_.push_back(currentFrame);
    allPoses.push_back(currentFramePose.clone());
    allPoses1.push_back(currentFramePose.clone());
    poseTimeStamps.push_back(currentFrame->timeStamp);
}

void MonoFrontEnd::processFrame(){
    currentFrame->extractFeatures();
    // initialize all the landmark indices to -1
    currentFrame->lIds = vector<int>( currentFrame->image_kps_undist.size(), -1);
}

cv::Mat MonoFrontEnd::solveF_8point(const vector<cv::Point2f> &kps1,const vector<cv::Point2f> &kps2)
{
    const int N = kps1.size();

    cv::Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = kps1[i].x;
        const float v1 = kps1[i].y;
        const float u2 = kps2[i].x;
        const float v2 = kps2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

void MonoFrontEnd::Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

void MonoFrontEnd::normalizeKps(vector<cv::Point2f>& kps, vector<cv::Point2f>& kps_n, Mat& T){
    float meanX = 0;
    float meanY = 0;
    const int N = kps.size();

    kps_n.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += kps[i].x;
        meanY += kps[i].y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        kps_n[i].x = kps[i].x - meanX;
        kps_n[i].y = kps[i].y - meanY;

        meanDevX += fabs(kps_n[i].x);
        meanDevY += fabs(kps_n[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        kps_n[i].x = kps_n[i].x * sX;
        kps_n[i].y = kps_n[i].y * sY;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}


float MonoFrontEnd::CheckFundamental(const cv::Mat &F, vector<cv::Point2f>& kps1,
                                 vector<cv::Point2f>& kps2, vector<bool> &inliers, float sigma)
{
    const int N = kps1.size();

    const float f11 = F.at<float>(0,0);
    const float f12 = F.at<float>(0,1);
    const float f13 = F.at<float>(0,2);
    const float f21 = F.at<float>(1,0);
    const float f22 = F.at<float>(1,1);
    const float f23 = F.at<float>(1,2);
    const float f31 = F.at<float>(2,0);
    const float f32 = F.at<float>(2,1);
    const float f33 = F.at<float>(2,2);

    inliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::Point2f &kp1 = kps1[i];
        const cv::Point2f &kp2 = kps2[i];

        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            inliers[i]=true;
        else
            inliers[i]=false;
    }

    return score;
}

void MonoFrontEnd::generateRandomIndices(int max_ind, vector<vector<int>>& indices_set, int num_sets){

    // Indices for minimum set selection
    vector<size_t> all_indices;
    all_indices.reserve(max_ind);
    vector<size_t> availableIndices;

    for(int i=0; i<max_ind; i++)
    {
        all_indices.push_back(i);
    }

    indices_set = vector< vector<int> >(num_sets,vector<int>(8,0));

    DUtils::Random::SeedRandOnce(0);

    for(int it=0; it<num_sets; it++)
    {
        availableIndices = all_indices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0,availableIndices.size()-1);
            int idx = availableIndices[randi];

            indices_set[it][j] = idx;

            availableIndices[randi] = availableIndices.back();
            availableIndices.pop_back();
        }
    }
}

void MonoFrontEnd::findMatchesMono(MonoFrame* prev, MonoFrame* cur, std::vector<DMatch>& matches){
    vector<vector<DMatch>> matches_mono;
    matches.clear();
    vector<DMatch> matches_ini;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->knnMatch( cur->image_descriptors, prev->image_descriptors,matches_mono, 2);
    Mat mask = Mat::zeros(prev->image_descriptors.rows, cur->image_descriptors.rows, CV_8UC1);
    for(auto &m : matches_mono){
        if(m[0].distance < 0.7*m[1].distance) {
            if (m[0].distance > 50){
                //cout<<"descriptor distance: "<<endl;
                continue;
            }
            Point2f p_prev = prev->image_kps_undist[m[0].trainIdx].pt;
            Point2f p_cur = cur->image_kps_undist[m[0].queryIdx].pt;
            //make sure that the points belong to static areas based on the segmasks
            if (prev->segMask_.at<float>(p_prev.y, p_prev.x) < 0.7 and cur->segMask_.at<float>(p_cur.y, p_cur.x) < 0.7){
                matches_ini.push_back(m[0]);
                mask.at<uchar>(m[0].trainIdx, m[0].queryIdx) = 255;
            }

        }

    }

    matcher->match(prev->image_descriptors, cur->image_descriptors,matches, mask);
}

void MonoFrontEnd::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

int MonoFrontEnd::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::Point2f>& kps1,
                      const vector<cv::Point2f>& kps2, vector<bool> &inliers,
                      const cv::Mat &K, vector<cv::Point3f> &P3D, float th2, vector<bool> &good,
                      float &parallax, const cv::Mat &R1=Mat(), const cv::Mat &t1=Mat())
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    good = vector<bool>(kps1.size(),false);
    P3D.resize(kps1.size());

    vector<float> cosParallaxVec;
    cosParallaxVec.reserve(kps1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    if(R1.empty() and t1.empty()){
        K.copyTo(P1.rowRange(0,3).colRange(0,3));
    }
    else{
        R1.copyTo(P1.rowRange(0,3).colRange(0,3));
        t1.copyTo(P1.rowRange(0,3).col(3));
        P1 = K*P1;
    }

    cv::Mat O1;
    if(R1.empty() and t1.empty()){
        O1 = cv::Mat::zeros(3,1,CV_32F);
    }
    else{
        O1 = -R1.t()*t1;
    }


    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    cv::Mat O2 = -R.t()*t;

    int nGood=0;
    double cosAngleThresh = cos(kf_triangulation_angle_threshold * CV_PI/180. );
    for(size_t i=0, iend=kps1.size();i<iend;i++)
    {
        if(!inliers[i])
            continue;

        const cv::Point2f &kp1 = kps1[i];
        const cv::Point2f &kp2 = kps2[i];
        cv::Mat p3dC1;

        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            good[i]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        Mat p3dC1_w = p3dC1.clone();
        if(!(R1.empty() and t1.empty())){
            p3dC1 = R1*p3dC1+t1;
        }


        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<float>(2)<=0 && cosParallax<cosAngleThresh)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1_w+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<cosAngleThresh)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.x)*(im1x-kp1.x)+(im1y-kp1.y)*(im1y-kp1.y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.x)*(im2x-kp2.x)+(im2y-kp2.y)*(im2y-kp2.y);

        if(squareError2>th2)
            continue;

        cosParallaxVec.push_back(cosParallax);
        P3D[i] = cv::Point3f(p3dC1_w.at<float>(0),p3dC1_w.at<float>(1),p3dC1_w.at<float>(2));
        nGood++;

        if(cosParallax<cosAngleThresh)
            good[i]=true;
    }

    if(nGood>0)
    {
        sort(cosParallaxVec.begin(),cosParallaxVec.end());

        size_t idx = min(50,int(cosParallaxVec.size()-1));
        parallax = acos(cosParallaxVec[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}


bool MonoFrontEnd::ReconstructF(const vector<cv::Point2f>& kps1, const vector<cv::Point2f>& kps2,
                            vector<bool> &inliers, cv::Mat &F, cv::Mat &K,float sigma,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D,
                            vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    float sigma2 = sigma*sigma;
    for(size_t i=0, iend = inliers.size() ; i<iend; i++)
        if(inliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E = K.t()*F*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    DecomposeE(E,R1,R2,t);
    Mat rotation, translation;
    Mat inlier_mat = Mat::zeros(inliers.size(), 1, CV_8U);
    for(size_t i=0, iend = inliers.size() ; i<iend; i++){
        if(inliers[i])
            inlier_mat.at<uchar>(i,0) = 255;
    }
    Mat E_f;
    E.convertTo(E_f, CV_64F);
    recoverPose(E_f, kps1, kps2,K, rotation, translation, inlier_mat);
    VLOG(2)<<"Decomposed rotation and translations"<<endl;
    VLOG(2)<<"R1: "<<endl;
    VLOG(2)<<R1<<endl;
    VLOG(2)<<"R2: "<<endl;
    VLOG(2)<<R2<<endl;
    VLOG(2)<<"t:"<<endl;
    VLOG(2)<<t<<endl;
    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> P3D1, P3D2, P3D3, P3D4;
    vector<bool> triangulated1,triangulated2,triangulated3, triangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1,t1,kps1,kps2,inliers,K, P3D1, 4.0*sigma2, triangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,kps1,kps2,inliers,K, P3D2, 4.0*sigma2, triangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,kps1,kps2,inliers,K, P3D3, 4.0*sigma2, triangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,kps1,kps2,inliers,K, P3D4, 4.0*sigma2, triangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = P3D1;
            vbTriangulated = triangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = P3D2;
            vbTriangulated = triangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = P3D3;
            vbTriangulated = triangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = P3D4;
            vbTriangulated = triangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}


void MonoFrontEnd::FindFundamentalMatrix(vector<Point2f> kps1, vector<Point2f> kps2, Mat& F, vector<bool> &inliers){

    // normalize the keypoints.
    int maxIter = 200;
    float sigma = 1.0;
    float score = 0.0;

    vector<cv::Point2f> kps1_n, kps2_n;
    cv::Mat T1, T2;
    normalizeKps(kps1,kps1_n, T1);
    normalizeKps(kps2,kps2_n, T2);
    cv::Mat T2t = T2.t();
    //number of matches
    const int N = kps1.size();
    //// Generate random seuence of 8 points within the range
    vector<vector<int>> indices_set;
    generateRandomIndices(N, indices_set, maxIter);
    inliers = vector<bool>(N,false);

    vector<cv::Point2f> kps1_cur(8);
    vector<cv::Point2f> kps2_cur(8);
    cv::Mat F_cur;
    vector<bool> inliers_cur(N,false);
    float score_cur;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<maxIter; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = indices_set[it][j];

            kps1_cur[j] = kps1_n[idx];
            kps2_cur[j] = kps2_n[idx];
        }

        cv::Mat Fn = solveF_8point(kps1_cur,kps2_cur);

        F_cur = T2t*Fn*T1;

        score_cur = CheckFundamental(F_cur,kps1,kps2, inliers_cur, sigma);

        if(score_cur>score)
        {
            F = F_cur.clone();
            inliers = inliers_cur;
            score = score_cur;
        }
    }

}



void MonoFrontEnd::initialization(){

    bool FUND = true;
    ///initializing purely as a monocular system
    if(frames_.size() == 0 ){
        currentFramePose = cv::Mat::eye(3, 4, CV_64F);
        currentFrame->pose = cv::Mat::eye(4,4, CV_64F);
        insertKeyFrame();
        return;
    }
    else
    {
        assert(frames_.size() == 1);
        // match the features between current frame and the prev frame
        MonoFrame* prevFrame = frames_.back();
        std::vector<DMatch> matches_mono;
        vector<Point2f> kp_pts1_mono, kp_pts2_mono;
        Mat mask_mono;
        findMatchesMono(prevFrame, currentFrame, matches_mono);

        for(auto &m : matches_mono){
            Point2f p_prev = prevFrame->image_kps_undist[m.queryIdx].pt;
            Point2f p_cur = currentFrame->image_kps_undist[m.trainIdx].pt;
            kp_pts1_mono.push_back(p_prev);
            kp_pts2_mono.push_back(p_cur);
        }
        VLOG(2)<<"Number of good matches: "<<matches_mono.size()<<endl;
        // if there not enough matches exit
        if(matches_mono.size() >= 100){

            cv::Mat rotation_t = Mat::zeros(3, 3, CV_32F), translation_t = Mat::zeros(3, 1, CV_32F);
            bool res = false;
            vector<cv::Point3f> P3D;
            vector<bool> Triangulated;
            vector<bool> inliers;
            int num_inliers = 0;
            int num_good_tri = 0;
            ///using normalized coordinates and fundMat
            if(FUND){
                Mat F,K;
                FindFundamentalMatrix(kp_pts1_mono, kp_pts2_mono, F , inliers );
                //Get the pose and triangulate the inliers
                // Compute Essential Matrix from Fundamental Matrix
                Kmat_.convertTo(K, CV_32F);
                res = ReconstructF(kp_pts1_mono,kp_pts2_mono, inliers, F, K,1.0,
                                   rotation_t, translation_t, P3D, Triangulated, 1.0, 50);
                for(int p=0; p < inliers.size() ; p++){
                    if(inliers[p])
                        num_inliers++;
                }
            }
            else{
                ///ESSENIAL MATRIX
                cv::Mat E_mat;
                //find the essential matrix
                E_mat = findEssentialMat(kp_pts1_mono, kp_pts2_mono , Kmat_,cv::RANSAC,0.97, 1.0, mask_mono);
                //recover pose
                recoverPose(E_mat, kp_pts1_mono, kp_pts2_mono, Kmat_, rotation_t, translation_t, mask_mono);
                //triangulate the points
                inliers.clear();
                int i=0;
                for(auto &m : matches_mono) {
                    if (mask_mono.at<uchar>(i)){
                        inliers.push_back(true);
                        num_inliers++;
                    }
                    else
                        inliers.push_back(false);
                    i++;
                }
                float parallax;
                int num_good = CheckRT(rotation_t,translation_t,kp_pts1_mono,kp_pts2_mono,inliers,
                                       Kmat_, P3D, 4.0*1.0*1.0, Triangulated, parallax);
                // if the trianglated points are a minimum number and have enough parallx
                if (num_good >= 50 and parallax > 1.0)
                {
                    res = true;

                }
            }
            // if there are enough inliers
            if(res){

                Rodrigues(rotation_t, prev_rvec);
                prev_tvec = translation_t.clone();

                // insert keyframe and initialize the map
                //get the number of inliers and initial pose estimate
                //int num_inliers = countNonZero(mask_mono);
                //convert the Tcw to Twc
                cv::Mat rotation, translation;
                rotation = rotation_t.t();
                translation = -rotation_t.t() *translation_t;

                //triangulate the matches to form landmarks
                Mat T_mono = Mat::eye(4, 4, CV_32F);
                rotation.copyTo(T_mono(Range(0, 3), Range(0, 3)));
                translation.copyTo(T_mono(Range(0, 3), Range(3, 4)));
                T_mono.convertTo(T_mono, CV_64F);

                // get the last estimated pose WRT world into WTp
                Mat rot_p = allPoses.back().colRange(0,3);
                Mat t_p = allPoses.back().colRange(3,4);
                Mat WTp = Mat::eye(4, 4, CV_64F);
                rot_p.copyTo(WTp(Range(0, 3), Range(0, 3)));
                t_p.copyTo(WTp(Range(0, 3), Range(3, 4)));

                // Now convert the current pose WRT world
                {
                    unique_lock<mutex> lock(mMutexPose);
                    Mat WTc =  WTp*T_mono;
                    currentFramePose = WTc.rowRange(0, 3).clone();
                    currentFrame->pose = WTc.clone();
                    VLOG(2) << " Current Frame pose MONO WRT to last KEYFRAME : " << currentFramePose << endl;
                }


                //insert the triangulated points into the map
                // Now denote the matches as tracks.
                int ii=0;
                for(auto& m : matches_mono){
                    if(Triangulated[ii]){
                        num_good_tri++;
                        int l_id = prevFrame->lIds[m.queryIdx];
                        if( l_id == -1){ // landmark has not been inserted
                            Mat p_world = Mat(P3D[ii]);
                            Mat p_world_d;
                            p_world.convertTo(p_world_d, CV_64F);
                            VLOG(3)<<"landmark "<<ii<<": "<<p_world << ","<<p_world_d<<endl;
                            l_id = map->insertLandmark(p_world_d, prevFrame, m.queryIdx , prevFrame->image_kps_undist[m.queryIdx].pt);
                            prevFrame->lIds[m.queryIdx] = l_id;
                            map->getLandmark(l_id)->addMonoFrame(currentFrame, m.trainIdx, currentFrame->image_kps_undist[m.trainIdx].pt);
                            currentFrame->lIds[m.trainIdx] = l_id;
                        }
                    }
                    ii++;
                }

                prevFrame->numTrackedLMs = num_good_tri;
                currentFrame->numTrackedLMs = num_good_tri;
                //insert the pose
                insertKeyFrame();

                VLOG(2)<<"INITIALIZED"<<endl;
                VLOG(2)<<"number of matches_mono : "<< matches_mono.size()<<endl;
                VLOG(2)<<"number of inliers : "<< num_inliers<<endl;
                VLOG(2)<<"number of good triangulated points : "<< num_good_tri<<endl;
                VLOG(2)<<"Rotation: "<<endl;
                VLOG(2)<<rotation<<endl;
                VLOG(2)<<"translation:"<<endl;
                VLOG(2)<<translation<<endl;
                initialized_ = INITIALIZED;
                return;
            } //end of if res

        } //end of number of matches

    }
    // the initialization was not successful
    // delete the frame from lfFrames and insert the currentframe.
    frames_.clear();
    allPoses.clear();
    allPoses1.clear();
    //allPoses2.clear();
    poseTimeStamps.clear();
    currentFramePose = cv::Mat::eye(3, 4, CV_64F);
    //currentFramePose1 = cv::Mat::eye(3, 4, CV_64F);
    // currentFramePose2 = cv::Mat::eye(3, 4, CV_64F);
    currentFrame->pose = cv::Mat::eye(4, 4, CV_64F);
    insertKeyFrame();


}

void MonoFrontEnd::CheckTriAngle(MonoFrame* prev_KF, MonoFrame* curFr, vector<DMatch> matches_with_lms, vector<bool>& inliers){
    inliers = vector<bool>(matches_with_lms.size(),false);
    int i =0;
    for (auto& m: matches_with_lms){
        Mat pose1 = prev_KF->pose.rowRange(0,3).colRange(3,4);
        Mat pose2 = curFr->pose.rowRange(0,3).colRange(3,4);
        int lid = prev_KF->lIds[m.queryIdx];
        Mat landmark = map->getLandmark(lid)->pt3D;

        ///Get the Ray1 between landmark and the previous pose
        Mat ray1 = landmark - pose1;
        double norm1 = cv::norm(ray1);

        ///Get the Ray2 between landmark and the current pose
        Mat ray2 = landmark - pose2;
        double norm2 = cv::norm(ray2);

        /// Compute the angle between the two rays
        double cosTheta = ray1.dot(ray2)/(norm1*norm2);
        double angle_deg = acos(cosTheta)*(180.0/3.141592653589793238463);

        if(angle_deg > kf_triangulation_angle_threshold) {
            inliers[i] = true;
        }

        i++;
    }
}

bool MonoFrontEnd::trackMono() {

    /// grab the current frame
    //MonoFrame* curFrame = currentFrame;

    /// Initialization ??
    if(initialized_ == NOT_INITIALIZED){
        initialization();
        return true;
    }

    /// After initialization we need to track each incoming frame
    /// WRT the last keyframe.
    /// grab the last keyframe
    MonoFrame* prev_KF = frames_.back();

    VLOG(3)<<"landmarks inds:"<<endl;
    for (auto l : prev_KF->lIds){
        if(l!=-1)
            VLOG(3)<<l<<",";
    }
    VLOG(3)<<endl;
    VLOG(2)<<"Number of landmarks: "<<prev_KF->numTrackedLMs<<endl;
    VLOG(2)<<prev_KF->countLandmarks()<<endl;


    ///Find matches between current and previosu KF
    std::vector<DMatch> matches_mono;
    vector<Point2d> kp_pts1_mono, kp_pts2_mono;
    vector<Point2f> kps1_mono_new, kps2_mono_new;
    vector<KeyPoint> kp_pts1_mono_f, kp_pts2_mono_f;
    vector<Landmark*> existing_lms;
    vector<Point3d> points1_3d;
    Mat mask_mono;
    findMatchesMono(prev_KF, currentFrame, matches_mono);

    std::vector<DMatch> inter_matches_with_landmarks, new_inter_matches;

    for(auto &m : matches_mono){
        //get the landmark corresponding to this match
        int lid = prev_KF->lIds[m.queryIdx];
        Landmark* l = map->getLandmark(lid);
        // if there exists a landmark
        if(l){
            // double x = (lf_cur->image_kps_undist[0][m.trainIdx].pt.x - camconfig_.K_mats_[0].at<double>(0,2))/camconfig_.K_mats_[0].at<double>(0,0);
            // double y  = (lf_cur->image_kps_undist[0][m.trainIdx].pt.y - camconfig_.K_mats_[0].at<double>(1,2)) / camconfig_.K_mats_[0].at<double>(1,1);

            Point2d p_cur = Point2d(currentFrame->image_kps_undist[m.trainIdx].pt);
            kp_pts2_mono.push_back(p_cur);
            kp_pts2_mono_f.push_back(currentFrame->image_kps_undist[m.trainIdx]);

            Point2d p_cur_1 = Point2d(prev_KF->image_kps_undist[m.queryIdx].pt);
            kp_pts1_mono.push_back(p_cur_1);

            kp_pts1_mono_f.push_back(prev_KF->image_kps_undist[m.queryIdx]);

            Point3d pt3d = Point3d(l->pt3D);

            points1_3d.push_back(pt3d);
            existing_lms.push_back(l);
            inter_matches_with_landmarks.push_back(m);
        }
        else{//landmark is not there
            //save these points for triangulation.
            //float x = (prev_KF->image_kps_undist[0][m.queryIdx].pt.x - camconfig_.K_mats_[0].at<double>(0,2))/camconfig_.K_mats_[0].at<double>(0,0);
            //float y  = (prev_KF->image_kps_undist[0][m.queryIdx].pt.y - camconfig_.K_mats_[0].at<double>(1,2)) / camconfig_.K_mats_[0].at<double>(1,1);

            Point2f p_prev = prev_KF->image_kps_undist[m.queryIdx].pt; //prev_KF->image_kps_undist[0][m.queryIdx].pt;
            kps1_mono_new.push_back(p_prev);


            Point2f p_cur = currentFrame->image_kps_undist[m.trainIdx].pt;
            kps2_mono_new.push_back(p_cur);
            new_inter_matches.push_back(m);

        }
    }

    //pnp ransac as opposed to essential matrix estimation
    ////// PNP RANSAC /////////////////////
    Mat rvec_cw, tvec_cw, rvec_cw_it, tvec_cw_it;
    vector<int> inliers_indices;
    Mat inliers_bool =  Mat::zeros(points1_3d.size(), 1,CV_8U);

    // estimate the pose of the new camera WRT world map points
    cv::solvePnPRansac(points1_3d, kp_pts2_mono, Kmat_, Mat::zeros(4,1,CV_64F),
                       prev_rvec, prev_tvec,true,
                       250, 2.0, 0.97, inliers_indices,SOLVEPNP_ITERATIVE);

    rvec_cw = prev_rvec.clone();
    tvec_cw = prev_tvec.clone();

    //convert the rotation vector into rotation matrix
    Mat rot_cw =  Mat::eye(3, 3, CV_64F);
    cv::Rodrigues(rvec_cw, rot_cw);

    // create a inlier boolean mask for the inliers indices in the matches
    for(auto &i: inliers_indices )
        inliers_bool.at<uchar>(i) = 255;
    VLOG(2)<<"Number of inliers after pnp: "<<inliers_indices.size()<<endl;
    VLOG(2)<<"//////////// PNP RANSAC MAT COMPUTATION ///////////"<<endl;

    //-- Draw matches
    Mat img_matches_mono;
    //drawMatches( img1, prev_KF->image_kps_undist[0], img2, lf_cur->image_kps_undist[0], good_matches, img_matches_mono,Scalar::all(-1),
    //            Scalar::all(-1), mask_mono, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    Mat img2;
    cvtColor(currentFrame->img_, img2 ,COLOR_GRAY2BGR);
    img_matches_mono =  img2.clone();
    mask_mono= Mat::ones(matches_mono.size(), 1,CV_8U);

//    drawMatchesArrows(img_matches_mono, prev_KF->image_kps_undist, currentFrame->image_kps_undist, matches_mono, mask_mono , cv::Scalar(150,200, 0));
    drawMatchesArrows(img_matches_mono, prev_KF->image_kps_undist, currentFrame->image_kps_undist, inter_matches_with_landmarks, inliers_bool, cv::Scalar(150,200, 0));
    //-- Show detected matches
    imshow("Matches_mono", img_matches_mono );
    waitKey(500);


    ///Convert the pose into world frame and save into the required variables
    Mat T_mono = Mat::eye(4, 4, CV_64F);
    Mat rotation= rot_cw.t();
    Mat translation = (-rot_cw.t() * tvec_cw);
    int num_inliers = inliers_indices.size();
    rotation.copyTo(T_mono(Range(0, 3), Range(0, 3)));
    translation.copyTo(T_mono(Range(0, 3), Range(3, 4)));


    currentFramePose = T_mono.rowRange(0, 3).clone();
    currentFrame->pose = T_mono.clone();
    VLOG(2) << " Current Frame pose : " << currentFramePose << endl;


    VLOG(1) << "Total Number of Matches "<<matches_mono.size()<<endl;
    VLOG(1) << "Number of inter_matches_with_landmarks : "<<inter_matches_with_landmarks.size()<<endl;
    VLOG(1) << "Number of new intermatches : "<<new_inter_matches.size()<<endl;
    VLOG(1) << "Number of tracked landmarks : " << num_inliers << ","<<(float)num_inliers/prev_KF->numTrackedLMs<<endl;


    //// KEYFRAME INSERTION ///
    int num_triangulated=0;
    int prevlms = prev_KF->numTrackedLMs;
    bool ret = false;

    Mat relativePose = prev_KF->pose.col(3) - T_mono.col(3);
    double baseline = norm(relativePose);
    Mat relative_transformation = prev_KF->pose.inv() * T_mono;

    double rotangle = abs(atan2(-1 * relative_transformation.at<double>(2,0),
                                sqrt(pow(relative_transformation.at<double>(2,1), 2) +
                                     pow(relative_transformation.at<double>(2,2), 2)))) *  (180.0/3.141592653589793238463);
    VLOG(2)<<" translation: "<<baseline<<endl;
    VLOG(2)<<" rot angle: "<<rotangle<<endl;


    //if(num_inliers < 15)
    //    return ret;


    if((float)num_inliers/prev_KF->numTrackedLMs <= 0.3 && (baseline >= kf_translation_threshold || rotangle >= kf_rotation_threshold)) {

        /// if num of inliers falls below a threshold insert the new keyframe
        /// - assign landmarks to features, optimize R|t and
        /// else we have enough tracked points. Save the current pose and move on to next frame


        /// Triangulate new landmarks and insert them as well.
        /// Actually we need to insert them only if they are seen in min_frames
        ///////////////////////////////////////////////////////////////////////////
        //Triangulate new matches and insert good triangulations into the map
        float parallax;
        vector<cv::Point3f> P3D;
        vector<bool> Triangulated;
        int num_tri_points=0;
        vector<bool> inliers_new = vector<bool>(kps1_mono_new.size(),true); // we wanna triangulate all the new points
        Mat K, rot_KF_cw, trans_KF_cw;

        // Get the rotation and translation of the previous keyFrame (The extrinsic params)
        rot_KF_cw = prev_KF->pose.rowRange(0,3).colRange(0,3).t();
        trans_KF_cw = -1 * rot_KF_cw * prev_KF->pose.rowRange(0,3).colRange(3,4);

        Kmat_.convertTo(K, CV_32F);
        rot_KF_cw.convertTo(rot_KF_cw, CV_32F);
        trans_KF_cw.convertTo(trans_KF_cw,CV_32F);

        rot_cw.convertTo(rot_cw, CV_32F);
        tvec_cw.convertTo(tvec_cw, CV_32F);

        /// Here num_good is the number of points for which the reprojection error is < thresh, but not necessarily the agle is preserved
        int num_good = CheckRT(rot_cw,tvec_cw ,kps1_mono_new, kps2_mono_new, inliers_new,
                               K, P3D, 4.0*1.0*1.0, Triangulated, parallax,rot_KF_cw, trans_KF_cw );





        /// For inter matches with landmarks add only the ones which have a good parallax
        vector<bool> mask_lm_matches = vector<bool>(kp_pts1_mono_f.size(),true);
        vector<cv::Point3f> P3D_tmp;
        vector<bool> inliers_lm_matches;
        /// Here num_good_lm_matches is the number of points for which the reprojection error is < thresh, but not necessarily the agle is preserved
        //int num_good_lm_matches = CheckRT(rot_cw,tvec_cw ,kp_pts1_mono_f, kp_pts2_mono_f, mask_lm_matches,
        //                                  K, P3D_tmp, 4.0*1.0*1.0, inliers_lm_matches, parallax, rot_KF_cw, trans_KF_cw );
        CheckTriAngle(prev_KF, currentFrame, inter_matches_with_landmarks, inliers_lm_matches);

        int num_inliers_good_parallax=0;
        for (auto& i : inliers_indices) {
            DMatch m = inter_matches_with_landmarks[i];
            if(inliers_lm_matches[i])
            {
                int l_id = prev_KF->lIds[m.queryIdx];
                map->getLandmark(l_id)->addMonoFrame(currentFrame, m.trainIdx, currentFrame->image_kps_undist[m.trainIdx].pt);
                currentFrame->lIds[m.trainIdx] = l_id;
                num_inliers_good_parallax++;
            }
        }

        for (int i = 0 ; i < new_inter_matches.size(); i++){
            if(Triangulated.at(i)){
                //create Landmarks for each inlier and insert them into global map and well as
                // the keyframes.
                Mat pt3d = Mat(P3D.at(i));
                pt3d.convertTo(pt3d, CV_64F);
                DMatch m = new_inter_matches[i];
                int l_id = map->insertLandmark( pt3d, prev_KF, m.queryIdx, prev_KF->image_kps_undist[m.queryIdx].pt);
                map->getLandmark(l_id)->addMonoFrame(currentFrame, m.trainIdx, currentFrame->image_kps_undist[m.trainIdx].pt);
                prev_KF->lIds[m.queryIdx] = l_id;
                currentFrame->lIds[m.trainIdx] = l_id;
                //triangulatedPts[i] = pt;
                num_triangulated++;
            }
        }
        VLOG(2)<<"Number of good new triangulated points : "<< num_triangulated<<endl;
        VLOG(2)<<"Number of tracked landmarks with good parallax : "<< num_inliers_good_parallax<<endl;

        prev_KF->numTrackedLMs = prev_KF->numTrackedLMs + num_triangulated;
        currentFrame->numTrackedLMs = num_inliers_good_parallax + num_triangulated;
        insertKeyFrame();
        ret = true;

    }

    std::stringstream ss;
    ss <<std::setprecision(6)<<std::fixed<<currentFrame->timeStamp<<","<<std::setw(4)<< frames_.size()<<","<<std::setw(12)  <<prevlms<<","<<std::setw(11)  <<inter_matches_with_landmarks.size()<<","
       <<std::setw(12)  <<new_inter_matches.size()<<","<<std::setw(11)  <<num_inliers<<","<<std::setw(11)  <<num_triangulated;
    std::string s = ss.str();
    VLOG(1)<<s<<endl;
    if(!ret){
        delete currentFrame;
    }
    return ret;

    /*Mat T_mono = Mat::eye(4, 4, CV_64F);
    //if(num_inliers > 15){
    rotation_t.copyTo(T_mono(Range(0, 3), Range(0, 3)));
    translation_t.copyTo(T_mono(Range(0, 3), Range(3, 4)));
    //}

    // Now denote the matches as tracks.
    int ii=0;
    for(auto& m : matches_mono){
        if(mask_mono.at<uchar>(ii)){
            int l_id = prev_KF->lIds_mono[m.queryIdx];
            if( l_id == -1){ // landmark has not been inserted
                Mat p_world = Mat::zeros(Size(3,1), CV_64F);
                l_id = map_mono->insertLandmark(p_world, prev_KF, prev_KF->image_kps_undist[0][m.queryIdx].pt);
                prev_KF->lIds_mono[m.queryIdx] = l_id;
            }
            //landmark already exists
            // add the observation
            // add the observation for the same landmark
            Landmark* l = map_mono->getLandmark(l_id);
            if (l) {
                l->addLfFrame(lf_cur, lf_cur->image_kps_undist[0][m.trainIdx].pt);
                lf_cur->lIds_mono[m.trainIdx] = l_id;
            }
        }
        ii++;
    }

    cout<<"total number of landmarks in mono : "<<map_mono->num_lms<<endl; */
}

void MonoFrontEnd::reset() {
    //clear the data structures which are no longer needed to store
    if(frames_.size() > 2){
        //cleanup unncessary data before inserting
        int ref_cam=0;
        MonoFrame* f = frames_[frames_.size()-2-1];
        f->img_ = cv::Mat();
        f->segMask_ = cv::Mat();

    }


}

//////////?Repeating ones

void MonoFrontEnd::drawMatchesArrows(cv::Mat& img, vector<KeyPoint>& kps_prev, vector<KeyPoint>& kps_cur, std::vector<DMatch> matches, Mat mask, cv::Scalar color){

    int i =0;
    if (img.channels() != 3){
        Mat imgColor;
        cvtColor(img,imgColor , COLOR_GRAY2BGR);
        img = imgColor;
    }

    for(auto &m : matches){
        Point2f p1 = kps_prev[m.queryIdx].pt;
        Point2f p2 = kps_cur[m.trainIdx].pt;

        if(mask.at<uchar>(i)){
            cv::circle(img,p1,2,Scalar(0,255,0),1);
            cv::arrowedLine(img, p1, p2, color);

        }
        else{
            cv::circle(img,p1,2,Scalar(0,0,255),1);
            cv::arrowedLine(img, p1, p2, Scalar(0,0,255));
        }
        i++;
    }

}