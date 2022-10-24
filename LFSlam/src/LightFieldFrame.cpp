//
// Created by Pushyami Kaveti on 6/29/20.
//
#include "LFSlam/LightFieldFrame.h"
#include <opencv2/sfm/projection.hpp>
#include <opencv2/sfm/triangulation.hpp>
#include <FeatureVector.h>
using namespace std;
using namespace cv;
//using namespace opengv;
using namespace std::chrono;


LightFieldFrame::LightFieldFrame(vector<Mat> img_set, vector<Mat> segmap_set,ORBVocabulary* vocab, ORBextractor* orb_obj, vector<ORBextractor*> orbs,
                                 CamArrayConfig& camconfig, int id,double tStamp,cv::Ptr<cv::CLAHE> clahe, bool debug):num_cams_(camconfig.num_cams_),img_size(camconfig.im_size_),
                                 camconfig_(camconfig), DEBUG_MODE(debug), orb_vocabulary(vocab), orBextractor(orb_obj), frameId(id), timeStamp(tStamp), clahe(clahe) {
    //create the ORBDatabase object to parse and store images into Bag of words vector
    //orb_database = new ORBDatabase(*orb_vocabulary, true , 2);
    orBextractors = orbs;
    numTrackedLMs =0;
    intramatch_size = 0;
    mono_size = 0;

    num_inliers_per_view = vector<int>(num_cams_, 0);
    num_triangulated_per_view = vector<int>(num_cams_, 0);

    //orbCuda = cuda::ORB::create(2000,1.2f, 8, 31, 0, 2, 0, 31, 20, true);

    //initialize all the camera matrices
    // make all rotations and translations a eigen vector3D and matrix3D
    rotations_ptr = new std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d >>(num_cams_);
    for (int i =0; i <num_cams_ ; i++){

        Eigen::Matrix3d rot;
        Eigen::Vector3d trans;
        // rotation matrix to eigen
        cv::cv2eigen(camconfig_.R_mats_[i], rot);
        (*rotations_ptr)[i] = rot.transpose();
        // translation to eigen
        //cv::cv2eigen(frontend.camconfig_.t_mats_[i], trans);
        cv::Mat t=camconfig_.t_mats_[i];
        trans<<t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0);
        trans = -rot.transpose() * trans;
        translations.push_back(trans);

    }
    pose = Mat::eye(4,4, CV_64F);
    cov =  Mat::zeros(6,6, CV_64F);
    setData(img_set, segmap_set);
}

LightFieldFrame::LightFieldFrame(CamArrayConfig& camconfig): camconfig_(camconfig),num_cams_(camconfig.num_cams_) {}

LightFieldFrame::~LightFieldFrame() {
    imgs.clear();
    segMasks.clear();
    image_descriptors.clear();
    intraMatches.clear();
}


/// Light Field Frame for refocus image estimation. n images, n segmasks, camera_array info.
///


/// Method to update the class variables with the current
/// image set from light field array
/// \param img_set
/// \param segmap_set
void LightFieldFrame::setData(vector<Mat> img_set, vector<Mat> segmap_set) {
    imgs.clear();
    segMasks.clear();
    if(img_set.size() != num_cams_ or segmap_set.size() !=num_cams_){
        cout<<"ERROR:: number of images is wrong";
        return;
    }
    // visualize the rectified frame and check
    //Mat all;
    //all.create(img_set[0].rows, img_set[0].cols * img_set.size(), CV_8UC1);

    cv::Size siz = cv::Size(img_set[0].cols, img_set[0].rows);
    for(uint im=0; im<img_set.size(); ++im){
        Mat img = img_set[im].clone();
        multiply(img, 255, img);
        img.convertTo(img,CV_8U);
        if (img.channels() == 3){
            Mat imgGray;
            cvtColor(img,imgGray , COLOR_BGR2GRAY);
            img = imgGray;
        }
        Mat segmap = segmap_set[im].clone(); // no need to convert this into 8U
        if (segmap.channels() == 3){
            vector<Mat> channels(3);
            split(segmap, channels);
            segmap = channels[0];
        }
        if(camconfig_.RECTIFY){
            //undistort before applying rectification transform
            Mat undistImg;
            cv::undistort(img, undistImg, camconfig_.K_mats_[im], camconfig_.dist_coeffs_[im] );
            //cv::warpPerspective(undistImg, undistImg, camconfig_.Rect_mats_[im], img.size() );
            //cv::imshow("image rectified", undistImg);
           // cvWaitKey(0);
            //undistImg.copyTo(all.colRange(siz.width*im, siz.width*(im+1)));
            img = undistImg;

            // do the same for segmaps
            Mat undistSegmap;
            cv::undistort(segmap, undistSegmap, camconfig_.K_mats_[im], camconfig_.dist_coeffs_[im] );
           // cv::warpPerspective(undistSegmap, undistSegmap, camconfig_.Rect_mats_[im], img.size() );
            segmap = undistSegmap;
        }

        //clahe->apply(img,img);
        imgs.push_back(img.clone());
        segMasks.push_back(segmap);
    }
    //correct the baseline
    //for (int j = 0; j < siz.height; j += 32)
     //   line(all, cv::Point(0, j), cv::Point(siz.width * img_set.size(), j),cv::Scalar(255));

    //cv::imshow("image rectified", all);
    //cv::waitKey(0);
    //cv::imshow("image ", imgs[0]);
    //cvWaitKey(0);

}

void LightFieldFrame::extractORBCuda(){
    assert(num_cams_ == imgs.size());
    image_descriptors.clear();
    image_descriptors.reserve(num_cams_);
    image_kps.clear();
    image_kps.reserve(num_cams_);
    image_kps_undist.clear();
    image_kps_undist.reserve(num_cams_);

    // each thread gets its own CUDA stream
    cv::cuda::Stream stream;

    for (int i =0; i <num_cams_; i++){
        vector<cv::KeyPoint> kps;
        cv::Mat descs;

        //auto start = high_resolution_clock::now();
        // upload to GPU
        cv::cuda::GpuMat frame_d;
        frame_d.upload(imgs[i], stream);
        // detect and compute features
        cv::cuda::GpuMat kp_d;
        cv::cuda::GpuMat desc_d;

        orbCuda->detectAndComputeAsync(frame_d, cv::cuda::GpuMat(), kp_d, desc_d, false, stream);
        stream.waitForCompletion();
        orbCuda->convert(kp_d, kps);
        desc_d.download(descs);
        //auto stop_intramatch = high_resolution_clock::now();
        //auto duration = duration_cast<milliseconds>(stop_intramatch - start);
        //cout<<"time taken ORB CUDA single image: "<<duration.count()<<endl;
        //convert the descriptors from Mat into a vector<mat> suitable for DBoW2
        image_kps.push_back(kps);
        //UNDISTORT THE KEY POINTS
        if(camconfig_.RECTIFY){
            image_kps_undist.push_back(kps);
        }
        else
            UndistortKeyPoints(i);
        std::vector<cv::Mat> vec_desc;
        vec_desc.reserve(descs.rows);
        for (int j=0;j<descs.rows;j++)
            vec_desc.push_back(descs.row(j));
        image_descriptors.push_back(vec_desc);

    }

}
void LightFieldFrame::extractFeaturesParallel(){
    assert(num_cams_ == imgs.size());
    image_descriptors.clear();
    image_descriptors.reserve(num_cams_);
    image_kps.clear();
    image_kps.reserve(num_cams_);
    image_kps_undist.clear();
    image_kps_undist.reserve(num_cams_);

    std::thread ths[num_cams_];
    for(int cam_ind = 0; cam_ind < num_cams_ ; cam_ind++) {
        image_kps.push_back(vector<cv::KeyPoint>());
        image_kps_undist.push_back(vector<cv::KeyPoint>());
        image_descriptors.push_back(vector<cv::Mat>());

        ths[cam_ind] = thread(&LightFieldFrame::extractFeatureSingle, this, cam_ind);

    }
    for(int cam_ind = 0; cam_ind < num_cams_ ; cam_ind++)
        ths[cam_ind].join();
}

void LightFieldFrame::extractFeatureSingle(int cam_ind){

    vector<cv::KeyPoint> kps;
    cv::Mat descs;
    ORBextractor* orbObj =orBextractors[cam_ind];

    (*orbObj)(imgs[cam_ind],cv::Mat(),kps,descs);
    //convert the descriptors from Mat into a vector<mat> suitable for DBoW2
    image_kps[cam_ind] = kps;
    //UNDISTORT THE KEY POINTS
    if(camconfig_.RECTIFY){
        image_kps_undist[cam_ind]= kps;
    }
    else
        UndistortKeyPoints(cam_ind);
    std::vector<cv::Mat> vec_desc;
    vec_desc.reserve(descs.rows);
    for (int j=0;j<descs.rows;j++)
        vec_desc.push_back(descs.row(j));
    image_descriptors[cam_ind]= vec_desc;
}

///Method to extract ORB features of the images
///The key points and image descriptors are updated
void LightFieldFrame::extractFeatures(){
    assert(num_cams_ == imgs.size());
    image_descriptors.clear();
    image_descriptors.reserve(num_cams_);
    image_kps.clear();
    image_kps.reserve(num_cams_);
    image_kps_undist.clear();
    image_kps_undist.reserve(num_cams_);

    for (int i =0; i <num_cams_; i++){
        vector<cv::KeyPoint> kps;
        cv::Mat descs;
        //cv::namedWindow("img");
        //imshow("img", imgs[i]);
        //cv::waitKey(0);
        (*orBextractor)(imgs[i],cv::Mat(),kps,descs);
        //convert the descriptors from Mat into a vector<mat> suitable for DBoW2
        image_kps.push_back(kps);
        //UNDISTORT THE KEY POINTS
        if(camconfig_.RECTIFY){
            image_kps_undist.push_back(kps);
        }
        else
            UndistortKeyPoints(i);
        std::vector<cv::Mat> vec_desc;
        vec_desc.reserve(descs.rows);
        for (int j=0;j<descs.rows;j++)
            vec_desc.push_back(descs.row(j));
        image_descriptors.push_back(vec_desc);

    }

}

void LightFieldFrame::UndistortKeyPoints(int cam)
{
    if(camconfig_.dist_coeffs_[cam].at<float>(0)==0.0)
    {
        image_kps_undist[cam] = image_kps[cam];
        return;
    }
    int N =  image_kps[cam].size();
    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=image_kps[cam][i].pt.x;
        mat.at<float>(i,1)=image_kps[cam][i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,camconfig_.K_mats_[cam],camconfig_.dist_coeffs_[cam]);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    vector<KeyPoint> undist_kps;
    undist_kps.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = image_kps[cam][i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        undist_kps[i]=kp;
    }
    image_kps_undist.push_back(undist_kps);
}


///Method to compute the bag of words vector of the images
///in light field array and add that to the bag of words data base

void LightFieldFrame::parseandadd_BoW(){
    set<DBoW2::NodeId> uniquewords;
    for(int i =0; i < num_cams_ ; i++){
        std::vector<cv::Mat> vec_desc = image_descriptors[i];
        // Bag of Words Vector structures.
        DBoW2::BowVector bowVec;
        DBoW2::FeatureVector featVec;

        orb_vocabulary->transform(vec_desc,bowVec,featVec,4);

        // add the image to the database
        BoW_feats.push_back(featVec);
        BoW_vecs.push_back(bowVec);

        DBoW2::BowVector::const_iterator itr ;
        for(itr = bowVec.begin() ; itr != bowVec.end() ; ++itr){
            uniquewords.insert(itr->first);
        }

    }

}


void LightFieldFrame::parseIntraMatchBoW() {

    std::vector<cv::Mat> vec_desc;

    for (auto &d : intraMatches)
        vec_desc.push_back(d.matchDesc);

    orb_vocabulary->transform(vec_desc,lfBoW,lfFeatVec,4);
}


void LightFieldFrame::testTriangulateIntraMatches(){
    double b = 0.165;
    Mat gt_pt = Mat(4, 1, CV_64FC1);
    gt_pt.at<double>(0, 0) = 0;
    gt_pt.at<double>(1, 0) = 0;
    gt_pt.at<double>(2, 0) = 6.0;
    gt_pt.at<double>(3, 0) = 1.0;
    Mat_<double> x(2, num_cams_);
    vector<int> view_inds;
    for(int i =0; i < num_cams_ ; i++){
        Mat Rt = build_Rt(camconfig_.R_mats_[i], camconfig_.t_mats_[i]);
        Mat Ps = camconfig_.K_mats_[i]*Rt;
        Mat projected = Ps * gt_pt;
        double expected_x = projected.at<double>(0,0) / projected.at<double>(2,0);
        double expected_y = projected.at<double>(1,0) / projected.at<double>(2,0);
        x(0, i) = expected_x;
        x(1, i) = expected_y;
        view_inds.push_back(i);
        cout<<"cam"<<i<<": ("<<expected_x<<","<<expected_y<<") , ";
    }
    Eigen::Vector3d pt;
    cout<<endl;
    triangulateIntraMatches(x, view_inds, pt);
    cout<<endl;
    cout<<"3D point : "<<pt<<endl;

}

void LightFieldFrame::triangulateIntraMatches(const Mat_<double> &x, vector<int> view_inds, Eigen::Vector3d& pt){
     // takes the projected 2D points in N views
     // the corresponding p[rojection matrices and
     //triangulates
    CV_Assert(x.rows == 2);
    unsigned nviews = x.cols;
    CV_Assert(nviews == view_inds.size());

    cv::Mat_<double> design = cv::Mat_<double>::zeros(3*nviews, 4 + nviews);
    for (unsigned i=0; i < nviews; ++i) {
        Mat Rt = build_Rt(camconfig_.R_mats_[view_inds[i]], camconfig_.t_mats_[view_inds[i]]);
        //cout<<Rt<<endl;

        Mat Ps = camconfig_.K_mats_[view_inds[i]]*Rt;
        //cout<<Ps<<endl;
        //cout<<x(0, i)<<","<<x(1, i)<<endl;
        for(char jj=0; jj<3; ++jj)
            for(char ii=0; ii<4; ++ii)
                design(3*i+jj, ii) = -Ps.at<double>(jj, ii);
        design(3*i + 0, 4 + i) = x(0, i);
        design(3*i + 1, 4 + i) = x(1, i);
        design(3*i + 2, 4 + i) = 1.0;
    }

    Mat X_and_alphas;
    cv::SVD::solveZ(design, X_and_alphas);
    Vec3d X;
    cv::sfm::homogeneousToEuclidean(X_and_alphas.rowRange(0, 4), X);
    for(int j=0; j<3; ++j)
        pt[j] = X[j];
}

///
/// \param kps_1
/// \param kps_2
void LightFieldFrame::triangulateIntraMatches(vector<vector<vector<Point2f>>>& kps_1,vector<vector<vector<Point2f>>>& kps_2){
    pts3D_.clear();
    for (int i =0; i < num_cams_-1 ; i++){
        int ind=0;
        for (int j = i+1 ; j < num_cams_ ; j++ ){
            vector<Point2f> k_points_1, k_points_2;
            k_points_1 = kps_1[i][ind];
            k_points_2 = kps_2[i][ind];
            vector<Point3f> pts;
            triangulate(k_points_1,k_points_2, pts, i, j);
            if (i != 0){
                // if the first camera is not the reference or left most cam
                // then the 3D points should be converted into the reference camera's coorfinate frame
                Mat R = camconfig_.R_mats_[i].t();
                Mat t = -R * camconfig_.t_mats_[i];
                R.convertTo(R,CV_32FC1);
                t.convertTo(t, CV_32FC1);
                cv::Affine3f T(R, t);
                for(int k = 0 ; k <pts.size(); k++){
                    pts[k] = T * pts[k];
                }
            }
            pts3D_.insert( pts3D_.end(), pts.begin(), pts.end() );
            ind++;
        }
    }
}

///
/// \param kps_1
/// \param kps_2
/// \param pts3D
/// \param cam1
/// \param cam2
void LightFieldFrame::
triangulate(vector<Point2f>& kps_1, vector<Point2f>& kps_2, vector<Point3f>& pts3D , int cam1, int cam2 ){

    Mat Rt = build_Rt(camconfig_.R_mats_[cam1], camconfig_.t_mats_[cam1]);
    Mat P1 = camconfig_.K_mats_[cam1]*Rt;
    Rt = build_Rt(camconfig_.R_mats_[cam2], camconfig_.t_mats_[cam2]);
    Mat P2 = camconfig_.K_mats_[cam2]*Rt;

    assert(kps_1.size() == kps_2.size());
    pts3D.clear();
    for (int i =0; i <kps_1.size(); i++){
        cv::Mat pt3D;
        //triangulate points DLT method
        cv::Mat A(4,4,CV_32F);

        A.row(0) = kps_1[i].x*P1.row(2)-P1.row(0);
        A.row(1) = kps_1[i].y*P1.row(2)-P1.row(1);
        A.row(2) = kps_2[i].x*P2.row(2)-P2.row(0);
        A.row(3) = kps_2[i].y*P2.row(2)-P2.row(1);

        cv::Mat u,w,vt;
        cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
        pt3D = vt.row(3).t();
        pt3D = pt3D.rowRange(0,3)/pt3D.at<float>(3);
        pts3D.push_back(cv::Point3f(pt3D.at<float>(0), pt3D.at<float>(1), pt3D.at<float>(2)));
    }

}

/// Method which returns the latest triangulated points
/// \param pts
void LightFieldFrame::getMapPoints(vector<Point3f>& pts){
    unique_lock<mutex> lock(mMutexPts);
    pts.clear();
    pts.insert( pts.end(), pts3D_.begin(), pts3D_.end() );
}

/// Method which returns the latest estimated pose
/// \return
/*cv::Mat LightFieldFrame::getPose(){
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat pose;
    if(currentFramePose.isZero()) {
        pose = cv::Mat::eye(3, 4, CV_32FC1);
    }
    else
        cv::eigen2cv(currentFramePose, pose);
    return pose;
}*/

void LightFieldFrame::computeRepresentativeDesc( vector<Mat> descs, cv::Mat& desc_out ){

    //THis is only temporary for debugging

    // Compute distances between them
    const size_t N = descs.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = orBextractor->DescriptorDistance(descs[i],descs[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    //cout<<"best index : "<<BestIdx<<"out of "<<N<<" best distance : "<<BestMedian<<endl;
    desc_out = descs[BestIdx].clone();
}

bool LightFieldFrame::checkItersEnd(DBoW2::FeatureVector::const_iterator* featVecIters, DBoW2::FeatureVector::const_iterator* featVecEnds) {
    bool res = true;
    for (int i = 0; i < num_cams_ ; i++) {
        res = res && featVecIters[i]->first >= featVecEnds[i]->first;
    }
    return res;
}

void LightFieldFrame::computeDisparity() {

}

/// Method which computes the intra matches between the  keypoints
/// in the component cameras of the light field array. The variables matches,
/// kps_1 and kps_2 are updated with the matches
/// \param matches  : It is a 2D array of size N x Num_cams. matches[i] represents iTH intra
///                   match and gives array of indices to the keypoints in respective cameras.
void LightFieldFrame::computeIntraMatches(vector<IntraMatch>& matches,vector<DBoW2::NodeId >& words_){

    //assert(num_cams_ == orb_database->size());
    assert(num_cams_ == BoW_feats.size());
    matches.reserve(10000);
    //initialize all the iterators
    DBoW2::FeatureVector* featVecs = new DBoW2::FeatureVector[num_cams_];
    DBoW2::FeatureVector::const_iterator* featVecIters = new  DBoW2::FeatureVector::const_iterator[num_cams_];
    DBoW2::FeatureVector::const_iterator* featVecEnds = new DBoW2::FeatureVector::const_iterator[num_cams_];

    ///For each camera get the bag of words feature vector
    for(unsigned int i =0; i < num_cams_ ; i++) {
        //featVecs[i] = orb_database->retrieveFeatures(i);
        DBoW2::FeatureVector b = BoW_feats.at(i);
        featVecs[i] = b;
        featVecIters[i] = featVecs[i].begin();
        featVecEnds[i] =  std::prev(featVecs[i].end());  //std::next(featVecs[i].begin(),200);  //featVecs[i].end();    //
        //cout<<"endID for cam "<<i<<": "<<featVecEnds[i]->first<<"vs true end: "<<std::prev(featVecs[i].end())->first<<endl;
    }
    //featVecIters[0] =featVecEnds[0];
    //featVecIters[3] =featVecEnds[3];
    //featVecIters[4] =featVecEnds[4];
    int intraMatchInd = 0;

    //set<DBoW2::NodeId > words_;
    // Check the terminating condition where all the iterators have reached the end.
    // RUn the loop if the end has not reached
    int itr=0;
    int itr_cng_match =0;
    auto start_intramatch = high_resolution_clock::now();
    /// We simultaneously iterate over all the 5 featute vectors by finding next common word node
    /// until we reach the end of all the iterators of the feature vectors
    while(!checkItersEnd(featVecIters,featVecEnds)) {
        itr++;
         // feat wordID-> bag

         /// sort the features in the ascending order of their word ids
         vector<DBoW2::NodeId> wordIds;
         for(int i = 0 ; i < num_cams_; i++){
             if(featVecIters[i] == featVecEnds[i])
                 wordIds.push_back(INT_MAX);
             else
                 wordIds.push_back(featVecIters[i]->first);
         }

         /// Find the lowest words with matching indices among all the data cameras
         int min_val=INT_MAX-1;
         int second_min_val = INT_MAX-1;
         vector<int> selected_cams;
         vector<vector<int>> matchedFlags;
         matchedFlags.reserve(num_cams_);

         for (int i =0 ;i < num_cams_ ; i++){
            //if (i !=0 and i != 1 and i !=2)
            //     continue;
            if (wordIds[i] < min_val){
                second_min_val = min_val;
                min_val = wordIds[i];
                selected_cams.clear();
                matchedFlags.clear();
                selected_cams.push_back(i); // selected_cams.push_back(i);
                vector<int> matchedBool(featVecIters[i]->second.size(),-1);
                matchedFlags.push_back(matchedBool);
            }
            else if(wordIds[i] == min_val){
                selected_cams.push_back(i); //  selected_cams.push_back(i);
                vector<int> matchedBool(featVecIters[i]->second.size(),-1);
                matchedFlags.push_back(matchedBool);
            }
            else if (wordIds[i] < second_min_val)
                second_min_val = wordIds[i];
         }


        if( selected_cams.size() >= 2){
            //DEBUG
            /*cout<<"WordIDs : ";
            for(int i =0; i <wordIds.size(); i++){

                cout<<wordIds[i]<<",";
            }
            cout<<endl;
            cout<<"selected Cams : "<<endl;
            for(int i = 0 ; i < selected_cams.size() ; i++){
                cout<<selected_cams[i]<<": ";
                vector<unsigned int> feat_cam1 = featVecIters[selected_cams[i]]->second;
                for(int j=0; j < feat_cam1.size(); j++)
                    cout<<feat_cam1[j]<<", ";
                cout<<endl;
            } */

            // compute matches between the features corresponding to
            // the matching node ID across all the matched cameras. The decriptors indices are present in the
            //corresponding featVec[it]->second.
            bool first_time = true;
            for (int i = 0 ; i < selected_cams.size() - 1; i++){
                int cam1 = selected_cams[i];
                vector<unsigned int> feat_cam1 = featVecIters[cam1]->second;

                //for each feature of cam1 in the current bag
                for (int cam1_feat_ind=0 ;  cam1_feat_ind < feat_cam1.size() ; cam1_feat_ind++){
                    //cout<<cam1<<": Finding match for feature : "<<feat_cam1[cam1_feat_ind]<<endl;
                    // check here if the current feature has an intra match. if so, get the corresponding entry
                    bool foundMatch= false;
                    int cam1_existing_intramatch = matchedFlags[i][cam1_feat_ind];
                    // if an in tra match already exists hen move on to next feature of cami
                    if (cam1_existing_intramatch != -1)
                        continue;

                    //else create an empty entry.
                    IntraMatch temp;
                    //fill(temp, temp+num_cams_, -1);
                    temp.matchIndex[cam1] = feat_cam1[cam1_feat_ind];
                    matches.push_back(temp);
                    matchedFlags[i][cam1_feat_ind] = intraMatchInd;


                    for (int j = i+1 ; j < selected_cams.size(); j++){

                        int cam2 = selected_cams[j];
                        vector<unsigned int> feat_cam2 = featVecIters[cam2]->second;

                        ////////////////////////////  FIND THE MATCHES////////////////////////////////////////////////
                        // initialize the index of best match in cam2_iter
                        int best_j_now = -1;
                        double best_dist_1 = 1e9;
                        double best_dist_2 = 1e9;


                        for (int cam2_feat_ind=0 ;  cam2_feat_ind < feat_cam2.size() ; cam2_feat_ind++)
                        {
                            //avoid matching the keypoints which have a bigger y coordinate difference.
                            //This is restricting the matches to the epipolar line.
                            if(abs(image_kps_undist[cam1][feat_cam1[cam1_feat_ind]].pt.y - image_kps_undist[cam2][feat_cam2[cam2_feat_ind]].pt.y) >= 50)
                                continue;
                            //find the distance between the current descriptors between cam1 and cam2
                            double d = orBextractor->DescriptorDistance(image_descriptors[cam1][feat_cam1[cam1_feat_ind]],  image_descriptors[cam2][feat_cam2[cam2_feat_ind]]);

                            if(d < best_dist_1)
                            {
                                best_j_now = cam2_feat_ind;
                                best_dist_2 = best_dist_1;
                                best_dist_1 = d;
                            }
                            else if(d < best_dist_2)
                            {
                                best_dist_2 = d;
                            }
                        }
                        //check if the distance qualifies for a match
                        if(best_dist_1 <= TH_LOW and best_dist_1 / best_dist_2 <= orBextractor->max_neighbor_ratio){
                            //We found the best distance and the best feature with index best_j_now in cam2
                            //Now, check if it is already matched with another feature

                            int existing_intramatch = matchedFlags[j][best_j_now];
                            if (existing_intramatch == -1 ){
                                // if it doesnt, this is a new m,atch, so add new match
                                //check if the match already exists for cam1
                                // if it does, get the intra match entry and add cam2 match
                                //else, this is a new intra match, add a new entry

                                matches[intraMatchInd].matchIndex[cam2] = feat_cam2[best_j_now];
                                matchedFlags[j][best_j_now] = intraMatchInd;
                                foundMatch= true;
                                //cout<<"New IntraMatch!! "<<cam2<<" : found match at feature : "<<feat_cam2[best_j_now]<<endl;
                            }
                            else{
                                // if it does, get the index of the cam1 match via matches[intramatch index][cam1]. compare the distnace and update accordingly
                                int old_cam1_match_ind = matches[existing_intramatch].matchIndex[cam1];
                                //check if old_cam1_match_ind is valid i.e != -1. if it is not valid, that mean
                                // ths cam2 feature is matched previously with another camera. For now,we ignore this match
                                // and move on. ideally, we should compare the distance between this feat and the older camera matches and update.-- May be
                                if (old_cam1_match_ind == -1){

                                    // cout<<"HEREHEREHEREHEREHEEHERE......................."<<endl;
                                    itr_cng_match++;
                                    continue;
                                }
                                double d = orBextractor->DescriptorDistance( image_descriptors[cam1][old_cam1_match_ind],  image_descriptors[cam2][feat_cam2[best_j_now]]);

                                if(best_dist_1 < d)
                                {
                                    // the new descriptor distance is smaller than the old.
                                    //so, update the intramatch - remove cam2 feature index from the old intramatch, add a new intramatch with current cam1_ind
                                    //it_intramatch->second[cam2] = -1;
                                    matches[existing_intramatch].matchIndex[cam2] = -1;
                                    //vector<int> temp(num_cams_ , -1);
                                    IntraMatch temp;
                                    //fill(temp, temp+num_cams_, -1);
                                    temp.matchIndex[cam1] = feat_cam1[cam1_feat_ind];
                                    temp.matchIndex[cam2] = feat_cam2[best_j_now];
                                    //matches.insert(make_pair(intraMatchInd, temp));
                                    matches[intraMatchInd] = temp;
                                    matchedFlags[j][best_j_now] = intraMatchInd;
                                    foundMatch = true;
                                    //cout<<"Overriding old intra match"<<cam2<<" : found match at feature : "<<feat_cam2[best_j_now]<<endl;
                                    // and matched flags
                                }
                                else{
                                    //cout<<"old intra match of "<<cam2<<"at "<<cam1 <<" : feature : "<<old_cam1_match_ind<<endl;
                                }
                            }
                        }
                    }
                    // after going through all camera images corresponding to this word
                    //if we found a match to this feature in cam1
                    if (foundMatch){
                        if(first_time){
                            //cout<<"WordID : "<< featVecIters[selected_cams[0]]->first<<endl;
                            //words_.insert(featVecIters[selected_cams[0]]->first);
                            /*cout<<"selected Cams : "<<endl;
                            for(int iii = 0 ; iii < selected_cams.size() ; iii++){
                                cout<<selected_cams[iii]<<": ";
                                vector<unsigned int> feat_cam1 = featVecIters[selected_cams[iii]]->second;
                                for(int jjj=0; jjj < feat_cam1.size(); jjj++)
                                    cout<<feat_cam1[jjj]<<", ";
                                cout<<endl;
                            } */
                            first_time = false;
                        }
                        words_.push_back(featVecIters[selected_cams[0]]->first);

                        //array<int, 5> temp = matches.back();
                        //cout<<"Match "<<intraMatchInd<<": "<< temp[0]<<", "<< temp[1]<<", "<< temp[2]<<", "<< temp[3]<<", "<< temp[4] <<endl;
                        intraMatchInd++;
                    }

                    else{
                        //if we did not find a match to this feature in cam1
                        // remove the inserted intramatch entry
                        //matches.erase(intraMatchInd);
                        matches.pop_back();
                        matchedFlags[i][cam1_feat_ind] = -1;
                    }
                    /////////////////////////////////////////////////////////////////////////////////////////////
                }
            }
        }


        // if there is only one minimum, increament that iterator and continue
        if (selected_cams.size() == num_cams_){
            for(int s_word = 0; s_word < selected_cams.size() ; s_word++){
                // move the lowest iterator forward
                ++featVecIters[selected_cams[s_word]];
            }
        }
        else{
            for(int s_word = 0; s_word < selected_cams.size() ; s_word++){
                ++featVecIters[selected_cams[s_word]];
            }

            // increment all selected iters to next word
            /*for(int s_word = 0; s_word < selected_cams.size() ; s_word++){
                // move the lowest iterator forward

                if(featVecs[selected_cams[s_word]].lower_bound(second_min_val) == featVecs[selected_cams[s_word]].end())
                    featVecIters[selected_cams[s_word]] = featVecEnds[selected_cams[s_word]];
                else
                    featVecIters[selected_cams[s_word]] = featVecs[selected_cams[s_word]].lower_bound(second_min_val);

            }*/
        }
    }
    /*set<DBoW2::NodeId>::iterator iter;
    for(iter =words_.begin() ; iter !=words_.end() ; ++iter){
        cout<<*iter<<endl;
    }*/

    delete[] featVecs;
    delete[] featVecIters;
    delete[] featVecEnds;
    //cout<<"matched words : "<<words_.size()<<endl;
    auto stop_intramatch = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_intramatch - start_intramatch);
    //cout<<"time taken to get the matching words: "<<duration.count()<<endl;
    // cout<<"total number of times HERE is printed "<<itr_cng_match<<endl;
    //cout<<"total number of iterations (common words): "<<itr<<endl;
}


void LightFieldFrame::BowMatching(int cam1_ind, int cam2_ind,vector<unsigned int>& indices_1, vector<unsigned int>& indices_2, vector<Point2f>& kps1, vector<Point2f>& kps2, set<DBoW2::NodeId>& words ){
    //matches.clear();
    indices_2.clear();
    indices_1.clear();
    //const DBoW2::FeatureVector &featVec1 = orb_database->retrieveFeatures(cam1_ind);
    const DBoW2::FeatureVector &featVec1 = BoW_feats.at(cam1_ind);
    //second image feature vector
    //const DBoW2::FeatureVector &featVec2 = orb_database->retrieveFeatures(cam2_ind);
    const DBoW2::FeatureVector &featVec2 = BoW_feats.at(cam2_ind);
    int itr_cng_match=0;
    // iterators for each of the feature vectors
    DBoW2::FeatureVector::const_iterator featVec1_it, featVec2_it;
    featVec1_it = featVec1.begin();
    featVec2_it = featVec2.begin();

    const DBoW2::FeatureVector::const_iterator featVec1_end = featVec1.end(); // std::next(featVec1.begin(),200);    //
    const DBoW2::FeatureVector::const_iterator featVec2_end = featVec2.end(); //std::next(featVec2.begin(),200);   //
    // until both the feature vectors iterators reach the end
    while(featVec1_it != featVec1_end && featVec2_it != featVec2_end)
    {
        // check if the node ID of both the vectors is same.
        if(featVec1_it->first == featVec2_it->first)
        {
            // compute matches between the features corresponding to
            // the matching node ID. The decriptors indices are present in the
            //corresponding featVec[it]->second. The actual descriptors of the key points
            // in the images are present in image_descriptors[i/j]
            vector<unsigned int> i_ind_tmp, j_ind_tmp;
            DBoW2::NodeId w1= featVec1_it->first;
            DBoW2::NodeId w2= featVec2_it->first;

            orBextractor->getMatches_distRatio(image_descriptors[cam1_ind], featVec1_it->second,
                                 image_descriptors[cam2_ind], featVec2_it->second, i_ind_tmp, j_ind_tmp,
                                 itr_cng_match);

            indices_1.insert(indices_1.end(), i_ind_tmp.begin(), i_ind_tmp.end());
            indices_2.insert(indices_2.end(), j_ind_tmp.begin(), j_ind_tmp.end());
            for(int m_ind=0; m_ind < i_ind_tmp.size() ; m_ind++){
                kps1.push_back(image_kps_undist[cam1_ind][i_ind_tmp[m_ind]].pt);
                kps2.push_back(image_kps_undist[cam2_ind][j_ind_tmp[m_ind]].pt);
            }

            if(i_ind_tmp.size()>= 1){
                //cout<<"WordID :"<< featVec1_it->first <<endl;
                words.insert(featVec1_it->first);
                /*for(int i=0; i < i_ind_tmp.size() ; i++){
                    for(int j =0; j < num_cams_ ; j++){
                        if(j == cam1_ind )
                            cout<<i_ind_tmp[i]<<", " ;
                        else if (j == cam2_ind)
                            cout<<j_ind_tmp[i]<<", " ;
                        else
                            cout<<"-1, ";
                    }
                    cout<<endl;
                } */

            }
            // move featVec1_it and featVec2_it forward
            ++featVec1_it;
            ++featVec2_it;
        }
        else if(featVec1_it->first < featVec2_it->first)
        {
            // move old_it forward
            featVec1_it = featVec1.lower_bound(featVec2_it->first);
            // old_it = (first element >= cur_it.id)
        }
        else
        {
            // move cur_it forward
            featVec2_it = featVec2.lower_bound(featVec1_it->first);
            // cur_it = (first element >= old_it.id)
        }
    }

}

void LightFieldFrame::computeIntraMatches( vector<IntraMatch>& matches, bool old){
    matches.reserve(10000);
    vector<vector<unsigned int>> match_inv_idx;

    for(int i =0; i < num_cams_ ; i++){
        match_inv_idx.push_back( vector<unsigned int>(image_kps[i].size(), -1));
    }
    int intramatches = 0;
    set<DBoW2::NodeId> wordIds;
    for(unsigned int i =0; i < (num_cams_ - 1 )  ; i++) {
        for (unsigned int j = i+1; j < num_cams_; j++) {
            vector<unsigned int> indices1, indices2;
            vector<Point2f> kp_pts1, kp_pts2;
            Mat F;
            Mat inliers;
            BowMatching(i, j,indices1, indices2, kp_pts1, kp_pts2, wordIds );
            /// filter the matches based on fundamental matrix
            if(old)
                cv::findFundamentalMat(kp_pts1, kp_pts2, cv::FM_RANSAC, 0.5, 0.99,inliers);
            else
                inliers = Mat::ones(kp_pts1.size(), 1, CV_8U);
            //now update the intra matches and flags for individual arrays
            for(int k=0; k <indices1.size(); k++){
                if (inliers.at<char>(k) == 0)
                    continue;
                int cami_feat = indices1[k];
                int camj_feat = indices2[k];
                int match_idx = match_inv_idx[i][cami_feat];
                int match_idx_2 = match_inv_idx[j][camj_feat];
                if(abs(image_kps_undist[i][cami_feat].pt.y - image_kps_undist[j][camj_feat].pt.y) >= 50)
                    continue;
                //check if 1st cam feature already has been matched
                // if it is not then create a new intramatch and update both feature indices
                if(match_idx == -1 and match_idx_2 == -1){
                    IntraMatch temp;
                    temp.matchIndex[i] = cami_feat;
                    temp.matchIndex[j] = camj_feat;
                    matches.push_back(temp);

                    match_inv_idx[i][cami_feat] = intramatches;
                    match_inv_idx[j][camj_feat] = intramatches;
                    intramatches++;
                }
                else{ // if the match already exists then update the new cam2 match here
                    if(match_idx_2 == -1){
                        matches[match_idx].matchIndex[j] = camj_feat;
                        match_inv_idx[j][camj_feat] = match_idx;
                    }

                }

            }
        }
    }
    /*set<DBoW2::NodeId>::iterator iter;
    for(iter =wordIds.begin() ; iter !=wordIds.end() ; ++iter){
        cout<<*iter<<endl;
    }*/
    cout<<"matched words : "<<wordIds.size()<<endl;
}

/// Method which computes the intra matches between the  keypoints detected
/// in the component cameras of the light field array. The variables matches,
/// kps_1 and kps_2 are updated with the matches
/// \param matches  : 3D vector of Dmatch. matches[i][j] gives vector of matches between the key points of camera i and camera j
/// \param kps_1 : 3D vector of matched key points indexed similar to matches
/// \param kps_2 : 3D vectors of matched key points indexed similar to matches
void LightFieldFrame::computeIntraMatches(vector<vector<vector<cv::DMatch>>>& matches, \
                                   vector<vector<vector<Point2f>>>& kps_1,vector<vector<vector<Point2f>>>& kps_2){
    //for each camera
    //assert(num_cams_ == orb_database->size());
    assert(num_cams_ == BoW_feats.size());

    matches.reserve(num_cams_);
    kps_1.clear();
    kps_2.clear();
    kps_1.reserve(num_cams_);
    kps_2.reserve(num_cams_);
    int itr_cng_match=0;
    //iterate
    for(unsigned int i =0; i < (num_cams_-1) ; i++){
        //first image feature vector
        //const DBoW2::FeatureVector &featVec1 = orb_database->retrieveFeatures(i);
        const DBoW2::FeatureVector &featVec1 = BoW_feats.at(i);
        matches.push_back(vector<vector<DMatch> >());
        kps_1.push_back(vector<vector<Point2f> >());
        kps_2.push_back(vector<vector<Point2f> >());
        for(unsigned int j = (i+1); j < num_cams_; j++){
            //second image feature vector
            //const DBoW2::FeatureVector &featVec2 = orb_database->retrieveFeatures(j);
            const DBoW2::FeatureVector &featVec2 = BoW_feats.at(j);

            // for each word in common, get the closest descriptors
            vector<unsigned int> i_ind, j_ind;
            // iterators for each of the feature vectors
            DBoW2::FeatureVector::const_iterator featVec1_it, featVec2_it;
            featVec1_it = featVec1.begin();
            featVec2_it = featVec2.begin();

            const DBoW2::FeatureVector::const_iterator featVec1_end = featVec1.end();
            const DBoW2::FeatureVector::const_iterator featVec2_end = featVec2.end();
            // until both the feature vectors iterators reach the end
            while(featVec1_it != featVec1_end && featVec2_it != featVec2_end)
            {
                // check if the node ID of both the vectors is same.
                if(featVec1_it->first == featVec2_it->first)
                {
                    // compute matches between the features corresponding to
                    // the matching node ID. The decriptors indices are present in the
                    //corresponding featVec[it]->second. The actual descriptors of the key points
                    // in the images are present in image_descriptors[i/j]
                    vector<unsigned int> i_ind_tmp, j_ind_tmp;

                    orBextractor->getMatches_distRatio(image_descriptors[i], featVec1_it->second,
                                         image_descriptors[j], featVec2_it->second, i_ind_tmp, j_ind_tmp,
                                         itr_cng_match);

                    i_ind.insert(i_ind.end(), i_ind_tmp.begin(), i_ind_tmp.end());
                    j_ind.insert(j_ind.end(), j_ind_tmp.begin(), j_ind_tmp.end());

                    // move featVec1_it and featVec2_it forward
                    ++featVec1_it;
                    ++featVec2_it;
                }
                else if(featVec1_it->first < featVec2_it->first)
                {
                    // move old_it forward
                    featVec1_it = featVec1.lower_bound(featVec2_it->first);
                    // old_it = (first element >= cur_it.id)
                }
                else
                {
                    // move cur_it forward
                    featVec2_it = featVec2.lower_bound(featVec1_it->first);
                    // cur_it = (first element >= old_it.id)
                }
            }

            //update the kps1 and kps2 vector
            vector<unsigned int>::const_iterator i_it, j_it;
            i_it = i_ind.begin();
            j_it = j_ind.begin();
            vector<DMatch> m_vec;
            //cout<<i_ind.size()<<","<<j_ind.size()<<","<<image_kps[i].size()<<","<<image_kps[j].size()<<endl;
            vector<cv::Point2f> k_points_1, k_points_2;
            for(; i_it != i_ind.end(); ++i_it, ++j_it)
            {
                DMatch m(*i_it, *j_it, 0.0);
                m_vec.push_back(m);

                const cv::KeyPoint &key1 = image_kps[i][*i_it];
                const cv::KeyPoint &key2 = image_kps[j][*j_it];

                k_points_1.push_back(key1.pt);
                k_points_2.push_back(key2.pt);
            }
            //triangulate the intra matches
            //vector<Point3f> pts3D;
            //triangulate(k_points_1,k_points_2, pts3D, i, j);
            cout<<"number of pair-wise matches between ("<<i<<" , "<<j<<") : "<<i_ind.size()<<endl;
            matches.back().push_back(m_vec);
            kps_1.back().push_back(k_points_1);
            kps_2.back().push_back(k_points_2);

        }// for loop j
    }// for loop i


}

// todo: sparse_disparity to intramatch->uv
void LightFieldFrame::drawIntraMatches(){
    assert(intraMatches.size() == sparse_disparity.size());
    Mat all;
    all.create(imgs[0].rows, imgs[0].cols * imgs.size(), CV_8UC3);
    for (int im = 0; im < imgs.size(); im++) {
        Mat imm, imm2;
        imgs[im].convertTo(imm, CV_32FC1);
        cv::add(imm,  0.2 * segMasks[im] , imm);
        cvtColor(imm, imm2,COLOR_GRAY2BGR);
        imm2.copyTo(all.colRange(img_size.width * im, img_size.width * (im + 1)));
    }
    std::vector<IntraMatch >::iterator matches_iter;
    RNG& rng=theRNG();

    // if color based on disparity
    cv::Mat sp_pts(Size(1,sparse_disparity.size()),CV_8UC1);
    for (int i =0; i <sparse_disparity.size(); i ++){
        support_pt pt = sparse_disparity[i];
        sp_pts.at<uint8_t>(i,0) = (uint8_t )pt.d;
        //cout<<"disp: "<<pt.d<<endl;
    }

    Mat out;
    cv::normalize(sp_pts, sp_pts, 40, 250, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(sp_pts, out, cv::COLORMAP_JET);

    int ind=0;
    int num_matches_all_cams = 0;
    for(matches_iter = intraMatches.begin() ;  matches_iter!= intraMatches.end(); ++matches_iter,ind++) {
        IntraMatch temp = *matches_iter;

        //draw the key points
        Point2f prev_pt;
        int more_than_1=0;
        Scalar color = Scalar( rng(256), rng(256), rng(256), 255 );
        Vec3b col1 = out.at<Vec3b>(ind,0);
        color = Scalar( col1[0], col1[1], col1[2], 255 );

        bool flag_todraw_only_reprojected = false;
        bool all_five=true;
        //DEBUG
        int num_views=0;
        vector<int> view_inds;
        Mat P_1 = build_Rt(camconfig_.R_mats_[1], camconfig_.t_mats_[1]);
        //TODO : fix intramatch class mess up
        for(int c=0; c < num_cams_; c++) {
            int featInd = temp.matchIndex[c];
            //////////THIS IS OPNLY FOR DEBUG. WILL BE REMOVED//////////////
            if(featInd != -1){
                num_views++;
                view_inds.push_back(c);
            }
            //////////////////////////////////////////////////////////////
            if (featInd != -1 and flag_todraw_only_reprojected) {
                cout <<c<<": "<< image_kps[c][featInd].pt << " || ";
                Point2f p = Point2f(img_size.width * c, 0) + image_kps[c][featInd].pt;
                circle(all, p, 3, color, 2);
                if(more_than_1!=0){

                    line(all, prev_pt, p, cv::Scalar(color));
                }
                prev_pt = p;
                more_than_1++;
            }
            else if (c==0){
                flag_todraw_only_reprojected = true;
                cout<<featInd<<endl;
                cout <<c<<": ["<< sparse_disparity[ind].u<<", "<<sparse_disparity[ind].v << "] || ";
                // this means the feature is not available in ref camera
                //get the support point here
                Point2f p = Point2f(sparse_disparity[ind].u, sparse_disparity[ind].v);
                circle(all, p, 3, color, 3);
                prev_pt = p;
                more_than_1++;
            }

        }

        cout<<endl;
        //// THIS IS ONLY FOR DEBUG. WILL BE REMOVED////
        Mat_<double> x(2, num_views);
        for(int k =0; k <num_views ; k++)
        {
            int cur_view_ind= view_inds[k];
            x(0, k) = image_kps[cur_view_ind][temp.matchIndex[cur_view_ind]].pt.x;
            x(1, k) = image_kps[cur_view_ind][temp.matchIndex[cur_view_ind]].pt.y;
            cout <<cur_view_ind<<": "<< image_kps[cur_view_ind][temp.matchIndex[cur_view_ind]].pt << " || ";
        }
        //triangulate the 3D point
        Eigen::Vector3d pt;
        triangulateIntraMatches(x, view_inds, pt);
        //try opencv sfm triagulation
        vector<Mat> PJs;
        std::vector<Mat_<double> >  xx;
        for(int ii=0; ii<num_views ; ii++){
            int indd = view_inds[ii];
            Mat Rt = build_Rt(camconfig_.R_mats_[indd], camconfig_.t_mats_[indd]);
            Mat P1 = cv::Mat_<double>(camconfig_.K_mats_[indd]*Rt);
            PJs.push_back(P1);
            Mat_<double> x1(2, 1);
            x1(0, 0) = image_kps[indd][temp.matchIndex[indd]].pt.x;
            x1(1, 0) = image_kps[indd][temp.matchIndex[indd]].pt.y;
            xx.push_back(x1.clone());
        }

        cv::Mat pt3d_sfm;
        cv::sfm::triangulatePoints(xx, PJs, pt3d_sfm);

        Mat projected_sfm = camconfig_.K_mats_[0] * pt3d_sfm;
        double expected_x_sfm = projected_sfm.at<double>(0,0) / projected_sfm.at<double>(2,0);
        double expected_y_sfm = projected_sfm.at<double>(1,0) / projected_sfm.at<double>(2,0);
        double inv_depth_sfm = 1.0/ projected_sfm.at<double>(2,0);
        //from multi-cam elas. just for comparison
        double base_0_sfm = -1*P_1.at<double>(0, 3);
        double f_0_sfm =  camconfig_.K_mats_[0].at<double>(0, 0);
        inv_depth_sfm = f_0_sfm * base_0_sfm / (double) pt3d_sfm.at<double>(2,0);
        cout<<"\n3D point SFM: "<<pt3d_sfm.at<double>(0,0)<<","<<pt3d_sfm.at<double>(1,0)<<","<<pt3d_sfm.at<double>(2,0)<<"  expected x,y : "<<expected_x_sfm<<","<<expected_y_sfm<<endl;
        for(int ii=0; ii<num_views ; ii++) {
            int indd = view_inds[ii];
            Mat Rt = build_Rt(camconfig_.R_mats_[indd], camconfig_.t_mats_[indd]);
            Mat P1 = camconfig_.K_mats_[indd] * Rt;
            Mat homo;
            sfm::euclideanToHomogeneous(pt3d_sfm, homo);
            Mat x_projected_sfm;
            sfm::homogeneousToEuclidean(P1 * homo, x_projected_sfm) ;
            cout<<indd<< "  : "<<x_projected_sfm.at<double>(0,0)<<","<<x_projected_sfm.at<double>(1,0)<<"||";
        }
        cout<<endl;
        //update the support point
        // specify in terms of inverse Z
        //Compute expected u,v in reference frame
        Mat w_in_rect = Mat(3, 1, CV_64FC1);
        w_in_rect.at<double>(0, 0) = pt[0];
        w_in_rect.at<double>(1, 0) = pt[1];
        w_in_rect.at<double>(2, 0) = pt[2];
        Mat projected = camconfig_.K_mats_[0] * w_in_rect;
        double expected_x = projected.at<double>(0,0) / projected.at<double>(2,0);
        double expected_y = projected.at<double>(1,0) / projected.at<double>(2,0);
        double inv_depth = 1.0/ projected.at<double>(2,0);
        //from multi-cam elas. just for comparison
        double base_0 = -1*P_1.at<double>(0, 3);
        double f_0 =  camconfig_.K_mats_[0].at<double>(0, 0);
        inv_depth = f_0 * base_0 / (double) w_in_rect.at<double>(2,0);
        cout<<"3D point: "<<pt[0]<<","<<pt[1]<<","<<pt[2]<<"  expected x,y : "<<expected_x<<","<<expected_y<<"  disparity : "<<inv_depth<<endl;
        ///////////////////////////////////////////////

        /*cv::imshow("Showing Intra match", all);
        cvWaitKey(0);
        all.create(imgs[0].rows, imgs[0].cols * imgs.size(), CV_8UC3);
        for (int im = 0; im < imgs.size(); im++) {
            Mat imm;
            cvtColor(imgs[im], imm,COLOR_GRAY2BGR);
            imm.copyTo(all.colRange(img_size.width * im, img_size.width * (im + 1)));
       }*/
    }

    cv::imshow("Showing Intra match", all);
    waitKey(5);
}


void LightFieldFrame::drawIntraMatches(bool all_matches){
    assert(intraMatches.size() == sparse_disparity.size());
    Mat all;
    all.create(imgs[0].rows, imgs[0].cols * imgs.size(), CV_8UC3);
    for (int im = 0; im < imgs.size(); im++) {
        Mat imm;
        cvtColor(imgs[im], imm,COLOR_GRAY2BGR);
        imm.copyTo(all.colRange(img_size.width * im, img_size.width * (im + 1)));
    }
    std::vector<IntraMatch >::iterator matches_iter;
    RNG& rng=theRNG();

    // if color based on disparity
    cv::Mat sp_pts(Size(1,sparse_disparity.size()),CV_8UC1);
    for (int i =0; i <sparse_disparity.size(); i ++){
        support_pt pt = sparse_disparity[i];
        sp_pts.at<uint8_t>(i,0) = (uint8_t )pt.d;
        //cout<<"disp: "<<pt.d<<endl;
    }

    Mat out;
    cv::normalize(sp_pts, sp_pts, 40, 250, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(sp_pts, out, cv::COLORMAP_JET);

    int ind=0;
    int num_matches_all_cams = 0;
    for(matches_iter = intraMatches.begin() ;  matches_iter!= intraMatches.end(); ++matches_iter,ind++) {
        IntraMatch temp = *matches_iter;

        //draw the key points
        Point2f prev_pt;
        int more_than_1=0;
        Scalar color = Scalar( rng(256), rng(256), rng(256), 255 );
        Vec3b col1 = out.at<Vec3b>(ind,0);
        color = Scalar( col1[0], col1[1], col1[2], 255 );

        bool all_five=true;
        Mat prev_all = all.clone();
        // TODO: fix IntraMatch class mess-up
        for(int c=0; c < num_cams_; c++) {
            int featInd = temp.matchIndex[c];
            if (featInd != -1) {
                cout <<c<<": "<< image_kps[c][featInd].pt << " || ";
                Point2f p = Point2f(img_size.width * c, 0) + image_kps[c][featInd].pt;
                circle(all, p, 3, color, 2);
                if(more_than_1!=0){

                    line(all, prev_pt, p, cv::Scalar(color));
                }
                prev_pt = p;
                more_than_1++;
            }
            else {
                all_five=false;
            }

        }

        cout<<endl;
        if(!all_five)
            all = prev_all.clone();
        // cv::imshow("Showing Intra match", all);
        // cvWaitKey(0);
        //all.create(imgs[0].rows, imgs[0].cols * imgs.size(), CV_8UC3);
        // for (int im = 0; im < imgs.size(); im++) {
        //     Mat imm;
        //     cvtColor(imgs[im], imm,COLOR_GRAY2BGR);
        //     imm.copyTo(all.colRange(img_size.width * im, img_size.width * (im + 1)));
        //  }
    }

    cv::imshow("Showing Intra match", all);
    waitKey(5);
}

int LightFieldFrame::countLandmarks (bool mono){
    int c=0;
    if(mono){
        for(auto& l : lIds){
            if(l != -1)
                c++;
        }
    }
    return c;
}

/// Not Implemented
void LightFieldFrame::reconstruct3D(){

}


