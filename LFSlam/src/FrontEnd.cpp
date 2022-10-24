//
// Created by Pushyami Kaveti on 6/6/20.
//
#include "LFSlam/FrontEnd.h"
#include "DUtils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//sac
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>

//opencv
#include <opencv2/sfm/triangulation.hpp>
#include <opencv2/sfm/projection.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace boost::assign;
//using namespace opengv;

// todo: Normalized coordinates for keypoints
// todo: Intramatch class instead of independent vectors
// todo: Delete images from memory, just save the intramatches, Map


void FrontEnd::createFrame(vector<Mat> img_set, vector<Mat> segmap_set, double timeStamp){
    currentFrame = new LightFieldFrame(img_set, segmap_set, orb_vocabulary, orBextractor, orBextractors, camconfig_, current_frameId++,
                                      timeStamp, clahe, false);
   // reconstructor->updateImages(segmap_set, img_set);

    //lfFrames.push_back(frame);
}

void FrontEnd::computeRefocusedImage(){
    //! compute the support points. multicam_elas-
    //! updatedata- remains same
    //! compute support points
    //!    - 1. compute matches between reference view & next view. For all the matches apply the disparity obtained and find other camera disparities.
    //!         Take +/- 5 pixels across and find the matches. Now we have intra matches across all the cameras for visible points.
    //!    - 2. For each other view - match the same way only for unmatched pixels.
    //!    - 3.
    //! disparity prior-
    //! disparity optimization.
    //!
}

void FrontEnd::obtainLfFeatures( std::vector<IntraMatch>& matches_map, LightFieldFrame* lf_frame,
                                   std::vector<IntraMatch>& filtered_intra_matches,
                                   vector<DBoW2::NodeId >& words_, set<DBoW2::NodeId >& filtered_words){

    std::vector<IntraMatch>::iterator matches_iter;
    int ind=0;
    set<DBoW2::NodeId > words_fil;

    vector<vector<bool>> keypoint_mask;
    for (int i = 0; i < lf_frame->num_cams_; ++i)
        keypoint_mask.emplace_back(lf_frame->image_kps_undist[i].size(), true);
    Mat P_1;
    if(lf_frame->num_cams_ > 1)
        P_1 = build_Rt(camconfig_.R_mats_[1], camconfig_.t_mats_[1]);
    std::vector<cv::Mat> vecDescs;

    vector<IntraMatch> mono_keypoints;
    vector<float> responses;
    vector<cv::Mat> mono_descs;
    float avg_depth_intrmatches = 0.0;
    std::map<int, int> intraMatchViews;
    for (int i = 0; i < lf_frame->num_cams_; ++i)
        intraMatchViews[i+1] = 0;

    /// Iterate over intramatches (matches_map)
    for(matches_iter = matches_map.begin() ;  matches_iter!= matches_map.end(); ++matches_iter, ++ind ){
        IntraMatch temp = *matches_iter;
        DBoW2::NodeId word_cur = words_.at(ind);

        int first=-1, last=-1; // initialize this with ref camera, since that is the index which we will be avoiding
        vector<Mat> descs;
        vector<int> view_inds;
        int num_views=0;

        ///
        for(auto i = 0; i < temp.matchIndex.size(); ++i)
        {
            //print the intra matches so that we know.
            int feat_ind = temp.matchIndex[i];
            // if it is a valid index
            if(feat_ind != -1) {
                Point2f p = lf_frame->image_kps[i][feat_ind].pt;
                if (lf_frame->segMasks[i].at<float>(p.y, p.x) < 0.7) {
                    keypoint_mask[i][feat_ind] = false;
                    descs.push_back(lf_frame->image_descriptors[i][feat_ind]);
                    view_inds.push_back(i);
                    num_views++;
                } else {
                    temp.matchIndex[i] = -1;
                }
            }
        }
        //cout<<endl;
        //get the disparity of those points WRT to the biggest baseline it is visible in
        if(num_views>1) {
            //cout<<"Number of valid intra matches : "<<num_views<<endl;
            //create the DS to hold the 2D points corresponding to this intra match

            ///// Traingulation using  opencv sfm and normalized coordinates///////////
            vector<Mat> PJs;
            std::vector<Mat_<double> > xx;
            vector<int> octs;
            vector<Point2f> kps;
            for (int ii = 0; ii < num_views; ii++) {
                int cur_view_ind = view_inds[ii];
                Mat Rt = build_Rt(camconfig_.R_mats_[cur_view_ind], camconfig_.t_mats_[cur_view_ind]);
                Mat P1 = cv::Mat_<double>(Rt);
                PJs.push_back(P1);
                Mat_<double> x1(2, 1);
                x1(0, 0) = (lf_frame->image_kps[cur_view_ind][temp.matchIndex[cur_view_ind]].pt.x -
                            camconfig_.K_mats_[cur_view_ind].at<double>(0, 2)) /
                           camconfig_.K_mats_[cur_view_ind].at<double>(0, 0);
                x1(1, 0) = (lf_frame->image_kps[cur_view_ind][temp.matchIndex[cur_view_ind]].pt.y -
                            camconfig_.K_mats_[cur_view_ind].at<double>(1, 2)) /
                           camconfig_.K_mats_[cur_view_ind].at<double>(1, 1);
                octs.push_back(lf_frame->image_kps[cur_view_ind][temp.matchIndex[cur_view_ind]].octave);
                kps.push_back( lf_frame->image_kps_undist[cur_view_ind][temp.matchIndex[cur_view_ind]].pt);
                xx.push_back(x1.clone());
            }

            cv::Mat pt3d_sfm;
            Eigen::Vector3d pt;
            cv::sfm::triangulatePoints(xx, PJs, pt3d_sfm);

            if(pt3d_sfm.at<double>(2, 0) < 40 and pt3d_sfm.at<double>(2, 0) > 0.5 ){

                ///////////////////////////CHECK THE TRIANGULATION/////////////////////////////////////////////////

                // check the reprojection errors of the points and the parallalx angle
                int jj=0;
                bool accept_intramatch = true;
//                if((currentFrame->frameId+1) % 100 ==0){   //// DO THIS ONLY During online calib
//                    for (auto P : PJs){
//                        int cam_ind = view_inds[jj];
//                        Mat p3d_conv = P(cv::Range(0, 3), cv::Range(0, 3)) * pt3d_sfm +  P(cv::Range(0, 3), cv::Range(3, 4));
//                        Mat projected = camconfig_.K_mats_[cam_ind] * p3d_conv;
//                        double expected_x = projected.at<double>(0,0) / projected.at<double>(2,0);
//                        double expected_y = projected.at<double>(1,0) / projected.at<double>(2,0);
//                        double err = (expected_x - kps[jj].x)*(expected_x - kps[jj].x)+(expected_y - kps[jj].y)*(expected_y - kps[jj].y);
//                        err = err*orBextractor->GetInverseScaleSigmaSquares()[ octs[jj]];
//                        jj++;
//                        if(err>5.991){
//                            accept_intramatch = false;
//                            // VLOG(2)<<"Bad Intramatch : ";
//                            break;
//                        }
//                    }
//                }

                if(accept_intramatch){
                    intraMatchViews[num_views]++;
                    Mat projected = camconfig_.K_mats_[0] * pt3d_sfm;
                    double expected_x = projected.at<double>(0, 0) / projected.at<double>(2, 0);
                    double expected_y = projected.at<double>(1, 0) / projected.at<double>(2, 0);
                    double inv_depth = 1.0 / projected.at<double>(2, 0);
                    //from multi-cam elas. just for comparison
                    double base_0 = -1 * P_1.at<double>(0, 3);
                    double f_0 = camconfig_.K_mats_[0].at<double>(0, 0);
                    inv_depth = f_0 * base_0 / (double) pt3d_sfm.at<double>(2, 0);
                    VLOG(3)<<"Intra Match Point: "<<pt3d_sfm.at<double>(0, 0)<<","<<pt3d_sfm.at<double>(1, 0)<<","<<pt3d_sfm.at<double>(2, 0);


                    words_fil.insert(word_cur);
                    lf_frame->lIds.push_back(-1);
                    //update the intraMatch descriptor
                    cv::Mat desc_out;
                    lf_frame->computeRepresentativeDesc(descs, desc_out);
                    temp.matchDesc = desc_out;
                    temp.point3D = pt3d_sfm;
                    temp.uv_ref = cv::Point2f(expected_x, expected_y);
                    temp.mono = false;
                    temp.n_rays = num_views;
                    filtered_intra_matches.push_back(temp);
                    vecDescs.push_back(desc_out);
                    lf_frame->intramatch_size++;

                    avg_depth_intrmatches = avg_depth_intrmatches + pt3d_sfm.at<double>(2, 0);

                }

                //////////////////////////////////////////////////////////////////////////
            }

        }
        else if(num_views == 1){
            intraMatchViews[num_views]++;
            temp.matchDesc = descs[0];
            temp.uv_ref = lf_frame->image_kps_undist[view_inds[0]][temp.matchIndex[view_inds[0]]].pt;
            temp.n_rays = num_views;
//            lf_frame->lIds.push_back(-1);
            mono_keypoints.push_back(temp);
            mono_descs.push_back(descs[0]);
            responses.push_back(lf_frame->image_kps_undist[view_inds[0]][temp.matchIndex[view_inds[0]]].response);
        }

    }
    VLOG(2)<<"Average Depth of the Intra Match features: "<<avg_depth_intrmatches / lf_frame->intramatch_size;
    VLOG(2)<<"Number of Intra Match features : "<<lf_frame->intramatch_size;
    for(int i=0 ; i < lf_frame->num_cams_ ; i++)
        VLOG(2)<<"num of intra matches with "<<(i+1)<<" views: "<<intraMatchViews[i+1];
   //if((currentFrame->frameId+1) % 100 ==0)
   //     bundleAdjustIntraMatches(filtered_intra_matches);
    ///Tiling of  features for good distribution
    int total_feats = 3000;
    int num_mono_feats = total_feats - lf_frame->intramatch_size;
    int cells_hor = 4;
    int cells_vert = 4;
    int feats_per_cell = ceil(num_mono_feats /(cells_hor * cells_vert*5));
    bool tiling = false;
    bool refview= false;
    std::vector<std::size_t> grid[cells_hor][cells_vert];
    std::vector<float> gridResponses[cells_hor][cells_vert];

    assert (lf_frame->img_size.width % cells_hor == 0);
    assert (lf_frame->img_size.height % cells_vert == 0);
    int cell_width = round(lf_frame->img_size.width / cells_hor);
    int cell_height = round(lf_frame->img_size.height / cells_vert);
    if(num_mono_feats <= 0)
        tiling= false;
    if(tiling){
        for(unsigned int i=0; i<cells_hor;i++){
            for (unsigned int j=0; j<cells_vert;j++){
                grid[i][j].reserve(feats_per_cell);
                gridResponses[i][j].reserve(feats_per_cell);
            }
        }

        int ind_to_mono =0;
        for(int i = 0; i < lf_frame->num_cams_; ++i){
            for(int j = 0; j < lf_frame->image_kps_undist[i].size(); ++j){
                Point2f p = lf_frame->image_kps[i][j].pt;
                if(keypoint_mask[i][j] && lf_frame->segMasks[i].at<float>(p.y, p.x) < 0.7){
                    IntraMatch intra_match;
                    intra_match.matchIndex[i] = j;
                    intra_match.matchDesc = lf_frame->image_descriptors[i][j];
                    intra_match.uv_ref = lf_frame->image_kps_undist[i][j].pt;
                    intra_match.n_rays = 1;
                    mono_keypoints.push_back(intra_match);
                    responses.push_back(lf_frame->image_kps_undist[i][j].response);
                    mono_descs.push_back(lf_frame->image_descriptors[i][j]);

                    int nGridPosX, nGridPosY;
                    int grid_x = round((p.x - 30)/ cell_width);
                    int grid_y = round((p.y - 30)/ cell_height);
                    if(grid_x >=0 and grid_x < cells_hor and grid_y >=0 and grid_y < cells_vert){
                        grid[grid_x][grid_y].push_back(ind_to_mono);
                        gridResponses[grid_x][grid_y].push_back(lf_frame->image_kps_undist[i][j].response);
                    }
                    ind_to_mono++;
                }
            }
        }

        // select the top feats_per_cell number of kps in each grid cell
        for (int i =0; i < cells_hor ; i++){
            for(int j =0 ; j <cells_vert ; j++){
                vector<size_t> kp_inds = grid[i][j];
                vector<int> sorted_kp_ind = argsorte(gridResponses[i][j], false);
                for(int k = 0; k < sorted_kp_ind.size() && k < feats_per_cell; ++k){
                    int mono_kp_ind = kp_inds[sorted_kp_ind[k]];
                    filtered_intra_matches.push_back(mono_keypoints[mono_kp_ind]);
                    vecDescs.push_back(mono_descs[mono_kp_ind]);
                    lf_frame->lIds.push_back(-1);
                    lf_frame->mono_size++;
                }
            }
        }
    }
    else if (refview){
        for(int j = 0; j < lf_frame->image_kps_undist[0].size(); ++j){
            Point2f p = lf_frame->image_kps_undist[0][j].pt;
            if(keypoint_mask[0][j] && lf_frame->segMasks[0].at<float>(p.y, p.x) < 0.7){
                IntraMatch intra_match;
                intra_match.matchIndex[0] = j;
                intra_match.matchDesc = lf_frame->image_descriptors[0][j];
                intra_match.uv_ref = lf_frame->image_kps_undist[0][j].pt;
                intra_match.n_rays = 1;
                //mono_keypoints.push_back(intra_match);
                filtered_intra_matches.push_back(intra_match);
                vecDescs.push_back(lf_frame->image_descriptors[0][j]);
                lf_frame->lIds.push_back(-1);
                lf_frame->mono_size++;
            }
        }
    }
    else{
        for(int i = 0; i < lf_frame->num_cams_; ++i){
            for(int j = 0; j < lf_frame->image_kps_undist[i].size(); ++j){
                Point2f p = lf_frame->image_kps_undist[i][j].pt;
                if(keypoint_mask[i][j] && lf_frame->segMasks[i].at<float>(p.y, p.x) < 0.7){
                    IntraMatch intra_match;
                    intra_match.matchIndex[i] = j;
                    intra_match.matchDesc = lf_frame->image_descriptors[i][j];
                    intra_match.uv_ref = lf_frame->image_kps_undist[i][j].pt;
                    intra_match.n_rays = 1;
                    mono_keypoints.push_back(intra_match);
                    responses.push_back(lf_frame->image_kps_undist[i][j].response);
                    mono_descs.push_back(lf_frame->image_descriptors[i][j]);
                }
            }
        }
        vector<int> sorted_kp_ind = argsorte(responses, false);
        for(int i = 0; i < sorted_kp_ind.size() && i < (3000 - lf_frame->intramatch_size); ++i){
            filtered_intra_matches.push_back(mono_keypoints[sorted_kp_ind[i]]);
            vecDescs.push_back(mono_descs[sorted_kp_ind[i]]);
            lf_frame->lIds.push_back(-1);
            lf_frame->mono_size++;
        }
    }

    orb_vocabulary->transform(vecDescs, lf_frame->lfBoW, lf_frame->lfFeatVec, 4);

    ///////draw the intramatches
    Mat all;
    all.create(lf_frame->imgs[0].rows, lf_frame->imgs[0].cols * lf_frame->num_cams_, CV_8UC3);
    for(int i=0; i < camconfig_.num_cams_ ; i++){
        Mat imgBGR;
        cvtColor(lf_frame->imgs[i],imgBGR , COLOR_GRAY2BGR);
        imgBGR.copyTo(all.colRange(camconfig_.im_size_.width*i, camconfig_.im_size_.width*(i+1)));

    }
    //Show the support matches
    cv::Mat dMap(Size(1,lf_frame->intramatch_size),CV_8UC1);
    int ind_Dmap=0;
    for(IntraMatch im : filtered_intra_matches){
        if(!im.mono) {
            dMap.at<uint8_t>(ind_Dmap, 0) = (uint8_t) im.point3D.at<double>(2, 0);
            ind_Dmap++;
        }

    }

    Mat out;
    Mat img_color;
    cv::normalize(dMap, dMap, 0, 220, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(dMap, out, cv::COLORMAP_RAINBOW);
    ind_Dmap=0;
    for(IntraMatch im : filtered_intra_matches){
        if(!im.mono){
             Vec3b col1 = out.at<Vec3b>(ind_Dmap,0);
             cv::circle(all, im.uv_ref , 3, Scalar(col1[0],col1[1],col1[2]), 2,8,0 );
            ind_Dmap++;
        }
        else{
            for (int i = 0; i < im.matchIndex.size(); ++i){
                if(im.matchIndex[i] != -1)
                    cv::circle(all, lf_frame->image_kps_undist[i][im.matchIndex[i]].pt + Point2f(i * lf_frame->imgs[0].cols , 0),  3, Scalar(255,255,255), 2 );
            }
        }
    }

    cv::resize(all, all, Size(lf_frame->imgs[0].cols * lf_frame->num_cams_/2 , lf_frame->imgs[0].rows/2));
    imshow("img with keypoints ", all);
    waitKey(5);


    filtered_words = words_fil;

}

void FrontEnd::filterIntraMatches( std::vector<IntraMatch>& matches_map, LightFieldFrame* lf_frame,
                                   std::vector<IntraMatch>& filtered_intra_matches,
                                   vector<DBoW2::NodeId >& words_, set<DBoW2::NodeId >& filtered_words){
    Mat P_1 = build_Rt(camconfig_.R_mats_[1], camconfig_.t_mats_[1]);
    std::vector<IntraMatch>::iterator matches_iter;
    int ind=0;
    set<DBoW2::NodeId > words_fil;

    /// Iterate over intramatches (matches_map)
    for(matches_iter = matches_map.begin() ;  matches_iter!= matches_map.end(); ++matches_iter, ++ind ){
        IntraMatch temp = *matches_iter;
        DBoW2::NodeId word_cur = words_.at(ind);
        int c=0;

        int first=-1, last=-1; // initialize this with ref camera, since that is the index which we will be avoiding
        vector<Mat> descs;
        vector<int> view_inds;
        int num_views=0;

        for_each(temp.matchIndex.begin(), temp.matchIndex.end(), [&c,&view_inds, &lf_frame, &descs, &num_views](int featInd)
        {
            //print the intra matches so that we know.
            if(featInd != -1){
                // if it is a valid index
                Point2f p = lf_frame->image_kps[c][featInd].pt;
                if (lf_frame->segMasks[c].at<float>(p.y, p.x) < 0.7){
                    descs.push_back(lf_frame->image_descriptors[c][featInd]);
                    view_inds.push_back(c);
                    num_views++;
                }
                else{
                    // cout<<"belongs to dynamic obj ||";
                }
            }
            else{
                //cout<<"Nan"<<" || ";
            }
            c = c+1;
        });
        //cout<<endl;
        //get the disparity of those points WRT to the biggest baseline it is visible in
        if(num_views>1){
            //cout<<"Number of valid intra matches : "<<num_views<<endl;
            //create the DS to hold the 2D points corresponding to this intra match
            Mat_<double> x(2, num_views);
            for(int k =0; k <num_views ; k++)
            {
                int cur_view_ind= view_inds[k];
                x(0, k) = lf_frame->image_kps[cur_view_ind][temp.matchIndex[cur_view_ind]].pt.x;
                x(1, k) = lf_frame->image_kps[cur_view_ind][temp.matchIndex[cur_view_ind]].pt.y;
                //cout <<cur_view_ind<<": "<< lf_frame->image_kps[cur_view_ind][temp[cur_view_ind]].pt << " || ";
            }

            //cout <<endl;
            //triangulate the 3D point
            Eigen::Vector3d pt;
            lf_frame->triangulateIntraMatches(x, view_inds, pt);

            ///// Traingulation using  opencv sfm and normalized coordinates///////////
            vector<Mat> PJs;
            std::vector<Mat_<double> >  xx;
            for(int ii=0; ii<num_views ; ii++){
                int cur_view_ind= view_inds[ii];
                Mat Rt = build_Rt(camconfig_.R_mats_[cur_view_ind], camconfig_.t_mats_[cur_view_ind]);
                Mat P1 = cv::Mat_<double>(Rt);
                PJs.push_back(P1);
                Mat_<double> x1(2, 1);
                x1(0, 0) = (lf_frame->image_kps[cur_view_ind][temp.matchIndex[cur_view_ind]].pt.x - camconfig_.K_mats_[cur_view_ind].at<double>(0,2))/camconfig_.K_mats_[cur_view_ind].at<double>(0,0);
                x1(1, 0) = (lf_frame->image_kps[cur_view_ind][temp.matchIndex[cur_view_ind]].pt.y - camconfig_.K_mats_[cur_view_ind].at<double>(1,2)) / camconfig_.K_mats_[cur_view_ind].at<double>(1,1);
                xx.push_back(x1.clone());
            }

            cv::Mat pt3d_sfm;
            cv::sfm::triangulatePoints(xx, PJs, pt3d_sfm);
            //////////////////////////////////////////////////////////////////////////
            //update the support point
            // specify in terms of inverse Z
            //Compute expected u,v in reference frame
            //Mat w_in_rect = Mat(3, 1, CV_64FC1);
            //w_in_rect.at<double>(0, 0) = pt[0];
            // w_in_rect.at<double>(1, 0) = pt[1];
            // w_in_rect.at<double>(2, 0) = pt[2];
            //Mat projected = camconfig_.K_mats_[0] * w_in_rect;

            Mat projected = camconfig_.K_mats_[0] * pt3d_sfm;
            double expected_x = projected.at<double>(0,0) / projected.at<double>(2,0);
            double expected_y = projected.at<double>(1,0) / projected.at<double>(2,0);
            double inv_depth = 1.0/ projected.at<double>(2,0);
            //from multi-cam elas. just for comparison
            double base_0 = -1*P_1.at<double>(0, 3);
            double f_0 =  camconfig_.K_mats_[0].at<double>(0, 0);
            inv_depth = f_0 * base_0 / (double) pt3d_sfm.at<double>(2,0);

            if(pt3d_sfm.at<double>(2,0) < 12 and pt3d_sfm.at<double>(2,0) > 0.5){
                //cout<<"3D point: "<<pt[0]<<","<<pt[1]<<","<<pt[2]<<"  expected x,y : "<<expected_x<<","<<expected_y<<"  disparity : "<<inv_depth<<endl;
                support_pt sp_pt = support_pt( (int32_t)expected_x,(int32_t)expected_y, (int32_t)inv_depth );
                lf_frame->sparse_disparity.push_back(sp_pt);
                //collect all the valid intra matches

                words_fil.insert(word_cur);
                //cout<<word_cur<<endl;
                //update the local 3D points
                for(int j=0; j<3; ++j)
                    pt[j] = pt3d_sfm.at<double>(j,0);
                lf_frame->points_3D.push_back(pt);
                lf_frame->lIds.push_back(-1);
                //update the intraMatch descriptor
                cv::Mat desc_out;
                lf_frame->computeRepresentativeDesc(descs, desc_out);
//                lf_frame->intraMatchDescs.push_back(desc_out);
                temp.matchDesc = desc_out;
                temp.point3D = pt3d_sfm;
                filtered_intra_matches.push_back(temp);
            }

        }

    }

    filtered_words = words_fil;

}

void FrontEnd::BruteForceMatching(Mat img1, Mat img2, vector<Mat> descs1, vector<Mat> descs2, vector<KeyPoint> kps1, vector<KeyPoint> kps2){
    vector<vector<DMatch>> matches_mono;
    vector<DMatch> good_matches;
    vector<KeyPoint> kp_pts1_mono, kp_pts2_mono;
    Mat mask_mono;

    Mat descs1_mono = Mat(descs1.size(), descs1[0].cols, CV_8U );
    Mat descs2_mono = Mat(descs2.size(), descs2[0].cols, CV_8U );
    int ind=0;
    for (auto& d : descs1){
        d.copyTo(descs1_mono.row(ind));
        ind++;
    }
    ind=0;
    for (auto& d : descs2){
        d.copyTo(descs2_mono.row(ind));
        ind++;
    }

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    auto start_intramatch = high_resolution_clock::now();
    matcher->knnMatch(descs1_mono, descs2_mono, matches_mono, 2);
    auto stop_intramatch = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_intramatch - start_intramatch);
    VLOG(1)<<"time taken for matching brute force "<<duration.count()<<endl;

    for(auto &m : matches_mono){
        if(m[0].distance < 0.7*m[1].distance) {
            if(m[0].distance > 50)
                continue;
            KeyPoint p_prev = kps1[m[0].queryIdx];
            KeyPoint p_cur = kps2[m[0].trainIdx];
            //make sure that the points belong to static areas based on the segmasks
           // if (currentFrame->segMasks[0].at<float>(p_prev.pt.y, p_prev.pt.x) < 0.7 and currentFrame->segMasks[1].at<float>(p_cur.pt.y, p_cur.pt.x) < 0.7){
                kp_pts1_mono.push_back(p_prev);
                kp_pts2_mono.push_back(p_cur);
                good_matches.push_back(m[0]);
           // }

        }

    }
    Mat matches_img;
    drawMatches( img1, kps1,img2, kps2, good_matches, matches_img, Scalar::all(-1),
                 Scalar::all(-1) );
    imshow("Good Matches:"+to_string(0)+"--"+to_string(1), matches_img );
    waitKey(5);

    cout<<"BRUTEFORCE Total Number of matches between cam0 and cam1: "<<good_matches.size()<<endl;
}

void FrontEnd::getMapPoints(vector<Point3f>& mapPoints){
    mapPoints.reserve(map->num_lms);
    std::map<int, Landmark*>::iterator it;
    unique_lock<mutex> lock(mMutexPose);
    for(it = map->mapPoints.begin() ; it != map->mapPoints.end() ; ++it) {
        Landmark *l = it->second;
        mapPoints.push_back(Point3f (l->pt3D.at<double>(0,0),l->pt3D.at<double>(1,0),l->pt3D.at<double>(2,0)) );
    }
}

void FrontEnd::processFrameNon(){
    auto start_intramatch = high_resolution_clock::now();
    // get the images and extract key points
    ///////////////////////////////////////
    ///extract the ORB  features
    //////////////////////////////////////
    currentFrame->extractFeaturesParallel();
    auto stop_intramatch = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_intramatch - start_intramatch);
    VLOG(3)<<"time taken for extracting ORB features: "<<duration.count()<<endl;

    start_intramatch = high_resolution_clock::now();
    // Now parse through the tree and form bag of words and feature vectors
    currentFrame->parseandadd_BoW();

    stop_intramatch = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop_intramatch - start_intramatch);
    VLOG(3)<<"time taken for parsing BoW "<<duration.count()<<endl;
    vector<bool> keypoint_mask(currentFrame->image_kps_undist[0].size(), true);
    std::vector<cv::Mat> vecDescs;

  // if(initialized_ == NOT_INITIALIZED){
       //// Find 3D matches for just the front camera. This is required for initialization
       vector<unsigned int> ind1, ind2;
       //Match the landmark descriptors with the current frame descriptors
       InterMatchingBow( currentFrame->BoW_feats[0],currentFrame->BoW_feats[1], currentFrame->image_descriptors[0],currentFrame->image_descriptors[1] , ind1, ind2 );

       ///visualize the matches
       /// filter them
       vector<Point2f> kp_pts1, kp_pts2;
       Mat mask_mono;

       /// triangulate them
       Mat P1_d = camconfig_.K_mats_[0] * build_Rt(camconfig_.R_mats_[0], camconfig_.t_mats_[0]);
       Mat P2_d = camconfig_.K_mats_[1] * build_Rt(camconfig_.R_mats_[1], camconfig_.t_mats_[1]);
       Mat o1  = -camconfig_.R_mats_[0].t() * camconfig_.t_mats_[0];
       Mat o2  = -camconfig_.R_mats_[1].t() * camconfig_.t_mats_[1];
       Mat P1, P2;
       P1_d.convertTo(P1, CV_32F);
       P2_d.convertTo(P2, CV_32F);

       for(size_t i=0, iend=ind1.size();i<iend;i++)
       {
           const cv::Point2f &kp1 = currentFrame->image_kps_undist[0][ind1[i]].pt;
           const cv::Point2f &kp2 = currentFrame->image_kps_undist[1][ind2[i]].pt;
           cv::Mat p3f, p3dC1;

           Triangulate(kp1,kp2,P1,P2,p3f);
           //cout<<"pt1 : "<<kp1<<" pt2: "<<kp2<<" p3f: "<<p3f<<endl;
           p3f.convertTo(p3dC1, CV_64F);
           if(!isfinite(p3dC1.at<double>(0)) || !isfinite(p3dC1.at<double>(1)) || !isfinite(p3dC1.at<double>(2)))
           {
               continue;
           }

           // Check parallax
           cv::Mat normal1 = p3dC1 - o1;
           float dist1 = cv::norm(normal1);

           cv::Mat normal2 = p3dC1 - o2;
           float dist2 = cv::norm(normal2);

           float cosParallax = normal1.dot(normal2)/(dist1*dist2);

           Mat p3dC1_w = p3dC1.clone();

           p3dC1 = camconfig_.R_mats_[0]*p3dC1_w + camconfig_.t_mats_[0];

           // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
           if(p3dC1.at<double>(2)<=0 && cosParallax<0.99998)
               continue;

           // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth
           cv::Mat p3dC2 = camconfig_.R_mats_[1] * p3dC1_w + camconfig_.t_mats_[1];

           if(p3dC2.at<double>(2)<=0 && cosParallax<0.99998)
               continue;

           // Check reprojection error in first image
           double im1x, im1y;
           double invZ1 = 1.0/p3dC1.at<double>(2);
           im1x = camconfig_.K_mats_[0].at<double>(0,0) * p3dC1.at<double>(0) * invZ1 +  camconfig_.K_mats_[0].at<double>(0,2) ;
           im1y =  camconfig_.K_mats_[0].at<double>(1,1)  * p3dC1.at<double>(1) * invZ1 +  camconfig_.K_mats_[0].at<double>(1,2) ;

           double squareError1 = (im1x-kp1.x)*(im1x-kp1.x)+(im1y-kp1.y)*(im1y-kp1.y);

           if(squareError1>4)
               continue;

           // Check reprojection error in second image
           double im2x, im2y;
           double invZ2 = 1.0/p3dC2.at<double>(2);
           im2x = camconfig_.K_mats_[1].at<double>(0,0) * p3dC2.at<double>(0) * invZ2 + camconfig_.K_mats_[1].at<double>(0,2);
           im2y = camconfig_.K_mats_[1].at<double>(1,1) * p3dC2.at<double>(1) * invZ2 + camconfig_.K_mats_[1].at<double>(1,2);

           double squareError2 = (im2x-kp2.x)*(im2x-kp2.x)+(im2y-kp2.y)*(im2y-kp2.y);

           if(squareError2>4)
               continue;

           IntraMatch intra_match;
           intra_match.matchIndex[0] = ind1[i];
           intra_match.matchIndex[1] = ind2[i];
           intra_match.matchDesc = currentFrame->image_descriptors[0][ind1[i]];
           intra_match.uv_ref = kp1;
           intra_match.n_rays = 2;
           intra_match.point3D = p3dC1_w;
           intra_match.mono = false;
           currentFrame->intraMatches.push_back(intra_match);
           vecDescs.push_back(currentFrame->image_descriptors[0][ind1[i]]);
           currentFrame->lIds.push_back(-1);
           currentFrame->intramatch_size++;
           keypoint_mask[ind1[i]] = false;
       }
  // }


    // Project intra-matches on to reference camera and visualize the disparity prior
    // Find out all the matches which do not have a correspondence in the reference frame
    //////////////DEBUGGING VARIABLES/////////////////////////
    //////////////////////////////////////////////////////////
    int real_num_intra=0;
    int points_on_seg = 0;
    int unaccounted = 0;
    int five_cam_intramatches=0;
    //////////////DEBUGGING VARIABLES/////////////////////////
    //////////////////////////////////////////////////////////

    for(int j = 0; j < currentFrame->image_kps_undist[0].size(); ++j){
        Point2f p = currentFrame->image_kps_undist[0][j].pt;
        if(currentFrame->segMasks[0].at<float>(p.y, p.x) < 0.7 and keypoint_mask[j]){
            IntraMatch intra_match;
            intra_match.matchIndex[0] = j;
            intra_match.matchDesc = currentFrame->image_descriptors[0][j];
            intra_match.uv_ref = currentFrame->image_kps_undist[0][j].pt;
            intra_match.n_rays = 1;
            currentFrame->intraMatches.push_back(intra_match);
            vecDescs.push_back(currentFrame->image_descriptors[0][j]);
            currentFrame->lIds.push_back(-1);
            currentFrame->mono_size++;
        }
    }

    for(int i = 2; i < currentFrame->num_cams_; ++i){
        for(int j = 0; j < currentFrame->image_kps_undist[i].size(); ++j){
            Point2f p = currentFrame->image_kps_undist[i][j].pt;
            if(currentFrame->segMasks[i].at<float>(p.y, p.x) < 0.7){
                IntraMatch intra_match;
                intra_match.matchIndex[i] = j;
                intra_match.matchDesc = currentFrame->image_descriptors[i][j];
                intra_match.uv_ref = currentFrame->image_kps_undist[i][j].pt;
                intra_match.n_rays = 1;
                currentFrame->intraMatches.push_back(intra_match);
                vecDescs.push_back(currentFrame->image_descriptors[i][j]);
                currentFrame->lIds.push_back(-1);
                currentFrame->mono_size++;
            }
        }
    }

    orb_vocabulary->transform(vecDescs, currentFrame->lfBoW, currentFrame->lfFeatVec, 4);

    ///////draw the intramatches
    Mat all;
    all.create(currentFrame->imgs[0].rows, currentFrame->imgs[0].cols * currentFrame->num_cams_, CV_8UC3);
    for(int i=0; i < camconfig_.num_cams_ ; i++){
        Mat imgBGR;
        cvtColor(currentFrame->imgs[i],imgBGR , COLOR_GRAY2BGR);
        imgBGR.copyTo(all.colRange(camconfig_.im_size_.width*i, camconfig_.im_size_.width*(i+1)));

    }
    //Show the support matches
    cv::Mat dMap(Size(1,currentFrame->intramatch_size),CV_8UC1);
    int ind_Dmap=0;
    for(IntraMatch im : currentFrame->intraMatches){
        if(!im.mono) {
            dMap.at<uint8_t>(ind_Dmap, 0) = (uint8_t) im.point3D.at<double>(2, 0);
            ind_Dmap++;
        }

    }

    Mat out;
    Mat img_color;
    if(currentFrame->intramatch_size > 0){
        cv::normalize(dMap, dMap, 0, 220, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(dMap, out, cv::COLORMAP_RAINBOW);
    }
    ind_Dmap=0;
    for(IntraMatch im : currentFrame->intraMatches){
        if(!im.mono){
            Vec3b col1 = out.at<Vec3b>(ind_Dmap,0);
            cv::circle(all, im.uv_ref , 3, Scalar(col1[0],col1[1],col1[2]), 2,8,0 );
            ind_Dmap++;
        }
        else{
            for (int i = 0; i < im.matchIndex.size(); ++i){
                if(im.matchIndex[i] != -1)
                    cv::circle(all, currentFrame->image_kps_undist[i][im.matchIndex[i]].pt + Point2f(i * currentFrame->imgs[0].cols , 0),  3, Scalar(255,255,255), 2 );
            }
        }
    }

    cv::resize(all, all, Size(currentFrame->imgs[0].cols * currentFrame->num_cams_/2 , currentFrame->imgs[0].rows/2));
    imshow("img with keypoints ", all);
    waitKey(5);

    //currentFrame->drawIntraMatches();

    // cout<<"Total number of intra matches before : "<<matches_map.size()<<endl;

    // currentFrame->parseIntraMatchBoW();

    stop_intramatch = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop_intramatch - start_intramatch);
    VLOG(3)<<"time taken for intra match computation: "<<duration.count()<<endl;
    VLOG(2)<<"Total number of intramatch features : "<<currentFrame->intramatch_size<<endl;
    VLOG(2)<<"Total number of LF features : "<<currentFrame->intraMatches.size()<<endl;
    ////todo keep track of num mono and num intra
    /// visualize intra matches across pairs - todo: maybe extract into a function ?
}

void FrontEnd::processFrame() {
    auto start_intramatch = high_resolution_clock::now();
    // get the images and extract key points
    ///////////////////////////////////////
    ///extract the ORB  features
    //////////////////////////////////////
    currentFrame->extractFeaturesParallel();
    //currentFrame->extractFeatures();
    //currentFrame->extractORBCuda();
    auto stop_intramatch = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop_intramatch - start_intramatch);
    VLOG(3) << "time taken for extracting ORB features: " << duration.count() << endl;

    start_intramatch = high_resolution_clock::now();
    // Now parse through the tree and form bag of words and feature vectors
    currentFrame->parseandadd_BoW();

    stop_intramatch = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop_intramatch - start_intramatch);
    VLOG(3) << "time taken for parsing BoW " << duration.count() << endl;

    start_intramatch = high_resolution_clock::now();

    std::vector<IntraMatch> matches_map;

    //start_intramatch = high_resolution_clock::now();

    vector<DBoW2::NodeId> words_;
    currentFrame->computeIntraMatches(matches_map, words_);
    //currentFrame->computeIntraMatches( matches_map, true);
    words_ = vector<DBoW2::NodeId>(matches_map.size(), 1);
    VLOG(2) << "Total Number of intra matches found before filtering :" << matches_map.size();
    stop_intramatch = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop_intramatch - start_intramatch);
    VLOG(3) << "time taken for Matching All Cams BoW " << duration.count() << endl;

    ////////////////////// Matching indiidual cameras for debugging///////////////////

//    vector<unsigned int> ind1, ind2;set<DBoW2::NodeId> words1;
//    start_intramatch = high_resolution_clock::now();
//    currentFrame->BowMatching(0,1,ind1, ind2,words1 );
//    stop_intramatch = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(stop_intramatch - start_intramatch);
//    VLOG(1)<<"time taken for Matching single Cam BoW "<<duration.count()<<endl;
//    vector<DMatch> good_matches_bow;
//    for(int i =0; i <ind1.size(); i++){
//        if(abs(currentFrame->image_kps_undist[0][ind1[i]].pt.y - currentFrame->image_kps_undist[1][ind2[i]].pt.y) < 50){
//            DMatch m = DMatch(ind1[i], ind2[i], 0);
//            good_matches_bow.push_back(m);
//        }
//    }
//    cout<<"BOW Total Number of matches between cam0 and cam1: "<<good_matches_bow.size()<<endl;
//    Mat matches_img_bow;
//    drawMatches( currentFrame->imgs[0], currentFrame->image_kps_undist[0], currentFrame->imgs[1], currentFrame->image_kps_undist[1], good_matches_bow, matches_img_bow, Scalar::all(-1),
//                 Scalar::all(-1) );
//    imshow("Good Matches Bow:"+to_string(0)+"--"+to_string(1), matches_img_bow );
//    waitKey(5);
//
//    BruteForceMatching(currentFrame->imgs[0], currentFrame->imgs[1], currentFrame->image_descriptors[0], currentFrame->image_descriptors[1],
//                       currentFrame->image_kps_undist[0], currentFrame->image_kps_undist[1]);
//
//    if(lfFrames.size()>1){
//        LightFieldFrame* prevLF = lfFrames.back();
//        BruteForceMatching(prevLF->imgs[0], currentFrame->imgs[0], prevLF->image_descriptors[0], currentFrame->image_descriptors[0],
//                           prevLF->image_kps_undist[0], currentFrame->image_kps_undist[0]);
//    }

    ////////////////////// Matching indiidual cameras for debugging///////////////////


    // Project intra-matches on to reference camera and visualize the disparity prior
    // Find out all the matches which do not have a correspondence in the reference frame
    //////////////DEBUGGING VARIABLES/////////////////////////
    //////////////////////////////////////////////////////////
    int real_num_intra = 0;
    int points_on_seg = 0;
    int unaccounted = 0;
    int five_cam_intramatches = 0;
    //////////////DEBUGGING VARIABLES/////////////////////////
    //////////////////////////////////////////////////////////
    // get the rays going through
    std::vector<IntraMatch> filtered_intra_matches;
    assert(words_.size() == matches_map.size());
    set<DBoW2::NodeId> filtered_words;
    obtainLfFeatures(matches_map, currentFrame, filtered_intra_matches, words_, filtered_words);
    // cout<<"Total number of filetered intra matches : "<<filtered_intra_matches.size()<<endl;
    currentFrame->intraMatches.insert(currentFrame->intraMatches.end(), filtered_intra_matches.begin(),
                                      filtered_intra_matches.end());
    //currentFrame->drawIntraMatches();

    // cout<<"Total number of intra matches before : "<<matches_map.size()<<endl;

    // currentFrame->parseIntraMatchBoW();

    stop_intramatch = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop_intramatch - start_intramatch);
    VLOG(3) << "time taken for intra match computation: " << duration.count() << endl;
    VLOG(2) << "Total number of LF features : " << currentFrame->intraMatches.size() << endl;
    ////todo keep track of num mono and num intra
    /// visualize intra matches across pairs - todo: maybe extract into a function ?
    // construct the keypoint matches
    std::vector<KeyPoint> keypoints1, keypoints2;
    vector<Point2f> kp_pts1, kp_pts2;
    vector<DMatch> good_matches;
    int matching_cam = 0;
    std::vector<IntraMatch>::iterator matches_iter;
   /* for (int matching_cam2 = matching_cam + 1; matching_cam2 < currentFrame->num_cams_; matching_cam2++) {
        int ind = 0;
        for (matches_iter = filtered_intra_matches.begin();
             matches_iter != filtered_intra_matches.end(); ++matches_iter) {
            array<int, 5> temp = matches_iter->matchIndex;
            if (temp[matching_cam] != -1 and temp[matching_cam2] != -1) {
                keypoints1.push_back(currentFrame->image_kps[matching_cam][temp[matching_cam]]);
                keypoints2.push_back(currentFrame->image_kps[matching_cam2][temp[matching_cam2]]);
                kp_pts1.push_back(currentFrame->image_kps[matching_cam][temp[matching_cam]].pt);
                kp_pts2.push_back(currentFrame->image_kps[matching_cam2][temp[matching_cam2]].pt);
                good_matches.push_back(DMatch(ind, ind, 0));
                ind++;
            }
        }
        Mat img_matches_mono1;
        cv::drawMatches(currentFrame->imgs[matching_cam], keypoints1, currentFrame->imgs[matching_cam2], keypoints2, good_matches, img_matches_mono1, Scalar::all(-1),
                        Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        ///draw the inatrmatch 2D points as well
        ////-- Show detected matches
        imshow("Inter Matches 1"+to_string(matching_cam)+to_string(matching_cam2), img_matches_mono1);
        waitKey(0);
        Mat inliers;
        cv::findFundamentalMat(kp_pts1, kp_pts2, cv::FM_RANSAC, 2, 0.99,inliers);

        Mat img_matches_mono;
        cv::drawMatches(currentFrame->imgs[matching_cam], keypoints1, currentFrame->imgs[matching_cam2], keypoints2, good_matches, img_matches_mono, Scalar::all(-1),
                        Scalar::all(-1),inliers, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        ///draw the inatrmatch 2D points as well
        ////-- Show detected matches
        imshow("Inter Matches"+to_string(matching_cam)+to_string(matching_cam2), img_matches_mono);
        waitKey(0);

        Mat img_matches_mono2;
        cv::bitwise_not(inliers,inliers);
        cv::drawMatches(currentFrame->imgs[matching_cam], keypoints1, currentFrame->imgs[matching_cam2], keypoints2, good_matches, img_matches_mono2, Scalar::all(-1),
                        Scalar::all(-1),inliers, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        ///draw the inatrmatch 2D points as well
        ////-- Show detected matches
        imshow("Inter Matches outliers"+to_string(matching_cam)+to_string(matching_cam2), img_matches_mono2);
        waitKey(0);
    } */
}


/// Method which returns the latest estimated pose
/// \return
cv::Mat FrontEnd::getPose(){
    unique_lock<mutex> lock(mMutexPose);
    return currentFramePose;
}
////////////// this is just for visualizing which poss estimation is better////////////
cv::Mat FrontEnd::getPose_seventeen(){
    unique_lock<mutex> lock(mMutexPose);
    return currentFramePose1;
}

cv::Mat FrontEnd::getPose_gp3p(){
    unique_lock<mutex> lock(mMutexPose);
    return currentFramePose2;
}


cv::Mat FrontEnd::getPose_Mono(){
    unique_lock<mutex> lock(mMutexPose);
    return currentFramePose_mono;
}
/*cv::Mat FrontEnd::getPose(){
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat pose;
    if(currentFramePose.isZero()) {
        pose = cv::Mat::eye(3, 4, CV_32FC1);
    }
    else
        cv::eigen2cv(currentFramePose, pose);
    return pose;
}*/

vector<cv::Mat> FrontEnd::getAllPoses(){
    unique_lock<mutex> lock(mMutexPose);
    return allPoses;
}

vector<cv::Mat> FrontEnd::getAllPoses_Mono(){
    unique_lock<mutex> lock(mMutexPose);
    return allPoses_mono;
}

////////////??EXTRA for pose vcisualizatrion
vector<cv::Mat> FrontEnd::getAllPoses1(){
    unique_lock<mutex> lock(mMutexPose);
    return allPoses1;
}
vector<cv::Mat> FrontEnd::getAllPoses_gp3p(){
    unique_lock<mutex> lock(mMutexPose);
    return allPoses2;
}

/// Method which clears all the vectors after
/// processing at time step t
void FrontEnd::reset() {
    //clear the data structures which are no longer needed to store
    if(lfFrames.size() > 2){
        //cleanup unncessary data before inserting
        int ref_cam=0;
        LightFieldFrame* f = lfFrames[lfFrames.size()-2-1];

        //f->imgs.clear();
        f->segMasks.clear();
        f->image_descriptors.clear();
        f->image_kps.clear();
    }
    //currentFrame->imgs.clear();
    //currentFrame->segMasks.clear();
    //currentFrame->image_descriptors.clear();
   // currentFrame->image_kps.clear();
   // currentFrame->image_kps_undist.clear();

}

void FrontEnd::writeTrajectoryToFile(const string &filename, bool mono)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;

    ofstream f;
    f.open(filename.c_str());
    f << fixed;
    int ind =0;
    vector<Mat> posesTowrite;
    vector<double> times;
    if(mono){
        posesTowrite = allPoses_mono;
        times = poseTimeStamps_mono;
    }
    else{
        posesTowrite = allPoses;
        times = poseTimeStamps;
    }

    for(auto p : posesTowrite){
        Mat Rwc = p.colRange(0,3);
        Mat twc = p.colRange(3,4);
        //cv::Mat C = (cv::Mat_<double>(3,3) << 0, 1, 0, 0, 0, -1, -1, 0, 0); //above 25_08_2020
        //cv::Mat C = (cv::Mat_<double>(3,3) << -1, 0, 0, 0, 0, 1, 0, 1, 0);
        //Rwc = Rwc * C ;
        vector<float> q = RotMatToQuat(Rwc);
        VLOG(2)<< setprecision(6) << times.at(ind) << " " <<  setprecision(9) << twc.at<double>(0) << " " << twc.at<double>(1) << " " << twc.at<double>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        f << setprecision(6) << times.at(ind) << " " <<  setprecision(9) << twc.at<double>(0) << " " << twc.at<double>(1) << " " << twc.at<double>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        ind++;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}



void FrontEnd::check_tri_sp(int num_views, vector<int> view_inds, IntraMatch& matches,  LightFieldFrame* lf){
    //// THIS IS ONLY FOR DEBUG. WILL BE REMOVED////
    Mat_<double> x(2, num_views);
    cout <<"Detected Key Points.";
    for(int k =0; k <num_views ; k++)
    {
        int cur_view_ind= view_inds[k];
        x(0, k) = lf->image_kps[cur_view_ind][matches.matchIndex[cur_view_ind]].pt.x;
        x(1, k) = lf->image_kps[cur_view_ind][matches.matchIndex[cur_view_ind]].pt.y;
        cout<< cur_view_ind<<": "<< lf->image_kps[cur_view_ind][matches.matchIndex[cur_view_ind]].pt << " || ";
    }
    //triangulate the 3D point
    Eigen::Vector3d pt;
    lf->triangulateIntraMatches(x, view_inds, pt);
    //try opencv sfm triagulation
    vector<Mat> PJs;
    std::vector<Mat_<double> >  xx;
    for(int ii=0; ii<num_views ; ii++){
        int indd = view_inds[ii];
        Mat Rt = build_Rt(camconfig_.R_mats_[indd], camconfig_.t_mats_[indd]);
        //Mat P1 = cv::Mat_<double>(camconfig_.K_mats_[indd]*Rt);
        Mat P1 = cv::Mat_<double>(Rt);
        PJs.push_back(P1);
        Mat_<double> x1(2, 1);
        x1(0, 0) = (lf->image_kps[indd][matches.matchIndex[indd]].pt.x - camconfig_.K_mats_[indd].at<double>(0,2))/camconfig_.K_mats_[indd].at<double>(0,0);
        x1(1, 0) = (lf->image_kps[indd][matches.matchIndex[indd]].pt.y - camconfig_.K_mats_[indd].at<double>(1,2)) / camconfig_.K_mats_[indd].at<double>(1,1);
        xx.push_back(x1.clone());
    }

    cv::Mat pt3d_sfm;
    cv::sfm::triangulatePoints(xx, PJs, pt3d_sfm);

    Mat projected_sfm = camconfig_.K_mats_[0] * pt3d_sfm;
    double expected_x_sfm = projected_sfm.at<double>(0,0) / projected_sfm.at<double>(2,0);
    double expected_y_sfm = projected_sfm.at<double>(1,0) / projected_sfm.at<double>(2,0);
    double inv_depth_sfm = 1.0/ projected_sfm.at<double>(2,0);
    //from multi-cam elas. just for comparison
    Mat P_1 = build_Rt(camconfig_.R_mats_[1], camconfig_.t_mats_[1]);
    double base_0_sfm = -1*P_1.at<double>(0, 3);
    double f_0_sfm =  camconfig_.K_mats_[0].at<double>(0, 0);
    inv_depth_sfm = f_0_sfm * base_0_sfm / (double) pt3d_sfm.at<double>(2,0);
    cout<<"\n3D point SFM: "<<pt3d_sfm.at<double>(0,0)<<","<<pt3d_sfm.at<double>(1,0)<<","<<pt3d_sfm.at<double>(2,0)<<"  expected x,y : "<<expected_x_sfm<<","<<expected_y_sfm<<endl;
    cout <<"Reprojected Key Points.";
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
    cout<<"\n--------------------"<<endl;

}
void FrontEnd::drawInterMatch(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur, int prev_ind, int cur_ind){
    Mat all = Mat::zeros(Size(lf_prev->num_cams_* lf_prev->img_size.width ,2 * lf_prev->img_size.height), CV_8UC3);
    Scalar color_sp = Scalar(0,0,255);
    Scalar color_kp = Scalar(255,0,0);

    //get the intra matches
    IntraMatch intramatches1 = lf_prev->intraMatches[prev_ind];
    IntraMatch intramatches2 = lf_cur->intraMatches[cur_ind];
    Point2f prev_pt1(-1,0);
    Point2f prev_pt2(-1,0);
    int num_views1=0;
    int num_views2=0;
    vector<int> view_inds1;
    vector<int> view_inds2;

    for(int i =0; i < lf_prev->num_cams_ ; i++){
        Mat img1;
        cvtColor(lf_prev->imgs[i], img1,COLOR_GRAY2BGR);
        img1.copyTo(all.rowRange(0,lf_prev->img_size.height ).colRange(i*lf_prev->img_size.width, (i+1)*lf_prev->img_size.width));
        Mat img2 ;
        cvtColor(lf_cur->imgs[i], img2 ,COLOR_GRAY2BGR);
        img2.copyTo(all.rowRange(lf_prev->img_size.height, 2*lf_prev->img_size.height ).colRange(i*lf_prev->img_size.width, (i+1)*lf_prev->img_size.width));
        if(i ==0){ //ref camera we need to plot the reprojected point.
            Point2f p1 = Point2f(lf_prev->sparse_disparity[prev_ind].u, lf_prev->sparse_disparity[prev_ind].v);
            circle(all, p1, 3, color_sp, 3);
            Point2f p2 = Point2f(lf_cur->sparse_disparity[cur_ind].u, lf_cur->sparse_disparity[cur_ind].v)+Point2f(0, lf_prev->img_size.height);
            circle(all, p2, 3, color_sp, 3);
        }

        if (intramatches1.matchIndex[i] != -1) {

            num_views1++;
            view_inds1.push_back(i);

            Point2f p1 = Point2f(lf_prev->img_size.width * i, 0) + lf_prev->image_kps[i][intramatches1.matchIndex[i]].pt;
            circle(all, p1, 3, color_kp, 2);

            if(prev_pt1.x !=-1){
                line(all, prev_pt1, p1, color_kp);
            }
            prev_pt1 = p1;


        }
        if(intramatches2.matchIndex[i] != -1){
            num_views2++;
            view_inds2.push_back(i);

            Point2f p2 = Point2f(lf_prev->img_size.width * i, lf_prev->img_size.height) +
                         lf_cur->image_kps[i][intramatches2.matchIndex[i]].pt;
            circle(all, p2, 3, color_kp, 2);
            if(prev_pt2.x !=-1){
                line(all, prev_pt2, p2, color_kp);
            }
            prev_pt2 = p2;
        }

    }

    check_tri_sp(num_views1, view_inds1, intramatches1,  lf_prev);
    check_tri_sp(num_views2, view_inds2, intramatches2,  lf_cur);

    cv::resize(all,all, Size(lf_prev->num_cams_* lf_prev->img_size.width/2 ,lf_prev->img_size.height) );
    cv::imshow("intermatch ", all);
    cv::waitKey(30);

}


void FrontEnd::insertKeyFrame(){
    VLOG(2)<<"INSERTING KEYFRAME...: "<<currentFrame->frameId<<endl;
    unique_lock<mutex> lock(mMutexPose);
    VLOG(2)<<"Pose of KF"<<currentFrame->pose<<endl;
    currentKFID++;
    lfFrames.push_back(currentFrame);
    allPoses.push_back(currentFramePose.clone());
    allPoses1.push_back(currentFramePose.clone());
    poseTimeStamps.push_back(currentFrame->timeStamp);
}

void FrontEnd::deleteCurrentFrame(){
    delete currentFrame;
}

void FrontEnd::insertKeyFrame_Mono(){
    lfFrames_mono.push_back(currentFrame);
    allPoses_mono.push_back(currentFramePose_mono.clone());
    poseTimeStamps_mono.push_back(currentFrame->timeStamp);
}


///INITIALIZATION AND POSE ESTIMATION and TRIANGULATE IN MONO///
///////////////////////////////////////////////////////////////

//////// TAKEN FROM ORBSLAM ///////////////////////////////////////
void FrontEnd::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
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

void FrontEnd::Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
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

int FrontEnd::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::Point2f>& kps1,
                      const vector<cv::Point2f>& kps2, vector<bool> &inliers,
                      const cv::Mat &K, vector<cv::Point3f> &P3D, float th2, vector<bool> &good,
                      float &parallax, const cv::Mat &R1=Mat(), const cv::Mat &t1=Mat())
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    /// create vector for output variables
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
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1_w+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
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

        if(cosParallax<0.99998)
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


bool FrontEnd::ReconstructF(const vector<cv::Point2f>& kps1, const vector<cv::Point2f>& kps2,
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
    cout<<"Decomposed rotation and translations"<<endl;
    cout<<"R1: "<<endl;
    cout<<R1<<endl;
    cout<<"R2: "<<endl;
    cout<<R2<<endl;
    cout<<"t:"<<endl;
    cout<<t<<endl;
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

void FrontEnd::normalizeKps(vector<cv::Point2f>& kps, vector<cv::Point2f>& kps_n, Mat& T){
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

float FrontEnd::CheckFundamental(const cv::Mat &F, vector<cv::Point2f>& kps1,
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

cv::Mat FrontEnd::solveF_8point(const vector<cv::Point2f> &kps1,const vector<cv::Point2f> &kps2)
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

void FrontEnd::generateRandomIndices(int max_ind, vector<vector<int>>& indices_set, int num_sets){

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

void FrontEnd::FindFundamentalMatrix(vector<Point2f> kps1, vector<Point2f> kps2, Mat& F, vector<bool> &inliers){

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
//void FrontEnd::triangulatePointTwoLFs(LightFieldFrame* lf1, LightFieldFrame* lf2, int featInd1, int featInd2, Mat pose1, Mat pose2,){
//}

void
FrontEnd::get3D_2DCorrs(LightFieldFrame *lf, int featInd, Mat pose, vector<Mat> &projmats, vector<Mat> &PJs, std::vector<Mat_<double>> &xs,
                        vector<int> &view_inds, vector<Point2f> &kps, vector<int> &octaves) {
    IntraMatch m = lf->intraMatches[featInd];
    for (int i1 = 0 ; i1 < lf->num_cams_ ; i1++){
        int ind = m.matchIndex[i1];
        if (ind != -1){
            /// for the LF frame we need to build projection matrices in world frame
//            Mat ref_R_cur = camconfig_.R_mats_[i1].t();
//            Mat ref_t_cur = -ref_R_cur*camconfig_.t_mats_[i1];
//            Mat ref_T_cur = Mat::eye(4, 4, CV_64F);
//            ref_R_cur.copyTo(ref_T_cur(cv::Range(0, 3), cv::Range(0, 3)));
//            ref_t_cur.copyTo(ref_T_cur(cv::Range(0, 3), cv::Range(3, 4)));
//            Mat W_T_cur = pose * ref_T_cur;
//            Mat cur_T_W = W_T_cur.inv();
//            Mat P1 = cv::Mat_<double>(cur_T_W(cv::Range(0,3), cv::Range(0,4)));
            PJs.push_back(projmats[i1]);
            Mat_<double> x1(2, 1);
            x1(0, 0) = (lf->image_kps_undist[i1][ind].pt.x - camconfig_.K_mats_[i1].at<double>(0,2))/camconfig_.K_mats_[i1].at<double>(0,0);
            x1(1, 0) = (lf->image_kps_undist[i1][ind].pt.y - camconfig_.K_mats_[i1].at<double>(1,2)) / camconfig_.K_mats_[i1].at<double>(1,1);
            xs.push_back(x1.clone());
            view_inds.push_back(i1);
            kps.push_back(lf->image_kps_undist[i1][ind].pt);
            octaves.push_back(lf->image_kps_undist[i1][ind].octave);
        }
    }
}
///////////////////////////////////////////////////////////////
void FrontEnd::initialization(){
    // if this is the first frame then insert it as a KF
    // todo: May be check if there are enough intra matches for no
    VLOG(2)<<"Initializing ......."<<endl;
    if (init_cond_ == MIN_FEATS){
        if(currentFrame->intraMatches.size() > 150){
            currentFramePose = cv::Mat::eye(3, 4, CV_64F);
           // currentFramePose1 = cv::Mat::eye(3, 4, CV_64F);
            //currentFramePose2 = cv::Mat::eye(3, 4, CV_64F);

            currentFrame->pose = cv::Mat::eye(4, 4, CV_64F);
            insertKeyFrame();
            initialized_ = INITIALIZED;
            VLOG(2)<<"Initialization DONE"<<endl;
        }
    }
    else{
        if(currentFrame->frameId == 0){
            currentFramePose = cv::Mat::eye(3, 4, CV_64F);
            //currentFramePose1 = cv::Mat::eye(3, 4, CV_64F);
            //currentFramePose2 = cv::Mat::eye(3, 4, CV_64F);
            currentFrame->pose = cv::Mat::eye(4, 4, CV_64F);
            insertKeyFrame();
            return;
        }
        else{
            assert(lfFrames.size() == 1);
            /// match the features between current frame and the prev frame
            LightFieldFrame* lf_prev = lfFrames.back();
            /// Find inter matches, and get the rotation and translation from last keyframe
            std::vector<DMatch> inter_matches, mono_matches;
            opengv::points_t interMatch_3d_1, interMatch_3d_2;
            switch (interMatch_) {
                case BF_MATCH:
                    findInterMatches(lf_prev, currentFrame, inter_matches, mono_matches, false);
                    break;
                case BoW_MATCH:
                    findInterMatchesBow(lf_prev, currentFrame, inter_matches, mono_matches, false);
                    break;
            }
            VLOG(2)<<"Intramatch Intermatches size: "<<inter_matches.size()<<endl;
            VLOG(2)<<"Monomatch Intermatches size: "<<mono_matches.size()<<endl;
            // if there are min number of matches
            if(inter_matches.size() > 50){
                // estimate the pose between the two LF frames and
                //get the inlier features
                Mat T = Mat::eye(4, 4, CV_64F);
                Mat T_17 = Mat::eye(4, 4, CV_64F);
                // find pose of the light field frame using 3D-3d PC alignment
                vector<bool> inliers;
                vector<bool> mono_inliers;
                vector<Mat> triangulatedPts = vector<Mat>(inter_matches.size() + mono_matches.size());
                //int num_inliers = poseFromPCAlignment(lf_prev, currentFrame, inter_matches, inliers, T, false );
                int num_inliers = estimatePoseLF(lf_prev, currentFrame, inter_matches, inliers, T, false);
                int num_mono_inliers = poseFromSeventeenPt(lf_prev, currentFrame, mono_matches, mono_inliers, T_17, false);
                //if the inliers are greater than a threshold (100)

                inter_matches.insert(inter_matches.end(), mono_matches.begin(), mono_matches.end());
                inliers.insert(inliers.end(), mono_inliers.begin(), mono_inliers.end());

                currentFrame->pose = T.clone();

                vector<Mat> ProjMats1, ProjMats2;
                 for(int c_ind=0; c_ind < currentFrame->num_cams_ ; c_ind++){
                     Mat P1 = build_Rt( camconfig_.R_mats_[c_ind],  camconfig_.t_mats_[c_ind]);
                     ProjMats1.push_back(P1.clone());
                 }
                for(int c_ind=0; c_ind < currentFrame->num_cams_ ; c_ind++){
                    Mat cur_T_ref = Mat::eye(4, 4, CV_64F);
                    camconfig_.R_mats_[c_ind].copyTo(cur_T_ref(cv::Range(0, 3), cv::Range(0, 3)));
                    camconfig_.t_mats_[c_ind].copyTo(cur_T_ref(cv::Range(0, 3), cv::Range(3, 4)));
                    Mat cur_T_W = cur_T_ref * T.inv();
                    Mat P1 = cv::Mat_<double>(cur_T_W(cv::Range(0,3), cv::Range(0,4)));
                    ProjMats2.push_back(P1.clone());
                }


                int num_triangulated=0;
                int num_initial_inliers = 0;
                vector<double> parllaxVec;
                //Do a triangulation with non linear refinement of the inliers using both the LF frames
                for (int i = 0 ; i < inter_matches.size(); i++){
                    /// if the current match is an outlier move on to next
                    if(!inliers[i])
                        continue;
                    num_initial_inliers++;
                    ///Get the intramatches and their corresponding 2D observations
                    DMatch m = inter_matches[i];
                    vector<Mat> PJs, centres;
                    vector<int> view_inds;
                    vector<Point2f> kps;
                    int num_views =0;
                    std::vector<Mat_<double> >  xs;
                    vector<int> kp_octaves;
                    Mat o1, o2 ;
                    //extract the 3D-2D correspondences in world coordintes
                    get3D_2DCorrs(lf_prev, m.queryIdx, Mat::eye(4, 4, CV_64F), ProjMats1, PJs, xs, view_inds, kps,kp_octaves);
                    o1 = -1 * PJs[0](cv::Range(0, 3), cv::Range(0, 3)).t() * PJs[0](cv::Range(0, 3), cv::Range(3, 4));
                    int nextI = PJs.size();
                    get3D_2DCorrs(currentFrame, m.trainIdx, T, ProjMats2, PJs, xs, view_inds, kps, kp_octaves);
                    o2 = -1 * PJs[nextI](cv::Range(0, 3), cv::Range(0, 3)).t() * PJs[nextI](cv::Range(0, 3), cv::Range(3, 4));

                    //todo: Triangulation methods implementation
                    ///// Triangulation using  opencv sfm and normalized coordinates///////////
                    //triangulate and get the 3D point
                    cv::Mat pt3d;
                    cv::sfm::triangulatePoints(xs, PJs, pt3d);
                    if(pt3d.at<double>(2,0) < 0){
                        inliers[i] = false;
                        continue;
                    }

                    // check the reprojection errors of the points and the parallalx angle
                    int jj=0;
                    for (auto P : PJs){

                        int cam_ind = view_inds[jj];
                        Mat p3d_conv = P(cv::Range(0, 3), cv::Range(0, 3)) * pt3d +  P(cv::Range(0, 3), cv::Range(3, 4));
                        Mat projected = camconfig_.K_mats_[cam_ind] * p3d_conv;
                        double expected_x = projected.at<double>(0,0) / projected.at<double>(2,0);
                        double expected_y = projected.at<double>(1,0) / projected.at<double>(2,0);
                        double err = (expected_x - kps[jj].x)*(expected_x - kps[jj].x)+(expected_y - kps[jj].y)*(expected_y - kps[jj].y);
                        err = err*orBextractor->GetInverseScaleSigmaSquares()[ kp_octaves[jj]];
                        jj++;
                        if(err>5.991){
                            inliers[i] = false;
                            break;
                        }

                    }
                    if(inliers[i]){
                        // Check parallax
                        cv::Mat normal1 = pt3d - o1;
                        double dist1 = cv::norm(normal1);

                        cv::Mat normal2 = pt3d - o2;
                        double dist2 = cv::norm(normal2);

                        double cosParallax = normal1.dot(normal2)/(dist1*dist2);
                        parllaxVec.push_back(cosParallax);
                        //Eigen::Vector3d pt;
                        //for(int j=0; j<3; ++j)
                        //    pt[j] = pt3d.at<double>(j,0);
                        triangulatedPts[i] = pt3d;
                        //triangulatedPts[i] = pt;
                        num_triangulated++;
                    }

                }

                // CHekc if we have enough parallax, only then initialize
                sort(parllaxVec.begin(),parllaxVec.end());

                size_t idx = min(50,int(parllaxVec.size()-1));
                double parallax = acos(parllaxVec[idx])*180/CV_PI;
                if(parallax < 1.0){
                    // Not enough parallax
                    return;
                }
                //if(norm(T.colRange(3,4).rowRange(0,3)) < kf_translation_threshold*0.5 )
                //    return;

                /// enough initial map points
                if(num_triangulated > 50)
                {

                    VLOG(1)<<"Initialization DONE"<<endl;
                    VLOG(1)<<"Initial Map with landmarks :"<<num_triangulated<<endl;
                    VLOG(2)<<"fraction of Triangulation/number of matches"<<(float)num_triangulated/inter_matches.size();
                    VLOG(2)<<"fraction of Triangulation/pose inliers"<<(float)num_triangulated/num_initial_inliers;
                    //insert the two LF frames as keyframes
                    currentFramePose = T(cv::Range(0,3),cv::Range(0,4)).clone();
                    //currentFramePose1 = T(Range(0,3),Range(0,4)).clone();
                   // currentFramePose2 = T(Range(0,3),Range(0,4)).clone();

                    lf_prev->numTrackedLMs = num_triangulated;
                    currentFrame->numTrackedLMs = num_triangulated;
                    insertKeyFrame();
                    //create Landmarks for each inlier and insert them into global map and well as
                    // the keyframes.
                    for (int i =0; i < inter_matches.size(); i++) {
                        DMatch m = inter_matches[i];
                        if(inliers[i])
                        {
                           // cout<<"3D point from Intra match triangulation"<<endl;
                           // cout<<lf_prev->points_3D[ m.queryIdx];
                           // cout<<"3D point form relative pose triangulation"<<endl;
                           // cout<<triangulatedPts[i];
                           // cout<<"descriptor distance : "<<m.distance<<endl;
                            int l_id = map->insertLandmark( triangulatedPts[i], lf_prev, m.queryIdx, lf_prev->intraMatches[m.queryIdx].uv_ref);
                            map->getLandmark(l_id)->addLfFrame(currentFrame, m.trainIdx, currentFrame->intraMatches[m.trainIdx].uv_ref);
                            lf_prev->lIds[m.queryIdx] = l_id;
                            currentFrame->lIds[m.trainIdx] = l_id;

                            //cout<<"prev_ind : "<<m.queryIdx<<"cur Ind: "<<m.trainIdx<<"Lid: "<<l_id<<endl;
                        }

                    }
                    //make the initialized flag as true
                    initialized_ = INITIALIZED;
                    return;
                }
            }

            // possible non overlapping configuration
            if(inter_matches.size() < 10){
                if(currentFrame->num_cams_ == 1){
                    //initialize monocular SLAM
                    initializationTrials++;
                    if(initialization_mono(mono_matches))
                        return;
                    else if(initializationTrials < 10)
                        return;
                    else
                        initializationTrials = 0;
                }
                else{
                    if(initialization_non_overlapping(mono_matches))
                        return;
                }

            }

            // the initialization was not successful
            // delete the frame from lfFrames and insert the currentframe.
            lfFrames.clear();
            allPoses.clear();
            allPoses1.clear();
            //allPoses2.clear();
            poseTimeStamps.clear();
            currentFramePose = cv::Mat::eye(3, 4, CV_64F);
            //currentFramePose1 = cv::Mat::eye(3, 4, CV_64F);
           // currentFramePose2 = cv::Mat::eye(3, 4, CV_64F);
            currentFrame->pose = cv::Mat::eye(4, 4, CV_64F);
            insertKeyFrame();
            return;

        }
    }
}

bool FrontEnd::initialization_non_overlapping(vector<DMatch> &mono_matches){
    LightFieldFrame* lf_prev = lfFrames.back();
    Mat T_17 = Mat::eye(4, 4, CV_64F);
    // find pose of the light field frame using 3D-3d PC alignment
    vector<bool> inliers;
    vector<Mat> triangulatedPts = vector<Mat>(mono_matches.size());
    int num_mono_inliers = poseFromSeventeenPt(lf_prev, currentFrame, mono_matches, inliers, T_17, false);
    //if the inliers are greater than a threshold (100)

    currentFrame->pose = T_17.clone();
    int num_triangulated=0;

    vector<Mat> ProjMats1, ProjMats2;
    for(int c_ind=0; c_ind < currentFrame->num_cams_ ; c_ind++){
        Mat P1 = build_Rt( camconfig_.R_mats_[c_ind],  camconfig_.t_mats_[c_ind]);
        ProjMats1.push_back(P1.clone());
    }
    for(int c_ind=0; c_ind < currentFrame->num_cams_ ; c_ind++){
        Mat cur_T_ref = Mat::eye(4, 4, CV_64F);
        camconfig_.R_mats_[c_ind].copyTo(cur_T_ref(cv::Range(0, 3), cv::Range(0, 3)));
        camconfig_.t_mats_[c_ind].copyTo(cur_T_ref(cv::Range(0, 3), cv::Range(3, 4)));
        Mat cur_T_W = cur_T_ref * T_17.inv();
        Mat P1 = cv::Mat_<double>(cur_T_W(cv::Range(0,3), cv::Range(0,4)));
        ProjMats2.push_back(P1.clone());
    }


    //Do a triangulation with non linear refinement of the inliers using both the LF frames
    for (int i = 0 ; i < mono_matches.size(); i++){
        /// if the current match is an outlier move on to next
        if(!inliers[i])
            continue;
        ///Get the intramatches and their corresponding 2D observations
        DMatch m = mono_matches[i];
        vector<Mat> PJs;
        vector<int> view_inds;
        vector<Point2f> kps;
        int num_views =0;
        std::vector<Mat_<double> >  xs;
        vector<int> kp_octaves;
        //extract the 3D-2D correspondences in world coordintes
        get3D_2DCorrs(lf_prev, m.queryIdx, Mat::eye(4, 4, CV_64F), ProjMats1, PJs, xs, view_inds, kps, kp_octaves);
        get3D_2DCorrs(currentFrame, m.trainIdx, T_17, ProjMats2, PJs, xs, view_inds, kps, kp_octaves);

        ///// Triangulation using  opencv sfm and normalized coordinates///////////
        //triangulate and get the 3D point
        cv::Mat pt3d;
        cv::sfm::triangulatePoints(xs, PJs, pt3d);
//        if(pt3d.at<double>(2,0) < 0){
//            inliers[i] = false;
//            continue;
//        }

        // check the reprojection errors of the points and the parallalx angle
        int jj=0;
        for (auto P : PJs){
            int cam_ind = view_inds[jj];
            Mat p3d_conv = P(cv::Range(0, 3), cv::Range(0, 3)) * pt3d +  P(cv::Range(0, 3), cv::Range(3, 4));
            Mat projected = camconfig_.K_mats_[cam_ind] * p3d_conv;
            double expected_x = projected.at<double>(0,0) / projected.at<double>(2,0);
            double expected_y = projected.at<double>(1,0) / projected.at<double>(2,0);
            double err = (expected_x - kps[jj].x)*(expected_x - kps[jj].x)+(expected_y - kps[jj].y)*(expected_y - kps[jj].y);
            jj++;
            if(err>4){
                inliers[i] = false;
                break;
            }

        }
        if(inliers[i]){
            //Eigen::Vector3d pt;
            //for(int j=0; j<3; ++j)
            //    pt[j] = pt3d.at<double>(j,0);
            triangulatedPts[i] = pt3d;
            //triangulatedPts[i] = pt;
            num_triangulated++;
        }

    }
    /// enough initial map points
    if(num_triangulated > 50)
    {

        VLOG(1)<<"Initialization DONE"<<endl;
        VLOG(1)<<"Initial Map with landmarks :"<<num_triangulated<<endl;
        //insert the two LF frames as keyframes
        currentFramePose = T_17(cv::Range(0,3),cv::Range(0,4)).clone();
        //currentFramePose1 = T(Range(0,3),Range(0,4)).clone();
        // currentFramePose2 = T(Range(0,3),Range(0,4)).clone();

        lf_prev->numTrackedLMs = num_triangulated;
        currentFrame->numTrackedLMs = num_triangulated;
        insertKeyFrame();
        VLOG(1)<<"lid    lf_prev    lf_cur    pt3d"<<endl;
        //create Landmarks for each inlier and insert them into global map and well as
        // the keyframes.
        for (int i =0; i < mono_matches.size(); i++) {
            DMatch m = mono_matches[i];
            if(inliers[i])
            {
                // cout<<"3D point from Intra match triangulation"<<endl;
                // cout<<lf_prev->points_3D[ m.queryIdx];
                // cout<<"3D point form relative pose triangulation"<<endl;
                // cout<<triangulatedPts[i];
                // cout<<"descriptor distance : "<<m.distance<<endl;
                int l_id = map->insertLandmark( triangulatedPts[i], lf_prev, m.queryIdx, lf_prev->intraMatches[m.queryIdx].uv_ref);
                map->getLandmark(l_id)->addLfFrame(currentFrame, m.trainIdx, currentFrame->intraMatches[m.trainIdx].uv_ref);
                lf_prev->lIds[m.queryIdx] = l_id;
                currentFrame->lIds[m.trainIdx] = l_id;

                //cout<<"prev_ind : "<<m.queryIdx<<"cur Ind: "<<m.trainIdx<<"Lid: "<<l_id<<endl;
            }

        }
        //make the initialized flag as true
        initialized_ = INITIALIZED;
        //VLOG(1)<<"         time          ,  KF, Lms Prev Fr, Tracked lms, New matches, Inlier Lms, Inlier tri"<<endl;
        return true;
    }
    return false;
}

bool FrontEnd::initialization_mono(vector<DMatch> &matches_mono){
    bool FUND = false;

    // if there not enough matches exit
    if(matches_mono.size() >= 100){

        cv::Mat rotation_t = Mat::zeros(3, 3, CV_32F), translation_t = Mat::zeros(3, 1, CV_32F);
        bool res = false;
        vector<cv::Point3f> P3D;
        vector<bool> Triangulated;
        vector<bool> inliers;
        int num_inliers = 0;
        int num_good_tri = 0;

        LightFieldFrame* prevFrame = lfFrames.back();
        vector<Point2f> kp_pts1_mono, kp_pts2_mono;
        Mat mask_mono;

        for(auto &m : matches_mono){
            Point2f p_prev = prevFrame->intraMatches[m.queryIdx].uv_ref;
            Point2f p_cur = currentFrame->intraMatches[m.trainIdx].uv_ref;
            kp_pts1_mono.push_back(p_prev);
            kp_pts2_mono.push_back(p_cur);
        }

        ///using normalized coordinates and fundMat
        if(FUND){
            Mat F,K;
            FindFundamentalMatrix(kp_pts1_mono, kp_pts2_mono, F , inliers );
            //Get the pose and triangulate the inliers
            // Compute Essential Matrix from Fundamental Matrix
            camconfig_.K_mats_[0].convertTo(K, CV_32F);
            res = ReconstructF(kp_pts1_mono,kp_pts2_mono, inliers, F, K,1.0,
                               rotation_t, translation_t, P3D, Triangulated, 1.0, 50);
            for(int p=0; p < Triangulated.size() ; p++){
                if(Triangulated[p])
                    num_good_tri++;
            }
        }
        else{
            ///ESSENIAL MATRIX
            cv::Mat E_mat;
            //find the essential matrix
            E_mat = findEssentialMat(kp_pts1_mono, kp_pts2_mono , camconfig_.K_mats_[0],cv::RANSAC,0.97, 1.0, mask_mono);
            //recover pose
            recoverPose(E_mat, kp_pts1_mono, kp_pts2_mono, camconfig_.K_mats_[0], rotation_t, translation_t, mask_mono);
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
            Mat K;
            camconfig_.K_mats_[0].convertTo(K, CV_32F);
            rotation_t.convertTo(rotation_t, CV_32F);
            translation_t.convertTo(translation_t, CV_32F);
            num_good_tri = CheckRT(rotation_t,translation_t,kp_pts1_mono,kp_pts2_mono,inliers,
                                   K, P3D, 4.0*1.0*1.0, Triangulated, parallax);
            // if the trianglated points are a minimum number and have enough parallx
            if (num_good_tri >= 50 and parallax > 1.0)
            {
                res = true;

            }
        }


        // if there are enough inliers
        if(res){
            VLOG(1)<<"Initialization DONE"<<endl;
            VLOG(1)<<"Initial Map with landmarks :"<<num_good_tri<<endl;

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
            rotation.copyTo(T_mono(cv::Range(0, 3), cv::Range(0, 3)));
            translation.copyTo(T_mono(cv::Range(0, 3), cv::Range(3, 4)));
            T_mono.convertTo(T_mono, CV_64F);

            currentFrame->pose = T_mono.clone();
            currentFramePose = T_mono(cv::Range(0,3),cv::Range(0,4)).clone();

            prevFrame->numTrackedLMs = num_good_tri;
            currentFrame->numTrackedLMs = num_good_tri;
            insertKeyFrame();

            //insert the triangulated points into the map
            // Now denote the matches as tracks.
            int ii=0;
            for(auto& m : matches_mono){
                if(Triangulated[ii]){
                    int l_id = prevFrame->lIds[m.queryIdx];
                    if( l_id == -1){ // landmark has not been inserted
                        Mat p_world = Mat(P3D[ii]);
                        Mat p_world_d;
                        p_world.convertTo(p_world_d, CV_64F);

                        l_id = map->insertLandmark( p_world_d, prevFrame, m.queryIdx, prevFrame->intraMatches[m.queryIdx].uv_ref);
                        prevFrame->lIds[m.queryIdx] = l_id;
                        map->getLandmark(l_id)->addLfFrame(currentFrame, m.trainIdx, currentFrame->intraMatches[m.trainIdx].uv_ref);
                        currentFrame->lIds[m.trainIdx] = l_id;
                    }
                }
                ii++;
            }
            cout<<"INITIALIZED"<<endl;
            initialized_ = INITIALIZED;
            return true;
        } //end of if res

    } //end of number of matches

    return false;

}


void FrontEnd::findMatchesMono(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur, int cam_ind, std::vector<DMatch>& matches){
    vector<vector<DMatch>> matches_mono;
    matches.clear();
    vector<Point2f> kp_pts1_mono, kp_pts2_mono;
    Mat mask_mono;

    Mat descs1_mono = Mat(lf_prev->image_descriptors[cam_ind].size(), lf_prev->image_descriptors[cam_ind][0].cols, CV_8U );
    Mat descs2_mono = Mat(lf_cur->image_descriptors[cam_ind].size(), lf_cur->image_descriptors[cam_ind][0].cols, CV_8U );
    int ind=0;
    for (auto& d : lf_prev->image_descriptors[cam_ind]){
        d.copyTo(descs1_mono.row(ind));
        ind++;
    }
    ind=0;
    for (auto& d : lf_cur->image_descriptors[cam_ind]){
        d.copyTo(descs2_mono.row(ind));
        ind++;
    }

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->knnMatch(descs1_mono, descs2_mono, matches_mono, 2);

    for(auto &m : matches_mono){
        if(m[0].distance < 0.7*m[1].distance) {
            if (m[0].distance > 50){
                //cout<<"descriptor distance: "<<endl;
                continue;
            }
            Point2f p_prev = lf_prev->image_kps_undist[cam_ind][m[0].queryIdx].pt;
            Point2f p_cur = lf_cur->image_kps_undist[cam_ind][m[0].trainIdx].pt;
            //make sure that the points belong to static areas based on the segmasks
            if (lf_prev->segMasks[cam_ind].at<float>(p_prev.y, p_prev.x) < 0.7 and lf_cur->segMasks[cam_ind].at<float>(p_cur.y, p_cur.x) < 0.7){
                kp_pts1_mono.push_back(p_prev);
                kp_pts2_mono.push_back(p_cur);
                matches.push_back(m[0]);
            }

        }

    }
}


void FrontEnd::findInterMatches(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur,
                                std::vector<DMatch>& matches_z_filtered, std::vector<DMatch>& matches_mono_z_filtered,
                                bool viz){

    ///go through all the descriptors and support points
    /// put the descs into a Mat and create a keypoint for each support point.
    /// by default support point is in the reference frame

    // initialize a BF matcher
    Mat descs1 = Mat(lf_prev->intraMatches.size(), lf_prev->intraMatches[0].matchDesc.cols, CV_8U);
    Mat descs2 = Mat(lf_cur->intraMatches.size(), lf_cur->intraMatches[0].matchDesc.cols, CV_8U);
    vector<cv::KeyPoint> kps1, kps2;

    /// todo: mono uv_ref not in ref cam. fix
    /// todo: use vecDesc instead of recreating desc vector
    int ind = 0;
    for (auto &d : lf_prev->intraMatches) {
        d.matchDesc.copyTo(descs1.row(ind));
        kps1.push_back(KeyPoint(d.uv_ref, 3.0));
        ind++;
    }
    ind = 0;
    for (auto &d : lf_cur->intraMatches) {
        d.matchDesc.copyTo(descs2.row(ind));
        kps2.push_back(KeyPoint(d.uv_ref, 3.0));
        ind++;
    }

    std::vector<DMatch> good_matches;
    std::vector<DMatch> matches_z_outliers;
    vector<vector<vector<DMatch>>> vizMatches(lf_prev->num_cams_,vector<vector<DMatch>>(lf_cur->num_cams_, vector<DMatch>()));
    matches_z_filtered.clear();

    std::vector<int> inds1, inds2;
    std::vector<int> mono_inds1, mono_inds2;
    vector<int>::const_iterator inds1_it, inds2_it;
    vector<int>::const_iterator mono_inds1_it, mono_inds2_it;

    /// create a feature matcher and match the features
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<vector<DMatch>> matches;
    matcher->knnMatch(descs1, descs2, matches, 2);

    ///filter the 3D points to get the matches
    vector<Point2f> ref_pts1, ref_pts2;
    int z_filtered_matches_cnt=0;
    vector<uint8_t> lm_flags;
    int num_lm_matches=0;
    int num_lm_matches_unique=0;
    int num_lm_matches_zfil = 0;

    /// go over the matches and get the unique matches
    for (auto &m : matches) {

        // if the match is a landmark, use it as it is even if it is not unique
        if(lf_prev->lIds[m[0].queryIdx] != -1){
            num_lm_matches++;
        }
        else{
            // if the match fails the ratio test move on
            if (m[0].distance > 0.7 * m[1].distance)
                continue;
        }
        good_matches.push_back(m[0]);

        /// if both the features are intramatches, check the difference in depth
        if(!lf_prev->intraMatches[m[0].queryIdx].mono and !lf_cur->intraMatches[m[0].trainIdx].mono) {
            cv::Mat  dif= lf_prev->intraMatches[m[0].queryIdx].point3D - lf_cur->intraMatches[m[0].trainIdx].point3D;
            float distance = cv::norm(dif);
            if(distance <= 2.0){   //
                z_filtered_matches_cnt++;
            }
            else{
                matches_z_outliers.push_back(m[0]);
                continue;
            }
        }

        ///find if the intermatch is already exising
        if(m[0].trainIdx < lf_cur->intramatch_size) {
            inds2_it = find(inds2.begin(), inds2.end(), m[0].trainIdx);
            mono_inds2_it = find(mono_inds2.begin(), mono_inds2.end(), m[0].trainIdx);
        }
        else{
            inds2_it = inds2.end();
            mono_inds2_it = find(mono_inds2.begin(), mono_inds2.end(), m[0].trainIdx);
        }
        ///if this is a new match
        if(inds2_it == inds2.end() && mono_inds2_it == mono_inds2.end())
        {
            int ind1 = m[0].queryIdx, ind2 = m[0].trainIdx;
            if(ind1 < lf_prev->intramatch_size && ind2 < lf_cur->intramatch_size) {
                inds2.push_back(ind2);
                inds1.push_back(ind1);
                matches_z_filtered.push_back(m[0]);
            }
            else {
                mono_inds2.push_back(ind2);
                mono_inds1.push_back(ind1);
                matches_mono_z_filtered.push_back((m[0]));
            }
            /// push matches into the viz matches structure
            IntraMatch im1 = lf_prev->intraMatches[m[0].queryIdx];
            IntraMatch im2 = lf_cur->intraMatches[m[0].trainIdx];
            int ii=0, ii2=0;
            if (im1.mono){
                ii=0;
                for(int featInd : im1.matchIndex){
                    if(featInd != -1)
                        break;
                    ii++;
                }
                assert(ii != lf_prev->num_cams_);

            }

            if(im2.mono){
                ii2=0;
                for(int featInd : im2.matchIndex){
                    if(featInd != -1)
                        break;
                    ii2++;
                }
                assert(ii2 != lf_cur->num_cams_);
            }

            vizMatches[ii][ii2].push_back(m[0]);

        }
        else if(mono_inds2_it == mono_inds2.end())/// the feature in second LF frame s already matched previouslly with an intramatch-intermatch
        {
            int idx_1 = inds1[ inds2_it - inds2.begin() ];
            double d = matches_z_filtered[inds2_it - inds2.begin()].distance; //norm( descs1.row(idx_1), descs2.row(m[0].trainIdx), NORM_HAMMING);
            if(m[0].distance < d)
            {
                inds1[  inds2_it - inds2.begin()] = m[0].queryIdx;
                matches_z_filtered[inds2_it - inds2.begin()] = m[0];
            }
        }
        else/// the feature in second LF frame s already matched previouslly with a mono-intermatch
        {
            int idx_1 = mono_inds1[ mono_inds2_it - mono_inds2.begin() ];
            double d = matches_mono_z_filtered[mono_inds2_it - mono_inds2.begin()].distance; //norm( descs1.row(idx_1), descs2.row(m[0].trainIdx), NORM_HAMMING);
            if(m[0].distance < d)
            {
                mono_inds1[  mono_inds2_it - mono_inds2.begin()] = m[0].queryIdx;
                matches_mono_z_filtered[mono_inds2_it - mono_inds2.begin()] = m[0];
            }
        }

    }


    for(int i =0; i >inds1.size(); i++){
        if(lf_prev->lIds[inds1[i]] != -1)
            num_lm_matches_zfil++;
        if(lf_prev->lIds[inds1[i]] != -1){
            lm_flags.push_back(255);
        }
        else{
            lm_flags.push_back(0);
        }
    }

    VLOG(1)<<"number of inter matches after ratio test : "<< good_matches.size()<<endl;
    VLOG(1) << "number of inter matches of Intramatch features after z test: " << z_filtered_matches_cnt << endl;
    //cout << "percentage of good points" << (z_filtered_matches_cnt / (float)good_matches.size()) * 100.0 << endl;
    VLOG(1)<<"Number of landmark matches"<<num_lm_matches_zfil<<endl;
    //cout<<"Number of landmark matches unique"<<num_lm_matches_unique<<endl;
    //VLOG(1)<<"Number of landmark matches z filtered"<<num_lm_matches_zfil<<endl;
    if( viz){
        for(int i=0; i <lf_prev->num_cams_; i++){
            for(int j=0; j < lf_cur->num_cams_; j++){
                Mat img_matches_mono;
                cv::drawMatches(lf_prev->imgs[i], kps1, lf_cur->imgs[j], kps2, vizMatches[i][j], img_matches_mono, Scalar::all(-1),
                                Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                ///draw the inatrmatch 2D points as well
                ////-- Show detected matches
                imshow("Inter Matches"+to_string(i)+to_string(j), img_matches_mono);
                waitKey(30);
            }
        }
    }
}


void FrontEnd::findInterMatches(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur, std::vector<DMatch>& matches_z_filtered,  bool viz){

     ///go through all the descriptors and support points
    /// put the descs into a Mat and create a keypoint for each support point.
    /// by default support point is in the reference frame

    // initialize a BF matcher
    Mat descs1 = Mat(lf_prev->intraMatches.size(), lf_prev->intraMatches[0].matchDesc.cols, CV_8U);
    Mat descs2 = Mat(lf_cur->intraMatches.size(), lf_cur->intraMatches[0].matchDesc.cols, CV_8U);
    vector<cv::KeyPoint> kps1, kps2;

    /// todo: mono uv_ref not in ref cam. fix
    /// todo: use vecDesc instead of recreating desc vector
    int ind = 0;
    for (auto &d : lf_prev->intraMatches) {
        d.matchDesc.copyTo(descs1.row(ind));
        kps1.push_back(KeyPoint(d.uv_ref, 3.0));
        ind++;
    }
    ind = 0;
    for (auto &d : lf_cur->intraMatches) {
        d.matchDesc.copyTo(descs2.row(ind));
        kps2.push_back(KeyPoint(d.uv_ref, 3.0));
        ind++;
    }

    std::vector<DMatch> good_matches;
    std::vector<DMatch> matches_z_outliers;
    vector<vector<vector<DMatch>>> vizMatches(lf_prev->num_cams_,vector<vector<DMatch>>(lf_cur->num_cams_, vector<DMatch>()));
    matches_z_filtered.clear();

    std::vector<int> inds1, inds2;
    vector<int>::const_iterator inds1_it, inds2_it;

    /// create a feature matcher and match the features
     Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
     vector<vector<DMatch>> matches;
     matcher->knnMatch(descs1, descs2, matches, 2);

    ///filter the 3D points to get the matches
    vector<Point2f> ref_pts1, ref_pts2;
    int z_filtered_matches_cnt=0;
    vector<uint8_t> lm_flags;
    int num_lm_matches=0;
    int num_lm_matches_unique=0;
    int num_lm_matches_zfil = 0;

    /// go over the matches and get the unique matches
    for (auto &m : matches) {

        // if the match is a landmark, use it as it is even if it is not unique
        if(lf_prev->lIds[m[0].queryIdx] != -1){
            num_lm_matches++;
        }
        else{
            // if the match fails the ratio test move on
            if (m[0].distance > 0.7 * m[1].distance)
                 continue;
        }
        good_matches.push_back(m[0]);

        /// if both the features are intramatches, check the difference in depth
        if(!lf_prev->intraMatches[m[0].queryIdx].mono and !lf_cur->intraMatches[m[0].trainIdx].mono) {
            cv::Mat  dif= lf_prev->intraMatches[m[0].queryIdx].point3D - lf_cur->intraMatches[m[0].trainIdx].point3D;
            float distance = cv::norm(dif);
            if(distance <= 2.0){   //
                z_filtered_matches_cnt++;
            }
            else{
                matches_z_outliers.push_back(m[0]);
                continue;
            }
        }

        ///find if the intermatch is already exising
        inds2_it = find(inds2.begin(), inds2.end(), m[0].trainIdx);
        ///if this is a new match
        if(inds2_it == inds2.end())
        {
            inds2.push_back(m[0].trainIdx);
            inds1.push_back(m[0].queryIdx);
            matches_z_filtered.push_back(m[0]);
            /// push matches into the viz matches structure
            IntraMatch im1 = lf_prev->intraMatches[m[0].queryIdx];
            IntraMatch im2 = lf_cur->intraMatches[m[0].trainIdx];
            int ii=0, ii2=0;
            if (im1.mono){
                ii=0;
                for(int featInd : im1.matchIndex){
                    if(featInd != -1)
                        break;
                    ii++;
                }
                assert(ii != lf_prev->num_cams_);

            }

            if(im2.mono){
                ii2=0;
                for(int featInd : im2.matchIndex){
                    if(featInd != -1)
                        break;
                    ii2++;
                }
                assert(ii2 != lf_cur->num_cams_);
            }

            vizMatches[ii][ii2].push_back(m[0]);

        }
        else /// the feature in second LF frame s already matched previouslly
        {
            int idx_1 = inds1[ inds2_it - inds2.begin() ];
            double d = matches_z_filtered[inds2_it - inds2.begin()].distance; //norm( descs1.row(idx_1), descs2.row(m[0].trainIdx), NORM_HAMMING);
            if(m[0].distance < d)
            {
                inds1[  inds2_it - inds2.begin()] = m[0].queryIdx;
                matches_z_filtered[inds2_it - inds2.begin()] = m[0];
            }
        }

    }


    for(int i =0; i >inds1.size(); i++){
        if(lf_prev->lIds[inds1[i]] != -1)
            num_lm_matches_zfil++;
        if(lf_prev->lIds[inds1[i]] != -1){
            lm_flags.push_back(255);
        }
        else{
            lm_flags.push_back(0);
        }
    }

    VLOG(1)<<"number of inter matches after ratio test : "<< good_matches.size()<<endl;
    VLOG(1) << "number of inter matches of Intramatch features after z test: " << z_filtered_matches_cnt << endl;
    //cout << "percentage of good points" << (z_filtered_matches_cnt / (float)good_matches.size()) * 100.0 << endl;
    VLOG(1)<<"Number of landmark matches"<<num_lm_matches_zfil<<endl;
    //cout<<"Number of landmark matches unique"<<num_lm_matches_unique<<endl;
    //VLOG(1)<<"Number of landmark matches z filtered"<<num_lm_matches_zfil<<endl;
    if( viz){
        for(int i=0; i <lf_prev->num_cams_; i++){
            for(int j=0; j < lf_cur->num_cams_; j++){
                Mat img_matches_mono;
                cv::drawMatches(lf_prev->imgs[i], kps1, lf_cur->imgs[j], kps2, vizMatches[i][j], img_matches_mono, Scalar::all(-1),
                                Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                ///draw the inatrmatch 2D points as well
                ////-- Show detected matches
                imshow("Inter Matches"+to_string(i)+to_string(j), img_matches_mono);
                waitKey(30);
            }
        }
    }
}


void FrontEnd::findInterMatchesBow(LightFieldFrame *lf_prev, LightFieldFrame *lf_cur, std::vector<DMatch> &matches,
                                   std::vector<DMatch> &mono_matches, bool viz) {

    vector<unsigned int> ind1, ind2;
    set<DBoW2::NodeId> words;
    InterMatchingBow(lf_prev, lf_cur, ind1, ind2, words);
    assert(ind1.size() == ind2.size());
    vector<vector<vector<DMatch>>> vizMatches(lf_prev->num_cams_, vector<vector<DMatch>>(lf_cur->num_cams_, vector<DMatch>()));

    for (int i = 0; i < ind1.size(); i++) {
        DMatch m;
        m.queryIdx = ind1[i];
        m.trainIdx = ind2[i];
        if (ind1[i] < lf_prev->intramatch_size) // && ind2[i] < lf_cur->intramatch_size)
            matches.push_back(m);
        else
            mono_matches.push_back(m);

        IntraMatch im1 = lf_prev->intraMatches[m.queryIdx];
        IntraMatch im2 = lf_cur->intraMatches[m.trainIdx];
        if (viz) {
            int ii = 0, ii2 = 0;
            if (im1.mono) {
                ii = 0;
                for (int featInd: im1.matchIndex) {
                    if (featInd != -1)
                        break;
                    ii++;
                }
                assert(ii != lf_prev->num_cams_);

            }

            if (im2.mono) {
                ii2 = 0;
                for (int featInd: im2.matchIndex) {
                    if (featInd != -1)
                        break;
                    ii2++;
                }
                assert(ii2 != lf_cur->num_cams_);
            }

            vizMatches[ii][ii2].push_back(m);
        }

    }

    cout << "Total Number of Inter matches" << matches.size() << endl;

    if(viz){

        Mat all;
        vector<cv::KeyPoint> kps1, kps2;
        int ind = 0;
        for (auto &d: lf_prev->intraMatches) {
            kps1.push_back(KeyPoint(d.uv_ref, 3.0));

            ind++;
        }
        ind = 0;
        for (auto &d: lf_cur->intraMatches) {
            kps2.push_back(KeyPoint(d.uv_ref, 3.0));
            ind++;
        }

        Mat img_matches_mono;
        cv::drawMatches(lf_prev->imgs[0], kps1, lf_cur->imgs[0], kps2, vizMatches[0][0], all,
                        Scalar(150,150,150),
                        Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);



        //Show the support matches
        cv::Mat dMap(Size(1,lf_prev->intramatch_size),CV_8UC1);
        int ind_Dmap=0;
        for(IntraMatch im : lf_prev->intraMatches){
            if(!im.mono) {
                dMap.at<uint8_t>(ind_Dmap, 0) = (uint8_t) im.point3D.at<double>(2, 0);
                ind_Dmap++;
            }

        }

        Mat out;
        //all.create(lf_prev->imgs[0].rows, lf_prev->imgs[0].cols * 2, CV_8UC3);
        cv::normalize(dMap, dMap, 0, 220, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(dMap, out, cv::COLORMAP_RAINBOW);
        ind_Dmap=0;
        for(IntraMatch im : lf_prev->intraMatches){
            if(!im.mono){
                Vec3b col1 = out.at<Vec3b>(ind_Dmap,0);
                cv::circle(all, im.uv_ref , 3, Scalar(col1[0],col1[1],col1[2]), 1,8,0 );
                ind_Dmap++;
            }
            else{

                if(im.matchIndex[0] != -1)
                    cv::circle(all, lf_prev->image_kps_undist[0][im.matchIndex[0]].pt ,  3, Scalar(255,255,255), 1 );

            }
        }

        //Show the support matches
        cv::Mat dMap2(Size(1,lf_cur->intramatch_size),CV_8UC1);
        int ind_Dmap2=0;
        for(IntraMatch im : lf_cur->intraMatches){
            if(!im.mono) {
                dMap2.at<uint8_t>(ind_Dmap2, 0) = (uint8_t) im.point3D.at<double>(2, 0);
                ind_Dmap2++;
            }

        }

        Mat out2;
        cv::normalize(dMap2, dMap2, 0, 220, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(dMap2, out2, cv::COLORMAP_RAINBOW);
        ind_Dmap2=0;
        for(IntraMatch im : lf_cur->intraMatches){
            if(!im.mono){
                Vec3b col1 = out2.at<Vec3b>(ind_Dmap2,0);
                cv::circle(all, im.uv_ref+ Point2f( lf_cur->imgs[0].cols , 0) , 3, Scalar(col1[0],col1[1],col1[2]), 1,8,0 );
                ind_Dmap2++;
            }
            else{
                if(im.matchIndex[0] != -1)
                    cv::circle(all, lf_cur->image_kps_undist[0][im.matchIndex[0]].pt+ Point2f( lf_cur->imgs[0].cols , 0),  3, Scalar(255,255,255), 1 );
            }
        }
        imshow("Inter Matches", all);
        waitKey(0);

    }


    if (false) {

        vector<cv::KeyPoint> kps1, kps2;
        int ind = 0;
        for (auto &d: lf_prev->intraMatches) {
            kps1.push_back(KeyPoint(d.uv_ref, 3.0));

            ind++;
        }
        ind = 0;
        for (auto &d: lf_cur->intraMatches) {
            kps2.push_back(KeyPoint(d.uv_ref, 3.0));
            ind++;
        }

        Mat img_matches_mono;


        Mat mask = cv::Mat::zeros(matches.size(), 1, CV_8UC1);
        if(matches.size() == 0)
            mask = cv::Mat::zeros(mono_matches.size(), 1, CV_8UC1);
        mask = mask * 255;
        for (int i = 0; i < lf_prev->num_cams_; i++) {
            for (int j = 0; j < lf_cur->num_cams_; j++) {
                Mat img_matches_mono;
                cv::drawMatches(lf_prev->imgs[i], kps1, lf_cur->imgs[j], kps2, vizMatches[i][j], img_matches_mono,
                                Scalar::all(-1),
                                Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                ///draw the inatrmatch 2D points as well
                ////-- Show detected matches
                imshow("Inter Matches" + to_string(i) + to_string(j), img_matches_mono);
                waitKey(0);
            }
        }
    }

}

void FrontEnd::findInterMatchesBow(LightFieldFrame *lf_prev, LightFieldFrame *lf_cur, std::vector<DMatch> &matches,
                                   bool viz) {

    vector<unsigned int> ind1, ind2;
    set<DBoW2::NodeId> words;
    InterMatchingBow(lf_prev, lf_cur, ind1, ind2, words);
//    ind1.clear();
//    ind2.clear();
//    InterMatchingBow(lf_prev->BoW_feats[0], lf_cur->BoW_feats[0], lf_prev->image_descriptors[0], lf_cur->image_descriptors[0], ind1, ind2);
//    VLOG(2)<<"Number of Bow Matches between prev left- cur left :"<<ind1.size();
//    ind1.clear();
//    ind2.clear();
//    InterMatchingBow(lf_prev->BoW_feats[0], lf_cur->BoW_feats[1], lf_prev->image_descriptors[0], lf_cur->image_descriptors[1], ind1, ind2);
//    VLOG(2)<<"Number of Bow Matches between prev left- cur right :"<<ind1.size();

    assert(ind1.size() == ind2.size());
    vector<vector<vector<DMatch>>> vizMatches(lf_prev->num_cams_,vector<vector<DMatch>>(lf_cur->num_cams_, vector<DMatch>()));

    for(int i=0; i<ind1.size(); i++){
        DMatch m;
        m.queryIdx = ind1[i];
        m.trainIdx = ind2[i];
        matches.push_back(m);

        IntraMatch im1 = lf_prev->intraMatches[m.queryIdx];
        IntraMatch im2 = lf_cur->intraMatches[m.trainIdx];
        if(viz){
            int ii=0, ii2=0;
            if (im1.mono){
                ii=0;
                for(int featInd : im1.matchIndex){
                    if(featInd != -1)
                        break;
                    ii++;
                }
                assert(ii != lf_prev->num_cams_);

            }

            if(im2.mono){
                ii2=0;
                for(int featInd : im2.matchIndex){
                    if(featInd != -1)
                        break;
                    ii2++;
                }
                assert(ii2 != lf_cur->num_cams_);
            }

            vizMatches[ii][ii2].push_back(m);
        }

    }

    cout<<"Total Number of Inter matches"<<matches.size()<<endl;
    if(viz){

        vector<cv::KeyPoint> kps1, kps2;
        int ind = 0;
        for (auto &d : lf_prev->intraMatches) {
            kps1.push_back(KeyPoint(d.uv_ref, 3.0));

            ind++;
        }
        ind = 0;
        for (auto &d : lf_cur->intraMatches) {
            kps2.push_back(KeyPoint(d.uv_ref, 3.0));
            ind++;
        }

        Mat img_matches_mono;
        Mat mask = cv::Mat::zeros(matches.size(),1,CV_8UC1);
        mask = mask *255;
        for(int i=0; i <lf_prev->num_cams_; i++){
            for(int j=0; j < lf_cur->num_cams_; j++){
                Mat img_matches_mono;
                cv::drawMatches(lf_prev->imgs[i], kps1, lf_cur->imgs[j], kps2, vizMatches[i][j], img_matches_mono, Scalar::all(-1),
                                Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                ///draw the inatrmatch 2D points as well
                ////-- Show detected matches
                imshow("Inter Matches"+to_string(i)+to_string(j), img_matches_mono);
                waitKey(30);
            }
        }
    }

}


void FrontEnd::InterMatchingBow( DBoW2::FeatureVector lfFeatVec1, DBoW2::FeatureVector lfFeatVec2, std::vector<cv::Mat> img_desc1, std::vector<cv::Mat> img_desc2, vector<unsigned int>& indices_1,
                                vector<unsigned int>& indices_2 ){
    indices_2.clear();
    indices_1.clear();

    int itr_cng_match=0;
    // iterators for each of the feature vectors
    DBoW2::FeatureVector::const_iterator featVec1_it, featVec2_it;
    featVec1_it = lfFeatVec1.begin();
    featVec2_it = lfFeatVec2.begin();

    const auto featVec1_end = lfFeatVec1.end(); // std::next(featVec1.begin(),200);    //
    const auto featVec2_end = lfFeatVec2.end(); //std::next(featVec2.begin(),200);   //
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

            orBextractor->getMatches_distRatio(img_desc1, featVec1_it->second,
                                         img_desc2, featVec2_it->second, i_ind_tmp, j_ind_tmp,
                                         itr_cng_match);

            indices_1.insert(indices_1.end(), i_ind_tmp.begin(), i_ind_tmp.end());
            indices_2.insert(indices_2.end(), j_ind_tmp.begin(), j_ind_tmp.end());

            // move featVec1_it and featVec2_it forward
            ++featVec1_it;
            ++featVec2_it;
        }
        else if(featVec1_it->first < featVec2_it->first)
        {
            // move old_it forward
            featVec1_it = lfFeatVec1.lower_bound(featVec2_it->first);
            // old_it = (first element >= cur_it.id)
        }
        else
        {
            // move cur_it forward
            featVec2_it = lfFeatVec2.lower_bound(featVec1_it->first);
            // cur_it = (first element >= old_it.id)
        }
    }
}


void FrontEnd::InterMatchingBow(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur, vector<unsigned int>& indices_1,
                                vector<unsigned int>& indices_2,set<DBoW2::NodeId>& words ){
    indices_2.clear();
    indices_1.clear();

    int itr_cng_match=0;
    // iterators for each of the feature vectors
    DBoW2::FeatureVector::const_iterator featVec1_it, featVec2_it;
    featVec1_it = lf_prev->lfFeatVec.begin();
    featVec2_it = lf_cur->lfFeatVec.begin();

    const auto featVec1_end = lf_prev->lfFeatVec.end(); // std::next(featVec1.begin(),200);    //
    const auto featVec2_end = lf_cur->lfFeatVec.end(); //std::next(featVec2.begin(),200);   //
    // until both the feature vectors iterators reach the end

    std::vector<cv::Mat> img_desc1, img_desc2;

    for (auto &d : lf_prev->intraMatches)
        img_desc1.push_back(d.matchDesc);

    for (auto &d : lf_cur->intraMatches)
        img_desc2.push_back(d.matchDesc);



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

            orBextractor->getMatches_distRatio(img_desc1, featVec1_it->second,
                                 img_desc2, featVec2_it->second, i_ind_tmp, j_ind_tmp,
                                 itr_cng_match);

            indices_1.insert(indices_1.end(), i_ind_tmp.begin(), i_ind_tmp.end());
            indices_2.insert(indices_2.end(), j_ind_tmp.begin(), j_ind_tmp.end());

            if(i_ind_tmp.size()>= 1){
                //cout<<"WordID :"<< featVec1_it->first <<endl;
                words.insert(featVec1_it->first);
            }
            // move featVec1_it and featVec2_it forward
            ++featVec1_it;
            ++featVec2_it;
        }
        else if(featVec1_it->first < featVec2_it->first)
        {
            // move old_it forward
            featVec1_it = lf_prev->lfFeatVec.lower_bound(featVec2_it->first);
            // old_it = (first element >= cur_it.id)
        }
        else
        {
            // move cur_it forward
            featVec2_it = lf_cur->lfFeatVec.lower_bound(featVec1_it->first);
            // cur_it = (first element >= old_it.id)
        }
    }
}

//void FrontEnd::poseFromPCAlignment(opengv::points_t points1, opengv::points_t points2, Mat& T


void FrontEnd::bundleAdjustIntraMatches(std::vector<IntraMatch>& tmp_intra_matches) {

    NonlinearFactorGraph graph;
    // Define the camera observation noise model
    auto measurementNoise =
            noiseModel::Isotropic::Sigma(2, 2.0);  // one pixel in u and v
    noiseModel::Robust::shared_ptr huberModel = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(
            sqrt(5.991)), measurementNoise);
    bool first= true;
    int intraInd=0;
    vector<int> octaves;
    gtsam::Values initialEstimate;
    int optim_type = 2; //0 - constant lms,1- string prior on lm,  2- non constant lms + rigid, 3- projectionPPP

    if(optim_type == 0){
        for (auto im: tmp_intra_matches) {
            gtsam::Point3 P_ = gtsam::Point3(im.point3D.at<double>(0,0), im.point3D.at<double>(1,0), im.point3D.at<double>(2,0));
            initialEstimate.insert(gtsam::Symbol('l', intraInd), P_);
            for (int i = 0; i < currentFrame->num_cams_; ++i) {
                if (im.matchIndex[i] != -1) {
                    Cal3_S2::shared_ptr K(new Cal3_S2(camconfig_.K_mats_[i].at<double>(0,0),
                                                      camconfig_.K_mats_[i].at<double>(1,1), 0.0,
                                                      camconfig_.K_mats_[i].at<double>(0,2),
                                                      camconfig_.K_mats_[i].at<double>(1,2)));
                    cv::KeyPoint kp = currentFrame->image_kps_undist[i][im.matchIndex[i]];
                    octaves.push_back(currentFrame->image_kps_undist[i][im.matchIndex[i]].octave);
                    gtsam::Point2 p_(kp.pt.x, kp.pt.y);
                    graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                            (p_, huberModel, gtsam::Symbol('x', i),gtsam::Symbol('l',intraInd), K ));

                }
            }
            intraInd++;
        }

        for(int i=0; i <currentFrame->num_cams_ ; i++){
            gtsam::Pose3 bTc = RT_Mats_init[i];
            if(i ==0){
                /// Add a prior on pose x0
                gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas
                        ((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.0001),gtsam::Vector3::Constant(0.0001)).finished());
                graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', i), bTc, poseNoise));
                VLOG(2)<<"Inserted prior for the state x"<<i<<endl;
            }
            initialEstimate.insert(Symbol('x', i), bTc);
        }
        intraInd=0;
        for (auto im: tmp_intra_matches) {
            gtsam::Point3 P_ = gtsam::Point3(im.point3D.at<double>(0,0), im.point3D.at<double>(1,0), im.point3D.at<double>(2,0));
            gtsam::noiseModel::Isotropic::shared_ptr pointNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.001);
            //cout<<lid<< landmark<<endl;
            graph.push_back(gtsam::PriorFactor<gtsam::Point3>(gtsam::Symbol('l', intraInd), P_, pointNoise)); // add directly to graph
            intraInd++;
        }

    }
    else if (optim_type == 1){
        for (auto im: tmp_intra_matches) {
            gtsam::Point3 P_ = gtsam::Point3(im.point3D.at<double>(0,0), im.point3D.at<double>(1,0), im.point3D.at<double>(2,0));
            initialEstimate.insert(gtsam::Symbol('l', intraInd), P_);
            for (int i = 0; i < currentFrame->num_cams_; ++i) {
                if (im.matchIndex[i] != -1) {
                    if(first){
                        first= false;
                        gtsam::noiseModel::Isotropic::shared_ptr pointNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.001);
                        graph.push_back(gtsam::PriorFactor<gtsam::Point3>(gtsam::Symbol('l', intraInd), P_, pointNoise)); // add directly to graph
                    }

                    Cal3_S2::shared_ptr K(new Cal3_S2(camconfig_.K_mats_[i].at<double>(0,0),
                                                      camconfig_.K_mats_[i].at<double>(1,1), 0.0,
                                                      camconfig_.K_mats_[i].at<double>(0,2),
                                                      camconfig_.K_mats_[i].at<double>(1,2)));
                    cv::KeyPoint kp = currentFrame->image_kps_undist[i][im.matchIndex[i]];
                    octaves.push_back(currentFrame->image_kps_undist[i][im.matchIndex[i]].octave);
                    gtsam::Point2 p_(kp.pt.x, kp.pt.y);
                    graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                            (p_, huberModel, gtsam::Symbol('x', i),gtsam::Symbol('l',intraInd), K ));

                }
            }
            intraInd++;
        }

        for(int i=0; i <currentFrame->num_cams_ ; i++){
            gtsam::Pose3 bTc = RT_Mats_init[i];
            if(i ==0){
                /// Add a prior on pose x0
                gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas
                        ((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.0001),gtsam::Vector3::Constant(0.0001)).finished());
                graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', i), bTc, poseNoise));
                VLOG(2)<<"Inserted prior for the state x"<<i<<endl;
            }
            initialEstimate.insert(Symbol('x', i), bTc);
        }
    }
    else if (optim_type == 2){
        /////////// add prior on first landmark and pose, add projection errors between all landmarks and poses, add rigid edges between the cameras
        for (auto im: tmp_intra_matches) {
            gtsam::Point3 P_ = gtsam::Point3(im.point3D.at<double>(0,0), im.point3D.at<double>(1,0), im.point3D.at<double>(2,0));
            initialEstimate.insert(gtsam::Symbol('l', intraInd), P_);
            for (int i = 0; i < currentFrame->num_cams_; ++i) {
                if (im.matchIndex[i] != -1) {
                    if(first){
                        first= false;
                        gtsam::noiseModel::Isotropic::shared_ptr pointNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.001);
                        graph.push_back(gtsam::PriorFactor<gtsam::Point3>(gtsam::Symbol('l', intraInd), P_, pointNoise)); // add directly to graph
                    }

                    Cal3_S2::shared_ptr K(new Cal3_S2(camconfig_.K_mats_[i].at<double>(0,0),
                                                      camconfig_.K_mats_[i].at<double>(1,1), 0.0,
                                                      camconfig_.K_mats_[i].at<double>(0,2),
                                                      camconfig_.K_mats_[i].at<double>(1,2)));
                    cv::KeyPoint kp = currentFrame->image_kps_undist[i][im.matchIndex[i]];
                    octaves.push_back(currentFrame->image_kps_undist[i][im.matchIndex[i]].octave);
                    gtsam::Point2 p_(kp.pt.x, kp.pt.y);
                    graph.push_back(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                                            (p_, huberModel, gtsam::Symbol('x', i),gtsam::Symbol('l',intraInd), K ));

                }
            }
            intraInd++;
        }

        gtsam::noiseModel::Diagonal::shared_ptr  betweenNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6)<<gtsam::Vector3::Constant(0.05), gtsam::Vector3::Constant(0.005)).finished());
        for(int i=0; i <currentFrame->num_cams_ ; i++){
            gtsam::Pose3 bTc = RT_Mats_init[i];
            if(i ==0){
                /// Add a prior on pose x0
                gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas
                        ((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.0001),gtsam::Vector3::Constant(0.0001)).finished());
                graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', i), bTc, poseNoise));
                VLOG(2)<<"Inserted prior for the state x"<<i<<endl;
            }
            if(i != 0){
                gtsam::Pose3 betweenPose = RT_Mats_init[i-1].inverse() * bTc;
                VLOG(2)<<betweenPose;
                graph.push_back(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('x', (i-1)), gtsam::Symbol('x', i),  betweenPose, betweenNoise));
            }
            initialEstimate.insert(Symbol('x', i), bTc);
        }

        //graph.print();
        //initialEstimate.print();


    }
    else{
        for (auto im: tmp_intra_matches) {
            gtsam::Point3 P_ = gtsam::Point3(im.point3D.at<double>(0,0), im.point3D.at<double>(1,0), im.point3D.at<double>(2,0));
            initialEstimate.insert(gtsam::Symbol('l', intraInd), P_);
            for (int i = 0; i < currentFrame->num_cams_; ++i) {
                if (im.matchIndex[i] != -1) {
                    if(first){
                        first= false;
                        gtsam::noiseModel::Isotropic::shared_ptr pointNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.001);
                        graph.push_back(gtsam::PriorFactor<gtsam::Point3>(gtsam::Symbol('l', intraInd), P_, pointNoise)); // add directly to graph
                    }

                    Cal3_S2::shared_ptr K(new Cal3_S2(camconfig_.K_mats_[i].at<double>(0,0),
                                                      camconfig_.K_mats_[i].at<double>(1,1), 0.0,
                                                      camconfig_.K_mats_[i].at<double>(0,2),
                                                      camconfig_.K_mats_[i].at<double>(1,2)));
                    cv::KeyPoint kp = currentFrame->image_kps_undist[i][im.matchIndex[i]];
                    octaves.push_back(currentFrame->image_kps_undist[i][im.matchIndex[i]].octave);
                    gtsam::Point2 p_(kp.pt.x, kp.pt.y);
                    graph.push_back(gtsam::ProjectionFactorPPP<Pose3, Point3, Cal3_S2>(p_, huberModel, Symbol('b',0), Symbol('x',i),Symbol('l',intraInd),K));
                    //graph.push_back(gtsam::ProjectionFactorPPPC<Pose3, Point3, Cal3_S2>(p_, huberModel, Symbol('b',0), Symbol('x',i),Symbol('l',intraInd),Symbol('k',i)));

                }
            }
            intraInd++;
        }
        for(int i=0; i <currentFrame->num_cams_ ; i++){
            gtsam::Pose3 bTc = RT_Mats_init[i];
            if(i ==0){
                /// Add a prior on pose x0
                gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas
                        ((gtsam::Vector(6)<< gtsam::Vector3::Constant(0.0001),gtsam::Vector3::Constant(0.0001)).finished());
                graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('b', 0), bTc, poseNoise));
                graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 0), bTc, poseNoise));
                VLOG(2)<<"Inserted prior for the state x"<<i<<endl;
                initialEstimate.insert(Symbol('b', 0), bTc);

            }
//            auto calNoise = noiseModel::Diagonal::Sigmas(
//                    (Vector(5) << 200, 200, 0.01, 30, 30).finished());
//            graph.addPrior(Symbol('k', i), Cal3_S2(camconfig_.K_mats_[i].at<double>(0,0),
//                                                   camconfig_.K_mats_[i].at<double>(1,1), 0.0,
//                                                   camconfig_.K_mats_[i].at<double>(0,2),
//                                                   camconfig_.K_mats_[i].at<double>(1,2)), calNoise);
//            initialEstimate.insert(Symbol('k', i), Cal3_S2(camconfig_.K_mats_[i].at<double>(0,0),
//                                                           camconfig_.K_mats_[i].at<double>(1,1), 0.0,
//                                                           camconfig_.K_mats_[i].at<double>(0,2),
//                                                           camconfig_.K_mats_[i].at<double>(1,2)));
            initialEstimate.insert(Symbol('x', i), bTc);
        }

    }


    LevenbergMarquardtParams lm_params;
    LevenbergMarquardtParams::SetCeresDefaults(&lm_params);
    lm_params.maxIterations = 50;
    Values result;
    set<int> removedIntra;
    for( int iter =0; iter<1 ; iter++){

        result = LevenbergMarquardtOptimizer(graph, initialEstimate, lm_params).optimize();
        cout << "final error = " << graph.error(result) << endl;
        //result.print("Final Result");
       /* for(int facInd =0 ; facInd < (graph.size() -5 - intraInd); facInd++) {
            if (graph.at(facInd)) {
                boost::shared_ptr<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>> fac = reinterpret_pointer_cast<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(
                        graph.at(facInd));
                // check if this is an oulier?
                //cout<<"key1 "<<symbolChr(fac->key1())<<symbolIndex(fac->key1())<<endl;
                //cout<<"key2 "<<symbolChr(fac->key2())<<symbolIndex(fac->key2())<<endl;

                gtsam::Pose3 poseVal = result.at<gtsam::Pose3>( fac->key1());
                gtsam::Point3 pointVal = result.at<gtsam::Point3>(fac->key2());
                gtsam::Point2 errVec = fac->evaluateError(poseVal, pointVal);

                gtsam::Matrix infMat =
                        gtsam::Matrix::Identity(2, 2) * orBextractor->GetInverseScaleSigmaSquares()[octaves[facInd]];
                double error = dot(errVec, infMat * errVec);

                if (error > 5.991) {
                    // this is an outlier. mark it
                    graph.remove(facInd);
                    removedIntra.insert(symbolIndex(fac->key2()));
                    VLOG(2)<<"removed the factor at "<<symbolChr(fac->key1())<<symbolIndex(fac->key1())<<" "<<symbolChr(fac->key2())<<symbolIndex(fac->key2())<<endl;

                }
            }
        }*/

    }
    for (gtsam::Values::iterator it = result.begin(); it != result.end(); ++it) {
        gtsam::Symbol sid = it->key;
        if (optim_type != 0){
            if (sid.chr() == 'l') {
                gtsam::Point3 b = result.at<gtsam::Point3>(sid);
                double aa = b.x();
                double bb = b.y();
                double cc = b.z();
                double a[3][1] = {aa, bb, cc};
                cv::Mat point3(3, 1, CV_64F, a);
                Mat oldpt = tmp_intra_matches[sid.index()].point3D;
                double diff_norm;
                diff_norm = cv::norm(oldpt- point3);
                VLOG(2)<<"intramatch: "<<sid.index()<<", ("<<oldpt.at<double>(0,0)<<","<<oldpt.at<double>(1,0)<<","<<oldpt.at<double>(2,0)
                       <<")--->("<<point3.at<double>(0,0)<<","<<point3.at<double>(1,0)<<","<<point3.at<double>(2,0)<<")"<<endl;
                tmp_intra_matches[sid.index()].point3D = point3;
            }
        }
        if (sid.chr() == 'x'){
            gtsam::Pose3 pose = result.at<gtsam::Pose3>( sid);
            gtsam::Matrix mat = pose.matrix();
            Mat Pose_wc;
            cv::eigen2cv(mat, Pose_wc);
            VLOG(2) <<"old pose : "<<RT_Mats_init[sid.index()];
            VLOG(2)<<"New pose : "<<Pose_wc;
            VLOG(2) <<"------------------------------------";
        }
        if (sid.chr() == 'k'){
            Cal3_S2 k = result.at<Cal3_S2>(sid);
            VLOG(2)<<k.K();
        }

    }

    for(int i=0; i <currentFrame->num_cams_ ; i++){
        //Update th calibration matrices
        RT_Mats[i] =  result.at<gtsam::Pose3>( Symbol('x', i));
        gtsam::Matrix mat = RT_Mats[i].matrix();
        Mat Pose_wc;
        cv::eigen2cv(mat, Pose_wc);
        Mat R2 = Pose_wc.rowRange(0,3).colRange(0,3).clone();
        Mat t2 = Pose_wc.rowRange(0,3).colRange(3,4).clone();
        camconfig_.R_mats_[i] = R2.t();
        camconfig_.t_mats_[i] = -1* R2.t()*t2;
        Pose3 kalibrPose = RT_Mats[i].inverse() *  RT_Mats[i-1];
        mat = kalibrPose.matrix();
        Mat kalibrCV;
        cv::eigen2cv(mat, kalibrCV);
        VLOG(2)<<camconfig_.Kalibr_R_mats_[i];
        VLOG(2)<<camconfig_.Kalibr_t_mats_[i];
        VLOG(2)<<kalibrCV;
        camconfig_.Kalibr_R_mats_[i] = kalibrCV.rowRange(0,3).colRange(0,3);
        camconfig_.Kalibr_t_mats_[i] =  kalibrCV.rowRange(0,3).colRange(3,4);
    }

}



void FrontEnd::OptimizePose(vector<DMatch> matches, vector<int> lids, vector<bool>& inliers, transformation_t nonlinear_transformation, Mat& poseOut ){
    inliers = vector<bool>(matches.size(),true);
    NonlinearFactorGraph graph;
    // Define the camera observation noise model
    auto measurementNoise =
            noiseModel::Isotropic::Sigma(2, 2.0);  // one pixel in u and v
    noiseModel::Robust::shared_ptr huberModel = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(
            sqrt(5.991)), measurementNoise);

    vector<int> fac_to_match;
    int Idx =0;
    int factorInd =0;
    //for(size_t i = 0; i < ransac.inliers_.size(); i++)
    for(int i =0; i < matches.size(); i++)
    {
        DMatch m = matches[i];
        Mat pt3d = map->getLandmark(lids[m.queryIdx])->pt3D;
        gtsam::Point3 P_ = gtsam::Point3(pt3d.at<double>(0,0), pt3d.at<double>(1,0), pt3d.at<double>(2,0));

        IntraMatch iMatch_cur = currentFrame->intraMatches[m.trainIdx];
        for (int camInd = 0; camInd <currentFrame->num_cams_ ; camInd++){
            if(iMatch_cur.matchIndex[camInd] != -1){
                gtsam::Pose3 bTc = RT_Mats[camInd];
                Cal3_S2::shared_ptr K(new Cal3_S2(camconfig_.K_mats_[camInd].at<double>(0,0),
                                                  camconfig_.K_mats_[camInd].at<double>(1,1), 0.0,
                                                  camconfig_.K_mats_[camInd].at<double>(0,2),
                                                  camconfig_.K_mats_[camInd].at<double>(1,2)));
                cv::KeyPoint kp = currentFrame->image_kps_undist[camInd][iMatch_cur.matchIndex[camInd]];
                gtsam::Point2 p_(kp.pt.x, kp.pt.y);

                //// This is to account for keypoint scale
                //auto measurementNoise1 =
                //        noiseModel::Isotropic::Sigma(2, orBextractor->GetInverseScaleSigmaSquares()[ kp.octave]);
                //noiseModel::Robust::shared_ptr huberModel1 = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(
                //        sqrt(5.991)), measurementNoise1);
                graph.emplace_shared<RigResectioningFactor>(huberModel, Symbol('x', 0), K, p_, P_, kp.octave, camInd, bTc);
                fac_to_match.push_back(Idx);
                factorInd++;
            }
        }
        Idx++;
    }
    LevenbergMarquardtParams lm_params;
    LevenbergMarquardtParams::SetCeresDefaults(&lm_params);
    lm_params.maxIterations = 25;
    Values result;

    for(int iter =0; iter < 2 ; iter++){
        Values initialEstimate;
        gtsam::Pose3 initPose(gtsam::Rot3(nonlinear_transformation.block<3,3>(0,0)),
                              gtsam::Point3(nonlinear_transformation.col(3)));
        initialEstimate.insert(Symbol('x', 0), initPose);

        result = LevenbergMarquardtOptimizer(graph, initialEstimate, lm_params).optimize();
        //cout<<"Graph Num Factors: "<<graph.nrFactors()<<endl;
        //result.print("Iteration "+ to_string(iter)+"results:\n");
        //cout << "initial error = " << graph.error(initialEstimate) << endl;
        cout << "final error = " << graph.error(result) << endl;


        gtsam::Pose3 curest = result.at<gtsam::Pose3>(Symbol('x', 0));
        for(int facInd =0 ; facInd < graph.size(); facInd++){
            if(graph.at(facInd)){
                boost::shared_ptr<RigResectioningFactor> fac = reinterpret_pointer_cast<RigResectioningFactor>(graph.at(facInd));
                // check if this is an oulier?
                gtsam::Point2 errVec = fac->evaluateError(curest);

                gtsam::Matrix infMat = gtsam::Matrix::Identity(2,2) * orBextractor->GetInverseScaleSigmaSquares()[ fac->octave()];
                double error = dot(errVec, infMat*errVec );


                if(error > 5.991){
                    // this is an outlier. mark it
                    graph.remove(facInd);

                    /*std::cout <<"error: "<<error<<endl;
                    std::cout<<"reproj error: "<<errVec<<endl;
                    std::cout<<"Sigma inv: "<<orBextractor->GetInverseScaleSigmaSquares()[ fac->octave()]<<endl;
                    std::cout<<"3D Point : "<<fac->landmark().x()<<","<<fac->landmark().y()<<","<<fac->landmark().z()<<endl;
                    std::cout<<"CamInd : "<<fac->cameraID()<<endl;
                    std::cout<<"2D obs : "<<fac->measured().x()<<","<<fac->measured().y()<<endl;*/
                    inliers[fac_to_match[facInd]] = false;
                    //show the points
                   // DMatch m = matches[fac_to_match[facInd]];
                   // IntraMatch iMatch_cur = currentFrame->intraMatches[m.trainIdx];
                    /*for (int camInd = 0; camInd <currentFrame->num_cams_ ; camInd++) {
                        if (iMatch_cur.matchIndex[camInd] != -1) {
                            cv::Point2f kp = lf_cur->image_kps_undist[camInd][iMatch_cur.matchIndex[camInd]].pt;
                            Mat img = lf_cur->imgs[camInd].clone();
                            cvtColor(img, img ,COLOR_GRAY2BGR);
                            cv::circle(img,kp,4,Scalar(0,255, 100),3);
                            cout<<"CAMID : "<<camInd<<endl;
                            cout<<"2D Obs: "<<kp.x<<", "<<kp.y<<endl;
                            //imshow("Outlier ", img );
                            //waitKey(0);
                        }
                    } */

                   // currentFrame->intraMatches[m.trainIdx].matchIndex[fac->cameraID()]= -1;
                   // iMatch_cur =  currentFrame->intraMatches[m.trainIdx];
                   // int raycnt =0;
                   // for (int camInd = 0; camInd <currentFrame->num_cams_ ; camInd++) {
                   //     if (iMatch_cur.matchIndex[camInd] != -1)
                   //         raycnt++;
                   // }
                   // if(raycnt <= 1){
                        // if the ray count of the current intra match is just 1
                        // mark it as an outlier.
                   //     inliers[fac_to_match[facInd]] = false;
                   // }

                    //std::cout<<"--------------------------------------------"<<endl;
                }
            }

        }
    }

    gtsam::Pose3 pose = result.at<gtsam::Pose3>(  gtsam::Symbol('x', 0));
    gtsam::Matrix matPose= pose.matrix();
    cv::eigen2cv(matPose, poseOut);
}

int FrontEnd::estimatePoseLF(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur,
                             std::vector<DMatch>& matches, vector<bool>& inliers, Mat& T, bool SAC){
    int num_inliers=0;
    switch (posestAlgo_){
        case PC_ALIGN:
            num_inliers=poseFromPCAlignment(lf_prev,lf_cur,matches, inliers, T, SAC );
            break;
        case SEVENTEEN_PT:
            num_inliers=poseFromSeventeenPt(lf_prev,lf_cur,matches, inliers, T, SAC);
            break;
        case G_P3P:
            num_inliers=absolutePoseFromGP3P(lf_prev,lf_cur,matches, inliers, T, SAC);
            break;
        default:
            break;
    }
    return num_inliers;


}
int FrontEnd::poseFromPCAlignment(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur,
                                   std::vector<DMatch>& matches, vector<bool>& inliers, Mat& T, bool SAC ){
    opengv::points_t points1, points2;
    inliers = vector<bool>(matches.size(),false);
    if(initialized_== INITIALIZED)
    {
        for (auto &m : matches) {
            int l_id = lf_prev->lIds[m.queryIdx];
            Mat pt3d= map->getLandmark(l_id)->pt3D;
            Eigen::Vector3d pt;
            for(int j=0; j<3; ++j)
                pt[j] = pt3d.at<double>(j,0);
            points1.push_back(pt);
            points2.push_back( lf_cur->points_3D[m.trainIdx]);
        }
    }
    else{
        for (auto &m : matches) {
            points1.push_back(lf_prev->points_3D[m.queryIdx]);
            points2.push_back( lf_cur->points_3D[m.trainIdx]);
        }
    }


    // normal
    opengv::point_cloud::PointCloudAdapter adapter(points1, points2);
    // run the non-linear optimization over all correspondences
    transformation_t nonlinear_transformation = point_cloud::optimize_nonlinear(adapter);
    cout<<"estimated pose : "<<endl;
    cout<<nonlinear_transformation<<endl;


    //sac style orientation

    //Create a PointCloudSacProblem and Ransac
    /*sac::Ransac<
            sac_problems::point_cloud::PointCloudSacProblem> ransac;
    std::shared_ptr<
            sac_problems::point_cloud::PointCloudSacProblem> relposeproblem_ptr(
            new sac_problems::point_cloud::PointCloudSacProblem(adapter));
    ransac.sac_model_ = relposeproblem_ptr;
    ransac.threshold_ = 0.2;
    ransac.max_iterations_ = 100;

    ransac.computeModel(0);
    transformation_t nonlinear_transformation = ransac.model_coefficients_;
    cout<<"estimated pose sac :"<<endl;
    std::cout << ransac.model_coefficients_ << std::endl << std::endl;
    std::cout << "the number of inliers in RANSAC is: " << ransac.inliers_.size()<<endl;*/


    //CHECK/DEBUG THE 3d-3d pose estimation
    int ii=0;
    int inlier_count = 0;
    float avg_error = 0;
    for (auto p : points2){
        opengv::point_t  p2_in_1 = nonlinear_transformation.block<3,3>(0,0) * p + nonlinear_transformation.col(3);
        opengv::point_t delta = points1.at(ii) - p2_in_1;
        avg_error = avg_error + delta.norm();
        if(delta.norm() < 0.1){
            inlier_count++;
            inliers[ii] = true;
           // cout<<"delta : "<< delta.norm()<<endl;
            //cout<<"P1 : "<<points1.at(ii) ;
            //cout<<"P2 in 1 : "<<p2_in_1<<endl;
        }
        else{
           // cout<<"delta : "<< delta.norm()<<endl;
            //cout<<"P1 : "<<points1.at(ii) ;
            //cout<<"P2 in 1 : "<<p2_in_1<<endl;
        }

        ii++;
    }
    cout<<"Average error : "<<avg_error/points1.size()<<endl;
    cout<<"Number of inlier intra matches after 3pt : "<<inlier_count<<","<<(float)inlier_count/points1.size()*100.0<<"%"<<endl;


    Mat curPos;
    cv::eigen2cv(nonlinear_transformation, curPos);
    double n = cv::norm(curPos.colRange(3,4));
    /*if(n > 1e-6){
        Mat norm_t = curPos.colRange(3,4) * 2 ;
        norm_t.copyTo(curPos.colRange(3,4));
    }*/

    T = Mat::eye(4, 4, CV_64F);
    curPos.copyTo(T(cv::Range(0, 3), cv::Range(0, 4)));
    return inlier_count;
}

int FrontEnd::poseFromSeventeenPt(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur,
                        std::vector<DMatch>& matches, vector<bool>& inliers, Mat& T,  bool SAC){


    bearingVectors_t bearingVectors1;
    bearingVectors_t bearingVectors2;
    std::vector<int> camCorrespondences1;
    std::vector<int> camCorrespondences2;

    ////////////////////////////////////////////////////

    inliers = vector<bool>(matches.size(),false);

    for (auto &m : matches) {
        IntraMatch iMatch = lf_prev->intraMatches[m.queryIdx];
        for (int i =0; i <currentFrame->num_cams_ ; i++){
            if(iMatch.matchIndex[i] != -1){
                Point2f uv_coord = lf_prev->image_kps_undist[i][iMatch.matchIndex[i]].pt;
                //camconfig_.K_mats_[i].inv();
                double norm_x = (uv_coord.x - camconfig_.K_mats_[i].at<double>(0,2))/camconfig_.K_mats_[i].at<double>(0,0);
                double norm_y = (uv_coord.y - camconfig_.K_mats_[i].at<double>(1,2)) / camconfig_.K_mats_[i].at<double>(1,1);
                Eigen::Vector3d pt;
                pt[0] = norm_x;
                pt[1] = norm_y;
                pt[2] = 1.0;
                bearingVectors1.push_back(pt/pt.norm());
                camCorrespondences1.push_back(i);
                break;
            }
        }

        IntraMatch iMatch_cur = lf_cur->intraMatches[m.trainIdx];
        for (int i =0; i <currentFrame->num_cams_ ; i++){
            if(iMatch_cur.matchIndex[i] != -1){
                Point2f uv_coord = lf_cur->image_kps_undist[i][iMatch_cur.matchIndex[i]].pt;
                //camconfig_.K_mats_[i].inv();
                double norm_x = (uv_coord.x - camconfig_.K_mats_[i].at<double>(0,2))/camconfig_.K_mats_[i].at<double>(0,0);
                double norm_y = (uv_coord.y - camconfig_.K_mats_[i].at<double>(1,2)) / camconfig_.K_mats_[i].at<double>(1,1);
                Eigen::Vector3d pt;
                pt[0] = norm_x;
                pt[1] = norm_y;
                pt[2] = 1.0;
                bearingVectors2.push_back(pt/pt.norm());
                camCorrespondences2.push_back(i);
                break;
            }
        }

    }

    ////////////////////////////////////////////////////

    //for (int i =0; i <currentFrame->num_cams_ ; i++){
   //    (*currentFrame->rotations_ptr)[i];
   // }
    //create non-central relative adapter
    relative_pose::NoncentralRelativeAdapter adapter(
            bearingVectors1,
            bearingVectors2,
            camCorrespondences1,
            camCorrespondences2,
            currentFrame->translations,
            *currentFrame->rotations_ptr);

    //Create a NoncentralRelativePoseSacProblem and Ransac
    sac::Ransac<
            sac_problems::relative_pose::NoncentralRelativePoseSacProblem> ransac;
    std::shared_ptr<sac_problems::relative_pose::NoncentralRelativePoseSacProblem>relposeproblem_ptr(
            new sac_problems::relative_pose::NoncentralRelativePoseSacProblem(
                    adapter,
                    sac_problems::relative_pose::NoncentralRelativePoseSacProblem::SEVENTEENPT));
    ransac.sac_model_ = relposeproblem_ptr;
    ransac.threshold_ = 2.0*(1.0 - cos(atan(sqrt(2.0)*0.5/camconfig_.K_mats_[0].at<double>(0,2))));
    ransac.max_iterations_ = 10000;

    //Run the experiment
    struct timeval tic;
    struct timeval toc;
    gettimeofday( &tic, 0 );
    ransac.computeModel();
    gettimeofday( &toc, 0 );
    double ransac_time = TIMETODOUBLE(timeval_minus(toc,tic));

    //print the results
    std::cout<<"/////////////////////SEVENTEEN PT RANSAC/////////////////////"<<endl;
    //std::cout << "the ransac threshold is: " << ransac.threshold_ << std::endl;
    //std::cout << "the ransac results is: " << std::endl;
    //std::cout << ransac.model_coefficients_ << std::endl << std::endl;
    //std::cout << "Ransac needed " << ransac.iterations_ << " iterations and ";
    //std::cout << ransac_time << " seconds" << std::endl << std::endl;
    std::cout << "the number of inliers is: " << ransac.inliers_.size();
    std::cout << std::endl << std::endl;
    //std::cout << "the found inliers are: " << std::endl;
    //for(size_t i = 0; i < ransac.inliers_.size(); i++)
    //    std::cout << ransac.inliers_[i] << " ";
    //std::cout << std::endl << std::endl;

    transformation_t nonlinear_transformation = ransac.model_coefficients_;
    Mat curPos;
    cv::eigen2cv(nonlinear_transformation, curPos);
    T = Mat::eye(4, 4, CV_64F);
    curPos.copyTo(T(cv::Range(0, 3), cv::Range(0, 4)));



    // get the last estimated pose WRT world into WTp
    Mat rot_p = allPoses.back().colRange(0,3);
    Mat t_p = allPoses.back().colRange(3,4);
    Mat WTp = Mat::eye(4, 4, CV_64F);
    rot_p.copyTo(WTp(cv::Range(0, 3), cv::Range(0, 3)));
    t_p.copyTo(WTp(cv::Range(0, 3), cv::Range(3, 4)));

    // Now convert the current pose WRT world
    unique_lock<mutex> lock(mMutexPose);
    Mat WTc =  WTp*T;
    T = WTc.clone();
    cout<<"Previous pose: "<<WTp<<endl;
    cout<<"POSE ESTIMATED FROM SEVENTEEN WRT world: "<<endl;
    cout<<T<<endl;
    cout<<"translation :"<< cv::norm(t_p - T(cv::Range(0, 3), cv::Range(3, 4)))<<endl;

    for(size_t i = 0; i < ransac.inliers_.size(); i++)
        inliers[ransac.inliers_[i]] = true;
    return ransac.inliers_.size();

}


int FrontEnd::absolutePoseFromGP3P(LightFieldFrame* lf_prev, LightFieldFrame* lf_cur,
                         std::vector<DMatch>& matches, vector<bool>& inliers, Mat& T,  bool SAC){


    bearingVectors_t bearingVectors;
    points_t points;
    std::vector<int> camCorrespondences;
    inliers = vector<bool>(matches.size(),false);
    int num_inliers =0;

    // For each match
    for (auto &m : matches) {
        Mat pt3d;
        if(initialized_==INITIALIZED){
            int l_id = lf_prev->lIds[m.queryIdx];
            pt3d= map->getLandmark(l_id)->pt3D;
        }
        else{
            pt3d = lf_prev->intraMatches[m.queryIdx].point3D;
        }
        Eigen::Vector3d pt;
        for(int j=0; j<3; ++j)
            pt[j] = pt3d.at<double>(j,0);
        points.push_back(pt);
        IntraMatch iMatch_cur = lf_cur->intraMatches[m.trainIdx];
        for (int i =0; i <currentFrame->num_cams_ ; i++){
            if(iMatch_cur.matchIndex[i] != -1){
                Point2f uv_coord = lf_cur->image_kps_undist[i][iMatch_cur.matchIndex[i]].pt;
                //camconfig_.K_mats_[i].inv();
                double norm_x = (uv_coord.x - camconfig_.K_mats_[i].at<double>(0,2))/camconfig_.K_mats_[i].at<double>(0,0);
                double norm_y = (uv_coord.y - camconfig_.K_mats_[i].at<double>(1,2)) / camconfig_.K_mats_[i].at<double>(1,1);
                Eigen::Vector3d pt_n;
                pt_n[0] = norm_x;
                pt_n[1] = norm_y;
                pt_n[2] = 1.0;
                bearingVectors.push_back(pt_n/pt_n.norm());
                camCorrespondences.push_back(i);
                break;
            }
        }
    }

    //create a non-central absolute adapter
    absolute_pose::NoncentralAbsoluteAdapter adapter(
            bearingVectors,
            camCorrespondences,
            points,
            currentFrame->translations,
            *currentFrame->rotations_ptr);
     //////////////////////////////////////
     ////////  RANSAC //////////////////////
    //Create a AbsolutePoseSacProblem and Ransac
    //The method is set to GP3P
    sac::Ransac<
            sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
    std::shared_ptr<sac_problems::absolute_pose::AbsolutePoseSacProblem> absposeproblem_ptr(
            new sac_problems::absolute_pose::AbsolutePoseSacProblem( adapter,
                    sac_problems::absolute_pose::AbsolutePoseSacProblem::GP3P));
    ransac.sac_model_ = absposeproblem_ptr;
    ransac.threshold_ = 1.0 - cos(atan(sqrt(2.0)*0.5/camconfig_.K_mats_[0].at<double>(0,2)));
    ransac.max_iterations_ = 50;


    //Run the experiment
    struct timeval tic;
    struct timeval toc;
    gettimeofday( &tic, 0 );
    ransac.computeModel();
    gettimeofday( &toc, 0 );
    double ransac_time = TIMETODOUBLE(timeval_minus(toc,tic));

    //print the results
    VLOG(3)<<"/////////////////////ABSOLUTE POSE RANSAC/////////////////////"<<endl;
    VLOG(3) << "the ransac results is: " << std::endl;
    VLOG(3) << ransac.model_coefficients_ << std::endl << std::endl;
    VLOG(3) << "Ransac needed " << ransac.iterations_ << " iterations and ";
    VLOG(3) << ransac_time << " seconds" << std::endl << std::endl;
    VLOG(3) << "the number of inliers is: " << ransac.inliers_.size();
    VLOG(3) << std::endl << std::endl;
    VLOG(3) << "the found inliers are: " << std::endl;
    for(size_t i = 0; i < ransac.inliers_.size(); i++)
        VLOG(3) << ransac.inliers_[i] << " ";
    VLOG(3) << std::endl << std::endl;
    for(size_t i = 0; i < ransac.inliers_.size(); i++)
        inliers[ransac.inliers_[i]] = true;

    transformation_t nonlinear_transformation = ransac.model_coefficients_;
    Mat curPos;
    cv::eigen2cv(nonlinear_transformation, curPos);
    T = Mat::eye(4, 4, CV_64F);
    curPos.copyTo(T(cv::Range(0, 3), cv::Range(0, 4)));

    /////////////////////////////////////////TESTING PNP RANSAC///////////////////////////
    /*vector<Point3d> points1_3d;
    vector<Point2d> kp_pts2_mono;
    vector<int> inliers_indices;
    for (auto m : matches){

        IntraMatch iMatch_cur = lf_cur->intraMatches[m.trainIdx];
        //for (int i =0; i <currentFrame->num_cams_ ; i++){
            int i =0;
            if(iMatch_cur.matchIndex[i] != -1){
                points1_3d.push_back(Point3d(lf_prev->intraMatches[m.queryIdx].point3D));
                Point2d uv_coord = Point2d(lf_cur->image_kps_undist[i][iMatch_cur.matchIndex[i]].pt);
                kp_pts2_mono.push_back(uv_coord);
            }
        //}

    }
    Mat rotation_t =  Mat::eye(3, 3, CV_64F);
    Rodrigues(rotation_t, prev_rvec);
    prev_tvec = Mat::zeros(3,1,CV_64F);

    // estimate the pose of the new camera WRT world map points
    cv::solvePnPRansac(points1_3d, kp_pts2_mono, camconfig_.K_mats_[0], Mat::zeros(4,1,CV_64F),
                       prev_rvec, prev_tvec,true,
                       250, 2.0, 0.97, inliers_indices,SOLVEPNP_ITERATIVE);

    Mat rvec_cw = prev_rvec.clone();
    Mat tvec_cw = prev_tvec.clone();

    //convert the rotation vector into rotation matrix
    Mat rot_cw =  Mat::eye(3, 3, CV_64F);
    cv::Rodrigues(rvec_cw, rot_cw);


    Mat rotation_pnp= rot_cw.t();
    Mat translation_pnp = (-rot_cw.t() * tvec_cw);

    int cnt_pnp_inliers=0;
    for(auto &i: inliers_indices )
        cnt_pnp_inliers++;

    VLOG(2)<<"Pnp Inliers : "<<cnt_pnp_inliers<<" out of "<<points1_3d.size()<<" matches";
    VLOG(2)<<"Pnp Rotation: "<< rotation_pnp;
    VLOG(2)<<"Pnp translation: "<<translation_pnp;*/

    ///////////////////////////////////////////////////////////////////////////////////////

    ///////////// Optimization //////////////////////

    if(initialized_ == INITIALIZED) {
        //cv::cv2eigen(lf_prev->pose.rowRange(0, 3), nonlinear_transformation);
        OptimizePose(matches, lf_prev->lIds, inliers, nonlinear_transformation, T);
    }
    /*else{
        //nonlinear_transformation.block<3,3>(0,0) = Eigen::MatrixXd::Identity(3,3);
        //nonlinear_transformation.col(3).setZero();
       // cout<<nonlinear_transformation<<endl;
        inliers = vector<bool>(matches.size(),true);
        NonlinearFactorGraph graph;
        // Define the camera observation noise model
        auto measurementNoise =
                noiseModel::Isotropic::Sigma(2, 2.0);  // one pixel in u and v
        noiseModel::Robust::shared_ptr huberModel = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(
                sqrt(5.991)), measurementNoise);

        vector<int> fac_to_match;
        int Idx =0;
        int factorInd =0;
        //for(size_t i = 0; i < ransac.inliers_.size(); i++)
        for(auto& m : matches)
        {
            //DMatch m = matches[ransac.inliers_[i]];
            Mat pt3d = lf_prev->intraMatches[m.queryIdx].point3D;
            //int l_id = lf_prev->lIds[m.queryIdx];
            //Mat pt3d= map->getLandmark(l_id)->pt3D;
            gtsam::Point3 P_ = gtsam::Point3(pt3d.at<double>(0,0), pt3d.at<double>(1,0), pt3d.at<double>(2,0));

            IntraMatch iMatch_cur = lf_cur->intraMatches[m.trainIdx];
            for (int camInd = 0; camInd <currentFrame->num_cams_ ; camInd++){
                if(iMatch_cur.matchIndex[camInd] != -1){
                    gtsam::Pose3 bTc = RT_Mats[camInd];
                    Cal3_S2::shared_ptr K(new Cal3_S2(camconfig_.K_mats_[camInd].at<double>(0,0),
                                                      camconfig_.K_mats_[camInd].at<double>(1,1), 0.0,
                                                      camconfig_.K_mats_[camInd].at<double>(0,2),
                                                      camconfig_.K_mats_[camInd].at<double>(1,2)));
                    cv::KeyPoint kp = lf_cur->image_kps_undist[camInd][iMatch_cur.matchIndex[camInd]];
                    gtsam::Point2 p_(kp.pt.x, kp.pt.y);

                    graph.emplace_shared<RigResectioningFactor>(huberModel, Symbol('x', 0), K, p_, P_, kp.octave, camInd, bTc);
                    fac_to_match.push_back(Idx);
                    factorInd++;
                }
            }
            Idx++;
        }
        LevenbergMarquardtParams lm_params;
        LevenbergMarquardtParams::SetCeresDefaults(&lm_params);
        lm_params.maxIterations = 25;
        Values result;

        for(int iter =0; iter < 4 ; iter++){
            Values initialEstimate;
            gtsam::Pose3 initPose(gtsam::Rot3(nonlinear_transformation.block<3,3>(0,0)),
                                  gtsam::Point3(nonlinear_transformation.col(3)));
            initialEstimate.insert(Symbol('x', 0), initPose);

            result = LevenbergMarquardtOptimizer(graph, initialEstimate, lm_params).optimize();
            //cout<<"Graph Num Factors: "<<graph.nrFactors()<<endl;
            //result.print("Iteration "+ to_string(iter)+"results:\n");
            //cout << "initial error = " << graph.error(initialEstimate) << endl;
            cout << "final error = " << graph.error(result) << endl;


            gtsam::Pose3 curest = result.at<gtsam::Pose3>(Symbol('x', 0));
            for(int facInd =0 ; facInd < graph.size(); facInd++){
                if(graph.at(facInd)){
                    boost::shared_ptr<RigResectioningFactor> fac = reinterpret_pointer_cast<RigResectioningFactor>(graph.at(facInd));
                    // check if this is an oulier?
                    gtsam::Point2 errVec = fac->evaluateError(curest);

                    gtsam::Matrix infMat = gtsam::Matrix::Identity(2,2) * orBextractor->GetInverseScaleSigmaSquares()[ fac->octave()];
                    double error = dot(errVec, infMat*errVec );


                    if(error > 5.991){
                        // this is an outlier. mark it
                        graph.remove(facInd);

                        //std::cout <<"error: "<<error<<endl;
                        //std::cout<<"reproj error: "<<errVec<<endl;
                        //std::cout<<"Sigma inv: "<<orBextractor->GetInverseScaleSigmaSquares()[ fac->octave()]<<endl;
                        //std::cout<<"3D Point : "<<fac->landmark().x()<<","<<fac->landmark().y()<<","<<fac->landmark().z()<<endl;
                        //std::cout<<"CamInd : "<<fac->cameraID()<<endl;
                        //std::cout<<"2D obs : "<<fac->measured().x()<<","<<fac->measured().y()<<endl;

                        //show the points

                        //for (int camInd = 0; camInd <currentFrame->num_cams_ ; camInd++) {
                        //    if (iMatch_cur.matchIndex[camInd] != -1) {
                        //        cv::Point2f kp = lf_cur->image_kps_undist[camInd][iMatch_cur.matchIndex[camInd]].pt;
                        //        Mat img = lf_cur->imgs[camInd].clone();
                        //        cvtColor(img, img ,COLOR_GRAY2BGR);
                        //        cv::circle(img,kp,4,Scalar(0,255, 100),3);
                        //        cout<<"CAMID : "<<camInd<<endl;
                        //        cout<<"2D Obs: "<<kp.x<<", "<<kp.y<<endl;
                        //        //imshow("Outlier ", img );
                        //        //waitKey(0);
                        //    }
                        //}
                        // if the ray count of the current intra match is just 1
                        // mark it as an outlier.
                        inliers[fac_to_match[facInd]] = false;


                        //std::cout<<"--------------------------------------------"<<endl;
                    }
                }

            }
        }

        gtsam::Pose3 pose = result.at<gtsam::Pose3>(  gtsam::Symbol('x', 0));
        gtsam::Matrix mat = pose.matrix();
        cv::eigen2cv(mat, T);
    }  */

    ////////////////////////////////////////////////

    for(int i=0; i <inliers.size(); i++){
        if(inliers[i])
            num_inliers++;
    }

    std::cout<<"Ransac Inliers: "<<ransac.inliers_.size()<<endl;
    std::cout<<"Optimization inliers :"<<num_inliers<<endl;

    return num_inliers;
}
///////////////////////////////////// Optimization stuff /////////////////////////////////////

//void FrontEnd::RigPoseOptimization(){

//}


void FrontEnd::getDataForGTSAMTriangulation(LightFieldFrame *lf, int iMatchInd, Cameras &cameras,
                                            gtsam::Point2Vector &measurements, vector<int> &compCamInds,
                                            vector<int> &octaves) {
    IntraMatch iMatch = lf->intraMatches[iMatchInd];
    for (int i1 = 0 ; i1 < lf->num_cams_ ; i1++){
        int ind = iMatch.matchIndex[i1];
        if (ind != -1){
            /// for the LF frame we need to build projection matrices in world frame
            Mat ref_R_cur = camconfig_.R_mats_[i1].t();
            Mat ref_t_cur = -ref_R_cur*camconfig_.t_mats_[i1];
            Mat ref_T_cur = Mat::eye(4, 4, CV_64F);
            ref_R_cur.copyTo(ref_T_cur(cv::Range(0, 3), cv::Range(0, 3)));
            ref_t_cur.copyTo(ref_T_cur(cv::Range(0, 3), cv::Range(3, 4)));
            Mat W_T_cur = lf->pose * ref_T_cur;
            gtsam::Cal3_S2 camK(camconfig_.K_mats_[i1].at<double>(0, 0),
                                camconfig_.K_mats_[i1].at<double>(1, 1), 0,
                                camconfig_.K_mats_[i1].at<double>(0, 2),
                                camconfig_.K_mats_[i1].at<double>(1, 2));
            gtsam::PinholeCamera<gtsam::Cal3_S2> camera1(convertPose3_CV2GTSAM(W_T_cur),camK);
            cameras += camera1;
            measurements += convertPoint2_CV2GTSAM(lf->image_kps_undist[i1][ind]);
            octaves.push_back(lf->image_kps_undist[i1][ind].octave);
            compCamInds.push_back(i1);
        }
    }
}

/////////////////////////////// optimization stufff /////////////////////////////////////////////////
void FrontEnd::TriangulateGTSAM(){

}

void
FrontEnd::searchLocalMap(LightFieldFrame *prevKF, vector<DMatch> inter_matches_with_landmarks, vector<bool> inliers,
                         vector<DMatch> &inlierAllMapMatches, Mat &refinedPose, vector<int> &alllids) {
    // get the set of KeyFrames in which the current inliers have been seen
   std::set<LightFieldFrame*> kfSet;
   std::set<Landmark*> matchedlmset;
   vector<bool> matchedFeatsCurFrame = vector<bool>(currentFrame->intraMatches.size(),false);
    vector<DMatch> allMapMatches;
    vector<int> lids;
    int old_inlier_lms = 0;
    for(int i=0; i <inter_matches_with_landmarks.size(); i++){
        if(inliers[i]) {
            DMatch m = inter_matches_with_landmarks[i];
            int l_id = prevKF->lIds[m.queryIdx];
            matchedlmset.insert(map->getLandmark(l_id));
            matchedFeatsCurFrame[m.trainIdx] = true; // mark the features which have already been matched.
            vector<LightFieldFrame*> observedKFs = map->getLandmark(l_id)->KFs;
            std::copy(observedKFs.begin(), observedKFs.end(), std::inserter(kfSet, kfSet.end()));

            alllids.push_back(l_id);
            DMatch oldm;
            oldm.queryIdx = (alllids.size()-1);
            oldm.trainIdx = m.trainIdx;
            allMapMatches.push_back(oldm);
        }
    }
    old_inlier_lms = alllids.size();
     VLOG(2)<<"Number of neghbor Keyframes"<<kfSet.size();
    //Get the landmarks that have been observed in the above keyframe set
    std::set<Landmark*> lmSet;
    for(auto kf : kfSet){
        for(int i =0; i <kf->lIds.size(); i++){
            if(kf->lIds[i] != -1){
                lmSet.insert(map->getLandmark(kf->lIds[i]));
            }
        }
    }
    VLOG(2)<<"Number of local map points found"<<lmSet.size();
    // Collect the descriptors of all the valid landmnarks
    std::vector<cv::Mat> vecDescs;

    for(auto& lm : lmSet){
        // If this landmark is already added to the lm->current feature matches move on
        if(matchedlmset.find(lm) != matchedlmset.end())
            continue;
        Mat Tcw = currentFrame->pose.inv();
        Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
        Mat tcw = Tcw.col(3).rowRange(0,3);
        Mat pt3d_body = Rcw * lm->pt3D + tcw;
        bool acceptlm = false;
        //Check if the landmark will be seen in the current frame.
        for(int camID = 0; camID < currentFrame->num_cams_ ; camID++){
            Mat pt3d_c = camconfig_.R_mats_[camID] * pt3d_body+ camconfig_.t_mats_[camID];
            double z = pt3d_c.at<double>(2,0);
            if(z < 0) // negative depth
                continue;
            // project the landmark into the camera and check the bounds
            Mat tmp = camconfig_.K_mats_[camID] * pt3d_c;
            tmp = tmp/ tmp.at<double>(2,0);
            if(tmp.at<double>(0,0) < 30 || tmp.at<double>(0,0) > (camconfig_.im_size_.width-30) )
                continue;
            if(tmp.at<double>(1,0) < 30 || tmp.at<double>(1,0) > (camconfig_.im_size_.height-30) )
                continue;

            // This is a valid landmark visible in this camera frame
            // accept lm will remain false if the landmark is not seen in any of the cameras
            // if it atleast in the frustum of one camera we can accept it
            acceptlm = true;
        }

        if(acceptlm){
            // get the descriptor of this landmark for further matching
            // for convenienjce we take the descriptor of the latest KF
            LightFieldFrame* kf = lm->KFs.back();
            vecDescs.push_back(kf->intraMatches[lm->featInds.back()].matchDesc);
            //pt3dVec.push_back(lm->pt3D);
            lids.push_back(lm->lId);
            alllids.push_back(lm->lId);

        }
    }
    VLOG(2)<<"Number of accepted Landmarks new : "<<lids.size();

    // transform all the decriptors of the accepted landmarks into BoW v ectors for matching
    DBoW2::BowVector lmBoW;
    DBoW2::FeatureVector lmFeatVec;

    orb_vocabulary->transform(vecDescs, lmBoW, lmFeatVec, 4);
    vector<Mat> img_desc2;
    for (auto d :currentFrame->intraMatches){
        img_desc2.push_back(d.matchDesc);
    }

    vector<unsigned int> ind1, ind2;
    //Match the landmark descriptors with the current frame descriptors
    InterMatchingBow( lmFeatVec,currentFrame->lfFeatVec, vecDescs,img_desc2 , ind1, ind2 );
    vector<DMatch> localMapMatches;
    for(int i=0; i<ind1.size(); i++){
        // if the matched feature in current frame is not already matched only then add it as a new match
        // We could additionally check if this is a good match than what has been matched before. but not doing that right now.
        if( !matchedFeatsCurFrame[ind2[i]]){
            DMatch m;
            m.queryIdx = old_inlier_lms + ind1[i];
            m.trainIdx = ind2[i];
            allMapMatches.push_back(m);
        }


    }
    vector<bool> localMapInliers, allMapInliers;
    transformation_t poseIn;
    cv::cv2eigen( currentFrame->pose.rowRange(0,3), poseIn);

    if(currentFrame->num_cams_ == -1){
        vector<Point3d> points1_3d;
        vector<Point2d> kp_pts2_mono;
        vector<int> inliers_indices;
        for(auto m: allMapMatches){
            Mat pt3d = map->getLandmark(alllids[m.queryIdx])->pt3D;
            points1_3d.push_back(Point3d(pt3d));
            Point2d p_cur = Point2d(currentFrame->intraMatches[m.trainIdx].uv_ref);
            kp_pts2_mono.push_back(p_cur);

        }

        // estimate the pose of the new camera WRT world map points
        cv::solvePnPRansac(points1_3d, kp_pts2_mono, camconfig_.K_mats_[0], Mat::zeros(4,1,CV_64F),
                           prev_rvec, prev_tvec,true,
                           250, 2.0, 0.97, inliers_indices,SOLVEPNP_ITERATIVE);

        Mat rvec_cw = prev_rvec.clone();
        Mat tvec_cw = prev_tvec.clone();

        //convert the rotation vector into rotation matrix
        Mat rot_cw =  Mat::eye(3, 3, CV_64F);
        cv::Rodrigues(rvec_cw, rot_cw);

        refinedPose = Mat::eye(4, 4, CV_64F);
        Mat rotation_pnp= rot_cw.t();
        Mat translation_pnp = (-rot_cw.t() * tvec_cw);
        rotation_pnp.copyTo(refinedPose(cv::Range(0, 3), cv::Range(0, 3)));
        translation_pnp.copyTo(refinedPose(cv::Range(0, 3), cv::Range(3, 4)));

        allMapInliers = vector<bool>(allMapMatches.size(),false);
        for(auto &i: inliers_indices )
            allMapInliers[i] = true;
    }
    else{
        OptimizePose(allMapMatches, alllids, allMapInliers, poseIn, refinedPose );
    }

    int num_inliers=0;
    for(int i=0; i <allMapInliers.size(); i++){
        if(allMapInliers[i]){
            num_inliers++;
            inlierAllMapMatches.push_back(allMapMatches[i]);
        }

    }
    VLOG(2)<<"Number of new matches in local Map : "<<(allMapMatches.size() - old_inlier_lms);
    VLOG(2)<<"Number of inliers in local Map : "<<(num_inliers - old_inlier_lms);
    VLOG(2)<<"Refined Pose from localMap: "<<refinedPose<<endl;
}

bool FrontEnd::trackLF(){
    bool viz_track = false;
    log_KF_ =0;
    Mat all;
    ///////////////////////////////////////////////////////////
    //////////////  This //////////////////////////////////////
    ///////////////////////////////////////////////////////////
    all.create(currentFrame->imgs[0].rows, currentFrame->imgs[0].cols * 3, CV_8UC3);
    for(int i=0; i < camconfig_.num_cams_ ; i++){
        Mat imgBGR;
        if(i<1){
            cvtColor(currentFrame->imgs[i], imgBGR, COLOR_GRAY2BGR);
            imgBGR.copyTo(all.colRange(camconfig_.im_size_.width * i, camconfig_.im_size_.width * (i + 1)));
        }
        if(i>1)
        {
            cvtColor(currentFrame->imgs[i], imgBGR, COLOR_GRAY2BGR);
            imgBGR.copyTo(all.colRange(camconfig_.im_size_.width * (i-1), camconfig_.im_size_.width * (i)));
        }

    }
    /// grab the current frame
    LightFieldFrame* lf_cur = currentFrame;
    //lf_cur->lIds_mono.assign( lf_cur->image_kps[0].size(), -1);

    /// Initialization ??
    if(initialized_ == NOT_INITIALIZED){
        initialization();
        //initialized_ = INITIALIZED;
        return true;
    }

    /// After initialization we need to track each incoming frame
    /// WRT the last keyframe.
    /// grab the last keyframe
    LightFieldFrame* prev_KF = lfFrames.back();
    //VLOG(3)<<"landmaks inds:"<<endl;
    //for (auto l : prev_KF->lIds){
    //    if(l!=-1)
    //        VLOG(3)<<l<<",";
   // }
   // VLOG(3)<<endl;
    VLOG(2)<<"------------------- NEW FRAME --------------------"<<endl;
    VLOG(2)<<"Number of landmarks: "<<prev_KF->numTrackedLMs<<endl;
   // VLOG(2)<<prev_KF->countLandmarks(true)<<endl;

    auto startT = high_resolution_clock::now();

    /// Find inter matches between current frame and the last keyframe
    std::vector<DMatch> inter_matches;
    opengv::points_t interMatch_3d_1, interMatch_3d_2;

    switch (interMatch_) {
        case BF_MATCH:
            findInterMatches(prev_KF, currentFrame, inter_matches, false);
            break;
        case BoW_MATCH:
            findInterMatchesBow(prev_KF, currentFrame, inter_matches, false);
            break;
    }

    if(inter_matches.size() < 100)
    {
        VLOG(2)<<"Not Enough Matches between frames";
        return false;
    }

    /// We will get two sets of matches
    /// 1) already tracked landmarks 2) new matches which are tracked in the current frame
    ///extract the matches associated with tracked landmarks
    std::vector<DMatch> inter_matches_with_landmarks, new_inter_matches;
    for(auto m : inter_matches){
        ///check if the feature corresponding to reference keyframe has a landmark
        int l_id = prev_KF->lIds[m.queryIdx];
        if (l_id == -1){
            new_inter_matches.push_back(m);
            continue;
        }

        inter_matches_with_landmarks.push_back(m);
    }

    Mat T = Mat::eye(4, 4, CV_64F);

    /// Get the landmarks corresponding to th inter matches (maintain vec<lm*>, vec<lmid>)
    /// align the current frame to these landmarks

    T = Mat::eye(4, 4, CV_64F);
    /// find pose of the light field frame using 3D-3d PC alignment
    vector<bool> inliers;
    //VLOG(2)<<"trackLF before estimate pose"<<endl;
    int num_inliers = 0;
    int num_inliers_refKF=0;
    if(currentFrame->num_cams_ == -1){
        vector<Point3d> points1_3d;
        vector<Point2d> kp_pts2_mono;
        vector<int> inliers_indices;

        for (auto m : inter_matches_with_landmarks){
            int l_id = prev_KF->lIds[m.queryIdx];
            points1_3d.push_back(Point3d(map->getLandmark(l_id)->pt3D));
            Point2d p_cur = Point2d(currentFrame->intraMatches[m.trainIdx].uv_ref);
            kp_pts2_mono.push_back(p_cur);

        }

        // estimate the pose of the new camera WRT world map points
        cv::solvePnPRansac(points1_3d, kp_pts2_mono, camconfig_.K_mats_[0], Mat::zeros(4,1,CV_64F),
                           prev_rvec, prev_tvec,true,
                           250, 2.0, 0.97, inliers_indices,SOLVEPNP_ITERATIVE);

        Mat rvec_cw = prev_rvec.clone();
        Mat tvec_cw = prev_tvec.clone();

        //convert the rotation vector into rotation matrix
        Mat rot_cw =  Mat::eye(3, 3, CV_64F);
        cv::Rodrigues(rvec_cw, rot_cw);


        Mat rotation_pnp= rot_cw.t();
        Mat translation_pnp = (-rot_cw.t() * tvec_cw);
        rotation_pnp.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
        translation_pnp.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));

        inliers = vector<bool>(inter_matches_with_landmarks.size(), false);
        for(auto &i: inliers_indices )
            inliers[i] = true;

        num_inliers = inliers_indices.size();

    }
    else{
        num_inliers = estimatePoseLF(prev_KF, lf_cur, inter_matches_with_landmarks, inliers, T, true );
    };

    auto stopT = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stopT - startT);
    VLOG(3)<<"time taken for feature matching and pose estimation : "<<duration.count()<<endl;

    VLOG(2) << "Total Number of intermatches "<<inter_matches.size()<<endl;
    VLOG(2) << "Number of inter_matches_with_landmarks : "<<inter_matches_with_landmarks.size()<<endl;
    VLOG(2) << "Number of new intermatches : "<<new_inter_matches.size()<<endl;
    VLOG(2) << "Number of tracked landmarks : " << num_inliers << ","<<(float)num_inliers/prev_KF->numTrackedLMs<<endl;
    log_num_intramatches_ = currentFrame->intraMatches.size();
    log_num_matches_ = inter_matches.size() ;
    log_num_lm_matches_ = inter_matches_with_landmarks.size();
    log_num_tracked_lms_ = num_inliers;
    log_num_new_matches_ = new_inter_matches.size();

    int pp=0;
    for(auto& m : inter_matches_with_landmarks){
        if(inliers[pp]){
            IntraMatch im2 = lf_cur->intraMatches[m.trainIdx];
            Point2f p2;
            int camIndCur=0;
            for(int featInd: im2.matchIndex){
                if(featInd != -1){
                    p2 = lf_cur->image_kps_undist[camIndCur][featInd].pt;
                    if(camIndCur == 0){
                        cv::circle(all,p2+Point2f(camIndCur * camconfig_.im_size_.width, 0),4,Scalar(0,255,0), 2);
                    }

                    if(camIndCur>1){
                        cv::circle(all,p2+Point2f((camIndCur-1) * camconfig_.im_size_.width, 0),4,Scalar(0,255,0), 2);
                    }
                }
                camIndCur++;
            }
        }
        pp++;
    }
    cv::resize(all,all,cv::Size(camconfig_.im_size_.width*3/2, camconfig_.im_size_.height/2));
    cv::imshow("images", all);
    if(currentFrame->frameId < 7)
        cv::waitKey(0);
    else
        cv::waitKey(4);


    /// we are directly localizing WRT the landmraks in global frame.
    /// no need of transform chaining

    currentFrame->pose = T.clone();

    VLOG(2) << "Initial Pose estimate "<< currentFrame->pose <<endl;

    Mat relativePose = prev_KF->pose.col(3) - T.col(3);
    double baseline = norm(relativePose);
    Mat relative_transformation = prev_KF->pose.inv() * T;


    double gamma = abs(atan2(relative_transformation.at<double>(2,1), relative_transformation.at<double>(2,2))) *  (180.0/3.141592653589793238463);

    double rotation = abs(atan2(-1 * relative_transformation.at<double>(2,0),
                                sqrt(pow(relative_transformation.at<double>(2,1), 2) +
                                     pow(relative_transformation.at<double>(2,2), 2)))) *  (180.0/3.141592653589793238463);
    VLOG(2)<<" translation: "<<baseline<<endl;
    VLOG(2)<<" rot angle: "<<rotation<<", gamma: "<<gamma<<endl;
    /////////////////// Check if this has a potential to be inserted as a keyframe ////////////////////////////
    /// 1. number of tracked points
    /// 2. number of tracked close points


    bool cond1 = (baseline >= kf_translation_threshold || rotation >= kf_rotation_threshold);
    bool cond2 = !cond1 and  num_inliers < 25;
    if(!cond1 and !cond2)
        return false;

    vector<DMatch> allMapMatches;
    vector<int> inlierLids;
    Mat refinedPose;
    searchLocalMap(prev_KF, inter_matches_with_landmarks, inliers, allMapMatches, refinedPose, inlierLids);
    currentFrame->pose = refinedPose.clone();
    ///////////////// Get the local map points from neighboring keyframes and refine the matches as well as pose estimates//////////////////////


//    std::vector<DMatch> inliers_lm_matches, outlier_lm_matches;
//    for(int i=0; i <inter_matches_with_landmarks.size(); i++){
//        if(inliers[i]){
//            inliers_lm_matches.push_back(inter_matches_with_landmarks[i]);
//        }
//        else{
//            outlier_lm_matches.push_back(inter_matches_with_landmarks[i]);
//        }
//
//    }

    if(viz_track){
        vector<cv::KeyPoint> kps1, kps2;
        int ind = 0;
        for (auto &d : prev_KF->intraMatches) {

            kps1.push_back(KeyPoint(d.uv_ref, 3.0));
            ind++;
        }
        ind = 0;
        for (auto &d : lf_cur->intraMatches) {

            kps2.push_back(KeyPoint(d.uv_ref, 3.0));
            ind++;
        }

        ///////////////////////////////////////////////////////////
        //////////////  This //////////////////////////////////////
        ///////////////////////////////////////////////////////////
        all.create(prev_KF->imgs[0].rows, prev_KF->imgs[0].cols * prev_KF->num_cams_, CV_8UC3);
        for(int i=0; i < camconfig_.num_cams_ ; i++){
            Mat imgBGR;
            cvtColor(lf_cur->imgs[i],imgBGR , COLOR_GRAY2BGR);
            imgBGR.copyTo(all.colRange(camconfig_.im_size_.width*i, camconfig_.im_size_.width*(i+1)));

        }
        int pp=0;
        for(auto& m : inter_matches_with_landmarks){
            IntraMatch im1 = prev_KF->intraMatches[m.queryIdx];
            IntraMatch im2 = lf_cur->intraMatches[m.trainIdx];
            Point2f p1, p2;
            int camIndCur=0;
            if(im2.mono){
                for(int featInd: im2.matchIndex){
                    if(featInd != -1)
                        break;
                    camIndCur++;
                }

            }
            p2 = im2.uv_ref;
            ///check if the im1's cam Ind matches current intra match cam
            if(im1.matchIndex[camIndCur] != -1){
                /// if the cam Indices of both the features match. We just draw the arrow
                p1 = prev_KF->image_kps_undist[camIndCur][im1.matchIndex[camIndCur]].pt;
            }
            else if(!im1.mono and camIndCur == 0){
                /// if the cam indices do not match, but it is because the prev feature is an intra match and the current feature's ref cam is 0
                /// draw the arrow in 0th camera
                p1 = im1.uv_ref;
            }
            else{
                /// all others- the cam indices dont match
                /// the previous feat is a landmark. get the projection of this landmark on the cur feature cam index in prev frame
                int l_id = prev_KF->lIds[m.queryIdx];
                Mat p3d = map->getLandmark(l_id)->pt3D;
                Mat p3d_in_body = prev_KF->pose.rowRange(0,3).colRange(0,3).t() *( p3d - prev_KF->pose.rowRange(0,3).colRange(3,4));
                Mat p3d_cam =  camconfig_.R_mats_[camIndCur] * p3d_in_body + camconfig_.t_mats_[camIndCur];
                Mat projected = camconfig_.K_mats_[camIndCur] * p3d_cam;
                double expected_x = projected.at<double>(0,0) / projected.at<double>(2,0);
                double expected_y = projected.at<double>(1,0) / projected.at<double>(2,0);
                p1 = Point2f(expected_x, expected_y);

            }
            if(inliers[pp]){
                //cv::circle(all,p2+Point2f(camIndCur * camconfig_.im_size_.width, 0),3,Scalar(0,0,130), 2);
                cv::arrowedLine(all, p1+ Point2f(camIndCur * camconfig_.im_size_.width, 0), p2+Point2f(camIndCur * camconfig_.im_size_.width, 0), Scalar(0,0,170), 2);
            }
            else{
                //cv::circle(all,p2+Point2f(camIndCur * camconfig_.im_size_.width, 0),3,Scalar(0,0,130), 2);
                cv::arrowedLine(all, p1+ Point2f(camIndCur * camconfig_.im_size_.width, 0), p2+Point2f(camIndCur * camconfig_.im_size_.width, 0), Scalar(50,0,100), 2);
            }
            pp++;

        }

        for(auto& m : new_inter_matches){
            IntraMatch im1 = prev_KF->intraMatches[m.queryIdx];
            IntraMatch im2 = lf_cur->intraMatches[m.trainIdx];
            Point2f p1, p2;
            int camIndCur=0;
            p1 = Point2f (0,0);
            if(im2.mono){
                for(int featInd: im2.matchIndex){
                    if(featInd != -1)
                        break;
                    camIndCur++;
                }

            }
            //// In this visualization P2 is always drawn in the current component camera
            //// if the p1 is also seen in the same component camera the arrow is drawn in that comp camera
            //// if p1 and p2 are both intra matches it is drawn in the reference camera
            //// if
            p2 = im2.uv_ref;
            ///check if the im1's cam Ind matches current intra match cam
            if(im1.matchIndex[camIndCur] != -1){
                /// if the cam Indices of both the features match. We just draw the arrow
                p1 = prev_KF->image_kps_undist[camIndCur][im1.matchIndex[camIndCur]].pt;
            }
            else if(!im1.mono and camIndCur == 0){
                /// if the cam indices do not match, but it is because the prev feature is an intra match and the current feature's ref cam is 0
                /// draw the arrow in 0th camera
                p1 = im1.uv_ref;
            }

            //cv::circle(all,p2+Point2f(camIndCur * camconfig_.im_size_.width, 0),3,Scalar(0,170,30), 2);
            if(p1.x !=0 and p1.y != 0)
                cv::arrowedLine(all, p1+ Point2f(camIndCur * camconfig_.im_size_.width, 0), p2+Point2f(camIndCur * camconfig_.im_size_.width, 0), Scalar(0,170,30), 2);
        }

        for (int i =0; i < allMapMatches.size(); i++) {
            DMatch m = allMapMatches[i];
            IntraMatch im2 = currentFrame->intraMatches[m.trainIdx];
            Point2f  p2 = im2.uv_ref;
            int camInd= 0;
            if(im2.mono){
                for(int featInd: im2.matchIndex){
                    if(featInd != -1)
                        break;
                    camInd++;
                }

            }
            cv::circle(all,p2+Point2f(camInd * camconfig_.im_size_.width, 0),3,Scalar(0,0,170), 2);
        }
    }

    ///we can get the inliers (featids in current frame) through reprojection error check as well. Right now
    /// we are only sticking with 3D-3D error within PCAlignment method

    // check the percentage of intramatches in the last frame tracked
    // if the tracked fall below 25% of intramatches
    // compare the inliers in the currentframe WRT the Map points in last keyframe
    int num_triangulated=0;
    int prevlms=prev_KF->numTrackedLMs;
    bool ret = false;

    relativePose = prev_KF->pose.col(3) - T.col(3);
    baseline = norm(relativePose);
    relative_transformation = prev_KF->pose.inv() * T;


    gamma = abs(atan2(relative_transformation.at<double>(2,1), relative_transformation.at<double>(2,2))) *  (180.0/3.141592653589793238463);

    rotation = abs(atan2(-1 * relative_transformation.at<double>(2,0),
                                sqrt(pow(relative_transformation.at<double>(2,1), 2) +
                                     pow(relative_transformation.at<double>(2,2), 2)))) *  (180.0/3.141592653589793238463);
    VLOG(2)<<" translation: "<<baseline<<endl;
    VLOG(2)<<" rot angle: "<<rotation<<", gamma: "<<gamma<<endl;

    if (num_inliers == 0)
        return ret;
    std::vector<char> inliers_triangulated, outliers_triangulated;
    num_inliers_refKF = num_inliers;
    /// Conditions for key frame insertion
    cond1 = (float)num_inliers/prev_KF->numTrackedLMs <= 0.4 && (baseline >= kf_translation_threshold || rotation >= kf_rotation_threshold);

    cond2 = !cond1 and  num_inliers < 35;
    num_inliers = allMapMatches.size();
    //LOGGING
    vector<int> n_rays (currentFrame->num_cams_, 0);
    if(cond1 or cond2 ) {

        /// if num of inliers falls below a threshold insert the new keyframe
        /// - assign landmarks to features, optimize R|t and
        /// else we have enough tracked points. Save the current pose and move on to next frame
        //Add additional local Map matches found
        for (int i =0; i < allMapMatches.size(); i++) {
            DMatch m = allMapMatches[i];
            int l_id = inlierLids[m.queryIdx];
            map->getLandmark(l_id)->addLfFrame(lf_cur, m.trainIdx, lf_cur->intraMatches[m.trainIdx].uv_ref);
            lf_cur->lIds[m.trainIdx] = l_id;
            ///logging
            n_rays[ lf_cur->intraMatches[m.trainIdx].n_rays - 1]++;
        }

        lf_cur->num_inliers_per_view = n_rays;
        n_rays = vector<int>(currentFrame->num_cams_, 0);
        /// Triangulate new landmarks and insert them as well.
        /// Actually we need to insert them only if they are seen in min_frames
        ///////////////////////////////////////////////////////////////////////////
        startT = high_resolution_clock::now();

        vector<bool> triangulation_inliers(new_inter_matches.size(), true);

        //Do a triangulation with non linear refinement of the inliers using both the LF frames
        for (int i = 0 ; i < new_inter_matches.size(); i++){

            ///Get the intramatches and their corresponding 2D observations
            DMatch m = new_inter_matches[i];
            vector<Mat> PJs;
            vector<int> view_inds;
            vector<Point2f> kps;
            int num_views =0;
            std::vector<Mat_<double> >  xs;


            ///// Triangulation using  opencv sfm and normalized coordinates///////////
            //triangulate and get the 3D point   ,
            cv::Mat pt3d;
            /// Non Linear optimization ///
            Cameras cameras;
            gtsam::Point2Vector measurements;
            vector<int> compCamInds, octaves;
            int firstLFEnd=0;
            int firstLFRaysCnt =0;
            int secLFRaysCnt =0;
            bool optimize_tri = true;

            getDataForGTSAMTriangulation(prev_KF, m.queryIdx, cameras, measurements, compCamInds, octaves);
            firstLFEnd = cameras.size();
            firstLFRaysCnt = cameras.size();

            getDataForGTSAMTriangulation(currentFrame, m.trainIdx, cameras, measurements, compCamInds, octaves);
            secLFRaysCnt = cameras.size() - firstLFRaysCnt;
            double distanceThresh = 40;
            double outlierReprojThresh = 0; // we do not apply reprojection error check here. but use it later.
            gtsam::TriangulationParameters triParams(0.001, false, distanceThresh,outlierReprojThresh);
            gtsam::TriangulationResult TriResult = triangulateSafe(cameras,measurements,triParams);

            if (TriResult.valid()){

                if(optimize_tri){
                    ////////////////////// OPTIMIZATION ////////////////////////
                    NonlinearFactorGraph graph;
                    static SharedNoiseModel unit(noiseModel::Unit::Create(2));
                    for (size_t i = 0; i < measurements.size(); i++) {
                        PinholeCamera<Cal3_S2> camera_i = cameras[i];
                        graph.emplace_shared<TriangulationFactor<PinholeCamera<Cal3_S2>> > //
                                (camera_i, measurements[i], unit, Symbol('P', 0));
                    }

                    LevenbergMarquardtParams params;
                    params.verbosityLM = LevenbergMarquardtParams::TRYLAMBDA;
                    params.verbosity = NonlinearOptimizerParams::ERROR;
                    params.lambdaInitial = 1;
                    params.lambdaFactor = 10;
                    params.maxIterations = 25;
                    params.absoluteErrorTol = 1.0;
                    params.verbosityLM = LevenbergMarquardtParams::SILENT;
                    params.verbosity = NonlinearOptimizerParams::SILENT;
                    params.linearSolverType = NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY;

                    Values result;

                    for(int iter =0; iter < 2 ; iter++) {
                        Values initialEstimate;
                        initialEstimate.insert(Symbol('P', 0), *TriResult);
                        try{
                            result = LevenbergMarquardtOptimizer(graph, initialEstimate, params).optimize();

                            //cout << "Triangulation final error- iter:"<<iter<<" = " << graph.error(result) << endl;
                            gtsam::Point3 curest = result.at<gtsam::Point3>(Symbol('P', 0));
                            for(int facInd =0 ; facInd < graph.size(); facInd++) {
                                if (graph.at(facInd)) {
                                    boost::shared_ptr<TriangulationFactor<PinholeCamera<Cal3_S2>>> fac = reinterpret_pointer_cast<TriangulationFactor<PinholeCamera<Cal3_S2>>>(
                                            graph.at(facInd));
                                    // check if this is an oulier?
                                    gtsam::Point2 errVec = fac->evaluateError(curest);
                                    gtsam::Matrix infMat = gtsam::Matrix::Identity(2,2) * orBextractor->GetInverseScaleSigmaSquares()[ octaves[facInd]];
                                    double error = dot(errVec, infMat*errVec );
                                    //double error = dot(errVec, errVec);
                                    if (error > 5.991) {
                                        // this is an outlier. mark it
                                        triangulation_inliers[i] = false;
                                        break;

                                    }
                                }
                            }


                        }
                        catch(gtsam::CheiralityException e){
                            VLOG(2)<<"Cheirality Exception at "<< e.nearbyVariable();
                            triangulation_inliers[i] = false;
                            break;
                        }

                    }
                    if(result.exists(Symbol('P', 0))){
                        gtsam::Point3 res = result.at<gtsam::Point3>(Symbol('P', 0));
                        pt3d = (cv::Mat_<double>(3,1) << res.x(), res.y(), res.z());
                    }

                    ////////////////////////////////////////////////////////////
                }
                else{

                    gtsam::Point3 actual = *TriResult;
                    triangulation_inliers[i] = true;
                    pt3d = (cv::Mat_<double>(3,1) << actual.x(), actual.y(), actual.z());
                }

            }
            else{
                triangulation_inliers[i] = false;
                if(viz_track){
                    IntraMatch im2 = currentFrame->intraMatches[m.trainIdx];
                    Point2f  p2 = im2.uv_ref;
                    int camInd= 0;
                    if(im2.mono){
                        for(int featInd: im2.matchIndex){
                            if(featInd != -1)
                                break;
                            camInd++;
                        }

                    }
                    cv::circle(all,p2+Point2f(camInd * camconfig_.im_size_.width, 0),3,Scalar(150,0,0), 2);
                }

            }

            if(triangulation_inliers[i]){
                if(viz_track){
                    IntraMatch im2 = currentFrame->intraMatches[m.trainIdx];
                    Point2f  p2 = im2.uv_ref;
                    int camInd= 0;
                    if(im2.mono){
                        for(int featInd: im2.matchIndex){
                            if(featInd != -1)
                                break;
                            camInd++;
                        }

                    }

                    cv::circle(all,p2+Point2f(camInd * camconfig_.im_size_.width, 0),3,Scalar(0,170,30), 2);
                    inliers_triangulated.push_back(255);
                    outliers_triangulated.push_back(0);
                }


                //Eigen::Vector3d pt;
                //for(int j=0; j<3; ++j)
                //    pt[j] = pt3d.at<double>(j,0);
                //create Landmarks for each inlier and insert them into global map and well as
                // the keyframes.
                int l_id = map->insertLandmark( pt3d, prev_KF, m.queryIdx, prev_KF->intraMatches[m.queryIdx].uv_ref);
                map->getLandmark(l_id)->addLfFrame(currentFrame, m.trainIdx, currentFrame->intraMatches[m.trainIdx].uv_ref);
                prev_KF->lIds[m.queryIdx] = l_id;
                currentFrame->lIds[m.trainIdx] = l_id;
                //triangulatedPts[i] = pt;
                num_triangulated++;
                n_rays[ currentFrame->intraMatches[m.trainIdx].n_rays - 1]++;

            }
            else{
                triangulation_inliers[i] = false;
                if(viz_track){
                    IntraMatch im2 = currentFrame->intraMatches[m.trainIdx];
                    Point2f  p2 = im2.uv_ref;
                    int camInd= 0;
                    if(im2.mono){
                        for(int featInd: im2.matchIndex){
                            if(featInd != -1)
                                break;
                            camInd++;
                        }

                    }
                    cv::circle(all,p2+Point2f(camInd * camconfig_.im_size_.width, 0),6,Scalar(150,0,0), 2);
                    outliers_triangulated.push_back(255);
                    inliers_triangulated.push_back(0);
                }

            }

        }
        if(viz_track){
            imshow("Tracking inliers", all);
            waitKey(30);
        }

        log_num_triangulated_ = num_triangulated;
        log_KF_=1;
        stopT = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stopT - startT);
        VLOG(1)<<"time taken for Triangulating new features: "<<duration.count()<<endl;

        VLOG(1)<<"Number of newly triangulated points : "<<num_triangulated<<endl;

        //if( num_triangulated < 20){
        //    VLOG(2)<<"Not enough landmarks have been created. BAD Frame";
        //    return false;
        //}

        prev_KF->numTrackedLMs = prev_KF->numTrackedLMs + num_triangulated;
        currentFrame->numTrackedLMs = num_inliers + num_triangulated;
        currentFramePose =  currentFrame->pose.rowRange(0,3).clone();        //T.rowRange(0, 3).clone();

        currentFrame->num_matches_refKF = inter_matches.size();
        currentFrame->num_matches_refKF_lms = inter_matches_with_landmarks.size();
        currentFrame->num_matches_refKF_new = new_inter_matches.size();
        currentFrame->num_matches_localMap = allMapMatches.size();
        currentFrame->num_inliers_refKF_lms = num_inliers_refKF;
        currentFrame->num_inliers_localMap = num_inliers;
        currentFrame->num_triangulated = num_triangulated;
        currentFrame->num_triangulated_per_view = n_rays ;
        n_rays.clear();


        insertKeyFrame();
        ret = true;
        //////////////////////////////////////////////////////////////////////////

    }

    std::stringstream ss;
    ss <<std::setprecision(6)<<std::fixed<<currentFrame->timeStamp<<","<<std::setw(4)<< lfFrames.size()<<","<<std::setw(12)  <<prevlms<<","<<std::setw(11)  <<inter_matches_with_landmarks.size()<<","
                        <<std::setw(12)  <<new_inter_matches.size()<<","<<std::setw(11)  <<num_inliers<<","<<std::setw(11)  <<num_triangulated;
    std::string s = ss.str();
    VLOG(3)<<s<<endl;
    if(!ret)
        deleteCurrentFrame();
    return ret;
}

void FrontEnd::estimatePose_Mono(){

    // grab the current frame
    LightFieldFrame* lf_cur = currentFrame;
    lf_cur->lIds_mono.assign( lf_cur->image_kps_undist[0].size(), -1);

    // Initialization ??
    if(!initialized_mono_){
        //initialization_mono();
        return;
    }

    // grab the last keyframe
    //LightFieldFrame* prev_KF = lfFrames[lfFrames.size()-2];
    LightFieldFrame* prev_KF = lfFrames_mono.back();
    cout<<"Number of landmarks in KF: "<< prev_KF->countLandmarks(true)<<endl;
    Mat rot_KF_wc = allPoses_mono.back().colRange(0,3);
    Mat trans_KF_wc = allPoses_mono.back().colRange(3,4);

    std::vector<DMatch> matches_mono,matches_mono_existing,  matches_mono_new;
    vector<Point2d> kp_pts1_mono, kp_pts2_mono;
    vector<Point2f> kps1_mono_new, kps2_mono_new, kp_pts1_mono_f, kp_pts2_mono_f;
    vector<Point3d> points1_3d;
    vector<Landmark*> existing_lms;

    findMatchesMono(prev_KF, lf_cur, 0, matches_mono);

    for(auto &m : matches_mono){
        //get the landmark corresponding to this match
        int lid = prev_KF->lIds_mono[m.queryIdx];
        Landmark* l = map_mono->getLandmark(lid);
        // if there exists a landmark
        if(l){
           // double x = (lf_cur->image_kps_undist[0][m.trainIdx].pt.x - camconfig_.K_mats_[0].at<double>(0,2))/camconfig_.K_mats_[0].at<double>(0,0);
           // double y  = (lf_cur->image_kps_undist[0][m.trainIdx].pt.y - camconfig_.K_mats_[0].at<double>(1,2)) / camconfig_.K_mats_[0].at<double>(1,1);

            Point2d p_cur = Point2d(lf_cur->image_kps_undist[0][m.trainIdx].pt);
            kp_pts2_mono.push_back(p_cur);
            kp_pts2_mono_f.push_back(lf_cur->image_kps_undist[0][m.trainIdx].pt);

            Point2d p_cur_1 = Point2d(prev_KF->image_kps_undist[0][m.queryIdx].pt);
            kp_pts1_mono.push_back(p_cur_1);

            kp_pts1_mono_f.push_back(prev_KF->image_kps_undist[0][m.queryIdx].pt);

            Point3d pt3d = Point3d(l->pt3D);

            points1_3d.push_back(pt3d);
            existing_lms.push_back(l);
            matches_mono_existing.push_back(m);
        }
        else{//landmark is not there
             //save these points for triangulation.
            //float x = (prev_KF->image_kps_undist[0][m.queryIdx].pt.x - camconfig_.K_mats_[0].at<double>(0,2))/camconfig_.K_mats_[0].at<double>(0,0);
            //float y  = (prev_KF->image_kps_undist[0][m.queryIdx].pt.y - camconfig_.K_mats_[0].at<double>(1,2)) / camconfig_.K_mats_[0].at<double>(1,1);

            Point2f p_prev = prev_KF->image_kps_undist[0][m.queryIdx].pt; //prev_KF->image_kps_undist[0][m.queryIdx].pt;
            kps1_mono_new.push_back(p_prev);


            Point2f p_cur = lf_cur->image_kps_undist[0][m.trainIdx].pt;
            kps2_mono_new.push_back(p_cur);
            matches_mono_new.push_back(m);

        }
    }

    //pnp ransac as opposed to essential matrix estimation
    ////// PNP RANSAC /////////////////////
    Mat rvec_cw, tvec_cw, rvec_cw_it, tvec_cw_it;
    vector<int> inliers;
    vector<bool> inliers_bool =  vector<bool>(points1_3d.size(),false);
   // cv::Rodrigues( rot_KF_wc.t(), rvec_cw);
    //tvec_cw_it =  -rot_KF_wc.t()*trans_KF_wc;
    //cv::solvePnP(points1_3d, kp_pts2_mono, camconfig_.K_mats_[0], Mat::zeros(4,1,CV_64F), rvec_cw_it, tvec_cw_it);

    // estimate the pose of the new camera WRT world map points
    cv::solvePnPRansac(points1_3d, kp_pts2_mono, camconfig_.K_mats_[0], Mat::zeros(4,1,CV_64F),
               rvec_cw, tvec_cw,false,
                       200, 1.0, 0.97, inliers,SOLVEPNP_EPNP);

    //convert the rotation vector into rotation matrix
    Mat rot_cw =  Mat::eye(3, 3, CV_64F);
    cv::Rodrigues(rvec_cw, rot_cw);

    // create a inlier boolean mask for the inliers indices in the matches
    for(auto &i: inliers )
        inliers_bool.at(i) = true;

    cout<<"//////////// PNP RANSAC MAT COMPUTATION ///////////"<<endl;
    cout<<"number of matches : "<< kp_pts1_mono.size()<<endl;
    cout<<"number of inliers : "<< inliers.size()<<endl;
    cout<<"///ransac//"<<endl;
    cout<<"Rotation: "<<endl;
    cout<<rot_cw.t()<<endl;
    cout<<"translation:"<<endl;
    cout<<(-rot_cw.t() * tvec_cw)<<endl;


    //-- Draw matches
    Mat img_matches_mono;

    img_matches_mono = lf_cur->imgs[0].clone();
    Mat mask_mono= Mat::ones(matches_mono.size(), 1,CV_8U);
    drawMatchesArrows(img_matches_mono, prev_KF->image_kps_undist[0], lf_cur->image_kps_undist[0], matches_mono, mask_mono , cv::Scalar(150,200, 0));
    //-- Show detected matches
    imshow("Matches_mono", img_matches_mono );
    waitKey(100);


    /*vector<cv::Point3f> P3D;
    vector<bool> Triangulated;
    float parallax;
    Mat K,rot_KF_cw, trans_KF_cw;

    rot_KF_cw = rot_KF_wc.t();
    trans_KF_cw = -rot_KF_wc.t() * trans_KF_wc;

    camconfig_.K_mats_[0].convertTo(K, CV_32F);
    rot_KF_cw.convertTo(rot_KF_cw, CV_32F);
    trans_KF_cw.convertTo(trans_KF_cw,CV_32F);

    rot_cw.convertTo(rot_cw, CV_32F);
    tvec_cw.convertTo(tvec_cw, CV_32F);

    int num_good_existing = CheckRT(rot_cw,tvec_cw ,kp_pts1_mono_f, kp_pts2_mono_f, inliers_bool,
                                    K, P3D, 4.0*1.0*1.0, Triangulated, parallax,rot_KF_cw, trans_KF_cw ); */




    //make the inliers bool mask
    /*for(auto &i: inliers )
        inliers_bool.at(i) = true;
    //////////////////////////////////////////////////////////
    /////////////////// ESSENTIAL //////////////////////////////
    cv::Mat E_mat;
    Mat mask_mono;
    cv::Mat rot_E = Mat::zeros(3, 3, CV_64F), trans_E = Mat::zeros(3, 1, CV_64F);
    //find the essential matrix
    E_mat = findEssentialMat(kp_pts1_mono, kp_pts2_mono , Mat::eye(3,3,CV_64F),cv::RANSAC,0.97, 1.0, mask_mono);
    recoverPose(E_mat, kp_pts1_mono, kp_pts2_mono, Mat::eye(3,3,CV_64F), rot_E, trans_E, mask_mono);
    int num_inliers = countNonZero(mask_mono);

    //-- Draw matches
    Mat img_matches_mono;
    //drawMatches( img1, prev_KF->image_kps_undist[0], img2, lf_cur->image_kps_undist[0], good_matches, img_matches_mono,Scalar::all(-1),
    //            Scalar::all(-1), mask_mono, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    img_matches_mono = lf_cur->imgs[0].clone();
    mask_mono= Mat::ones(matches_mono.size(), 1,CV_8U);
    drawMatchesArrows(img_matches_mono, prev_KF->image_kps_undist[0], lf_cur->image_kps_undist[0], matches_mono, mask_mono , cv::Scalar(150,200, 0));
    //-- Show detected matches
    imshow("Matches_mono", img_matches_mono );
    waitKey(100);

    Mat rotation_E_t = rot_E.t();
    Mat translation_E_t = -rot_E.t() *trans_E;
    cout<<"//////////// ESSENTIAL MAT COMPUTATION ///////////"<<endl;
    cout<<"number of matches_mono : "<< kp_pts1_mono.size()<<endl;
    cout<<"number of inliers : "<< num_inliers<<endl;
    cout<<"Rotation: "<<endl;
    cout<<rotation_E_t<<endl;
    cout<<"translation:"<<endl;
    cout<<translation_E_t<<endl;
    ////////////////////////////////////
    */
    // if we have atleast 80% of inliers i.e atleast 80% map points are tracked
    if(inliers.size()  >= 0.3*points1_3d.size()){ //


        //Triangulate new matches and insert good triangulations into the map
        float parallax;
        vector<cv::Point3f> P3D;
        vector<bool> Triangulated;
        int num_tri_points=0;
        vector<bool> inliers_new = vector<bool>(kps1_mono_new.size(),true); // we wanna triangulate all the new points
        Mat K, rot_KF_cw, trans_KF_cw;

        rot_KF_cw = rot_KF_wc.t();
        trans_KF_cw = -rot_KF_wc.t() *trans_KF_wc;

        camconfig_.K_mats_[0].convertTo(K, CV_32F);
        rot_KF_cw.convertTo(rot_KF_cw, CV_32F);
        trans_KF_cw.convertTo(trans_KF_cw,CV_32F);

        rot_cw.convertTo(rot_cw, CV_32F);
        tvec_cw.convertTo(tvec_cw, CV_32F);


        int num_good = CheckRT(rot_cw,tvec_cw ,kps1_mono_new, kps2_mono_new, inliers_new,
                               Mat::eye(3,3,CV_32F), P3D, 4.0*1.0*1.0, Triangulated, parallax,rot_KF_cw, trans_KF_cw );

        cout<<"Number of triangulated points : "<< num_good<<endl;
        if( (num_good + inliers.size() ) <= 0.25 * matches_mono.size()  ){
            //insert a keyframe
            Mat pose_cur = Mat::eye(3, 4, CV_64F);
            Mat rot_wc = rot_cw.t();
            Mat trans_wc = -rot_cw.t() *tvec_cw;
            rot_wc.copyTo(pose_cur(cv::Range(0, 3), cv::Range(0, 3)));
            trans_wc.copyTo(pose_cur(cv::Range(0, 3), cv::Range(3, 4)));
            currentFramePose_mono = pose_cur.clone();
            cout<<"///////////////DEBUG/////////////"<<endl;
            cout<<"pose 1 :"<<endl;
            cout<<"rotation: "<<endl;
            cout<<rot_KF_wc<<endl;
            cout<<"translation: "<<endl;
            cout<<trans_KF_wc<<endl;
            cout<<"-----------------------"<<endl;
            cout << " Current Frame pose MONO WRT to last KEYFRAME : " << currentFramePose_mono << endl;


            //add observations to the old matches
            int l_ind=0;
            for(auto&l : existing_lms){
                if(inliers_bool[l_ind]){
                    //l->addLfFrame(currentFrame, kp_pts2_mono[l_ind]);
                    currentFrame->lIds_mono[matches_mono_existing[l_ind].trainIdx] = l->lId;

                }
                l_ind++;
            }
            insertKeyFrame_Mono();
            //insert the landmarks
            //insert the new triangulated landmarks
            int ii=0;
            for(auto& m : matches_mono_new){
                if(Triangulated[ii]){
                    num_tri_points++;
                    Mat p_cur = Mat(P3D[ii]);
                    Mat p_world;
                    p_cur.convertTo(p_world, CV_64F);
                    cout<<"new Landmark : "<<p_world<<","<<p_cur<<endl;
                    int l_id;
                    //int l_id = map_mono->insertLandmark(p_world, prev_KF, kps1_mono_new.at(ii));
                    prev_KF->lIds_mono[m.queryIdx] = l_id;

                    //landmark already exists
                    // add the observation
                    // add the observation for the same landmark
                    Landmark* l = map_mono->getLandmark(l_id);
                    if (l) {
                       // l->addLfFrame(currentFrame, currentFrame->image_kps_undist[0][m.trainIdx].pt);
                        currentFrame->lIds_mono[m.trainIdx] = l_id;
                    }
                }

                ii++;
            }

        }

    }

    /*cv::Mat E_mat;
    cv::Mat rotation = Mat::zeros(3, 3, CV_64F), translation = Mat::zeros(3, 1, CV_64F);
    //find the essential matrix
    E_mat = findEssentialMat(kp_pts1_mono, kp_pts2_mono , camconfig_.K_mats_[0],cv::RANSAC,0.97, 1.0, mask_mono);
    recoverPose(E_mat, kp_pts1_mono, kp_pts2_mono, camconfig_.K_mats_[0], rotation, translation, mask_mono);
    int num_inliers = countNonZero(mask_mono);

    //-- Draw matches
    /*Mat img_matches_mono;
    //drawMatches( img1, prev_KF->image_kps_undist[0], img2, lf_cur->image_kps_undist[0], good_matches, img_matches_mono,Scalar::all(-1),
    //            Scalar::all(-1), mask_mono, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    img_matches_mono = img2.clone();
    drawMatchesArrows(img_matches_mono, prev_KF->image_kps_undist[0], lf_cur->image_kps_undist[0], good_matches, mask_mono , cv::Scalar(150,200, 0));
    //-- Show detected matches
    imshow("Matches_mono", img_matches_mono );
    waitKey(100);*/

    /*Mat rotation_t = rotation.t();
    Mat translation_t = -rotation.t() *translation;
    cout<<"number of matches_mono : "<< matches_mono.size()<<endl;
    cout<<"number of inliers : "<< num_inliers<<endl;
    cout<<"Rotation: "<<endl;
    cout<<rotation_t<<endl;
    cout<<"translation:"<<endl;
    cout<<translation_t<<endl;

    Mat T_mono = Mat::eye(4, 4, CV_64F);
    //if(num_inliers > 15){
    rotation_t.copyTo(T_mono(Range(0, 3), Range(0, 3)));
    translation_t.copyTo(T_mono(Range(0, 3), Range(3, 4)));
    //}

    // get the last estimated pose WRT world into WTp
    Mat rot_p = allPoses_mono.back().colRange(0,3);
    Mat t_p = allPoses_mono.back().colRange(3,4);
    Mat WTp = Mat::eye(4, 4, CV_64F);
    rot_p.copyTo(WTp(Range(0, 3), Range(0, 3)));
    t_p.copyTo(WTp(Range(0, 3), Range(3, 4)));

    // Now convert the current pose WRT world
    unique_lock<mutex> lock(mMutexPose);
    Mat WTc =  WTp*T_mono;
    currentFramePose_mono = WTc.rowRange(0, 3).clone();
    cout << " Current Frame pose MONO WRT to last KEYFRAME : " << currentFramePose_mono << endl; */


    // check the percentage of intramatches in the last frame tracked
    // if the tracked fall below 25% of intramatches
    // compare the inliers in the currentframe WRT the Map points in last keyframe
    /*int matches_num = matches_mono.size();
    int lastKFMapPoints = prev_KF->intraMatches.size();

    if((float)num_inliers/lastKFMapPoints < 0.25){
        insertKeyFrame_Mono();

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

        cout<<"total number of landmarks in mono : "<<map_mono->num_lms<<endl;

    }*/

}

//////////////////////////////////////////////////////////////////////
/////////////// VISUALIZATION AND ANALYSIS FUNCTIONS ////////////////
////////////////////////////////////////////////////////////////////
//synchronized callback
void FrontEnd::findLoopCandidates(vector<Mat>& images, vector<Mat>& segMasks){
    LightFieldFrame* fr = new LightFieldFrame(camconfig_) ;//new LightFieldFrame(images, segMasks, orb_vocabulary,
                         // orBextractor, orBextractors, camconfig_, current_frameId++,
                          //0.0, false);
    fr->extractFeaturesParallel();
    fr->parseandadd_BoW();

    /*std::vector<array<int, 5>> matches_map;
    vector<DBoW2::NodeId > words_;
    fr->computeIntraMatches(matches_map, words_);

    std::vector<array<int, 5> > filtered_intra_matches;
    assert(words_.size() == matches_map.size());
    set<DBoW2::NodeId > filtered_words;
    filterIntraMatches( matches_map, fr, filtered_intra_matches, words_,filtered_words );
    cout<<"Total number of filetered intra matches : "<<filtered_intra_matches.size()<<endl;

    orb_vocabulary->transform(fr->intraMatchDescs,fr->lfBoW,fr->lfFeatVec,2);*/
    vector<Mat> all_descriptors;
    for(int i =0 ; i < fr->num_cams_; i++){
        for(int j=0; j <fr->image_descriptors[i].size(); j++){
            Point2f p = fr->image_kps[i][j].pt;
            if(fr->segMasks[i].at<float>(p.y, p.x) < 0.7){
                all_descriptors.push_back(fr->image_descriptors[i][j]);
            }
        }
        //all_descriptors.insert(all_descriptors.end(), currentFrame->image_descriptors[i].begin(), currentFrame->image_descriptors[i].end());
    }
    DBoW2::BowVector bowVec_all;
    DBoW2::FeatureVector featVec_all;
    orb_vocabulary->transform(all_descriptors,bowVec_all,featVec_all,2);


    vector<double> scores;
    vector<double> stamps;
    // For LF frame
    lcLF_->queryDatabase(bowVec_all,7,scores, stamps);
    cout.precision(10);
    for(int i =0; i <scores.size(); i++){
        cout<<" image "<<(i+1)<<" score: "<<scores.at(i)<<" stamp: "<<stamps.at(i)<<endl;
    }
    //for monocular case
    DBoW2::BowVector bowVec;
    DBoW2::FeatureVector featVec;
    vector<double> scores_mono;
    vector<double> stamps_mono;
    orb_vocabulary->transform(fr->image_descriptors[2],bowVec,featVec,2);
    lcMono_->queryDatabase(bowVec,7,scores_mono, stamps_mono);
    cout.precision(10);
    for(int i =0; i <scores_mono.size(); i++){
        cout<<" image "<<(i+1)<<" score: "<<scores_mono.at(i)<<" stamp: "<<stamps_mono.at(i)<<endl;
    }



}


void FrontEnd::tracksForHist()
{

    // grab the current frame
    LightFieldFrame* lf_cur = currentFrame;
    lf_cur->lIds_mono.assign( lf_cur->image_kps[0].size(), -1);

    // Initialization ??
    if(initialized_ == NOT_INITIALIZED){
        currentFramePose = cv::Mat::eye(3, 4, CV_64F);
        insertKeyFrame();
        initialized_ = INITIALIZED;
        return;
    }

    // grab the last keyframe
    //LightFieldFrame* lf_prev = lfFrames[lfFrames.size()-2];
    LightFieldFrame* lf_prev = lfFrames.back();

    // Find inter matches, perform point cloud alignment to get the R and t
    // filter the inter matches
    std::vector<DMatch> inter_matches;
    opengv::points_t interMatch_3d_1, interMatch_3d_2;
    findInterMatches(lf_prev, lf_cur, inter_matches, false);

    for (auto &m : inter_matches) {
        interMatch_3d_1.push_back( lf_prev->points_3D[m.queryIdx]);
        interMatch_3d_2.push_back( lf_cur->points_3D[m.trainIdx]);
    }

    Mat T = Mat::eye(4, 4, CV_64F);
    // find pose of the light field frame using 3D-3d PC alignment
    // poseFromPCAlignment(interMatch_3d_1, interMatch_3d_2, T );

    //bool LF_METHOD=false;
    // bool PC_METHOD= true;
    bool MONO= true;

    /*if(LF_METHOD) {
         Mat mask;
         int num_inliers_LF_METHOD =0;
         E_mat = findEssentialMat(kp_pts1, kp_pts2, camconfig_.K_mats_[0], cv::RANSAC, 0.99, 0.5, mask);
         recoverPose(E_mat, kp_pts1, kp_pts2, camconfig_.K_mats_[0], rotation, translation, mask);
         num_inliers_LF_METHOD = countNonZero(mask);

         //-- Draw matches
         Mat img_matches_mono;
         //drawMatches( img1, lf_prev->image_kps_undist[0], img2, lf_cur->image_kps_undist[0], good_matches, img_matches_mono,Scalar::all(-1),
         //            Scalar::all(-1), mask_mono, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

         img_matches_mono = img2.clone();
         drawMatchesArrows(img_matches_mono, kps1, kps2, good_matches, mask, cv::Scalar(150, 200, 0));
         //-- Show detected matches
         imshow("Matches_mono", img_matches_mono);
         waitKey(100);
     } */
    //Debug matches between img 1 of two LF snapshots
    if(MONO){
        std::vector<DMatch> matches_mono;
        vector<Point2f> kp_pts1_mono, kp_pts2_mono;
        Mat mask_mono;
        findMatchesMono(lf_prev, lf_cur, 0, matches_mono);
        for(auto &m : matches_mono){
            Point2f p_prev = lf_prev->image_kps_undist[0][m.queryIdx].pt;
            Point2f p_cur = lf_cur->image_kps_undist[0][m.trainIdx].pt;
            kp_pts1_mono.push_back(p_prev);
            kp_pts2_mono.push_back(p_cur);
        }
        cv::Mat E_mat;
        cv::Mat rotation = Mat::zeros(3, 3, CV_64F), translation = Mat::zeros(3, 1, CV_64F);
        //find the essential matrix
        E_mat = findEssentialMat(kp_pts1_mono, kp_pts2_mono , camconfig_.K_mats_[0],cv::RANSAC,0.99, 0.5, mask_mono);
        recoverPose(E_mat, kp_pts1_mono, kp_pts2_mono, camconfig_.K_mats_[0], rotation, translation, mask_mono);
        int num_inliers = countNonZero(mask_mono);

        //-- Draw matches
        Mat img_matches_mono;
        //drawMatches( img1, lf_prev->image_kps_undist[0], img2, lf_cur->image_kps_undist[0], good_matches, img_matches_mono,Scalar::all(-1),
        //            Scalar::all(-1), mask_mono, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        img_matches_mono = lf_cur->imgs[0].clone();
        drawMatchesArrows(img_matches_mono, lf_prev->image_kps_undist[0], lf_cur->image_kps_undist[0], matches_mono, mask_mono , cv::Scalar(150,200, 0));
        //-- Show detected matches
        imshow("Matches_mono", img_matches_mono );
        waitKey(100);
        cout<<"number of matches_mono : "<< matches_mono.size()<<endl;
        cout<<"number of inliers : "<< num_inliers<<endl;

        // Now denote the matches as tracks.
        int ii=0;
        for(auto& m : matches_mono){
            if(mask_mono.at<uchar>(ii)){
                int l_id = lf_prev->lIds_mono[m.queryIdx];
                if( l_id == -1){ // landmark has not been inserted
                    Mat p_world = Mat::zeros(Size(3,1), CV_64F);
                    //l_id = map_mono->insertLandmark( p_world,lf_prev, lf_prev->image_kps_undist[0][m.queryIdx].pt);
                    lf_prev->lIds_mono[m.queryIdx] = l_id;
                }
                //landmark already exists
                // add the observation
                // add the observation for the same landmark
                Landmark* l = map_mono->getLandmark(l_id);
                if (l) {
                    //l->addLfFrame(lf_cur, lf_cur->image_kps_undist[0][m.trainIdx].pt);
                    lf_cur->lIds_mono[m.trainIdx] = l_id;
                }
            }
            ii++;
        }

        cout<<"total number of landmarks in mono : "<<map_mono->num_lms<<endl;
    }

    T = Mat::eye(4, 4, CV_64F);
    // find pose of the light field frame using 3D-3d PC alignment
    vector<bool> inliers;
    int num_inliers = poseFromPCAlignment(lf_prev, lf_cur, inter_matches, inliers, T, false );
    // get the last estimated pose WRT world into WTp
    Mat rot_p = allPoses.back().colRange(0,3);
    Mat t_p = allPoses.back().colRange(3,4);
    Mat WTp = Mat::eye(4, 4, CV_64F);
    rot_p.copyTo(WTp(cv::Range(0, 3), cv::Range(0, 3)));
    t_p.copyTo(WTp(cv::Range(0, 3), cv::Range(3, 4)));

    // Now convert the current pose WRT world
    unique_lock<mutex> lock(mMutexPose);
    Mat WTc =  WTp*T;
    currentFramePose = WTc.rowRange(0, 3).clone();
    cout << "Current Frame pose WRT to last KEYFRAME : " << currentFramePose << endl;

    insertKeyFrame();

    // insert a new keyframe
    //else move on to next frame

    //update the map with matched points
    //////////////////////////////////////////////////////////////////
    // find the inter frame matches.
    // filter the matches based on model- later
    // insert the matched 3D points in a structure - globalMap
    // Global Map : dict/set of MapPoints{3D locations in worldframe, list<frames> seen in, for each frame observations}
    // For each new frame :
    //   find the matches between 1) already matched points of the previous frame 2) new matches which are not there in the global Mao
    //   insert the new matches in the global map and mark which points are inserted in global Map or future matches.
    // In the end of all frames.
    // for each Map point in globalMap
    //    construct the histogram of the length of the features tracked.
    // Average age of the features
    // Average number of tracked feaures between two frames
    // Both for monocular + multi-cam setting with masked people.
    //////////////////////////////////////////////////////////////////
    for(auto& m : inter_matches){

        int l_id = lf_prev->lIds[m.queryIdx];
        if( l_id == -1){ // landmark has not been inserted
            Mat p_cur = Mat::ones(4, 1, CV_64F);
            Mat pp;
            cv::eigen2cv(lf_prev->points_3D[m.queryIdx], pp);
            pp.copyTo(p_cur.rowRange(0,3));
            Mat p_world = WTc * p_cur;
            //l_id = map->insertLandmark( p_world,lf_prev, Point2f(lf_prev->sparse_disparity[m.queryIdx].u, lf_prev->sparse_disparity[m.queryIdx].v));
            lf_prev->lIds[m.queryIdx] = l_id;
        }
        //landmark already exists
        // add the observation
        // add the observation for the same landmark
        Landmark* l = map->getLandmark(l_id);
        if (l) {
            //conl->addLfFrame(lf_cur, Point2f(lf_cur->sparse_disparity[m.trainIdx].u, lf_cur->sparse_disparity[m.trainIdx].v));
            lf_cur->lIds[m.trainIdx] = l_id;
        }
    }

    cout<<"total number of landmarks: "<<map->num_lms<<endl;

    //map->printMap();
    //show only the tracks of features matched in first two frames
    /*Mat img_tracks;
    img_tracks = img2.clone();
    cvtColor(img2, img_tracks, CV_GRAY2BGR);
    std::map<int, Landmark*>::iterator it;

    for(it = map_mono->mapPoints.begin() ; it != map_mono->mapPoints.end() ; ++it) {
        Landmark *l = it->second;
        if(l->frame_ids[0] == 0){
            cout << "landmark " << it->first << "," << l->lId << " pt: " << l->pt << " frame seen : "<<l->frame_ids[0]<<"at ("<<l->uv_ref[0]<<") ,";
            int prev_fr=l->frame_ids[0];
            //get the first frame point
            Point2f p_1 = l->uv_ref[0];
            for (auto i = 1; i < l->frame_ids.size(); i++){
                 if((l->frame_ids[i] - prev_fr) != 1){
                     cout<<"FRAMES SKIPPED"<<endl;
                     break;
                 }
                 else{
                     cout<<l->frame_ids[i]<<"at ("<<l->uv_ref[i]<<") ," ;
                     Point2f p = l->uv_ref[i];
                     cv::arrowedLine(img_tracks, p_1, p, Scalar(255, 100, 0));
                     prev_fr = l->frame_ids[i];
                     p_1 =  p;
                 }
            }
            cout<<endl;
        }

    }
    cv::imshow("tracks", img_tracks);
    cv::waitKey(50); */

}

void FrontEnd::featureStats(){
    cout<<"AT HISTOGRAM PLOT"<<endl;
    std::map<int, Landmark*>::iterator it;
    vector<int> hist_data, hist_data_mono;
    for(it = map->mapPoints.begin() ; it != map->mapPoints.end() ; ++it) {
        Landmark *l = it->second;
        //hist_data.push_back(l->frame_ids.size());
    }
    for(it = map_mono->mapPoints.begin() ; it != map_mono->mapPoints.end() ; ++it) {
        Landmark *l = it->second;
        //hist_data_mono.push_back(l->frame_ids.size());
    }
    vector<int> bins = {2,3,4,5,6,7,8,9,10,11,12,13};
    //plot the histogram of tracked points
    matplotlibcpp::hist_bins(hist_data, bins, "b", 0.2, true);
    matplotlibcpp::hist_bins(hist_data_mono, bins, "r", 0.2, true);
    matplotlibcpp::show();
}

void FrontEnd::writeLogs(string stats_file){
    VLOG(2)<<"Writing LOG File"<<endl;
    std::map<int, Landmark*>::iterator it;
    logFile_<<lfFrames.size()<<"\n";
    logFile2_.open (stats_file);
    logFile2_<<"FrameID, stamp,  Num_matches_KF, Num_matches_KF_lms, Num_matches_KF_new, Num_matches_localmap, Num_inliers_KF, Num_inliers_localmap,"
               "Num_triangulated, cov_00, cov_01, cov_02, cov_10, cov_11, cov_12, cov_20, cov_21, cov_22, pose_00, pose_01, pose_02, pose_03,"
               "pose_10, pose_11, pose_12, pose_13, pose_20, pose_21, pose_22, pose23, pose_30, pose_31, pose_32, pose33";
    for (int i=0; i < currentFrame->num_cams_ ; i++){
        logFile2_<<", inliers_view_"<<i;
    }
    for (int i=0; i < currentFrame->num_cams_ ; i++){
        logFile2_<<", triangulated_view_"<<i;
    }
    logFile2_<<"\n";

    int ind =0;
    for(auto f: lfFrames){
        Mat covPose = f->cov.colRange(3,6).rowRange(3,6);
        logFile2_<< f->frameId <<","<<std::fixed<< setprecision(6) << poseTimeStamps.at(ind)<<","<<f->num_matches_refKF<<","<<f->num_matches_refKF_lms<<","<<f->num_matches_refKF_new
                 <<","<<f->num_matches_localMap<<","<<f->num_inliers_refKF_lms<<","<<f->num_inliers_localMap
                 <<","<<f->num_triangulated<<","<<covPose.at<double>(0,0)<<","<<covPose.at<double>(0,1)<<","<<covPose.at<double>(0,2)<<
                 ","<<covPose.at<double>(1,0)<<","<<covPose.at<double>(1,1)<<","<<covPose.at<double>(1,2)<<
                 ","<<covPose.at<double>(2,0)<<","<<covPose.at<double>(2,1)<<","<<covPose.at<double>(2,2)<<
                 ","<<f->pose.at<double>(0, 0) << "," << f->pose.at<double>(0, 1) << "," << f->pose.at<double>(0, 2)
                 << "," << f->pose.at<double>(0, 3) << "," << f->pose.at<double>(1, 0) << "," << f->pose.at<double>(1, 1)
                 << "," << f->pose.at<double>(1, 2) << "," << f->pose.at<double>(1, 3) << "," << f->pose.at<double>(2, 0)
                 << "," << f->pose.at<double>(2, 1) << "," << f->pose.at<double>(2, 2) << "," << f->pose.at<double>(2, 3)
                 << "," << f->pose.at<double>(3, 0) << "," << f->pose.at<double>(3, 1) << "," << f->pose.at<double>(3, 2)
                 << "," << f->pose.at<double>(3, 3);
        for (int i=0; i < f->num_cams_ ; i++){
            logFile2_<<","<<f->num_inliers_per_view[i];
        }
        for (int i=0; i < f->num_cams_ ; i++){
            logFile2_<<","<<f->num_triangulated_per_view[i];
        }
        logFile2_<<"\n";


                 //f->num_inliers_per_view[0] << "," <<f->num_inliers_per_view[1]
                 //<<"," <<f->num_inliers_per_view[2]<<"," <<f->num_inliers_per_view[3]<<"," <<f->num_inliers_per_view[4]
                 //<<"," <<f->num_triangulated_per_view[0]<<"," <<f->num_triangulated_per_view[1]<<"," <<f->num_triangulated_per_view[2]
                 //<<"," <<f->num_triangulated_per_view[3]<<"," <<f->num_triangulated_per_view[4]<<"\n";

        logFile_ << f->frameId <<" "<<setprecision(6) << poseTimeStamps.at(ind)<< " " << f->pose.at<double>(0, 0) << " " << f->pose.at<double>(0, 1) << " " << f->pose.at<double>(0, 2)
                 << " " << f->pose.at<double>(0, 3) << " " << f->pose.at<double>(1, 0) << " " << f->pose.at<double>(1, 1)
                 << " " << f->pose.at<double>(1, 2) << " " << f->pose.at<double>(1, 3) << " " << f->pose.at<double>(2, 0)
                 << " " << f->pose.at<double>(2, 1) << " " << f->pose.at<double>(2, 2) << " " << f->pose.at<double>(2, 3)
                 << " " << f->pose.at<double>(3, 0) << " " << f->pose.at<double>(3, 1) << " " << f->pose.at<double>(3, 2) << " " << f->pose.at<double>(3, 3) << "\n";


        VLOG(1) <<f->pose<<endl;
        ind++;
    }
    for(it = map->mapPoints.begin() ; it != map->mapPoints.end() ; ++it) {
        Landmark *l = it->second;
        int frameIdx =0;
        //write the landmark ID and initial value
        logFile_<<"l"<<" "<<l->lId<<" "<<l->pt3D.at<double>(0,0)<<" "<<l->pt3D.at<double>(1,0)<<" "<<l->pt3D.at<double>(2,0)<<"\n";
        for(auto f: l->KFs){
            //get the key point index of each component camera where this landmark is seen
            IntraMatch* intraMatch = &f->intraMatches[l->featInds[frameIdx]];
            for(int i =0; i < f->num_cams_ ; i++){
                int kp_ind = intraMatch->matchIndex[i];
                if(kp_ind != -1){
                    VLOG(3)<<f->frameId<<" "<<i<<" "<<f->image_kps_undist[i][kp_ind].pt.x<<" "<<f->image_kps_undist[i][kp_ind].pt.y<<"\n";
                    logFile_ <<"e"<<" "<<f->frameId<<" "<<i<<" "<<f->image_kps_undist[i][kp_ind].pt.x<<" "<<f->image_kps_undist[i][kp_ind].pt.y<<"\n";
                }
                // logFile_ <<f->frameId<<" "<<i<<" "<<l->lId<<" "<<f->image_kps_undist[i][kp_ind].pt.x<<" "<<f->image_kps_undist[i][kp_ind].pt.y<<" "<<l->pt3D.at<double>(0,0)<<" "<<l->pt3D.at<double>(1,0)<<" "<<l->pt3D.at<double>(2,0)<<"\n";
            }
            //// This is to record only the reference camera
            //logFile_ <<f->frameId<<" "<<l->lId<<" "<<l->uv_ref[c].x<<" "<<l->uv_ref[c].y<<" "<<l->pt3D.at<double>(0,0)<<" "<<l->pt3D.at<double>(1,0)<<" "<<l->pt3D.at<double>(2,0)<<"\n";
            frameIdx++;
        }
    }
}

void FrontEnd::drawMatchesArrows(cv::Mat& img, vector<KeyPoint>& kps_prev, vector<KeyPoint>& kps_cur, std::vector<DMatch> matches, Mat mask, cv::Scalar color){

    int i =0;
    if (img.channels() != 3){
        Mat imgColor;
        cvtColor(img,imgColor , COLOR_GRAY2BGR);
        img = imgColor;
    }

    for(auto &m : matches){
        Point2f p1 = kps_prev[m.queryIdx].pt;
        Point2f p2 = kps_cur[m.trainIdx].pt;

        cv::arrowedLine(img, p1, p2, color, 1);
        if(mask.at<uchar>(i)){
            cv::circle(img,p1,2,color,2);
        }
        i++;
    }

}

/// This method takes key points of two images corresponding to two cameras
/// Undistorts them returns their normalized coordinates or bearing vectors
/// \param kps_1  : vector of key points in first image
/// \param kps_2  : vector of key points in second image
/// \param cam1  : index of first camera
/// \param cam2  : index of second camera
void FrontEnd::prepareOpengvData(vector<Point2f>& kps_1, vector<Point2f>& kps_2, int cam1, int cam2){

    //undistort the keypoints
    vector<Point2f> undist_1, undist_2;
    undistortPoints(kps_1, undist_1, camconfig_.K_mats_[cam1], camconfig_.dist_coeffs_[cam1] );
    undistortPoints(kps_2, undist_2, camconfig_.K_mats_[cam2], camconfig_.dist_coeffs_[cam2] );

    // convert these undistorted points into bdaring vectors to give into openGV
    //compute the normalized / undistorted coordinates of the key points
    for ( int kp_ind = 0; kp_ind < undist_1.size(); kp_ind++){
        opengv::bearingVector_t p1, p2 ;
        p1<<(double)undist_1[kp_ind].x, (double)undist_1[kp_ind].y, (double)1.0;
        p2<<(double)undist_2[kp_ind].x, (double)undist_2[kp_ind].y, (double)1.0;

        bearings_1.push_back(p1);
        bearings_2.push_back(p2);

        correspondences_1.push_back(cam1);
        correspondences_2.push_back(cam2);

    }
}
