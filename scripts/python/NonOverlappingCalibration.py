import utilities as ut
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import os
import gtsam
from gtsam import symbol_shorthand
L = symbol_shorthand.L
X = symbol_shorthand.X
B = symbol_shorthand.B
M = symbol_shorthand.M
import gtsam_unstable
from gtsam_unstable import ProjectionFactorPPPCal3_S2, ProjectionFactorPPPCCal3_S2
from gtsam import (Cal3_S2, LevenbergMarquardtOptimizer,
                         GenericProjectionFactorCal3_S2, Marginals,
                         NonlinearFactorGraph, PinholeCameraCal3_S2, Point3, Point2,
                         Pose3, PriorFactorPoint3, PriorFactorPose3, Rot3, Values)


def compute_extrinsic_estimate(image_path, tStamp, df, k, dist,T_opti_initial, T_board_initial):
    """
    We compute the rigid transform between the ref (front) and side cameras
    Args:
        image_path:  path of the side camera image
        tStamp: the corresponding timestamp
        df: dataframe of the VO trajectory
        k: intrinsic matrix of the image
        dist: distortion coefficients of the image
        T_opti_initial: initial pose of the ref camera in optitrack frame/VO frame
        T_board_initial: initial pose of the ref camera WRT the chessboard

    Returns: NOne.. This prints the initial estimate of the transform between
              the side and ref camera for each call of this function
    """
    #Go through other camera images one by one.
    # when calibration board is detected get pose T_board_side
    #print(imgname)imgg
    img_gray = cv2.imread(image_path, 0)
    cv2.imshow("image", img_gray)
    cv2.waitKey(0)
    T_board_side = ut.getPoseFromChessBoard(img_gray, k, dist, 0.081)
    if np.linalg.norm(T_board_side) == 0:
        #print("Checkerboard not found")
        return
    #Get the corresponding optitrack pose T_opti_rig.
    #We want relative motion T_rig1_rig = T_rig1_opti * T_opti_rig
    print("Image : ", tStamp)
    ind_df = df[0].sub(tStamp).abs().idxmin()
    print("index : ", ind_df)
    quat = np.array(df.iloc[ind_df, 4:8])
    rot = R.from_quat(quat).as_matrix()
    trans = np.array(df.iloc[ind_df,1:4]).reshape(3,1)
    T_opti_rig = np.vstack((np.hstack((rot, trans)), np.array([0,0,0,1])))
    T_rel_cur_1 = np.linalg.inv(T_opti_rig) @ T_opti_initial
    #compute the extrinsics T_rig_side =  T_rig_rig1 * T_rig1_board* T_board_side
    T_front_side = T_rel_cur_1 @ np.linalg.inv(T_board_initial) @ T_board_side
    print("Extrinsic T_front_side: ", T_front_side)

"""
This portion is required to get an initial estimate of the
rigid transformation between the side and front cameras
using the VO/optitrack groundtruth for relative motion,
chessboard for getting the pose of the front and side 
cameras at different time steps.
We have T_board_rig1 = pose of the ref camera (front camera) in chessboard at timestep1
T_opti_rig1 = pose of the ref camera in optitrack frame at timestep1
For each image in side cameras where the chessboard is found in the sequence
We compute the rigid transform between the ref (front) and side cameras
Get the list of image names in the dataset.
"""

def compute_initial_transform_allimages(data_path_left,T_opti_rig1, T_board_front1, df, k_s5, dist_s5, k_s6, dist_s6):
    imgnames = os.listdir(data_path_left)
    imgnames.sort()
    imgstamps = np.array([ float(img[:-4])*1e-9 for img in imgnames])

    #for imgname, tStamp in zip(imgnames[312 :412], imgstamps[312:412]):
    #    #get the pose of the camera WRT calib board . This is initial pose. T_board_rig1
    #    img_first = cv2.imread(os.path.join(data_path_front,imgname), 0)
    #    T_board_1 = ut.getPoseFromChessBoard(img_first, k_f, dist_f, 0.081)
    #    #get the corresponding optitrack pose. T_opti_rig1
    #    print("Image : ", tStamp)
    #    ind_df = df[0].sub(tStamp).abs().idxmin()
    #    print("index : ", ind_df)
    #    quat = np.array(df.iloc[ind_df, 4:8])
    #    rot = R.from_quat(quat).as_matrix()
    #    trans = np.array(df.iloc[ind_df,1:4]).reshape(3,1)
    #    T_opti_1 = np.vstack((np.hstack((rot, trans)), np.array([0,0,0,1])))
    #    T_opti_board = T_opti_1 @ np.linalg.inv(T_board_1)
    #    print("T_opti_Board: data_path_right",T_opti_board )
    #    #get_extrinsic_estimate(os.path.join(data_path_front,imgname),tStamp, df, k_f, dist_f, T_opti_rig1, T_board_front1)

    for imgname, tStamp in zip(imgnames[:100], imgstamps[:100]):
        compute_extrinsic_estimate(os.path.join(data_path_right,imgname),tStamp, df, k_s6, dist_s6, T_opti_rig1, T_board_front1)

    for imgname, tStamp in zip(imgnames[400:500], imgstamps[400:500]):
        compute_extrinsic_estimate(os.path.join(data_path_left,imgname),tStamp, df, k_s5, dist_s5, T_opti_rig1, T_board_front1)

"""
#### Optimization ###########
"""

# Define the camera observation noise model
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)  # one pixel in u and v
# 0.3 rad and 0.1 m

pose_noise_vo = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
# 0.1 m
point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)


def build_graph(graph, initial_estimate, img_stamp_list, indices, data_path, kmat, dist, df, landmarks, cam_ind, T_w_refinit = np.eye(4), psize=(16,9) ):
    gtsam_K = Cal3_S2(kmat[0, 0], kmat[1, 1], 0.0, kmat[0, 2], kmat[1, 2])
    # For each image
    for img_tup_i in indices:
        imgname, tStamp = img_stamp_list[img_tup_i]
        #       print(imgname, tStamp)
        # read the image
        img = cv2.imread(os.path.join(data_path, imgname), 0)
        # undistort the image
        imgg = cv2.undistort(img, kmat, dist, None, kmat)
        # get the pose of the image WRT board and the corresponding image points
        T_board_imgg, imgPoints = ut.getPoseFromChessBoard(imgg, kmat, np.zeros((1, 4)), 0.081, psize)
        # if no board detected in the front camera go to left cam
        if len(imgPoints) != 0:
            cv2.drawChessboardCorners(imgg, psize, imgPoints, True)
            cv2.imshow('img', imgg)
            cv2.waitKey(0) & 0xff
        else:
            continue

        # get the corresponding optitrack/VO pose. T_opti_rig1
        diff_stamps = df[0].sub(tStamp).abs()
        ind_df = diff_stamps.idxmin()
        if diff_stamps[ind_df] >= 0.01:
            continue
        print("index : ", ind_df)
        quat = np.array(df.iloc[ind_df, 4:8])
        rot = R.from_quat(quat).as_matrix()
        trans = np.array(df.iloc[ind_df, 1:4]).reshape(3, 1)
        T_w_cur = np.vstack((np.hstack((rot, trans)), np.array([0, 0, 0, 1])))
        # get the relative pose of the current frame WRT the refrence frame .
        T_1_cur = np.linalg.inv(T_w_refinit) @ T_w_cur
        #  depending on the camera it is seen in get the 3d-2d correspondences
        # For each landmark add a prior if needed
        assert (len(imgPoints) == landmarks.shape[1])

        # Insert the pose of the rig/front camera WRT the starting position
        initial_estimate.insert(X(img_tup_i), Pose3(T_1_cur))
        factor = PriorFactorPose3(X(img_tup_i), Pose3(T_1_cur), pose_noise_vo)
        graph.push_back(factor)

        # Add the projectionPPP factor for each landmark with the measured corners
        for i in range(landmarks.shape[1]):
            lm = Point3(landmarks[0:3, i])
            # factor = PriorFactorPoint3(L(i), lm, point_noise)
            # graph.push_back(factor)
            meas = Point2(imgPoints[i].ravel())
            factor = ProjectionFactorPPPCal3_S2(meas, measurement_noise, X(img_tup_i), B(cam_ind), L(i), gtsam_K)
            #factor = ProjectionFactorPPPCCal3_S2(meas, measurement_noise, X(img_tup_i), B(cam_ind), L(i), M(cam_ind))
            graph.push_back(factor)


"""

Variables for paths of the images
intrinsic calibration parameters
the first time stamp to be considered
"""
#data_path_front="/data/05_16_2022_calib/bag3/cam0"
#data_path_left="/data/05_16_2022_calib/bag3/cam5"
#data_path_right="/data/05_16_2022_calib/bag3/cam6"
#traj_path = "/data/05_16_2022_calib/bag3/2022_05_16_18_34_56_gt.txt"
#
#k_f = np.array([[874.039438699, 0, 364.550740159484],
#              [0, 874.787225463577, 286.6490604574],
#              [0, 0, 1]])
#dist_f = np.array([-0.24772900267450307, 0.1921058006879624, 2.2598774552204495e-05, 0.00309486514093092])
#
#k_s6 = np.array([[882.9391342187388, 0, 384.1878684084419],
#              [0, 882.8153341385736, 278.9079081016814],
#              [0, 0, 1]])
#dist_s6 = np.array([-0.21446184288970324, 0.14293621778218552, -0.0008679687005863199, 0.0013624375889589688])
#
#k_s5 = np.array([[877.1880881359045, 0, 367.273509202746],
#              [0, 877.57764402193, 285.9607389654299],
#              [0, 0, 1]])
#dist_s5 = np.array([-0.2228949253370626, 0.18736719172646038, 0.0010781783131958086, -0.0005436650475248478])
#
#first_stamp =  1652740514096921457 #side6 1652740503496921457


# img_list[0:600:10]
# first stamp index = 354

data_path_front="/data/04_21_2022_isec_curry/bag_outdoor_nu/cam0"
data_path_left="/data/04_21_2022_isec_curry/bag_outdoor_nu/cam5"
data_path_right="/data/04_21_2022_isec_curry/bag_outdoor_nu/cam6"
traj_path = "/data/04_21_2022_isec_curry/bag_outdoor_nu/KeyFrameTrajectory.txt"

k_f = np.array([[887.616499789026, 0, 361.240435102855],
              [0, 888.0072466034, 283.140828388],
              [0, 0, 1]])
dist_f = np.array([-0.21046072327790924, 0.15577979446322998, -0.0001708844706763513, -0.00022739337206347906])

k_s6 = np.array([[891.0490139190758, 0, 372.419471646802],
              [0, 889.3197721578968, 281.45452062943343],
              [0, 0, 1]])
dist_s6 = np.array([-0.21075142970197697, 0.16651989783667467, 4.0118732787539406e-05, 0.00019295049175684507])

k_s5 = np.array([[889.3028919015703, 0, 361.438647489639],
              [0, 887.8018514663769, 280.934458778716],
              [0, 0, 1]])
dist_s5 = np.array([-0.2086607385484864, 0.1539408933033932, 0.00046369996633142854, -0.0006303507193805352])

first_stamp = 1650576194697326789

'''
Read the optitrack trajectory file or a VO 
trajectory file in TUM format
'''
df = pd.read_csv(traj_path, header=None,skipinitialspace=True, sep=' ')
#df[0] = df[0] * 1e-9
"""
get the pose of the front camera WRT calib board . This is initial pose. T_board_rig1
"""
img_first = cv2.imread(os.path.join(data_path_front,str(first_stamp)+".jpg"), 0)
T_board_front1, _ = ut.getPoseFromChessBoard(img_first, k_f, dist_f, 0.081)
print(T_board_front1)

"""
get the corresponding optitrack or VO pose from the dataframe T_opti_rig1
"""
t = round(first_stamp*1e-9, 6)
print("Image : ", first_stamp)
ind_df = df[0].sub(t).abs().idxmin()
print("index : ", ind_df)
quat = np.array(df.iloc[ind_df, 4:8])
rot = R.from_quat(quat).as_matrix()
trans = np.array(df.iloc[ind_df,1:4]).reshape(3,1)
T_opti_rig1 = np.vstack((np.hstack((rot, trans)), np.array([0,0,0,1])))


# We have initial estimates for landmarks(calibration target points)
# rig poses frok optitrack and the relative poses between the rig and the individual cameras
T_front_left = np.array([[-0.00728608, -0.02427516, -0.99967876, 0.27547358],
                         [0.02150753, 0.99947024, -0.02442685, -0.13939047],
                         [0.99974214, -0.0216786, -0.00676012, -0.03593664],
                         [0., 0., 0., 1.]])

T_front_right = np.array([[0.00935536, 0.01434981, 0.99985327, 0.37059141],
                          [-0.01839862, 0.99973023, -0.01417589, -0.13330094],
                          [-0.99978696, -0.0182633, 0.00961685, 0.00726508],
                          [0., 0., 0., 1.]])


lms_board = np.zeros((16 * 9, 3), np.float32)
lms_board[:, :2] = np.mgrid[0:16, 0:9].T.reshape(-1, 2)
lms_board = lms_board * 0.081
lms_board = np.hstack((lms_board, np.ones((16 * 9, 1))))

# print(lms_board)
lms_1 = np.linalg.inv(T_board_front1) @ lms_board.T


lms_board_s = np.zeros((16 * 7, 3), np.float32)
lms_board_s[:, :2] = np.mgrid[0:16, 0:7].T.reshape(-1, 2)
lms_board_s = lms_board_s * 0.081
lms_board_s = np.hstack((lms_board_s, np.ones((16 * 7, 1))))

# print(lms_board)
lms_1_s = np.linalg.inv(T_board_front1) @ lms_board_s.T

# We build a factor graph with projectfactorPPP which optimizes over point, pose and pose
# Read the front camera images. For first pose get T_first_board. This pose is the origin. We also have T_opti_rig1

# for each of the other images, get the VO from optitrack T_rig1_opti * T_opti_rig = T_1_cur
# insert the projection factor where the lms are WRT first cam = T_first_board * lms_board,
# pose of the rig which coincides with first camera T_1_cur (this is indentity for the first frame)
# pose of the component camera WRT first camera T_cur_left, T_cur_right

# Initialze graph and initial values
graph = NonlinearFactorGraph()
initial_estimate = Values()

imgnames = os.listdir(data_path_left)
imgnames.sort()
imgstamps = np.array([ float(img[:-4])*1e-9 for img in imgnames])
img_list = list(zip(imgnames, imgstamps))
img_index = 0

##curry center front camera chessboard 90-120,
## left cam5 - 4167 - 4200,
## right cam6 227 - 300

"""
When we have the visual odometry indo, we should search for images corresponding 
to the time stamps
"""



build_graph(graph, initial_estimate, img_list,list(range(90, 160, 2))+list(range(3850, 4200, 2)), data_path_front, k_f, dist_f, df, lms_1, 0, T_opti_rig1 )
build_graph(graph, initial_estimate, img_list,list(range(3900, 4250, 1)), data_path_left, k_s5, dist_s5, df, lms_1, 1, T_opti_rig1 )
build_graph(graph, initial_estimate, img_list,list(range(160, 600, 1)), data_path_right, k_s6, dist_s6, df, lms_1_s, 2, T_opti_rig1, (16,7) )

# add prior on the first pose and the front camera ad identity. This our refernce frame. We assume
# here that the front camera frame is coinciding with the first image frame
pose_noise2 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.001, 0.001, 0.001]))
factor = PriorFactorPose3(B(0), Pose3(np.eye(4)), pose_noise2)
graph.push_back(factor)
initiPose = Pose3(np.eye(4))
factor = PriorFactorPose3(X(90), initiPose, pose_noise2)
graph.push_back(factor)

pose_noise_bl = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.02, 0.015, 0.02]))
pose_noise_br = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.015, 0.02]))
factor = PriorFactorPose3(B(1), Pose3(T_front_left), pose_noise_bl)
graph.push_back(factor)
factor = PriorFactorPose3(B(2), Pose3(T_front_right), pose_noise_br)
graph.push_back(factor)
#factor = PriorFactorPoint3(L(0), Point3(lms_1[0:3, 0]), point_noise)
#graph.push_back(factor)
"""
Prior on intrinsic matrices

"""
'''
calnoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([200, 200, 0, 50, 50]))
gtsam_K = Cal3_S2(k_f[0, 0], k_f[1, 1], 0.0, k_f[0, 2], k_f[1, 2])
gtsam_s5 = Cal3_S2(k_s5[0, 0], k_s5[1, 1], 0.0, k_s5[0, 2], k_s5[1, 2])
gtsam_s6 = Cal3_S2(k_s6[0, 0], k_s6[1, 1], 0.0, k_s6[0, 2], k_s6[1, 2])

factor = gtsam.PriorFactorCal3_S2(M(0), gtsam_K, calnoise )
graph.push_back(factor)
factor = gtsam.PriorFactorCal3_S2(M(1), gtsam_s5, calnoise )
graph.push_back(factor)
factor = gtsam.PriorFactorCal3_S2(M(2), gtsam_s6, calnoise )
graph.push_back(factor)
'''

############################
## Initial Estimates########
############################
#initial_estimate.insert(X(90), Pose3(np.eye(4)))
initial_estimate.insert(B(0), Pose3(np.eye(4)))
initial_estimate.insert(B(1), Pose3(T_front_left))
initial_estimate.insert(B(2), Pose3(T_front_right))

'''
initial_estimate.insert(M(0), gtsam_K)
initial_estimate.insert(M(1), gtsam_s5)
initial_estimate.insert(M(2), gtsam_s6)
'''


for i in range(lms_1.shape[1]):
    lm = Point3(lms_1[0:3, i])
    initial_estimate.insert(L(i), lm)
    factor = PriorFactorPoint3(L(i), lm, point_noise)
    graph.push_back(factor)

params = gtsam.LevenbergMarquardtParams()
params.setDiagonalDamping(True)
params.setVerbosityLM("SUMMARY")
optimizer = LevenbergMarquardtOptimizer(graph, initial_estimate, params)
result = optimizer.optimize()
result.print('Final results:\n')
print('initial error = {}'.format(graph.error(initial_estimate)))
print('final error = {}'.format(graph.error(result)))
print(graph.at(B(0)))

