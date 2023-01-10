###################################################################
#####  PROCESS the drift error from checkerboard              #####
###################################################################
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial.transform import Rotation as R
import cv2


def draw(img, corners, imgpts):
    corners = corners.astype(np.int32)
    imgpts = imgpts.astype(np.int32)
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 5)
    return img


def getPoseFromChessBoard(img_inp, k, dist, squarSize):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((16 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:16, 0:9].T.reshape(-1, 2)
    objp = objp * squarSize
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    axis = axis * squarSize
    T_ci = np.zeros((4, 4), np.float)
    ## detect chesssboard
    ret, corners = cv2.findChessboardCorners(img_inp, (16, 9), None)

    if ret == True:
        print("corners found")
        corners2 = cv2.cornerSubPix(img_inp, corners, (11, 11), (-1, -1), criteria)
        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, k, dist)
        rot_mat, _ = cv2.Rodrigues(rvecs)
        ## get pose WRT chessboard T_ci
        T_ic = np.vstack((np.hstack((rot_mat, tvecs)), np.array([0, 0, 0, 1])))
        T_ci = np.linalg.inv(T_ic)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, k, dist)
        cv2.drawChessboardCorners(img_inp, (16, 9), corners2, ret)
        img = cv2.cvtColor(img_inp, cv2.COLOR_GRAY2RGB)
        img = draw(img, corners2, imgpts)
        cv2.imshow('img', img)
        cv2.waitKey(0) & 0xff
    return T_ci


def computeError(df_traj, tStamp, k, dist, squareSize, patternSize, T_c0=np.eye(4)):
    img_gray = cv2.imread(os.path.join(data_path, str(tStamp) + ".png"), 0)
    # clahe = cv2.createCLAHE(clipLimit = 5)
    # img_gray = clahe.apply(img_gray)
    T_ci_gt = getPoseFromChessBoard(img_gray, k, dist, squareSize)
    if np.linalg.norm(T_ci_gt) == 0:
        # print("Checkerboard not found")
        return -1, -1
    t = round(tStamp * 1e-9, 6)
    print("Image : ", tStamp)
    ind_df = df_traj[0].sub(t).abs().idxmin()

    # ind_df = (df_traj[0].tolist().index(t))
    print("index : ", ind_df)
    quat = np.array(df_traj.iloc[ind_df, 4:8])
    rot = R.from_quat(quat).as_matrix()
    trans = np.array(df_traj.iloc[ind_df, 1:4]).reshape(3, 1)
    T_0i = np.vstack((np.hstack((rot, trans)), np.array([0, 0, 0, 1])))
    ## In Tum trajectory we have T_0i, pose WRT start,
    ## GT = T_ci, expected T_ci^ = T_c0 * T_0i=
    print("GT Pose: ")
    print(T_ci_gt)
    if T_ci_gt[3, 3] > 2.0:
        print("calibrationj board is too far. estimate wont be accurate")
        return -1, -1
    print("expected Pose Traj: ")
    T_ci_exp = T_c0 @ T_0i
    print(T_ci_exp)
    # find delta pose
    deltapose = np.linalg.inv(T_ci_exp) @ T_ci_gt
    rotvec_cv, _ = cv2.Rodrigues(deltapose[0:3, 0:3])
    trans_error = np.linalg.norm(T_ci_exp[0:3, 3] - T_ci_gt[0:3, 3])
    rot_error = np.rad2deg(np.linalg.norm(rotvec_cv))
    print("translation error: ", trans_error, )
    print("rotation error in degrees: ", rot_error)
    return trans_error, rot_error


# data_path="/data/04_14_2022_isec_lab_ground/isec_ground_night_2/cam0"
# traj_path = "/home/auv/ros_ws/devel/lib/LFApps/04_14_2022/Tum_trajectrory.txt"
# k = np.array([[886.6272107893, 0, 361.488867356257],
#              [0, 886.7929657865, 284.068219958725],
#              [0, 0, 1]])
# dist = np.array([-0.21286163901817143, 0.17023197326272427, -9.485363152188766e-06,
#    -0.0001386285980226874])
# five_cams = [1649979950837768693,1649979951838062494, 1649979953036801584,1649979954237361874,1649979955237576739 ]
# four_cams=[1649979951437556527, 1649979952637979681,1649979953837583386, 1649979954837175857,1649979955837720923  ]
# three_cams = [1649979951838062494,1649979953036801584, 1649979954237361874,1649979955237576739 ]
# two_cams = [1649979951437556527,1649979952436942290, 1649979953637951101, 1649979954636732071, 1649979955637119932]
# orb_slam = [1649979955186799326,1649979955787700488, 1649979955987961614,1649979957087622430 ]
# five_cams1 = [1649979954037088217, 1649979955038292781, 1649979956037060130 ]
#
# first_stamp = 1649979715237611442


# data_path="/data/04_25_2022_isec_ground_dynamic/isec_ground_day_2loops/cam0"
# traj_path = "/home/auv/ros_ws/devel/lib/LFApps/04_25_2022/Tum_trajectrory_04_25_2022_stereo.txt"

# k = np.array([[886.7286289305, 0, 364.893501171],
#              [0, 887.3000268141, 286.968036928],
#             [0, 0, 1]])
# dist = np.array([-0.21627336164628835, 0.17205009106186261, 1.633696614419175e-05, 0.000800860501631808])
# first_stamp = 1650907610204368120
# last_stamp =  1650907894553652056        #1650907892153231019


# data_path="/data/04_21_2022_isec_curry/bag_outdoor_nu/cam0"
# traj_path = "/home/auv/ros_ws/devel/lib/LFApps/04_21_2022/outdoor/Curry_center_Orbslam3_stereo.txt"
# k = np.array([[887.6164997890, 0, 361.2404351028],
#              [0, 888.007246603, 283.140828388],
#              [0, 0, 1]])
# dist = np.array([-0.21046072327790924, 0.15577979446322998, -0.0001708844706763513, -0.00022739337206347906])

# five_cams = [1650577604747234911,1650577605647659897, 1650577609847147436 ,1650577610747154366 , 1650577611947129604, 1650577613147546904]
# four_cams = [1650577604747234911, 1650577605647659897, 1650577609546509253, 1650577610447049664, 1650577611346696357, 1650577612547425164, 1650577613747880382]
# three_cams = [1650577604747234911, 1650577605647659897, 1650577609546509253, 1650577610447049664, 1650577611346696357, 1650577612547425164, 1650577613747880382]
# two_cams = [1650577604747234911,1650577605647659897, 1650577609847147436 ,1650577610747154366 , 1650577611947129604, 1650577613147546904]
# orb_slam = [ 1650577604047963739,1650577605047238159, 1650577611246613269 ]

# first_stamp = 1650576194947202033


# data_path="/data/04_21_2022_isec_curry/bag_indoor_isec/cam0"
# traj_path = "/home/auv/ros_ws/devel/lib/LFApps/04_21_2022_3cams_3.txt"
# k = np.array([[887.616499789026, 0, 361.2404351028],
#              [0, 888.007246603, 283.140828388],
#              [0, 0, 1]])
# dist = np.array([-0.21046072327790924, 0.15577979446322998, -0.0001708844706763513, -0.00022739337206347906])

# five_cams = [1650574607156144434]
# four_cams = [1650574607156144434]
# three_cams = [1650574606956391956]
# two_cams = [1650577604747234911,1650577605647659897, 1650577609847147436 ,1650577610747154366 , 1650577611947129604, 1650577613147546904]
# orb_slam = [ 1650577604047963739,1650577605047238159, 1650577611246613269 ]

# first_stamp = 1650574403456143497


# data_path="/data/04_25_2022_isec_ground_dynamic/bag1/cam0"
# traj_path = "/home/auv/ros_ws/devel/lib/LFApps/04_25_2022_isec_day_fixed/04_25_2022_isec_day_fixed_Orbslam3.txt"
# k = np.array([[886.728628930516, 0, 364.89350117142],
#              [0, 887.300026814151, 286.96803692817],
#              [0, 0, 1]])
# dist = np.array([-0.21627336164628835, 0.17205009106186261, 1.633696614419175e-05, 0.0008008605016318086])
# first_stamp = 1650907022320894118 #1650907004920428538

data_path = "/data/05_12_2022/auto_exp/cam0"
traj_path = "/home/auv/traj_05_12_2022_auto_4cams_4.txt"
k = np.array([[876.540399700669, 0, 360.8837461127],
              [0, 877.247558177220, 281.78108684230],
              [0, 0, 1]])
dist = np.array([-0.22638984819861208, 0.15111568439970843, -0.00036084571731219905, 0.001654352822682288])
first_stamp = 1652395744713111337  # fixed 1652382414704238766        #dyn1652394807360152512

## Get the time stamps and read images and check which ones have the checkerboard detected.
## Or get the timestamps manually and read the images

## read the tum format trajectory file
df = pd.read_csv(traj_path, header=None, skipinitialspace=True, sep=' ')
img_first = cv2.imread(os.path.join(data_path, str(first_stamp) + ".png"), 0)
T_cs = getPoseFromChessBoard(img_first, k, dist, 0.081)

# if start stamp is not the first pose
# t =round(first_stamp*1e-9, 6)
# ind_df = df[0].sub(t).abs().idxmin()
# print("index : ", ind_df)
# quat = np.array(df.iloc[ind_df, 4:8])
# rot = R.from_quat(quat).as_matrix()
# trans = np.array(df.iloc[ind_df,1:4]).reshape(3,1)
# T_0s = np.vstack((np.hstack((rot, trans)), np.array([0,0,0,1])))
# T_cs = T_cs @ np.linalg.inv(T_0s)

computeError(df, first_stamp, k, dist, 0.081, (9, 9), T_cs)
imgnames = os.listdir(data_path)
imgnames.sort()
imgstamps = np.array([float(img[:-4]) * 1e-9 for img in imgnames])

# check_stamp = 1650574602.756657
# print("Chec stamp: ", check_stamp)
avg_trans_e = 0
avg_rot_e = 0
cnt = 0
for stamp in df[0]:
    if stamp > (first_stamp * 1e-9 + 86):
        print("STamp:", stamp)
        diffs = np.absolute(imgstamps - stamp)
        image_index = diffs.argmin()
        trans_e, rot_e = computeError(df, int(imgnames[image_index][:-4]), k, dist, 0.081, (9, 9), T_cs)
        if trans_e != -1 and rot_e != -1:
            avg_trans_e = avg_trans_e + trans_e
            avg_rot_e = avg_rot_e + rot_e
            cnt = cnt + 1
print("Average translation error : ", avg_trans_e / cnt)
print("Average rotational error : ", avg_rot_e / cnt)


def iterate_timestamp_list(stamp_list, data_path, first_stamp, k, dist):
    for last_stamp in stamp_list:
        img_first = cv2.imread(os.path.join(data_path, str(first_stamp) + ".jpg"), 0)
        img_last = cv2.imread(os.path.join(data_path, str(last_stamp) + ".jpg"),
                              0)  # 1650907892203072353 #1650907892153231019
        # clahe = cv2.createCLAHE(clipLimit = 5)
        # img_first = clahe.apply(img_first)
        # img_last = clahe.apply(img_last)

        img_combined = np.hstack((img_first, img_last))
        cv2.imshow("first and last images", img_combined)
        cv2.waitKey(0)

        T_c0 = getPoseFromChessBoard(img_first, k, dist, 0.081)
        print("First Pose WRT chessboard: ")
        print(T_c0)

        print("----------------------------------------------------")
        computeError(df, first_stamp, k, dist, 0.081, (9, 9), T_c0)
        computeError(df, last_stamp, k, dist, 0.081, (9, 9), T_c0)

        cv2.destroyAllWindows()