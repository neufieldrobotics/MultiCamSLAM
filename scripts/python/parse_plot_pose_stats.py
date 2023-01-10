##############################################################
#PLOT TRACKING STATS
##############################################################

import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
from typing import Iterable, Optional, Tuple


def ellipsoid(rx: float, ry: float, rz: float,
              n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy equivalent of Matlab's ellipsoid function.

    Args:
        rx: Radius of ellipsoid in X-axis.
        ry: Radius of ellipsoid in Y-axis.
        rz: Radius of ellipsoid in Z-axis.
        n: The granularity of the ellipsoid plotted.

    Returns:
        The points in the x, y and z axes to use for the surface plot.
    """
    u = np.linspace(0, 2 * np.pi, n + 1)
    v = np.linspace(0, np.pi, n + 1)
    x = -rx * np.outer(np.cos(u), np.sin(v)).T
    y = -ry * np.outer(np.sin(u), np.sin(v)).T
    z = -rz * np.outer(np.ones_like(u), np.cos(v)).T

    return x, y, z


def plot_pose3_on_axes(axes, pose, color, axis_length=0.1, P=None, scale=1):
    """
    Plot a 3D pose on given axis `axes` with given `axis_length`.

    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        point (gtsam.Point3): The point to be plotted.
        linespec (string): String representing formatting options for Matplotlib.
        P (numpy.ndarray): Marginal covariance matrix to plot the uncertainty of the estimation.
    """
    # get rotation and translation (center)
    gRp = pose[0:3, 0:3]  # rotation from pose to global
    origin = pose[0:3,3]

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'g-')

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin[np.newaxis], z_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'b-')

    # plot the covariance
    if P is not None:
        # covariance matrix in pose coordinate frame
        # convert the covariance matrix to global coordinate frame
        gPp = gRp @ P @ gRp.T
        plot_covariance_ellipse_3d(axes, origin, gPp, color)

def plot_covariance_ellipse_3d(axes,
                               origin: np.ndarray,
                               P: np.ndarray,
                               color:np.ndarray,
                               scale: float = 1,
                               n: int = 8,
                               alpha: float = 0.3) -> None:
    """
    Plots a Gaussian as an uncertainty ellipse

    Based on Maybeck Vol 1, page 366
    k=2.296 corresponds to 1 std, 68.26% of all probability
    k=11.82 corresponds to 3 std, 99.74% of all probability

    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        origin: The origin in the world frame.
        P: The marginal covariance matrix of the 3D point
            which will be represented as an ellipse.
        scale: Scaling factor of the radii of the covariance ellipse.
        n: Defines the granularity of the ellipse. Higher values indicate finer ellipses.
        alpha: Transparency value for the plotted surface in the range [0, 1].
    """
    k = 11.82
    K= 1.0
    U, S, _ = np.linalg.svd(P)

    radii = k * np.sqrt(S)
    radii = radii * scale
    rx, ry, rz = radii

    # generate data for "unrotated" ellipsoid
    xc, yc, zc = ellipsoid(rx, ry, rz, n)

    # rotate data with orientation matrix U and center c
    data = np.kron(U[:, 0:1], xc) + np.kron(U[:, 1:2], yc) + \
        np.kron(U[:, 2:3], zc)
    n = data.shape[1]
    x = data[0:n, :] + origin[0]
    y = data[n:2 * n, :] + origin[1]
    z = data[2 * n:, :] + origin[2]

    axes.plot_surface(x, y, z, alpha=alpha,color= color ) #cmap='hot'
    
def parse_log(data_file):
    df = pd.read_csv(data_file, header=[0],skipinitialspace=True) 
    df_poses = df[["pose_00", "pose_01", 'pose_02', 'pose_03','pose_10', 'pose_11', 'pose_12',
               'pose_13', 'pose_20', 'pose_21', 'pose_22', 'pose_23', 'pose_30', 'pose_31', 'pose_32', 'pose_33']]
    df_cov = df[['cov_00', 'cov_01', 'cov_02', 'cov_10', 'cov_11', 'cov_12', 'cov_20', 'cov_21', 'cov_22']]

    poses=[]
    covs=[]
    #extract poses
    for row in range(df_poses.shape[0]):
        p = np.array(df_poses.iloc[row, :]).reshape(4,4)
        poses.append(p)

    #extract covariances
    for row in range(df_cov.shape[0]):
        c = np.array(df_cov.iloc[row, :]).reshape(3,3)
        covs.append(c)
    FrameID = df['FrameID'].tolist()
    stats_df = df[['FrameID','stamp', 'Num_matches_KF','Num_matches_KF_lms','Num_matches_KF_new','Num_matches_localmap', 'Num_inliers_KF','Num_inliers_localmap','Num_triangulated' ]]
    Num_matches_KF = df[['FrameID', 'Num_matches_KF']]
    Num_matches_KF_lms = df[['FrameID','Num_matches_KF_lms']]
    Num_matches_KF_new = df[['FrameID','Num_matches_KF_new']]
    Num_matches_localmap = df[['FrameID','Num_matches_localmap']]
    Num_inliers_KF = df[['FrameID','Num_inliers_KF']]
    Num_inliers_localmap = df[['FrameID','Num_inliers_localmap']]
    Num_triangulated = df[['FrameID','Num_triangulated']]
    return poses, covs, stats_df
    #return poses, covs, FrameID, Num_matches_KF, Num_matches_KF_lms, Num_matches_KF_new, Num_matches_localmap, Num_inliers_KF, Num_inliers_localmap,Num_triangulated

poses_2, covs_2, stats_2 = parse_log("../../log/02_23_2022/poses_stats.txt")
poses_3, covs_3, stats_3 = parse_log("../../log/02_23_2022/poses_stats_3cams.txt")
poses_4, covs_4, stats_4 = parse_log("../../log/02_23_2022/poses_stats_4cams.txt")
poses_5, covs_5, stats_5 = parse_log("../../log/02_23_2022/poses_stats_5cams_3.txt")



def plot_poses(poses_all, covs_all):
    fig = plt.figure()
    axes = fig.gca(projection='3d')
    axis_labels=["X axis", "Y axis", "Z axis"]
    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])
    axes.set_zlabel(axis_labels[2])
    cols=[np.array([1.0,0,0]),np.array([0,1.0,0])]
    i=0
    freq = 10
    for i in range(len(poses_all)):
        #color
        poses = poses_all[i]
        covs = covs_all[i]
        col = cols[i]
        poses= poses[:132]
        covs = covs[:132]
        print("color ", col)
        print("first pose")
        print(len(poses))
        ind=0
        for p,c in zip(poses, covs):
            #print(p)
            #print(c)
            if ind % freq == 0:
                plot_pose3_on_axes(axes, p,col, P=c)
            ind = ind + 1
    
    title="covariances"
    fig.suptitle(title)
    fig.canvas.set_window_title(title.lower())
plot_poses([poses_2,poses_3, poses_4, poses_5], [covs_2, covs_3, covs_4, covs_5])
plt.show()

