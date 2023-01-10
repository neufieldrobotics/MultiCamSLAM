import numpy as np
import cv2


def draw(img, corners, imgpts):
    """
    This function draw the corners of the checkboard pattern
    on an image and returns it
    Args:
        img: numpy array image
        corners: numpy array of corners
        imgpts: numpy array of projected points

    Returns:

    """
    corners = corners.astype(np.int32)
    imgpts = imgpts.astype(np.int32)
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)
    return img


def getPoseFromChessBoard(img_inp, k, dist, squarSize, psize=(16,9)):
    """
    This method find the chessboard corners and pose of the camera
    WRT the board. returns the transform and the detected chessboard corners
    Args:
        img_inp:  numpyarray image
        k:  3x3 numppy array intrinsic parameters
        dist: 1 x 4 or 1x5 distortion parameters
        squarSize: size of the side of the square on chessboard.

    Returns: numpy arrays transform, corners

    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((psize[0]*psize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:psize[0],0:psize[1]].T.reshape(-1,2)
    objp = objp * squarSize
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    axis = axis * squarSize
    T_ci = np.zeros((4,4), np.float)
    ## detect chesssboard
    ret, corners = cv2.findChessboardCorners(img_inp, psize,None,  cv2.CALIB_CB_FAST_CHECK)
    corners2 = []
    if ret == True:
        #print("corners found")
        corners2 = cv2.cornerSubPix(img_inp,corners,(7,7),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        _,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, k, dist)
        rot_mat, _ = cv2.Rodrigues(rvecs)
        ## get pose WRT chessboard T_ci
        T_ic = np.vstack((np.hstack((rot_mat, tvecs)), np.array([0, 0, 0, 1])))
        T_ci = np.linalg.inv(T_ic)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, k, dist)
        cv2.drawChessboardCorners(img_inp, psize, corners2, ret)
        img = cv2.cvtColor(img_inp,cv2.COLOR_GRAY2RGB)
        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        cv2.waitKey(0) & 0xff
    return T_ci, corners2
