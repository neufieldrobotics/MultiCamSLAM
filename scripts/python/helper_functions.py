#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from matplotlib import pyplot as plt
#from scipy import spatial
import matplotlib.patches as mpatches
import sys
from random_geometry_points.sphere import Sphere

def theta_2_rot2d(theta):
    ''' 
    Return 2D rotation matrix for angle given in radians
    If theta is a scalar, a 2x2 rot matrix is returned
    if theta is a N, array, then Nx2x2 array is returned
    '''
    c, s = np.cos(theta), np.sin(theta)
    if np.isscalar(theta):        
        return np.array(((c,-s), (s, c)))
    else:
        R = np.empty((len(theta),2,2))
        R[:,0,0]=c
        R[:,1,1]=c
        R[:,1,0]=s
        R[:,0,1]=-s
        return R

def rot2d_2_theta(R):
    ''' 
    Return rotation angle radians given a 2D rotation matrix
    '''
    return np.arctan2(R[1,0],R[0,0])

def compose_T(R,t):
    '''
    Given Rotation and translation return homogenous transformation matrix.

    Input
      R: 2x2 or 3x3 square array Rotation matrix
      t: 1x2 or 1x3 array rep. translation
     
    Return
      T: 3x3 or 4x4 Homogenous transformation matrix
    '''
    #return np.vstack((np.hstack((R,t)),np.array([0, 0, 0, 1])))
    if R.ndim == 2:
        Rt = np.hstack((R,t.T))
        T = np.vstack((Rt,np.zeros((1,R.shape[1]+1))))
        T[-1,-1]=1
        return T
    elif R.ndim == 3:
        T = np.zeros((len(R),R.shape[1]+1,R.shape[1]+1))
        T[:,:-1,:-1] = R
        T[:,:-1,-1] = t
        T[:,-1,-1] = 1.0
        return T

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def skew(v):
    return  np.array([[   0,  -v[2],  v[1] ],
                      [ v[2],    0 , -v[0] ],
                      [-v[1],  v[0],    0  ]])

def euclid_to_homo(v):
    return np.append(v, np.ones((1,v.shape[1])), axis=0)

def homo_to_euclid(v):
    v = v / v[-1]
    return v[:-1] 

def normal2rot(a, b=np.array([0.0,0.0,1.0])):
    '''
    Return Rotation matrix that rotates a vector to z-axis
    '''
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    v_skew = skew(v)
    R = np.eye(3) + v_skew + v_skew@v_skew * (1-c)/s**2
    return R

def plot_pose3_array_on_axes(axes, T_array, axis_length=0.1):
    line_obj_2d_list = []
    for T in T_array:
        line_obj_2d_list.append(plot_pose3RT_on_axes(axes, *decompose_T(T), axis_length))
    return line_obj_2d_list

def plot_pose3_on_axes(axes, T, axis_length=0.1, center_plot=False, line_obj_list=None):
    """Plot a 3D pose 4x4 homogenous transform  on given axis 'axes' with given 'axis_length'."""
    return plot_pose3RT_on_axes(axes, *decompose_T(T), axis_length, center_plot, line_obj_list)

def plot_pose3RT_on_axes(axes, gRp, origin, axis_length=0.1, center_plot=False, line_obj_list=None):
    """Plot a 3D pose on given axis 'axes' with given 'axis_length'."""
    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    linex = np.append(origin, x_axis, axis=0)
    
    y_axis = origin + gRp[:, 1] * axis_length
    liney = np.append(origin, y_axis, axis=0)

    z_axis = origin + gRp[:, 2] * axis_length
    linez = np.append(origin, z_axis, axis=0)


    if line_obj_list is None:
        xaplt = axes.plot(linex[:, 0], linex[:, 1], linex[:, 2], 'r-')    
        yaplt = axes.plot(liney[:, 0], liney[:, 1], liney[:, 2], 'g-')    
        zaplt = axes.plot(linez[:, 0], linez[:, 1], linez[:, 2], 'b-')
    
        if center_plot:
            center_3d_plot_around_pt(axes,origin[0])
        return [xaplt, yaplt, zaplt]
    
    else:
        line_obj_list[0][0].set_data(linex[:, 0], linex[:, 1])
        line_obj_list[0][0].set_3d_properties(linex[:,2])
        
        line_obj_list[1][0].set_data(liney[:, 0], liney[:, 1])
        line_obj_list[1][0].set_3d_properties(liney[:,2])
        
        line_obj_list[2][0].set_data(linez[:, 0], linez[:, 1])
        line_obj_list[2][0].set_3d_properties(linez[:,2])

        if center_plot:
            center_3d_plot_around_pt(axes,origin[0])
        return line_obj_list

def plot_3d_points(axes, vals, line_obj=None, *args, **kwargs):
    if line_obj is None:
        graph, = axes.plot(vals[:,0], vals[:,1], vals[:,2], *args, **kwargs)
        return graph

    else:
        line_obj.set_data(vals[:,0], vals[:,1])
        line_obj.set_3d_properties(vals[:,2])
        return line_obj


def decompose_T(T_in):
    return T_in[:3,:3], T_in[:3,[-1]].T

def pose_inv(R_in, t_in):
    t_out = -np.matmul((R_in).T,t_in)
    R_out = R_in.T
    return R_out,t_out

def T_inv(T_in):
    R_in = T_in[:3,:3]
    t_in = T_in[:3,[-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out,t_in)
    return np.vstack((np.hstack((R_out,t_out)),np.array([0, 0, 0, 1])))

def patch_legend(ax, *args, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    patches = []
    for handle, label in zip(handles, labels):
        patches.append(mpatches.Patch(color=handle.get_color(), label=label))
    legend = ax.legend(handles=patches, *args, **kwargs)

def initialize_3d_plot(number=None, title='Plot', axis_labels=['x', 'y', 'z'],view=[None,None],limits=None):
    fig = plt.figure(number)
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(0,0,1,1) # Make the plot tight
    fig.suptitle(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    if not (limits is None):
        ax.set_xlim(*limits[0,:])
        ax.set_ylim(*limits[1,:])
        ax.set_zlim(*limits[2,:])

    ax.view_init(*view)
    if sys.version_info[:2] == (3, 5):
        ax.set_aspect('equal')
    return fig,ax

def initialize_2d_plot(number=None, title='Plot', axis_labels=['x', 'y'],axis_equal=False):
    fig = plt.figure(number)
    fig.clf()
    ax = fig.add_subplot(111)  
    fig.suptitle(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    if axis_equal:
        ax.axis('equal')

    #fig.subplots_adjust(0.1,0.1,.9,.9) # Make the plot tight
    return fig,ax

def initialize_2d_plot_multi(number=None,num_rows=1,num_cols=2, title='Plot', axis_labels=['x', 'y'],axis_equal=False):
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.suptitle(title)
    for ax in axs:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        if axis_equal:
            ax.axis('equal')
    #fig.subplots_adjust(0.1,0.1,.9,.9) # Make the plot tight
    return fig,axs[np.newaxis, :]

def rectangle_xy(width = 1, height = 1):
    return np.array([[ width/2,  height/2, 0],
                     [-width/2,  height/2, 0],
                     [-width/2, -height/2, 0],
                     [ width/2, -height/2, 0]])
    
def transform_3d_pts(pts, T):
    '''
    Take 3D points a Nx3 array along with 4 x 4 Homogenous Tranform and return 
    transformed points
    '''
    if pts.ndim == 1:
        pts2d = np.expand_dims(pts, axis=0)
        hom_pts2d = np.column_stack((pts2d, np.ones(pts2d.shape[0])))
        trans_pts_hom2d = hom_pts2d @ T.T
        return trans_pts_hom2d[0,:-1]
    else:
        hom_pts = np.column_stack((pts, np.ones(pts.shape[0])))
        trans_pts_hom = hom_pts @ T.T
        return trans_pts_hom[:,:-1]

def plot_3d_rect(axes, vertices, *args, **kwargs):
     axes.plot(vertices[[0,-1],0], vertices[[0,-1],1], vertices[[0,-1],2], *args, **kwargs)
     axes.plot(vertices[0:2,0], vertices[0:2,1], vertices[0:2,2], *args, **kwargs)
     axes.plot(vertices[1:3,0], vertices[1:3,1], vertices[1:3,2], *args, **kwargs)
     axes.plot(vertices[2:4,0], vertices[2:4,1], vertices[2:4,2], *args, **kwargs)
     
def lines_between_rect(axes, vertices1, vertices2, *args, **kwargs):
    for v1,v2 in zip(vertices1,vertices2):
        pts = np.vstack((v1,v2))
        axes.plot(pts[:,0], pts[:,1], pts[:,2], *args, **kwargs)

def plot_camera(axes, pose_wT_cam, f=1.0, *args, **kwargs):
   
    opt_cent_rect_P_cam = rectangle_xy(width=0.1, height=0.1)
    img_plane_rect_P_cam = rectangle_xy(width=0.2, height=0.2)+np.array([0,0,f])
    opt_cent_rect_P_w = transform_3d_pts(opt_cent_rect_P_cam, pose_wT_cam)
    img_plane_rect_P_w = transform_3d_pts(img_plane_rect_P_cam, pose_wT_cam)
    plot_3d_rect(axes, opt_cent_rect_P_w, *args, **kwargs )
    plot_3d_rect(axes, img_plane_rect_P_w, *args, **kwargs )
    lines_between_rect(axes, opt_cent_rect_P_w, img_plane_rect_P_w, *args, **kwargs)
    plot_pose3_on_axes(axes, pose_wT_cam, axis_length=0.05)

def plot_3Dvector(axes, V, origin = np.zeros(3)):
    '''
    (x,y,z) = (x0, y0, z0) + t(a, b, c)
    X = X0 + tV
    Here t is a parameter describing a particular point on the line L
    Parametric Form
    x = x0 + t_a
    y = y0 + t_b
    z = z0 + t_c
    
    Interzection with plane z = z1
    z1 = z0 + t_c
    t1 = (z1 - z0)/c
    '''
    X0 = origin
    z1, z2 = axes.get_zlim3d()
    
    z0 = X0[2]
    c = V[2]
    t1 = (z1 - z0)/c
    t2 = (z2 - z0)/c
    
    pt1 = X0 + t1*V
    pt2 = X0 + t2*V
    
    pts = np.vstack((pt1,pt2))
    axes.plot(pts[:,0], pts[:,1], pts[:,2], color='grey',linewidth = 1 )
    
def plot_cam_array(axes, T_array, wTa, *args, **kwargs):
    '''
    Plot a camera array where:
      T_array is N x 4 x 4 array defining the camera poses in the array
      wTa is the position of the array in the world reference frame
    '''
    wT_array = wTa @ T_array
    for cam_num, wTc in enumerate(wT_array):
        plot_camera(axes, wTc, f=0.1, *args, **kwargs)
        if cam_num>0:
            wtc_prev = wT_array[cam_num-1]
            pts = np.vstack((wTc[:-1,-1], wtc_prev[:-1,-1]))
            axes.plot(pts[:,0],pts[:,1],pts[:,2], *args, **kwargs)

def generate_3D_points(num_points,typ="sphere", *args, **kwargs):
    sphere = Sphere(kwargs['x_c'], kwargs['y_c'], kwargs['z_c'], kwargs['rad'])
    random_sphere_points = sphere.create_random_points(num_points)
    pp = np.array(random_sphere_points)
    # normal_vec = (0, 0.0, 1.0)
    #ref_point = (0.0, 0.0, 5.0)
    #plane = Plane(normal_vec, 5.0, ref_point, 10.0)
    #random_plane_points = plane.create_random_points(50)
    return pp.T
