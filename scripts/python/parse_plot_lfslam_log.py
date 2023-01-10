import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
np.set_printoptions(precision=8,suppress=True)
import helper_functions
from scipy.spatial.transform import Rotation as R

### Use this method when monocular back-end is used for optimization

def read_and_convert(filename, conv_mat, outname):
    T_array = np.zeros((0, 4,4))
    with open(filename) as fil:
        lines = fil.readlines()
        op_array=np.zeros((len(lines), 8))
        cnt=0
        for line in lines:
            arr = line.split(' ')
            vals=np.array(arr[3:]).astype(np.float)
            print(vals)
            pose= vals.reshape((4,4))
            T_array = np.append(T_array,pose[None], axis=0)
            op_array[cnt, 0] = arr[1]
            op_array[cnt, 1:4] = pose[:3,3]
            aligned_rot = pose[:3,:3] @ conv_mat
            rotObj = R.from_matrix(aligned_rot)
            op_array[cnt, 4:] = rotObj.as_quat()
            cnt = cnt +1
    fil.close()
    np.savetxt(outname, op_array, fmt="%s")
    return T_array

### Use this method when the component cameras are also used
### in optimization and thus exist in the state ids in the log file  
def read_and_convert_allcams(filename, conv_mat, outname):
    T_array = np.zeros((0, 4,4))
    with open(filename) as fil:
        lines = fil.readlines()
        op_array=np.zeros((len(lines), 8))
        cnt=0
        for line in lines:
            arr = line.split(' ')
            camid = str(arr[2])[-1]
            poseid = str(arr[2])[:-1]
            if not camid == '0':
                continue
            vals = np.array(arr[3:]).astype(np.float)
            print(vals)
            pose = vals.reshape((4,4))
            T_array = np.append(T_array,pose[None], axis=0)
            op_array[cnt, 0] = arr[1]
            op_array[cnt, 1:4] = pose[:3,3]
            aligned_rot = pose[:3,:3] @ conv_mat
            rotObj = R.from_matrix(aligned_rot)
            op_array[cnt, 4:] = rotObj.as_quat()
            cnt = cnt +1
            print(cnt)
    fil.close()
    np.savetxt(outname, op_array, fmt="%s")
    return T_array


if __name__=="__main__":
    fig1,ax1 = helper_functions.initialize_3d_plot(number=1, limits = np.array([[-4,4],[-2,2],[0,5]]), view=[-30,-90])
    conv_mat = np.array([[-1, 0, 0],[ 0, 0, 1],[ 0, 1, 0]])
    T_array1 = read_and_convert('trajectories/poses_02_23_2022.txt', conv_mat, 'trajectories/front_end_tum.txt')

    #T_array2 = read_and_convert('trajectories/optimized_poses_mono.txt', conv_mat, 'trajectories/optimized_tum.txt')
    #T_array3 = read_and_convert_allcams('trajectories/optimized_poses_decoupled_rigid.txt', conv_mat, 'trajectories/optimized_decoupled_rigid_tum.txt')
    #T_array4 = read_and_convert_allcams('trajectories/optimized_poses_decoupled.txt', conv_mat, 'trajectories/optimized_decoupled_tum.txt')
    #T_array5 = read_and_convert_allcams('trajectories/optimized_poses_multi.txt', conv_mat, 'trajectories/optimized_multi_tum.txt')
    #T_array6 = read_and_convert_allcams('trajectories/optimized_poses_multi_rigid.txt', conv_mat, 'trajectories/optimized_multi_rigid_tum.txt')
    transX=[]
    transY=[]
    transZ=[]
    ini_ValX=0
    ini_ValY=0
    ini_ValZ=0

    with open('trajectories/filteredGroundTruth.txt') as fil:
        lines = fil.readlines()
        flg=True
        for line in lines:
            vals = np.array(line.split(' ')[1:]).astype(np.float)
            r = R.from_quat([vals[3], vals[4], vals[5], vals[6]])
            t=np.zeros((4,4))
            t[:3, :3] =r.as_matrix()
            t[:3, 3] = vals[:3]
            if flg:
                flg=False
                ini_ValX=vals[0]
                ini_ValY=vals[2]
                ini_ValZ=vals[1]
            
            transX.append(-1* (vals[0]- ini_ValX))
            transY.append((vals[2]- ini_ValY))
            transZ.append(-1*(vals[1]- ini_ValZ))        
    fil.close()

    wTa1 = np.eye(4)
    helper_functions.plot_cam_array(ax1, T_array1, wTa1, color = 'red')
    ax1.scatter3D(transX, transY, transZ, c='black', s= 1)
    #helper_functions.plot_cam_array(ax1, T_array2, wTa1, color = 'green')
    #helper_functions.plot_cam_array(ax1, T_array3, wTa1, color = 'blue')
    #helper_functions.plot_cam_array(ax1, T_array4, wTa1, color = 'magenta')
    #helper_functions.plot_cam_array(ax1, T_array3, wTa1, color = 'blue')
    #helper_functions.plot_cam_array(ax1, T_array4, wTa1, color = 'magenta')

    plt.show()


#with open('optimized_poses.txt') as fil:
#    lines = fil.readlines()
#    op_array=np.zeros((len(lines), 8))
#    cnt=0
#    for line in lines:
#        arr = line.split(' ')
#        vals = np.array(arr[3:]).astype(np.float)
#        pose = vals.reshape((4,4))
#        T_array2 = np.append(T_array2,pose[None], axis=0)
#        op_array[cnt, 0] = arr[1]
#        op_array[cnt, 1:4] = pose[:3,3]
#        aligned_rot = pose[:3,:3] @ conv_mat
#        rotObj = R.from_matrix(aligned_rot)
#        op_array[cnt, 4:] = rotObj.as_quat()
#        cnt = cnt +1
#fil.close()
#np.savetxt("optimized.txt", op_array, fmt="%s")
