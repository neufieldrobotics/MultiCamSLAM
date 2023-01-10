# Get Trajectories
* Run ORBSLAM2 Mono, Stereo and DynaSLAM RGBD on the dataset

* make directories and move the trajectory files there



* Ground Truth Processing
-----------------------------
* convert the groundtruth rosbag to txt file - light fields scripts

cd ~/ros_ws/src/light-fields/scripts
python ros2txt_pose.py -i ~/datasets/24_09_2020_isec_indoor_vio/bag10/2020-09-24-17-30-03.bag -t /mocap_node/Robot_1/pose  -o ~/datasets/24_09_2020_isec_indoor_vio/bag10/results/GroundTruth.txt


* associate and cleanup the groundtruth for each trajectory to remove noise . This should be done manually as the groundtruth sometimes has errors which arent removable automatically and move them into the folder - evo lffrontend eval

* Align the groundtruth and estimated trajectory first pose - evo lffrontend eval


* Run the script to estimate ape, rpe and save the plots
* Obtain box plots for errors for all the algos for multiple runs - evo lffrontend eval

* consolidate the results to obtain median, mean of various estimates - evo lffront-end eval


Vocabulary/ORBvoc.txt /home/auv/datasets/24_09_2020_isec_indoor_vio/calibration/24_09_2020_isec_indoor_orb_stereo.yaml /home/auv/datasets/24_09_2020_isec_indoor_vio/bag10/rgbd /home/auv/datasets/24_09_2020_isec_indoor_vio/bag10/rgbd/rgbd_assoc.txt /home/auv/datasets/24_09_2020_isec_indoor_vio/bag10/rgbd/masks /home/auv/datasets/24_09_2020_isec_indoor_vio/bag10/rgbd/test /home/auv/datasets/24_09_2020_isec_indoor_vio/bag10/results/DynaSLAM/DynaSLAM1.txt

