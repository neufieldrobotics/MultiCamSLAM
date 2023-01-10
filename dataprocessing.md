## Data Processing steps
## -----------------------
This file has all the processing steps required for preparing our data
to run the light field reconstruction as well as ORBSLAM and DynaSLAM 
algorithms for comparison


### calibration :
-----------------
The calibration data is colleted before each data collection run using the 9 x 16 checkerboard pattern with each square 0.81 m
The data is usually saved as images under cam0,cam1, cam2,cam3,cam4 folders.

* convert calibration images to rosbag
```bash
  git clone https://github.com/neufieldrobotics/rosbag_toolkit.git
  cd software/rosbag_toolkit/image2bag
  python image2bag.py -i /media/auv/neufrl_usb_0/calibration -c config/config.yaml
```

* Run kalibr : Make sure you have built and installed kalibr package
```bash
  git clone https://github.com/PushyamiKaveti/kalibr.git
  git checkout mpl_fix 
  cd kalibr_ws
  source devel/setup.bash
  kalibr_calibrate_cameras --target checkerboard.yaml --bag /home/auv/data/isec_2020_july/calib2/rosbag.bag --models pinhole-radtan pinhole-radtan pinhole-radtan pinhole-radtan pinhole-radtan --topics /   camera_array/cam0/image_raw /camera_array/cam1/image_raw /camera_array/cam2/image_raw /camera_array/cam3/image_raw /camera_array/cam4/image_raw
```

### Semantic Segmentation:
We can generate semantic segmnatation masks before hand for fast processing. Generation of masks can be donde using deeplab, maskRCNN or bodypix.

* deeplab :
```bash
export PYTHONPATH=$PYTHONPATH:/home/auv/software/segmentation/:/home/auv/software/segmentation/slim:/home/auv/ros_ws/src/light-fields-ros/src/python/
python pre_segment.py -i /media/auv/neufrl_usb_0/2020-07-15-20-14-52.bag -m /home/auv/software/segmentation/deeplab_model.tar.gz  -o /media/auv/neufrl_usb_0/segmasks/

```
* maskRCNN: Run segmentation is using masRCNN from dynaslam code. At one point we have to move it to our repo
```bash
 export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/auv/software/DynaSLAM/Examples/ROS
 rosrun ORB_SLAM2 Mono </media/auv/neufrl_usb_0/15_07_2020_isec_indoor/bag1/segmasks/cam0> , also can give just the path until segmasks to do segmentation on all the image frames.
```
* Bodypix :
```bash
  python pre_segment_bodypix.py -i /media/auv/neufrl_usb_0/2020-07-15-20-14-52.bag -o /media/auv/neufrl_usb_0/segmasks/
```
### stereo Reconstruction
run stereo reconstruction on the reference camera and the next camera to compute depth maps required by dynaslam to resemble RGB-D camera. This generates the rgb and depth images.

* stereo reconstruction:  Run stereo_app from the dense reconstruction package with following params 
```bash
  rosrun stereo_app
  --left_topic /camera_array/cam0/image_raw
  --right_topic /camera_array/cam1/image_raw
  --calib_file /home/auv/ros_ws/src/stereo_dense_reconstruction/calibration/15_07_2020_isec_indoor.yml
  --calib_width 720
  --calib_height 540
  --output_dir /media/auv/neufrl_usb_0/15_07_2020_isec_indoor/bag2

### ORBSLAM and DYNASLAM params

* generate rgb.txt and associations.txt using light-fields/scripts/genOrblist.sh
* export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH/ORB_SLAM2/Examples/ROS
* rosrun ORB_SLAM2 Mono Vocabulary/ORBvoc.txt /root/isec_dynaslam.yaml

### compute LF reconstruction

run app and generate refocused image and BG depth.
