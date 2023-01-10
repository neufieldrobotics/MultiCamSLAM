This document details the instructions of collecting Cam array data and the groundtruth.

###Required Packages
docker
mocap_optitrack
chrony or ntp 
motive software (optitrack GUI)

### Groundtruth collection
**skip this section if you do not have optitrack for groundtruth pose This is not a tutorial to setup optitrack system, rather some notes to refresh the todos.

### optitrack setup 
* Open motive software on optitrack system. 
* Calibrate the system as the cameras might have moved.
   * clear and re-mask any reflective surfaces
   * make sure the wand config is correct (500 or 250) on motive and start wanding.
   * make sure to collect uniform samples across cameras and calibrate.
   * Use the triangle to setup the ground plan. place the triangle (short side is x and long side is z axes respectively) 
   on the ground in the orientation you desire. select the markers in motive software, go to the ground plane tab and 
   select the ground plane. You should see the ground plane to have moved to align with the triangle. 
* Make sure a rigid body exists for your robot or sensor. If it is not present create one.
   * Stick reflective markers to the rigid body in somewhat asymmetrical way. This is because the optitrack should be able
   to disambiguate between different poses of the rigid body.
   * Select the markers in the motive software and create a rigid body out of it.
   * you can select one of the markers to represent your body frame and adjust the axis as you want in the motive software.

###Time sync
Before recording the optitrack data the time must be synced between the optitrack system and the system on which the sensor data is collected. This is important because for applications like SLAM we would like to extract the poses at specuiic time steps and evaluate the estimated trajectory.

You can do so using software likr chrony or ntp.

NEUFR Lab optitrack system already has the ntp server setup. Run <app name> application. You can specify the server's IP address (192.168.1.2) or any other server you would like. Make sure that the sensorshystem also sync with the same server.


### Data streaming and recording
* Select what data you want to record - assets etc
* You can click the red record buton to record the data on disk on the optitrack system as a csv.
* you can also stream the data over network and use ros package to record the data in the form of a rosbag.
     * get th mocap_optitrack ros package and install it on the client side i.e the system on which you would like to collect the data Usually this is the sensor system or the robot.
     * make sure the two systems are on the same network.
     * make changes to the config file in mocap_optitrack package - 
     * make sure multicast is enabled on motive softare and correct IP address is selected instead of loopback under
     streaming tab on motive.
* roslaunch the mocap_optitrack package, make sure the data is being streamed and record the rosbag.

##Data collection
### Via Docker Image

We have a docker image which has all the required packages installed to carry out the datacollection. 

From terminal issue these commands
```bash
$ xhost +local:root 

or

$xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerId`

$docker login (if you are not logged in)
$docker pull neufieldrobotics/lightfields:latest
```
Check the docker container ID
```bash
$docker ps -a 
$docker start $containerId
$docker run --privileged  -it -e "DISPLAY=unix:0.0" -v="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /dev:/dev -p "8888:8888"  --runtime=nvidia neufieldrobotics/lightfields /bin/bash
```
Run the spinnaker ros package on docker
```bash
roslaunch spinnaker_ros acquisition.launch
```
By default the acqusition is setup for the 5 camera linear array. you can give your own camera config file. Refer to the
spinnaker ros package for more details. <Link>



Record the rosbags

Move the bags from docker container to the pc's disk

```bash
docker cp -----
``` 


