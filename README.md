# MultiCam Visual Odometry

### Version 0.1, December 27th, 2022
**Authors:** Pushyami Kaveti et al.

<!-- Description of the work .. -->
<br/>

# 1. Getting started

## A. Prerequisites

We have tested this library in Ubuntu 16.04 and 20.04. 
The following external libraries are required for building Multicam Visual Odometry 
package. We have given the build/install instructions in the next section. 

**Dependencies:**
- Opencv 3.3.1/4.5.5 
- ROS Kinetic/Noetic
- Boost
- Eigen3
- GTSAM 4
- opengv
- gflags
- glog
- DBoW2
- DLib
- Pangolin
- python 2.7/3.8
- numpy
- YAML

##### [The entire package list can be built by writing a build.sh file.] 

## B. Build Instructions
### 1. <b>ROS</b>
Instructions to install ROS can be found in the links below: <br/>
- ROS Noetic
    -  http://wiki.ros.org/noetic/Installation/Ubuntu
- ROS Kinetic
    - http://wiki.ros.org/kinetic/Installation/Ubuntu

### 2. <b>Clone the repo</b>
- Create a ROS catkin workspace 
- Here on, we will assume that your catkin workspace location is ~/catkin_ws. 
    
    ```
    cd ~/catkin_ws/src
    ```
    ```
    git clone https://gitlab.com/neufieldrobotics/light-fields-pack
    ``` 

### 3. <b>OpenCV </b>
- **Tested with OpenCV 3.3.1 and 4.5.5**.
- For Ubuntu 20.04 + OpenCV 4.5.5 follow the below instructions. 
    ```
    sudo apt update && sudo apt install -y cmake 

    mkdir ~/catkin_ws/ThirdParty && cd ~/catkin_ws/ThirdParty

    git clone https://github.com/opencv/opencv.git
    git clone https://github.com/opencv/opencv_contrib.git
    cd opencv
    git checkout 4.5.5
    cd ../opencv_contrib
    git checkout 4.5.5
    cd ../opencv
    mkdir build && cd build

    cmake -D CMAKE_BUILD_TYPE=RELEASE   -D CMAKE_INSTALL_PREFIX=../../opencv/install -D CMAKE_BUILD_TYPE=RELEASE  -D WITH_OPENGL=ON       -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..

    make install -j 4
  ```

### 4. <b>Boost</b> 
-
    ```
    apt-get install cmake build-essential libboost-all-dev libgoogle-perftools-dev google-perftools  libatlas-base-dev libsuitesparse-dev libyaml-cpp-dev
    ```

### 5. <b> Eigen3 </b>
- 
    ```
    apt-get install wget unzip
    cd ~/catkin_ws/ThirdParty  
    wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
    tar zxf eigen-3.3.7.tar.gz
    mv eigen-3.3.7.tar.gz eigen
    ```
- In Eigen's CMakeLists.txt file, add the following below cmake_minimum_required

    ```
    add_compile_options(-std=c++17)
    ```
- 
    ```
    cd ~/catkin_ws/ThirdParty/eigen
    cd eigen
    mkdir build && cd build
    cmake .. 
    sudo make install
    ```

### 6. <b>GTSAM 4 </b>
-
    ```
    cd ~/catkin_ws/ThirdParty  
    wget https://github.com/borglab/gtsam/archive/refs/tags/4.1.1.zip 
    unzip 4.1.1.zip && rm 4.1.1.zip
    mv gtsam-4.1.1 gtsam
      
    cd gtsam
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install
    make check
    make install

    ```
### 7. <b>OpenGV</b>
- 
    ```
    cd ~/catkin_ws/ThirdParty 
    git clone https://github.com/laurentkneip/opengv
    ```
- In Opengv's CMakeLists.txt file, add the following below cmake_minimum_required
    ```
    add_compile_options(-std=c++17)
    ```
    
- 
    ```
    cd opengv
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install
    make install
    ```
- Run tests (Recommended)
    ```
    make test
    ```
### 8. <b> gflags and glog </b>
- This would have been installed as a part of the OpenCV build. If not, run this:

    -
        ```
        apt -y install libgoogle-glog-dev libgflags-dev
        ```

### 9. <b>DBoW2</b>
-  
    ```
    cd ~/catkin_ws/ThirdParty 
    git clone https://github.com/PushyamiKaveti/DBoW2
    cd DBoW2
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install
    make install
    ```

### 10. <b>DLib</b>
- Make sure you download the right one - there are 2 similar repositories - DLib and dlib
- 
    ```
    cd ~/catkin_ws/ThirdParty 
    git clone https://github.com/dorian3d/DLib
    cd DLib
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install
    make install
    ```
### 11. <b> Pangolin </b> 
- Build instructions can be found in this link: https://github.com/stevenlovegrove/Pangolin
- 
    ```
    cd ~/catkin_ws/ThirdParty 
    git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
    cd Pangolin

    ./scripts/install_prerequisites.sh recommended
    cmake -B build
    cmake --build build

    ctest
    ```

## C. Compile the ROS package
###  <b> catkin_make </b>

- 
    ```
    catkin_make -DOpenCV_DIR=/home/$USER/catkin_ws/Third_party/opencv/build \
    -DDBoW2_DIR=/home/$USER/catkin_ws/Third_party/DBoW2/build \
    -DDLib_DIR=/home/$USER/catkin_ws/Third_party/DLib/build \
    -Dopengv_DIR=/home/$USER/catkin_ws/Third_party/opengv/build \
    -DGTSAM_DIR=/home/$USER/catkin_ws/Third_party/gtsam/build \
    -DGTSAM_UNSTABLE_DIR=/home/$USER/catkin_ws/Third_party/gtsam/build  \
    -Dopengv_INC_DIR=/home/$USER/catkin_ws/Third_party/opengv/include 
    ```

# 2. Running an example

## a. Download the dataset
- Download the sample dataset from [here](https://drive.google.com/drive/folders/151_ifKEE8WYHAeZ9hGcC69iotIpevBf8?usp=sharing).

## b. Setup the config files

- edit the following paramters in src/light-fields-pack/LFApps/params/lf_frontend.yaml
    - <u>LogDir </u> - Provide the path to save the log files
        - Example: ~/catkin_ws/src/light-fields-pack/log/ 
    - <u>Vocabulary </u> - Provide the path to the ORB Vocabulary files
        - Example: ~/catkin_ws/src/light-fields-pack/LFApps/params/ORBvoc.txt

- edit the following parameters in src/light-fields-pack/LFApps/params/lf_slam_config.cfg
    - <u> data_path </u> - Provide the path to the downloaded dataset
        - Example: /home/marley/catkin_ws/ISEC_Lab1/
    - <u> calib_file_path </u> - Provide the path to the settings file for a particular multi-camera rig.
        - Example: /home/marley/catkin_ws/ISEC_Lab1/calib/02_23_2022_5cams_camchain.yaml
    - <u> images_path </u> - Provide the path to the images folder in the downloaded dataset
        - Example: /home/marley/catkin_ws/ISEC_Lab1/image_data/
    - <u> frontend_params_file </u> - Provide the complete path to the lf_frontend.yaml file in LFApps/params of the package.
        - Example: /home/marley/catkin_ws/src/light-fields-pack/LFApps/params/lf_frontend.yaml 
    - <u> backend_params_file </u> - Provide the complete path to the lf_backend.yaml file in LFApps/params of the package.
        - Example: /home/marley/catkin_ws/src/light-fields-pack/LFApps/params/lf_backend.yaml

## c. Run 

In Terminal 1
- 
    roscore
 
In Terminal 2
Edit the below command based on the path to the cfg file. 
-  
    ./devel/lib/LFApps/LFSlamapp --config_file /home/marley/neu_ws/src/light-fields-pack/LFApps/params/lf_slam_config.cfg --log_file /home/marley/log.txt --traj_file /home/marley/traj.txt











