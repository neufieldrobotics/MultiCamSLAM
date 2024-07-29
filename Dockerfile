FROM osrf/ros:noetic-desktop

# Install some basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-all-dev \
    libgoogle-perftools-dev \
    google-perftools \
    libatlas-base-dev \
    libsuitesparse-dev \
    libyaml-cpp-dev \
    wget \
    unzip \
    libgoogle-glog-dev \
    libgflags-dev

RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws/src
RUN git clone https://github.com/neufieldrobotics/MultiCamSLAM

RUN mkdir -p /root/catkin_ws/ThirdParty && \
    cd /root/catkin_ws/ThirdParty && \
    git clone -b 4.5.5 https://github.com/opencv/opencv.git && \
    git clone -b 4.5.5 https://github.com/opencv/opencv_contrib.git

RUN mkdir -p /root/catkin_ws/ThirdParty/opencv/build && \
    cd /root/catkin_ws/ThirdParty/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=../../opencv/install \
    -D WITH_OPENGL=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules .. && \
    make install -j $(nproc)

RUN cd /root/catkin_ws/ThirdParty && \
    wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz && \
    tar zxf eigen-3.3.7.tar.gz && \
    mv eigen-3.3.7 eigen && \
    rm eigen-3.3.7.tar.gz

RUN mkdir -p /root/catkin_ws/ThirdParty/eigen/build && \
    cd /root/catkin_ws/ThirdParty/eigen/build && \
    cmake .. && \
    make install -j $(nproc)

RUN cd /root/catkin_ws/ThirdParty && \
    wget https://github.com/borglab/gtsam/archive/refs/tags/4.1.1.zip && \
    unzip 4.1.1.zip && \
    rm 4.1.1.zip && \
    mv gtsam-4.1.1 gtsam 

RUN mkdir -p /root/catkin_ws/ThirdParty/gtsam/build && \
    cd /root/catkin_ws/ThirdParty/gtsam/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    make install -j $(nproc)

RUN cd /root/catkin_ws/ThirdParty && \
    git clone https://github.com/laurentkneip/opengv

RUN mkdir -p /root/catkin_ws/ThirdParty/opengv/build && \
    sed -i '2i add_compile_options(-std=c++17)' /root/catkin_ws/ThirdParty/opengv/CMakeLists.txt && \
    cd /root/catkin_ws/ThirdParty/opengv/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    make install -j $(nproc) && \
    make test 

RUN cd /root/catkin_ws/ThirdParty && \ 
    git clone https://github.com/PushyamiKaveti/DBoW2

RUN mkdir -p /root/catkin_ws/ThirdParty/DBoW2/build && \
    cd /root/catkin_ws/ThirdParty/DBoW2/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install && \
    make install -j $(nproc)

RUN cd /root/catkin_ws/ThirdParty && \
    git clone https://github.com/dorian3d/DLib

RUN mkdir -p /root/catkin_ws/ThirdParty/DLib/build && \
    cd /root/catkin_ws/ThirdParty/DLib/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install && \
    make install -j $(nproc)


RUN cd /root/catkin_ws/ThirdParty && \
    git clone --recursive https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && echo "y" | ./scripts/install_prerequisites.sh required && \
    cmake -B build && \
    cmake --build build && \
    ctest

RUN cd /root/catkin_ws && \
    . /opt/ros/noetic/setup.sh && \
    catkin_make -DOpenCV_DIR=/root/catkin_ws/ThirdParty/opencv/build \
    -DDBoW2_DIR=/root/catkin_ws/ThirdParty/DBoW2/build \
    -DDLib_DIR=/root/catkin_ws/ThirdParty/DLib/build \
    -Dopengv_DIR=/root/catkin_ws/ThirdParty/opengv/build \
    -DGTSAM_DIR=/root/catkin_ws/ThirdParty/gtsam/build \
    -DGTSAM_UNSTABLE_DIR=/root/catkin_ws/ThirdParty/gtsam/build  \
    -Dopengv_INC_DIR=/root/catkin_ws/ThirdParty/opengv/include \
    -DCUDA_TOOLKIT_INCLUDE=/home/$USER \
    -DCUDA_CUDART_LIBRARY=/home/$USER \


ENTRYPOINT [ "/bin/bash" ]