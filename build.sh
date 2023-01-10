#!/bin/bash


catkin_make -DOpenCV_DIR=/home/auv/software/opencv/build  -DDBoW2_DIR=/home/auv/software/DBoW2/build -DDLib_DIR=/home/auv/software/DLib/build -Dopengv_DIR=/home/auv/software/opengv/build -DGTSAM_DIR=/home/auv/software/gtsam/build -DGTSAM_UNSTABLE_DIR=/home/auv/software/gtsam/build -Dopengv_INC_DIR=/home/auv/software/opengv/include

