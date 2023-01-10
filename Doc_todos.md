Required packages 
------------------
ros
opencv
cv_bridge in ros package vision_opencv
Eigen
gflags
DBoW2
DLib
opengv
Pangolin
Boost
cuda

Installation Instructions
-------------------------
* Add recurive git submodules repos to be downloaded - vision_opencv,DBoW2, DLib, also opencv, opengv, Pangolin. these are for opencv dependencies if we use cuda.
* Change CMakelists such that there is no need to specify various package root directories
* May be add findPackage.cmakes for al the other packages
* complete the installation instructions

Execution Instructions
-------------------------
* explain the parameters in the .cfg files 
* explain the other param files i.e kalibr, front-end params, reconstruction params
* instructions to donwload sample data
* instructions to run and trobleshoot

Others
------
* Make sure docker works properly.
* Setup CI using travis.

