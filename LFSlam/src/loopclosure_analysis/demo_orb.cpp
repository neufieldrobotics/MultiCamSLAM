//
// Created by pushyami kaveti on 4/22/21.
//

#include <iostream>
#include <vector>
#include <string>


// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines BriefVocabulary
#include "DLoopDetector.h" // defines BriefLoopDetector
#include "ORBextractor.h"
#include "demoDetector.h"


// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

static const char *VOC_FILE = "/home/auv/ros_ws/src/light-fields/params/ORBvoc.txt";
static const char *IMAGE_DIR = "/home/auv/datasets/04_30_2021_library/bag4/refocused/rgb_original";
static const char *SEGMASK_DIR = "/home/auv/datasets/04_30_2021_library/bag4/refocused/segmask";
static const char *POSE_FILE = "/home/auv/software/DLoopDetector/build/resources/pose_rgb.txt";
static const int IMAGE_W = 720; // image size
static const int IMAGE_H = 540;


int main(int argc, char* argv[])
{
    bool show = true;
    if (argc > 1 && std::string(argv[1]) == "-noshow")
        show = false;

    // prepares the demo
    demoDetector<OrbVocabulary , OrbLoopDetector , FORB::TDescriptor>
            demo(VOC_FILE, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H, SEGMASK_DIR);

    try
    {
        int nFeatures = 1000;
        float fScaleFactor = 1.2;
        int nLevels = 8;
        int fIniThFAST = 20;
        int fMinThFAST = 7;

        // create the object to compute ORB features
        ORBextractor extractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
        demo.run("ORB", extractor);
    }
    catch(const std::string &ex)
    {
        cout << "Error: " << ex << endl;
    }

    return 0;
}
