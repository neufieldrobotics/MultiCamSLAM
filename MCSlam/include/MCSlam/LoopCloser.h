//
// Created by Pushyami Kaveti on 4/18/21.
//

#ifndef SRC_LOOPCLOSER_H
#define SRC_LOOPCLOSER_H

///Class which takes care of the loop closures
/// At this point iot is only used
/// 1. insert the LF frames into the database which stores TF-IDF
/// 2. find the similar images after the trajectory is run

#include "MCSlam/ORBVocabulary.h"

using namespace std;
using namespace DBoW2;

class LoopCloser {

public:
    LoopCloser(ORBVocabulary* voc);
    void addToDatabase(BowVector bow, FeatureVector featvec,  double tStamp);
    void queryDatabase(BowVector queryBow,int topN, vector<double>& bowScores,vector<double>& stamps);


    ORBVocabulary* orb_vocabulary;
    ORBDatabase* orbDatabase;
    vector<double> entryTimeStamps;


};


#endif //SRC_LOOPCLOSER_H
