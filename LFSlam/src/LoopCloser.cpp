//
// Created by Pushyami Kaveti on 4/18/21.
//

#include "LFSlam/LoopCloser.h"



LoopCloser::LoopCloser(ORBVocabulary* voc): orb_vocabulary(voc) {
    orbDatabase =  new ORBDatabase(*orb_vocabulary, true , 2);
}

void LoopCloser::addToDatabase(BowVector bow, FeatureVector featvec, double tStamp){
    assert(!bow.empty() and !featvec.empty());
    EntryId img_id = orbDatabase->add(bow, featvec);
    entryTimeStamps.push_back(tStamp);
    cout<<"Image inserted at : "<<img_id<<endl;

}
void LoopCloser::queryDatabase(BowVector queryBow,int topN, vector<double>& bowScores, vector<double>& stamps){
     assert(!queryBow.empty() and topN > 0);
     bowScores.clear();
     stamps.clear();
     QueryResults results;
     orbDatabase->query(queryBow, results, topN , -1);

     for( auto& res : results){
         bowScores.push_back(res.Score);
         stamps.push_back(entryTimeStamps.at(res.Id));
     }

}

