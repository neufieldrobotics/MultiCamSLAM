//
// Created by auv on 6/8/20.
//

#ifndef LIGHT_FIELDS_ROS_ORBVOCABULARY_H
#define LIGHT_FIELDS_ROS_ORBVOCABULARY_H

#include "TemplatedVocabulary.h"
#include "TemplatedDatabase.h"
#include "FORB.h"
#include "FORB.h"
/// Includes all the data structures to manage vocabularies and image databases
namespace DBoW2
{
}

/// ORB Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
        ORBVocabulary;

/// FORB Database
typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB>
        ORBDatabase;

#endif //LIGHT_FIELDS_ROS_ORBVOCABULARY_H
