//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                For Open Source Flow Visualization Library
//
// Copyright 2013-2017 Abhishek Bajpayee
//
// This file is part of OpenFV.
//
// OpenFV is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License version 2 as published by the Free Software Foundation.
//
// OpenFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License version 2 for more details.
//
// You should have received a copy of the GNU General Public License version 2 along with
// OpenFV. If not, see https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html.

#ifndef SERIALIZATION_LIBRARY
#define SERIALIZATION_LIBRARY

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;
using namespace cv;

BOOST_SERIALIZATION_SPLIT_FREE(Mat)
namespace boost {
    namespace serialization {

        /** Serialization support for cv::Mat */
        template<class Archive>
        void save(Archive & ar, const Mat& m, const unsigned int version) {
            size_t elem_size = m.elemSize();
            size_t elem_type = m.type();

            ar & m.cols;
            ar & m.rows;
            ar & elem_size;
            ar & elem_type;

            const size_t data_size = m.cols * m.rows * elem_size;
            ar & boost::serialization::make_array(m.ptr(), data_size);
        }

        /** Serialization support for cv::Mat */
        template<class Archive>
        void load(Archive & ar, Mat& m, const unsigned int version) {
            int cols, rows;
            size_t elem_size, elem_type;

            ar & cols;
            ar & rows;
            ar & elem_size;
            ar & elem_type;

            m.create(rows, cols, elem_type);

            size_t data_size = m.cols * m.rows * elem_size;
            ar & boost::serialization::make_array(m.ptr(), data_size);
        }

    }
}

BOOST_SERIALIZATION_SPLIT_FREE(Mat_<double>)
namespace boost {
    namespace serialization {

        /** Serialization support for cv::Mat_<double> */
        template<class Archive>
        void save(Archive & ar, const Mat_<double>& m, const unsigned int version) {
            size_t elem_size = m.elemSize();
            size_t elem_type = m.type();

            ar & m.cols;
            ar & m.rows;
            ar & elem_size;
            ar & elem_type;

            const size_t data_size = m.cols * m.rows * elem_size;
            ar & boost::serialization::make_array(m.ptr(), data_size);
        }

        /** Serialization support for cv::Mat_<double> */
        template<class Archive>
        void load(Archive & ar, Mat_<double>& m, const unsigned int version) {
            int cols, rows;
            size_t elem_size, elem_type;

            ar & cols;
            ar & rows;
            ar & elem_size;
            ar & elem_type;

            m = Mat_<double>(rows, cols); //m.create(rows, cols, elem_type);

            size_t data_size = m.cols * m.rows * elem_size;
            ar & boost::serialization::make_array(m.ptr(), data_size);
        }

    }
}

#endif
