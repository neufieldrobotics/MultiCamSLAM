//
// Created by pushyami on 2/7/22.
//

#ifndef SRC_GTSAMFACTORHELPERS_H
#define SRC_GTSAMFACTORHELPERS_H

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Cal3_S2.h>

using namespace gtsam;
using namespace std;

/**
 * Unary factor on the unknown pose, resulting from meauring the projection of
 * a known 3D point in the image
 */
class ResectioningFactor: public NoiseModelFactor1<Pose3> {
    typedef NoiseModelFactor1<Pose3> Base;

    Cal3_S2::shared_ptr K_; ///< camera's intrinsic parameters
    Point3 P_;              ///< 3D point on the calibration rig
    Point2 p_;              ///< 2D measurement of the 3D point

public:

    /// Construct factor given known point P and its projection p
    ResectioningFactor(const SharedNoiseModel& model, const Key& key,
                       const Cal3_S2::shared_ptr& calib, const Point2& p, const Point3& P) :
            Base(model, key), K_(calib), P_(P), p_(p) {
    }

    /// evaluate the error
    Vector evaluateError(const Pose3& pose, boost::optional<Matrix&> H =
    boost::none) const override {
        PinholeCamera<Cal3_S2> camera(pose, *K_);
        return camera.project(P_, H, boost::none, boost::none) - p_;
    }
};



class RigResectioningFactor: public NoiseModelFactor1<Pose3>{
    typedef NoiseModelFactor1<Pose3> Base;
    Cal3_S2::shared_ptr K_; ///< camera's intrinsic parameters
    Point3 P_;              ///< 3D point on the calibration rig
    Point2 p_;              ///< 2D measurement of the 3D point
    int octave_;
    int cameraID_;
    boost::optional<Pose3> body_P_sensor_;///< The pose of the sensor in the body frame

    // verbosity handling for Cheirality Exceptions
    bool throwCheirality_; ///< If true, rethrows Cheirality exceptions (default: false)
    bool verboseCheirality_; ///< If true, prints text for Cheirality exceptions (default: false)

public:

    /// Construct factor given known point P and its projection p
    RigResectioningFactor(const SharedNoiseModel& model, const Key& key,
                          const Cal3_S2::shared_ptr& calib, const Point2& p,
                          const Point3& P, int octave, int cameraID, boost::optional<Pose3> body_P_sensor = boost::none) :
            Base(model, key), K_(calib), P_(P), p_(p), octave_(octave), cameraID_(cameraID), body_P_sensor_(body_P_sensor),
            throwCheirality_(false), verboseCheirality_(false) {
    }

    /** Virtual destructor */
    ~RigResectioningFactor() override {}

    /// evaluate the error
    Vector evaluateError(const Pose3& pose, boost::optional<Matrix&> H =
    boost::none) const override {
        try{
            if(body_P_sensor_) {
                if(H) {
                    gtsam::Matrix H0;
                    PinholeCamera<Cal3_S2> camera(pose.compose(*body_P_sensor_, H0), *K_);
                    Point2 reprojectionError(camera.project(P_, H, boost::none, boost::none) - p_);
                    *H = *H * H0;
                    return reprojectionError;
                } else {
                    PinholeCamera<Cal3_S2> camera(pose.compose(*body_P_sensor_), *K_);
                    return camera.project(P_, H, boost::none, boost::none) - p_;
                }
            } else {
                PinholeCamera<Cal3_S2> camera(pose, *K_);
                return camera.project(P_, H, boost::none, boost::none) - p_;
            }
        }
        catch(CheiralityException& e){
            if (H) *H = Matrix::Zero(2,6);
            if (throwCheirality_)
                throw CheiralityException(this->key());
        }
        return Vector2::Constant(2.0 * K_->fx());
    }


    /** return the measurement */
    const Point2& measured() const {
        return p_;
    }

    /** return the measurement */
    const Point3& landmark() const {
        return P_;
    }

    /** return the measurement */
    const int& octave() const {
        return octave_;
    }

    /** return the measurement */
    const int& cameraID() const {
        return cameraID_;
    }

    /** return the calibration object */
    const boost::shared_ptr<Cal3_S2> calibration() const {
        return K_;
    }

    /** return the (optional) sensor pose with respect to the vehicle frame */
    const boost::optional<Pose3>& body_P_sensor() const {
        return body_P_sensor_;
    }

};



#endif //SRC_GTSAMFACTORHELPERS_H
