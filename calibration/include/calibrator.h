//opencv
#include <opencv/cv.h>

//AprilTags
#include "AprilTags/TagDetector.h"
#include "AprilTags/Tag36h11.h"

//Eigen
#include <Eigen/Dense>


class Calibrator {

    float fx_, fy_, cx_, cy_;
    int nmarkers_;

    Eigen::Matrix4f doCalibration(const std::vector<Eigen::Vector3f> &robot_points, const std::vector<Eigen::Vector3f> &tag_points);

public:

    Calibrator():fx_(575), fy_(575), cx_(319.5), cy_(239.5), nmarkers_(24) {}

    void calibrate(const cv::Mat &rgb, const cv::Mat &depth, const string &calib_file, const string &output_file);

    void setIntrinsics(float fx, float fy, float cx, float cy){
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
    }

    void setMarkers(int n){
        nmarkers_ = n;
    }

};
