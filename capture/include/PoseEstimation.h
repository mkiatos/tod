#ifndef _POSE_ESTIMATION_
#define _POSE_ESTIMATION_


#include <iostream>

#include <opencv2/opencv.hpp>
#include <vector>

#include <eigen3/Eigen/Geometry>

#include <apriltag.h>
#include <tag36h11.h>

#include "PinholeCamera.h"

class AprilTagDetector{
public:
    struct AprilTagParameters{
        cv::Size boardSize_;
        float tagSize_;
        float tagBorder_;

        AprilTagParameters()
        {
            boardSize_ = cv::Size(7, 10);
            tagSize_ = 0.04;
            tagBorder_ = 0.01;
        }
    };

    struct Result{
        uint64_t id;
        cv::Point2f pts[4];
    };

    AprilTagDetector();

    void detect(const cv::Mat &img,std::vector<AprilTagDetector::Result> &results);

    bool findPoints(std::vector<cv::Point2f> &pts, std::vector<cv::Point3f> &objs,
                    cv::Mat &img, const AprilTagParameters &params);
private:
    apriltag_detector* td_;
};


class ChessBoardDetector{
public:
    struct ChessBoardParamaters{
        cv::Size boardSize_;
        double squareSize_;

        ChessBoardParamaters()
        {
            boardSize_ = cv::Size(9, 6);
            squareSize_ = 0.025;//2.5cm
        }

    };

    ChessBoardDetector() {}

    bool findPoints(std::vector<cv::Point2f> &pts, std::vector<cv::Point3f> &objs,
                    const cv::Mat &img, const ChessBoardParamaters &params);
};



class PoseEstimator{

public:
    PoseEstimator() {}

    Eigen::Affine3d estimate(std::vector<cv::Point2f> &pts, std::vector<cv::Point3f> &objs,
                             const PinholeCamera &camera);

};


#endif
