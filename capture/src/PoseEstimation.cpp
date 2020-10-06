#include "PoseEstimation.h"

AprilTagDetector::AprilTagDetector()
{
    apriltag_family_t* tf = tag36h11_create();

    td_ = apriltag_detector_create();
    apriltag_detector_add_family(td_, tf);

    td_->debug = 0 ;
    td_->refine_pose = 1 ;
    td_->nthreads = 4 ;
}


void AprilTagDetector::detect(const cv::Mat &img, std::vector<AprilTagDetector::Result> &results)
{
    cv::Mat gray;
    //!< convert to gray scale
    if(img.type() == CV_8UC1) gray = img;
    else cv::cvtColor(img, gray, CV_BGR2GRAY);

    //!< convert to uint8
    //!< https://msdn.microsoft.com/en-us/library/windows/desktop/aa473780%28v=vs.85%29.aspx
    image_u8_t *im_u8 = image_u8_create(img.cols, img.rows);
    uint8_t *dst = im_u8->buf;

    for(int y = 0; y < im_u8->height; y++)
    {
        memcpy(dst, gray.ptr(y), im_u8->width);
        dst = dst + im_u8->stride;
    }

    //!< detect tags from an image
    zarray_t* detections = apriltag_detector_detect(td_, im_u8);

    for(int i = 0; i < zarray_size(detections); i++)
    {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        Result res;
        res.id = det->id;
        for(int j = 0; j < 4; j++)
        {
            res.pts[j].x = det->p[j][0];
            res.pts[j].y = det->p[j][1];
        }
        results.push_back(res);

        apriltag_detection_destroy(det);
    }

    zarray_destroy(detections);
    image_u8_destroy(im_u8);
}


bool AprilTagDetector::findPoints(std::vector<cv::Point2f> &pts, std::vector<cv::Point3f> &objs,
                                  cv::Mat &img, const AprilTagParameters &params)
{
    std::vector<AprilTagDetector::Result> results;
    detect(img,results);

    if(results.size() == 0)
    {
        return false;
    }

    for(int i = 0; i < results.size(); i++)
    {
        Result &res = results[i];

        //!<store 2d points
        for(int j = 0; j < 4; j++)
        {
            pts.push_back(res.pts[j]);
        }

        //!< store 3d points
        int row = res.id / params.boardSize_.width;
        int col = res.id % params.boardSize_.width;

        float x = col * (params.tagSize_ + params.tagBorder_);
        float y = row * (params.tagSize_ + params.tagBorder_);

        //!< counter-clockwise winding order
        objs.push_back(cv::Point3f(x, y, 0.0));
        objs.push_back(cv::Point3f(x + params.tagSize_, y, 0.0));
        objs.push_back(cv::Point3f(x + params.tagSize_ , y + params.tagSize_, 0.0));
        objs.push_back(cv::Point3f(x , y + params.tagSize_, 0.0));
    }

    return true;
}


bool ChessBoardDetector::findPoints(std::vector<cv::Point2f> &pts, std::vector<cv::Point3f> &objs,
                                    const cv::Mat &img, const ChessBoardParamaters &params)
{
    cv::Mat gray, tmpMat;
    tmpMat = img;

    cv::cvtColor(tmpMat, gray, CV_BGR2GRAY);

    bool found = cv::findChessboardCorners(gray, params.boardSize_, pts,
                                      CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

    if (found == 0 && (pts.size() != 0))
    {
        cv::cornerSubPix(gray, pts, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    }

    //cv::drawChessboardCorners(tmpMat, params.boardSize_, cv::Mat(corners_),found);

    for(int i = 0; i < params.boardSize_.height; i++)
    {
        for(int j = 0; j < params.boardSize_.width; j++)
        {
            objs.push_back(cv::Point3f(float(i * params.squareSize_), float(j * params.squareSize_), 0));
        }
    }

    return found;
}


Eigen::Affine3d PoseEstimator::estimate(std::vector<cv::Point2f> &pts, std::vector<cv::Point3f> &objs,
                                        const PinholeCamera &camera)
{   
    cv::Mat rVec = cv::Mat(cv::Size(3,1), CV_64F);
    cv::Mat tVec = cv::Mat(cv::Size(3,1), CV_64F);

    cv::solvePnPRansac(cv::Mat(objs).reshape(1), cv::Mat(pts).reshape(1), camera.getMatrix(),
                       camera.getDistortionCoeffs(), rVec, tVec);

    cv::Mat R = cv::Mat(cv::Size(3,3), CV_64F);
    cv::Rodrigues(rVec, R);

    Eigen::Matrix3d r;
    Eigen::Vector3d t;
    for(int i = 0; i < 3 ; i++)
    {
        for(int j = 0; j < 3 ; j++)
        {
            r(i,j) = R.at<double>(i,j);
        }
        t(i) = tVec.at<double>(0,i);
    }
    Eigen::Affine3d tr = Eigen::Translation3d(t) * r;

    return tr;
}
