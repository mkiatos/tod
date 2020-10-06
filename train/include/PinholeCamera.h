#ifndef _PINHOLE_CAMERA_H_
#define _PINHOLE_CAMERA_H_

#include <iostream>

#include <opencv2/opencv.hpp>


class PinholeCamera{
	private:
		double fx_, fy_, cx_, cy_;
		cv::Mat dist_;
		cv::Size size_;

	public:
		PinholeCamera(double fx, double fy, double cx, double cy, cv::Size size)
		{
			fx_ = fx;
			fy_ = fy;
			cx_ = cx;
			cy_ = cy;
			size_ = size;
		}
		
		
		cv::Mat getMatrix() const
		{
			cv::Mat camMatrix = cv::Mat::zeros(3, 3, CV_64FC1);
            camMatrix.at<double>(0, 0) = fx_; camMatrix.at<double>(1, 1) = fy_;
            camMatrix.at<double>(0, 2) = cx_; camMatrix.at<double>(1, 2) = cy_;
            camMatrix.at<double>(2, 2) = 1.0;
			
			return camMatrix;
		}
		
		
		cv::Mat getDistortionCoeffs() const
		{
            //dist_ = cv::Mat::zeros(1, 8, CV_32F);
			
			return dist_;
		}
		
		
		cv::Point3f backProject(cv::Point2d point, float depthValue) const
		{
		    float z = float(depthValue) / 1000.0f;
		    float x = (point.x - cx_) * z / fx_;
		    float y = (point.y - cy_) * z / fy_;
		    cv::Point3f bp(x,y,z);
		
		    return bp;
		}
		
		
		double fx() const {return fx_;}
		double fy() const {return fy_;}
		double cx() const {return cx_;}
		double cy() const {return cy_;}
        int height() const {return size_.height;}
        int width() const {return size_.width;}
        cv::Size size() const {return size_;}
};

#endif
